import torch
from data.dataset import Dataset
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy.sparse as sp
import time
from models import BasicModel
from util.logger import Logger

from torch_scatter import scatter

eps = 1e-12
# no backpropagation

class GradMulConst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)


class EliMRec(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: Dataset):
        super(EliMRec, self).__init__(dataset, config)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.temp = self.config["temp"]  # infonce temperature
        # ['cosin' (default), 'inner_product']
        self.logits = self.config["logits"]

        # predict type
        # ['NIE', 'TIE', 'normal']
        self.predict_type = self.config["predict_type"] if 'predict_type' in self.config else 'TIE'
        Logger.info('predict type: ' + self.predict_type)

        # multi - modal fusion
        self.mm_fusion_mode = self.config['mm_fusion_mode'] if 'mm_fusion_mode' in self.config else 'concat'
        Logger.info('mm fusion mode: ' + self.mm_fusion_mode)

        # user preference in single modality
        self.is_u_s = self.config['is_u_s'] if 'is_u_s' in self.config else False
        Logger.info('cf in single modal preference: ' + str(self.is_u_s))

        # fusion of m - m score and s - m score
        # ['rubi', 'hm', 'sum']
        self.fusion_mode = self.config["s_fusion_mode"] if 's_fusion_mode' in self.config else 'rubi'
        Logger.info('score fusion mode: ' + self.fusion_mode)

        # alpha of single modal loss
        Logger.info('alpha: ' + str(self.config.alpha))

        # ui_score ( * us_score ?)
        self.modified_ui_loss = self.config['modified_ui_loss'] if 'modified_ui_loss' in self.config else True
        Logger.info('modified_ui_loss: ' + str(self.modified_ui_loss))

        # learnable cf-mode
        self.modality = self.config['modality'] if 'modality' in self.config else 'vat' #v, a, t, va, vt, at
        Logger.info('Modality Ablation: ' + str(self.modality))
        # init
        self.create_u_embeding_i()

        self.all_items = self.all_users = None

        coo = self.create_adj_mat(self.config['adj_type']).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse.FloatTensor(
            indices, torch.FloatTensor(coo.data), coo.shape)
        self.norm_adj = self.norm_adj.to(self.config.device)
        self.f = nn.Sigmoid()
        
        # MLP
        self.s_dense_v = nn.Linear(self.latent_dim,self.latent_dim)
        self.s_dense_a = nn.Linear(self.latent_dim,self.latent_dim)
        self.s_dense_t = nn.Linear(self.latent_dim,self.latent_dim)
        nn.init.xavier_uniform_(self.s_dense_v.weight)
        nn.init.xavier_uniform_(self.s_dense_a.weight)
        nn.init.xavier_uniform_(self.s_dense_t.weight)
        

    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        items_emb = self.all_items
        # normal
        ui_score = self.f(torch.matmul(users_emb, items_emb.t()))
        
        if self.predict_type == "TE":
            TE = self.general_cm_fusion(ui_score, self.all_s_embs, users,normalize=True)
            return self.f(TE).detach().cpu()
        elif self.predict_type == "TIE":
            u_fixed_i_s = torch.mean(ui_score, -1, True)

            TE = self.general_cm_fusion(ui_score, self.all_s_embs, users,normalize=True)
            NDE = self.general_cm_fusion(u_fixed_i_s, self.all_s_embs, users,normalize=True)
            return self.f(TE-NDE).detach().cpu()
        
        return self.f(ui_score).detach().cpu()
    
    def bpr_loss(self, users, pos_items, neg_items):
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        
        self.all_s_embs = self.gcn_cf()
        
        if self.predict_type == "normal":
            return self.original_bpr_loss(users_emb, pos_emb, neg_emb)

        # multi-modal fusion loss:
        fusion_loss = self.original_bpr_loss(users_emb, pos_emb, neg_emb)

        # single-modal preference loss:
        p_loss = 0
        if self.config['data.input.dataset'] == "kwai":
            self.modality = 'v'
            
        for i in self.modality:
            users_emb_s = self.all_s_embs['pre_fusion_user_' + i][users]
            pos_emb_s = self.all_s_embs['pre_fusion_item_' + i][pos_items]
            neg_emb_s = self.all_s_embs['pre_fusion_item_' + i][neg_items]
            p_loss += self.original_bpr_loss(users_emb_s, pos_emb_s, neg_emb_s)

        return fusion_loss + self.config.alpha * p_loss

    def gcn_cf(self, detach=False):
        all_s_embs = {}
        v_emb = self.s_dense_v(self.v_emb) if not detach else self.s_dense_v(self.v_emb).clone().detach()
        all_s_embs['pre_fusion_user_v'],all_s_embs['pre_fusion_item_v']=  torch.split(v_emb, [self.num_users, self.num_items])
        if self.config['data.input.dataset'] != "kwai":
            a_emb = self.s_dense_a(self.a_emb) if not detach else self.s_dense_a(self.a_emb).clone().detach()
            all_s_embs['pre_fusion_user_a'],all_s_embs['pre_fusion_item_a']=  torch.split(a_emb, [self.num_users, self.num_items])
            t_emb = self.s_dense_t(self.t_emb) if not detach else self.s_dense_t(self.t_emb).clone().detach()
            all_s_embs['pre_fusion_user_t'],all_s_embs['pre_fusion_item_t']=  torch.split(t_emb, [self.num_users, self.num_items])
        return all_s_embs

    def general_cm_fusion(self, fusion_logits, all_s_embs, users, items=None, normalize=True):
        s_u_v = all_s_embs['pre_fusion_user_v'][users]
        s_i_v = all_s_embs['pre_fusion_item_v'][items] if items is not None else all_s_embs['pre_fusion_item_v']
        if self.config['data.input.dataset'] != "kwai":
            s_u_a = all_s_embs['pre_fusion_user_a'][users]
            s_i_a = all_s_embs['pre_fusion_item_a'][items] if items is not None else all_s_embs['pre_fusion_item_a']
            s_u_t = all_s_embs['pre_fusion_user_t'][users]
            s_i_t = all_s_embs['pre_fusion_item_t'][items] if items is not None else all_s_embs['pre_fusion_item_t']
        if normalize:
            s_u_v = F.normalize(s_u_v, dim=1)
            s_i_v = F.normalize(s_i_v, dim=1)
            if self.config['data.input.dataset'] != "kwai":
                s_u_a = F.normalize(s_u_a, dim=1)
                s_i_a = F.normalize(s_i_a, dim=1)
                s_u_t = F.normalize(s_u_t, dim=1)
                s_i_t = F.normalize(s_i_t, dim=1)
        if self.fusion_mode == 'rubi':
            if 'v' in self.modality:
                z_v = torch.sigmoid(torch.matmul(s_u_v, s_i_v.t()))
            else:
                z_v = 1
            if self.config['data.input.dataset'] != "kwai":
                if 'a' in self.modality:
                    z_a = torch.sigmoid(torch.matmul(s_u_a, s_i_a.t()))
                else:
                    z_a = 1
                if 't' in self.modality:
                    z_t = torch.sigmoid(torch.matmul(s_u_t, s_i_t.t()))
                else:
                    z_t = 1

                z = fusion_logits * z_v * z_a * z_t
            else:
                z = fusion_logits * z_v

        elif self.fusion_mode == 'hm':
            z_v = torch.sigmoid(torch.matmul(s_u_v, s_i_v.t()))
            if self.config['data.input.dataset'] != "kwai":
                z_a = torch.sigmoid(torch.matmul(s_u_a, s_i_a.t()))
                z_t = torch.sigmoid(torch.matmul(s_u_t, s_i_t.t()))

                z = torch.sigmoid(fusion_logits) * z_v * z_a * z_t
            else:
                z = torch.sigmoid(fusion_logits) * z_v
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z_v = torch.matmul(s_u_v, s_i_v.t())
            if self.config['data.input.dataset'] != "kwai":
                z_a = torch.matmul(s_u_a, s_i_a.t())
                z_t = torch.matmul(s_u_t, s_i_t.t())

                z = fusion_logits + z_v + z_a + z_t
            else:
                z = fusion_logits + z_v
            z = torch.log(torch.sigmoid(z) + eps)

        return z

    
    def matmul(self, a, b):
        return torch.matmul(a, b.t())

    def mul(self, a, b):
        return torch.sum(a * b, dim=1)
    
    def mm_fusion(self, reps: list):
        if self.mm_fusion_mode == "concat":
            z = torch.cat(reps, dim=1)
        elif self.mm_fusion_mode == "mean":
            z = torch.mean(torch.stack(reps), dim=0)
        return z

    def compute(self):
            # shape: user_size x embeding_size
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        self.v_dense_emb = self.v_dense(self.v_feat)  # v = >id
        if self.config["data.input.dataset"] != "kwai":
            self.a_dense_emb = self.a_dense(self.a_feat)  # a = >id
            self.t_dense_emb = self.t_dense(self.t_feat)  # t = >id

        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            # Light Graph Convolution
            g_droped = self.norm_adj
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out

        self.i_emb = compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.num_users, self.num_items])
        self.v_emb = compute_graph(users_emb, self.v_dense_emb)
        self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.num_users, self.num_items])
        if self.config["data.input.dataset"] != "kwai":
            self.a_emb = compute_graph(users_emb, self.a_dense_emb)
            self.t_emb = compute_graph(users_emb, self.t_dense_emb)
            self.a_emb_u, self.a_emb_i = torch.split(self.a_emb, [self.num_users, self.num_items])
            self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.num_users, self.num_items])

        # multi - modal features fusion
        if self.config["data.input.dataset"] == "kwai":
            user = self.embedding_user_after_GCN(
                self.mm_fusion([self.i_emb_u, self.v_emb_u]))
            item = self.embedding_item_after_GCN(
                self.mm_fusion([self.i_emb_i, self.v_emb_i]))
        else:
            user = self.embedding_user_after_GCN(self.mm_fusion(
                [self.i_emb_u, self.v_emb_u, self.a_emb_u, self.t_emb_u]))
            item = self.embedding_item_after_GCN(self.mm_fusion(
                [self.i_emb_i, self.v_emb_i, self.a_emb_i, self.t_emb_i]))
        self.final_emb = torch.concat([user, item], dim=0)
        return user, item

    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users, self.all_items = self.compute()

        users_emb = self.all_users[users]
        pos_emb = self.all_items[pos_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)

        if neg_items is None:
            neg_emb_ego = neg_emb = None
        else:
            neg_emb = self.all_items[neg_items]
            neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def original_bpr_loss(self, u_emb, p_emb, n_emb):
        u_emb = F.normalize(u_emb, dim=1)
        p_emb = F.normalize(p_emb, dim=1)
        n_emb = F.normalize(n_emb, dim=1)
        p_scores = torch.sum(u_emb * p_emb, dim=1)
        n_scores = torch.sum(u_emb * n_emb, dim=1)
        return torch.mean(F.softplus(n_scores - p_scores))

    def forward(self, users, items):
        # compute embedding

        all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma.detach()

    def create_adj_mat(self, adj_type):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, - 1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single - normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(
                adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, - 0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def create_u_embeding_i(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        Logger.info('[use Xavier initilizer]')

        self.v_feat = F.normalize(
            self.dataset.v_feat.to(self.config.device).float(), dim=1)
        if self.config["data.input.dataset"] != "kwai":
            self.a_feat = F.normalize(
                self.dataset.a_feat.to(self.config.device).float(), dim=1)
            if self.config["data.input.dataset"] == "tiktok":
                self.words_tensor = self.dataset.words_tensor.to(
                    self.config.device)
                self.word_embedding = torch.nn.Embedding(
                    11574, 128).to(self.config.device)
                torch.nn.init.xavier_normal_(self.word_embedding.weight)
                self.t_feat = scatter(self.word_embedding(
                    self.words_tensor[1]), self.words_tensor[0], reduce='mean', dim=0).to(self.config.device)
            else:
                self.t_feat = F.normalize(
                    self.dataset.t_feat.to(self.config.device).float(), dim=1)

        # visual feature dense
        self.v_dense = nn.Linear(self.v_feat.shape[1], self.latent_dim)
        if self.config["data.input.dataset"] != "kwai":
            # acoustic feature dense
            self.a_dense = nn.Linear(self.a_feat.shape[1], self.latent_dim)
            # textual feature dense
            self.t_dense = nn.Linear(self.t_feat.shape[1], self.latent_dim)

        if self.mm_fusion_mode == "concat":
            self.item_feat_dim = self.latent_dim * \
                4 if self.config["data.input.dataset"] != "kwai" else self.latent_dim * 2
        elif self.mm_fusion_mode == "mean":
            self.item_feat_dim = self.latent_dim

        nn.init.xavier_uniform_(self.v_dense.weight)
        if self.config["data.input.dataset"] != "kwai":
            nn.init.xavier_uniform_(self.a_dense.weight)
            nn.init.xavier_uniform_(self.t_dense.weight)

        self.embedding_user_after_GCN = nn.Linear(
            self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)
        self.embedding_item_after_GCN = nn.Linear(
            self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
