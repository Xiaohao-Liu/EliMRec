import torch
from data.dataset import Dataset
from torch import nn
import numpy as np
import scipy.sparse as sp
import time
from models import BasicModel
from util.logger import Logger
from util.mlp import MLP

from torch_scatter import scatter
from sklearn.cluster import KMeans

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


class Clickbait(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: Dataset):
        super(Clickbait, self).__init__(dataset, config)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.temp = self.config["temp"]  # infonce temperature
        # ['cosin' (default), 'inner_product']
        self.alpha = self.config['alpha']
        self.mm_fusion_mode = 'mean'
        
        # init
        self.create_u_embeding_i()

        self.all_items = self.all_users = None

        coo = self.create_adj_mat(self.config['adj_type']).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse.FloatTensor(
            indices, torch.FloatTensor(coo.data), coo.shape)
        self.norm_adj = self.norm_adj.to(self.config.device)
        self.f = nn.Sigmoid()
        
        

    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)

        pre_user_tensor = self.pre_embedding[:self.num_users]
        pre_item_tensor = self.pre_embedding[self.num_users:]
        post_user_tensor = self.post_embedding[:self.num_users]
        post_item_tensor = self.post_embedding[self.num_users:]
        
        temp_pre_user_tensor = pre_user_tensor[user_ids]
        temp_post_user_tensor = post_user_tensor[user_ids]
        
        pre_score_matrix = torch.matmul(temp_pre_user_tensor, pre_item_tensor.t())
        post_score_matrix = torch.matmul(temp_post_user_tensor, post_item_tensor.t())
        if self.predict_type == "TE":
            return (post_score_matrix*torch.sigmoid(pre_score_matrix)).detach().cpu()
        elif self.predict_type == "TIE":
            score_matrix = (post_score_matrix - torch.mean(post_score_matrix, 1, True))*torch.sigmoid(pre_score_matrix)
            return score_matrix.detach().cpu()
        elif self.predict_type == "NIE":
            return post_score_matrix.detach().cpu()
            
        return post_score_matrix.detach().cpu()
    
    def bpr_loss(self, users, pos_items, neg_items):
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        
        v_rep = self.v_emb
        a_rep = self.a_emb
        t_rep = self.t_emb
        
        # pre_interaction_score
        pre_representation = t_rep
        self.pre_embedding = pre_representation
        pre_user_tensor = pre_representation[users]
        pre_pos_item_tensor = pre_representation[self.num_users + pos_items]
        pre_neg_item_tensor = pre_representation[self.num_users + neg_items]
        pre_pos_scores = torch.sum(pre_user_tensor*pre_pos_item_tensor, dim=1)
        pre_neg_scores = torch.sum(pre_user_tensor*pre_neg_item_tensor, dim=1)

        # post_interaction_score
        post_representation = (v_rep+a_rep+t_rep)/3
        self.post_embedding = post_representation
        post_user_tensor = post_representation[users]
        post_pos_item_tensor = post_representation[self.num_users + pos_items]
        post_neg_item_tensor = post_representation[self.num_users + neg_items]
        post_pos_scores = torch.sum(post_user_tensor*post_pos_item_tensor, dim=1)
        post_neg_scores = torch.sum(post_user_tensor*post_neg_item_tensor, dim=1)
        
        # fusion of pre_ and post_ interaction scores
        # # post*sigmoid(pre)
        pos_scores = post_pos_scores*torch.sigmoid(pre_pos_scores)
        neg_scores = post_neg_scores*torch.sigmoid(pre_neg_scores)
        
#         loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)+eps))
        loss_value = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
#         loss_value_pre = -torch.mean(torch.log2(torch.sigmoid(pre_pos_scores - pre_neg_scores)+eps))
        loss_value_pre = torch.mean(torch.nn.functional.softplus(pre_neg_scores - pre_pos_scores))
        
        return loss_value + self.alpha * loss_value_pre
    
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
        self.i_emb_u, self.i_emb_i = torch.split(
            self.i_emb, [self.num_users, self.num_items])

        self.v_emb = compute_graph(users_emb, self.v_dense_emb)
        self.v_emb_u, self.v_emb_i = torch.split(
            self.v_emb, [self.num_users, self.num_items])
        if self.config["data.input.dataset"] != "kwai":
            self.a_emb = compute_graph(users_emb, self.a_dense_emb)
            self.t_emb = compute_graph(users_emb, self.t_dense_emb)

            self.a_emb_u, self.a_emb_i = torch.split(
                self.a_emb, [self.num_users, self.num_items])
            self.t_emb_u, self.t_emb_i = torch.split(
                self.t_emb, [self.num_users, self.num_items])

        # multi - modal features fusion
        if self.config["data.input.dataset"] == "kwai":
            users = self.embedding_user_after_GCN(
                self.mm_fusion([self.i_emb_u, self.v_emb_u]))
            items = self.embedding_item_after_GCN(
                self.mm_fusion([self.i_emb_i, self.v_emb_i]))
        else:
            users = self.embedding_user_after_GCN(self.mm_fusion(
                [self.i_emb_u, self.v_emb_u, self.a_emb_u, self.t_emb_u]))
            items = self.embedding_item_after_GCN(self.mm_fusion(
                [self.i_emb_i, self.v_emb_i, self.a_emb_i, self.t_emb_i]))
        return users, items

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

    def original_infonce(self, users_emb, pos_emb):
        users_emb = torch.nn.functional.normalize(users_emb, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=1)
        logits = torch.matmul(users_emb, pos_emb.t())
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(
            self.config.device)
        return self.infonce_criterion(logits, labels)
    
    def original_bpr_loss(self, u_emb, p_emb, n_emb):
        u_emb = torch.nn.functional.normalize(u_emb, dim=1)
        p_emb = torch.nn.functional.normalize(p_emb, dim=1)
        n_emb = torch.nn.functional.normalize(n_emb, dim=1)
        p_scores = torch.sum(u_emb * p_emb, dim=1)
        n_scores = torch.sum(u_emb * n_emb, dim=1)
        return torch.mean(torch.nn.functional.softplus(n_scores - p_scores))

    def forward(self, users, items):
        # compute embedding

        all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma.detach()

    def create_user_mono_modal_preference_matrix(self):
        # generate matrix
        start_time = time.time()
        user_train_dict = self.dataset.get_user_train_dict()

        def output(user_id):
            if not user_id in user_train_dict:
                return 1, 1, 0
            user_train = user_train_dict[user_id]
            return torch.var(self.v_feat[user_train]), torch.var(self.a_feat[user_train]), torch.var(self.t_feat[user_train])

        user_p_matrix = torch.zeros(
            (self.dataset.num_users, 3)).to(self.config.device)

        for i in range(0, self.dataset.num_users):
            v, a, t = output(i)
            user_p_matrix[i][torch.argmin(torch.tensor(
                [v, a, t]).to(self.config.device))] = 1
        Logger.info("generate user_mono_modal_preference_matrix: ",
                    time.time() - start_time)
        return torch.unsqueeze(user_p_matrix, 2)

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

        self.v_feat = torch.nn.functional.normalize(
            self.dataset.v_feat.to(self.config.device).float(), dim=1)
        if self.config["data.input.dataset"] != "kwai":
            self.a_feat = torch.nn.functional.normalize(
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
                self.t_feat = torch.nn.functional.normalize(
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

        self.embedding_item_after_GCN = nn.Linear(
            self.item_feat_dim, self.latent_dim)
        self.embedding_user_after_GCN = nn.Linear(
            self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)
