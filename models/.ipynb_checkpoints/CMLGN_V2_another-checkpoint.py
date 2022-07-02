import torch
from data.dataset import Dataset
from torch import nn
import numpy as np
import scipy.sparse as sp

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


class CMLGN_V2(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(CMLGN_V2, self).__init__(dataset,config)
        self.__init_weight()
    
    def __init_weight(self):
        self.num_users  = self.dataset.num_users
        self.num_items  = self.dataset.num_items
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.temp = self.config["temp"] # infonce temperature
        self.logits = "cosin" # ['cosin', 'inner_product']
        self.predict_type = self.config["predict_type"] if  'predict_type' in self.config else 'NIE' # ['NIE', 'TIE']
        self.fusion_mode = self.config["fusion_mode"] if  'fusion_mode' in self.config else 'rubi' # ['rubi', 'hm','sum']
        Logger.info('fusion mode: ' + self.fusion_mode)
        Logger.info('predict type: ' + self.predict_type)
        Logger.info('alpha: ' + str( self.config.alpha))
        
#         if self.predict_type == "TIE":
#             self.c = torch.nn.Embedding(
#                 num_embeddings=1, embedding_dim=self.latent_dim)
#             nn.init.xavier_uniform_(self.c.weight)
            
        # init
        self.create_u_embeding_i()

        self.all_items = self.all_users = None
        
        coo = self.create_adj_mat(self.config['adj_type']).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse.FloatTensor(indices, torch.FloatTensor(coo.data), coo.shape)
        self.norm_adj = self.norm_adj.to(self.config.device)
        self.f = nn.Sigmoid()
        if 'rubi2' in self.config or 'rubi3' in self.config:
            v_desnse_s_shape =a_desnse_s_shape =t_desnse_s_shape =  self.latent_dim
        else:
            v_desnse_s_shape = self.v_feat.shape[1]
            if self.config["data.input.dataset"] != "kwai":
                a_desnse_s_shape = self.a_feat.shape[1]
                t_desnse_s_shape = self.t_feat.shape[1]
        
        # MLP(input_dim, dimensions,activation='relu', dropout=0.)
        self.v_dense_s = MLP(v_desnse_s_shape, [self.latent_dim for _ in range(3)])
        if self.config["data.input.dataset"] != "kwai":
            # acoustic feature dense
            self.a_dense_s = MLP(a_desnse_s_shape, [self.latent_dim for _ in range(3)])
            # textual feature dense
            self.t_dense_s = MLP(t_desnse_s_shape, [self.latent_dim for _ in range(3)])
        self.v_dense_s.init_weight('xavier')
        if self.config["data.input.dataset"] != "kwai":
            self.a_dense_s.init_weight('xavier')
            self.t_dense_s.init_weight('xavier')
                
    def compute(self):
        """
        
        """
        # shape: user_size x embeding_size       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        self.v_dense_emb = self.v_dense(self.v_feat)# v=>id
        if self.config["data.input.dataset"] != "kwai":
            self.a_dense_emb = self.a_dense(self.a_feat)# a=>id
            self.t_dense_emb = self.t_dense(self.t_feat)# t=>id
        
        def compute_graph(u_emb,i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            # Light Graph Convolution
            g_droped = self.norm_adj
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
#             users, items = torch.split(light_out, [self.num_users, self.num_items])
            return light_out
        
        self.v_emb = compute_graph(users_emb,self.v_dense_emb)
        self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.num_users, self.num_items])
        if self.config["data.input.dataset"] != "kwai":
            self.a_emb = compute_graph(users_emb,self.a_dense_emb)
            self.t_emb = compute_graph(users_emb,self.t_dense_emb)
        
            self.a_emb_u, self.a_emb_i = torch.split(self.a_emb, [self.num_users, self.num_items])
            self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.num_users, self.num_items])
        
        self.i_emb = compute_graph(users_emb,items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.num_users, self.num_items])
        
        # multi-modal features fusion
        if self.config["data.input.dataset"] == "kwai":
            users = self.embedding_user_after_GCN( torch.cat([self.i_emb_u,self.v_emb_u], dim=1) )
            items = self.embedding_item_after_GCN( torch.cat([self.i_emb_i,self.v_emb_i], dim=1) )
        else:
            users = self.embedding_user_after_GCN( torch.cat([self.i_emb_u,self.v_emb_u,self.a_emb_u,self.t_emb_u], dim=1))
            items = self.embedding_item_after_GCN( torch.cat([self.i_emb_i,self.v_emb_i,self.a_emb_i,self.t_emb_i], dim=1))
        return users, items
 
    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        if candidate_items is None:
            items_emb = self.all_items
        else:
            items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
            
        ui_score = torch.matmul(users_emb, items_emb.t())
        if self.predict_type == "TIE":
            s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
            y_u_fixed_i_s = torch.matmul(users_emb, torch.unsqueeze(torch.mean(items_emb,dim=0).t(),dim=1))
#             print(ui_score.shape, y_u_fixed_i_s.shape)
            score = self.fusion((ui_score - y_u_fixed_i_s) ,torch.matmul(users_emb, s_emb.t()))
            return self.f(score).detach().cpu()
        elif self.predict_type == 'original':
            s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
            score = self.fusion(ui_score ,torch.matmul(users_emb, s_emb.t()))
            return self.f(score).detach().cpu()
        elif self.predict_type == 'unimodal':
            s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
            score = torch.matmul(users_emb, s_emb.t())
            return self.f(score).detach().cpu()

        return self.f(ui_score).detach().cpu()
        
        
        # - torch.matmul(users_emb, torch.mean(items_emb,dim=0).repeat(items_emb.shape[0]).reshape(items_emb.shape).t())
        
        scores = ui_score #*self.f(torch.matmul(users_emb, s_emb.t()))
#         print(scores)
        return self.f(scores).detach().cpu()
        
    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users, self.all_items = self.compute()
        users_emb = self.all_users[users]
        pos_emb = self.all_items[pos_items]
        
        users_emb_ego = self.embedding_user(users)
        if self.config["default_encoder"] == "embedding":
            pos_emb_ego = self.embedding_item(pos_items)
        elif self.config["default_encoder"] == "only_mm":
            pos_emb_ego = None
        else:
            pos_emb_ego = self.embedding_item(pos_items)
        if neg_items is None:
            neg_emb_ego = neg_emb =None
        else:
            neg_emb = self.all_items[neg_items]
            if self.config["default_encoder"] == "embedding":
                neg_emb_ego = self.embedding_item(neg_items)
            elif self.config["default_encoder"] == "only_mm":
                neg_emb_ego = None
            else:
                neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def original_bpr_loss(self,u_emb,p_emb,n_emb):
        u_emb=torch.nn.functional.normalize(u_emb,dim=1)
        p_emb=torch.nn.functional.normalize(p_emb,dim=1)
        n_emb=torch.nn.functional.normalize(n_emb,dim=1)
        p_scores = torch.mul(u_emb, p_emb)
        p_scores = torch.sum(p_scores, dim=1)
        n_scores = torch.mul(u_emb, n_emb)
        n_scores = torch.sum(n_scores, dim=1)
        return torch.mean(torch.nn.functional.softplus(n_scores - p_scores))
    
    def bpr_loss(self,users,pos_items,neg_items):
        users = users.long()
        pos_items =pos_items.long()
        neg_items = neg_items.long()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        users_emb = self.all_users[users]# self.user_content_feature(users)
        pos_emb_s = []
        if neg_items is None:
            neg_emb =None
        else:
            neg_emb = self.all_items[neg_items]
        
        # RUBI mode
        if 'rubi' in self.config or 'rubi2' in self.config or 'rubi3' in self.config:
            if 'rubi' in self.config and self.config['rubi']:
                self.all_s_embs = []
                if 'v' in self.config['rubi']:
                    self.all_s_embs.append(self.v_dense_s(self.v_feat))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi']:
                        self.all_s_embs.append(self.a_dense_s(self.a_feat))
                    if 't' in self.config['rubi']:
                        self.all_s_embs.append(self.t_dense_s(self.t_feat))
            if 'rubi2' in self.config and self.config['rubi2']:
                self.all_s_embs = []
                if 'v' in self.config['rubi2']:
                    self.all_s_embs.append(self.v_dense_s(self.v_emb_i))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi2']:
                        self.all_s_embs.append(self.a_dense_s(self.a_emb_i))
                    if 't' in self.config['rubi2']:
                        self.all_s_embs.append(self.t_dense_s(self.t_emb_i))
            if 'rubi3' in self.config and self.config['rubi3']:
                self.all_s_embs = []
                if 'v' in self.config['rubi3']:
                    self.all_s_embs.append(self.v_dense_s(grad_mul_const(self.v_emb_i, 0.0)))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi3']:
                        self.all_s_embs.append(self.a_dense_s(grad_mul_const(self.a_emb_i, 0.0)))
                    if 't' in self.config['rubi3']:
                        self.all_s_embs.append(self.t_dense_s(grad_mul_const(self.t_emb_i, 0.0)))
        else: 
            return self.original_bpr_loss(users_emb,pos_emb,neg_emb)
        
        s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
        pos_emb_s = s_emb[pos_items]
        pos_emb = self.all_items[pos_items]
        if neg_items is None:
            neg_emb_s =None
        else:
            neg_emb_s = s_emb[neg_items]
        
        users_emb=torch.nn.functional.normalize(users_emb,dim=1)
        
        users_emb_s = grad_mul_const(users_emb, 0.0)
        
        pos_emb_s=torch.nn.functional.normalize(pos_emb_s,dim=1)
        neg_emb_s=torch.nn.functional.normalize(neg_emb_s,dim=1)
        pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        neg_emb=torch.nn.functional.normalize(neg_emb,dim=1)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1) 
        pos_scores_s = torch.sum(torch.mul(users_emb_s, pos_emb_s),dim=1)
        pos_scores = self.fusion(pos_scores,pos_scores_s)
        
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        neg_scores_s = torch.sum(torch.mul(users_emb_s, neg_emb_s),dim=1)
        neg_scores = self.fusion(neg_scores,neg_scores_s)
        main_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        print(main_loss)
        return main_loss + \
                self.config.alpha* torch.mean(torch.nn.functional.softplus(neg_scores_s - pos_scores_s)) #+ \
                #self.similarity(pos_emb_s,pos_emb) + self.similarity(neg_emb_s,neg_emb)
    
    def original_infonce(self, users_emb, pos_emb):
#         users_emb=torch.nn.functional.normalize(users_emb,dim=1)
#         pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        if self.logits == "inner_product":
            pass
        elif  self.logits == "cosin":
            users_emb=torch.nn.functional.normalize(users_emb,dim=1)
            pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        logits = torch.mm(users_emb,pos_emb.T)
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(self.config.device)

        return  self.infonce_criterion(logits, labels)# + reg_loss

    def infonce(self,users,pos_items):
        users = users.long()
        pos_items =pos_items.long()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users, pos_items, None)
        
        # RUBI mode
        if 'rubi' in self.config or 'rubi2' in self.config or 'rubi3' in self.config:
            if 'rubi' in self.config and self.config['rubi']:
                self.all_s_embs = []
                if 'v' in self.config['rubi']:
                    self.all_s_embs.append(self.v_dense_s(self.v_feat))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi']:
                        self.all_s_embs.append(self.a_dense_s(self.a_feat))
                    if 't' in self.config['rubi']:
                        self.all_s_embs.append(self.t_dense_s(self.t_feat))
            if 'rubi2' in self.config and self.config['rubi2']:
                self.all_s_embs = []
                if 'v' in self.config['rubi2']:
                    self.all_s_embs.append(self.v_dense_s(self.v_emb_i))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi2']:
                        self.all_s_embs.append(self.a_dense_s(self.a_emb_i))
                    if 't' in self.config['rubi2']:
                        self.all_s_embs.append(self.t_dense_s(self.t_emb_i))
            if 'rubi3' in self.config and self.config['rubi3']:
                self.all_s_embs = []
                if 'v' in self.config['rubi3']:
                    self.all_s_embs.append(self.v_dense_s(grad_mul_const(self.v_emb_i, 0.0)))
                if self.config['data.input.dataset']!="kwai":
                    if 'a' in self.config['rubi3']:
                        self.all_s_embs.append(self.a_dense_s(grad_mul_const(self.a_emb_i, 0.0)))
                    if 't' in self.config['rubi3']:
                        self.all_s_embs.append(self.t_dense_s(grad_mul_const(self.t_emb_i, 0.0)))
        else: 
            return self.original_infonce(users_emb,pos_emb)
        
        s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
        pos_emb_s = s_emb[pos_items]
        pos_emb = self.all_items[pos_items]

#         users_emb=torch.nn.functional.normalize(users_emb,dim=1)
#         pos_emb_s=torch.nn.functional.normalize(pos_emb_s,dim=1)
#         pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        
        if self.logits == "inner_product":
            pass
        elif  self.logits == "cosin":
            users_emb=torch.nn.functional.normalize(users_emb,dim=1)
            pos_emb_s=torch.nn.functional.normalize(pos_emb_s,dim=1)
            pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        
        users_emb_s = grad_mul_const(users_emb, 0.0)
        
        z_m = torch.mm(users_emb,pos_emb.T)
        z_s = torch.mm(users_emb_s, pos_emb_s.T)
        logits = self.fusion(z_m, z_s)
        logits /= self.temp
        labels = torch.tensor(list(range(users.shape[0]))).to(self.config.device)
        
        logits_s = z_s#torch.mm(users_emb, pos_emb_s.T)
        logits_s /= self.temp

        return self.infonce_criterion(logits, labels) +self.config.alpha* self.infonce_criterion(logits_s, labels)# + reg_loss
    
    def fusion(self, z_m, z_s):

        if self.fusion_mode == 'rubi':
            z = z_m * torch.sigmoid(z_s)

        elif self.fusion_mode == 'hm':
            z_m = torch.sigmoid(z_m)
            z_s = torch.sigmoid(z_s)
            z = z_m * z_s
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = z_m + z_s
            z = torch.log(torch.sigmoid(z) + eps)

        return z
    
    def forward(self, users, items):
        # compute embedding
        
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma.detach()
    
    def create_adj_mat(self, adj_type):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
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
        
        self.v_feat = torch.nn.functional.normalize(self.dataset.v_feat.to(self.config.device).float(),dim=1)
        if self.config["data.input.dataset"] != "kwai":
            self.a_feat = torch.nn.functional.normalize(self.dataset.a_feat.to(self.config.device).float(),dim=1)
            if self.config["data.input.dataset"] == "tiktok":
                self.words_tensor = self.dataset.words_tensor.to(self.config.device)
                self.word_embedding = torch.nn.Embedding(11574, 128).to(self.config.device)
                torch.nn.init.xavier_normal_(self.word_embedding.weight)
                self.t_feat = scatter(self.word_embedding(self.words_tensor[1]),self.words_tensor[0],reduce='mean',dim=0).to(self.config.device)
            else:
                self.t_feat = torch.nn.functional.normalize(self.dataset.t_feat.to(self.config.device).float(),dim=1)#self.dataset.t_feat.to(self.config.device).float()
                    
        # visual feature dense
        self.v_dense = nn.Linear(self.v_feat.shape[1], self.latent_dim)
        if self.config["data.input.dataset"] != "kwai":
            # acoustic feature dense
            self.a_dense = nn.Linear(self.a_feat.shape[1], self.latent_dim)
            # textual feature dense
            self.t_dense = nn.Linear(self.t_feat.shape[1], self.latent_dim)
        
        self.item_feat_dim = self.latent_dim * 4 if self.config["data.input.dataset"] != "kwai" else self.latent_dim * 2
        
        nn.init.xavier_uniform_(self.v_dense.weight)
        if self.config["data.input.dataset"] != "kwai":
            nn.init.xavier_uniform_(self.a_dense.weight)
            nn.init.xavier_uniform_(self.t_dense.weight)
        
        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)
    