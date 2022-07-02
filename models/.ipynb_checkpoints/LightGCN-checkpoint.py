import torch
from data.dataset import Dataset
from torch import nn
import numpy as np
import scipy.sparse as sp
from models import BasicModel
from util.logger import Logger
from torch_scatter import scatter
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(LightGCN, self).__init__(dataset,config)
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.num_users
        self.num_items  = self.dataset.num_items
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.keep_prob = self.config['keepprob']
        self.A_split = self.config['A_split']
        self.temp = self.config['temp']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.config.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.config.device)
        
        # load multi-modal feature
        self.v_feat = self.dataset.v_feat.to(self.config.device).float()
        
        if self.config["data.input.dataset"] != "kwai":
            self.a_feat = self.dataset.a_feat.to(self.config.device).float()
        else :
            self.a_feat = None
        
        if self.config["data.input.dataset"] == "tiktok":
            self.words_tensor = self.dataset.words_tensor.to(self.config.device)
            self.word_embedding = torch.nn.Embedding(11574, 128).to(self.config.device)
            torch.nn.init.xavier_normal_(self.word_embedding.weight)
            self.t_feat = scatter(self.word_embedding(self.words_tensor[1]),self.words_tensor[0],reduce='mean',dim=0).to(self.config.device)
        else:
            if self.config["data.input.dataset"] != "kwai":
                self.t_feat = self.dataset.t_feat.to(self.config.device).float()
            else :
                self.t_feat = None
        # 
        feat_stack = [self.v_feat]
        if self.a_feat is not None:
            feat_stack.append(self.a_feat)
        if self.t_feat is not None:
            feat_stack.append(self.t_feat)
            
        self.feat = torch.cat(feat_stack, dim=1)
        self.item_embs = torch.cat([self.embedding_item.weight, self.feat],dim=1)
        
        self.dense_item = torch.nn.Linear(self.item_embs.shape[1], self.latent_dim)
        
        nn.init.xavier_uniform_(self.dense_item.weight, gain=1)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.all_items = self.all_users = None
        
        coo = self.create_adj_mat(self.config['adj_type']).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse.FloatTensor(indices, torch.FloatTensor(coo.data), coo.shape)
        self.norm_adj = self.norm_adj.to(self.config.device)
        
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
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def compute(self):
        users_emb = self.embedding_user.weight
#         items_emb = self.embedding_item.weight
        items_emb = self.dense_item(self.item_embs)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.norm_adj        
        else:
            g_droped = self.norm_adj    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        if candidate_items is None:
            items_emb = self.all_items
        else:
            items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores).detach().cpu()
       
    
    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users, self.all_items = self.compute()
        users_emb = self.all_users[users]
        pos_emb = self.all_items[pos_items]
        
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        if neg_items is None:
            neg_emb = neg_emb_ego =None
        else:
            neg_emb = self.all_items[neg_items]
            neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    # def bpr_loss(self, users, pos, neg)
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.compute()
        # print('forward')
        #all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma