import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import uniform, glorot, zeros

from models import BasicModel
from util.logger import Logger
from data.dataset import Dataset
from torch_scatter import scatter


class GraphGAT(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(GraphGAT, self).__init__(aggr=aggr, **kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.dropout = 0.2

		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()
		self.is_get_attention = False

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)
		uniform(self.in_channels, self.bias)


	def forward(self, x, edge_index, size=None):
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		x = torch.matmul(x, self.weight)

		return self.propagate(edge_index, size=size, x=x)

	def message(self, edge_index_i, x_i, x_j, size_i, edge_index, size):
		# Compute attention coefficients.
		x_i = x_i.view(-1, self.out_channels)
		x_j = x_j.view(-1, self.out_channels)
		inner_product = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)

		# gate
		row, col = edge_index
		deg = degree(row, size[0], dtype=x_i.dtype)
		deg_inv_sqrt = deg[row].pow(-0.5)
		tmp = torch.mul(deg_inv_sqrt, inner_product)
		gate_w = torch.sigmoid(tmp)
		# gate_w = F.dropout(gate_w, p=self.dropout, training=self.training)

		# attention
		tmp = torch.mul(inner_product, gate_w)
		attention_w = softmax(tmp, edge_index_i,num_nodes =  size_i)
		#attention_w = F.dropout(attention_w, p=self.dropout, training=self.training)
		return torch.mul(x_j, attention_w.view(-1, 1))

	def update(self, aggr_out):
		if self.bias is not None:
			aggr_out = aggr_out + self.bias
		if self.normalize:
			aggr_out = F.normalize(aggr_out, p=2, dim=-1)
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    
class GNN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat,dim_id, dim_latent=None):
        super(GNN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index

        self.preference = nn.Embedding(num_user, self.dim_latent)
        nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
        nn.init.xavier_normal_(self.g_layer2.weight)

    def forward(self,features, id_embedding):
        temp_features = torch.tanh(self.MLP(features)) if self.dim_latent else features
        x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()

        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
#         return x_1
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        x = torch.cat((x_1, x_2), dim=1)

        return x
    
class MGAT(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(MGAT, self).__init__(dataset,config)
        
        self.batch_size = self.config.batch_size
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        
        self.aggr_mode = self.config.aggr_mode
        self.concate = self.config.concat
        self.dim_x=self.latent_dim
        self.pre_train = self.config['pre_train']
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
        
        users_list, items_list = self.dataset.get_train_interactions()
        edge_index = np.array([users_list,items_list]).transpose()
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda().long()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        if self.config["data.input.dataset"] != "kwai":
            self.v_gcn = GNN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.v_feat.size(1),self.dim_x, dim_latent=256)
            self.a_gcn = GNN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.a_feat.size(1),self.dim_x, dim_latent=128)
            self.t_gcn = GNN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.t_feat.size(1),self.dim_x, dim_latent=100)
        else :
            self.v_gcn = GNN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.v_feat.size(1),self.dim_x, dim_latent=256)

        self.id_embedding = nn.Embedding(self.num_users + self.num_items, self.latent_dim).cuda()
        #if self.pre_train:
        nn.init.xavier_normal_(self.id_embedding.weight).cuda()
        self.f = nn.Sigmoid()

        self.result = nn.init.xavier_normal_(torch.rand((self.num_users+self.num_items, self.dim_x))).cuda()
        
#     def forward(self, user_nodes, pos_items, neg_items):
#         v_rep = self.v_gnn(self.id_embedding)
#         a_rep = self.a_gnn(self.id_embedding)
#         t_rep = self.t_gnn(self.id_embedding)
#         representation = (v_rep + a_rep + t_rep) / 3 #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
#         self.result_embed = representation
#         user_tensor = representation[user_nodes]
#         pos_tensor = representation[pos_items]
#         neg_tensor = representation[neg_items]
#         pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
#         neg_tensor = torch.sum(user_tensor * neg_tensor, dim=1)
#         return pos_scores, neg_tensor
    
    def compute(self):
        if self.config["data.input.dataset"] != "kwai":
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            a_rep = self.a_gcn(self.a_feat, self.id_embedding)
            t_rep = self.t_gcn(self.t_feat, self.id_embedding)
            representation = (v_rep+a_rep+t_rep)/3
        else :
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            representation = v_rep
        users, items = torch.split(representation, [self.num_users, self.num_items])

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
        users_emb = users_emb_ego = self.all_users[users.long()]
        pos_emb = pos_emb_ego = self.all_items[pos_items.long()]
        if neg_items is not None:
            neg_emb = neg_emb_ego = self.all_items[neg_items.long()]
        else:
            neg_emb = neg_emb_ego = None
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego





