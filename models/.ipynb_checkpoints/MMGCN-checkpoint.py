import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform

from models import BasicModel
from util.logger import Logger
from data.dataset import Dataset
from torch_scatter import scatter

eps = 1e-12
# add ssl

class BaseModel(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(BaseModel, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

		self.reset_parameters()

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)

	def forward(self, x, edge_index, size=None):
		x = torch.matmul(x, self.weight)
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		return x_j

	def update(self, aggr_out):
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)  
        

    def forward(self, features, id_embedding):

        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        if self.num_layer > 2:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        
        return x
    
class MMGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(MMGCN, self).__init__(dataset,config)
        
        self.batch_size = self.config.batch_size
        self.latent_dim = self.config['recdim']
        self.n_layers = self.config['layer_num']
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        
        self.aggr_mode = self.config.aggr_mode
        self.concate = self.config.concat
        self.dim_x=self.latent_dim
        self.pre_train = self.config['pre_train']
        self.cf_mode = self.config['cf_mode'] if 'cf_mode' in self.config else False
        
        self.alpha = self.config['alpha']
        
        self.v_feat = self.dataset.v_feat.to(self.config.device).float()
        
        self.a_feat = self.dataset.a_feat.to(self.config.device).float()
        
        self.words_tensor = self.dataset.words_tensor.to(self.config.device)
        self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128).to(self.config.device)
        nn.init.xavier_normal_(self.word_embedding.weight) 
        
        users_list, items_list = self.dataset.get_train_interactions()
        edge_index = np.array([users_list,items_list]).transpose()
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda().long()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.v_gcn = GCN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.v_feat.size(1), self.dim_x, self.aggr_mode, self.concate, self.n_layers, True, dim_latent=256)
        self.a_gcn = GCN(self.edge_index, self.batch_size, self.num_users, self.num_items, self.a_feat.size(1), self.dim_x, self.aggr_mode, self.concate, self.n_layers, True)
        self.t_gcn = GCN(self.edge_index, self.batch_size, self.num_users, self.num_items, 128, self.dim_x, self.aggr_mode, self.concate, self.n_layers, True)


        self.id_embedding = nn.init.xavier_normal_(torch.rand((self.num_users+self.num_items, self.dim_x), requires_grad=True)).cuda()
        self.pre_embedding = nn.init.xavier_normal_(torch.rand((self.num_users+self.num_items, self.dim_x))).cuda()
        self.post_embedding = nn.init.xavier_normal_(torch.rand((self.num_users+self.num_items, self.dim_x))).cuda()
        
        self.f = nn.Sigmoid()

#         self.result = nn.init.xavier_normal_(torch.rand((self.num_users+self.num_items, self.dim_x))).cuda()
    
    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = self.a_gcn(self.a_feat, self.id_embedding)
        self.t_feat = scatter(self.word_embedding(self.words_tensor[1]),self.words_tensor[0],reduce='mean',dim=0).cuda()
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        # pre_interaction_score
        pre_representation = t_rep
        self.pre_embedding = pre_representation
        pre_user_tensor = pre_representation[user_nodes]
        pre_pos_item_tensor = pre_representation[self.num_users + pos_item_nodes]
        pre_neg_item_tensor = pre_representation[self.num_users + neg_item_nodes]
        pre_pos_scores = torch.sum(pre_user_tensor*pre_pos_item_tensor, dim=1)
        pre_neg_scores = torch.sum(pre_user_tensor*pre_neg_item_tensor, dim=1)

        # post_interaction_score
        post_representation = (v_rep+a_rep+t_rep)/3
        self.post_embedding = post_representation
        post_user_tensor = post_representation[user_nodes]
        post_pos_item_tensor = post_representation[self.num_users + pos_item_nodes]
        post_neg_item_tensor = post_representation[self.num_users + neg_item_nodes]
        post_pos_scores = torch.sum(post_user_tensor*post_pos_item_tensor, dim=1)
        post_neg_scores = torch.sum(post_user_tensor*post_neg_item_tensor, dim=1)

        # fusion of pre_ and post_ interaction scores
        # # post*sigmoid(pre)
        if self.cf_mdoe:
            pos_scores = post_pos_scores*torch.sigmoid(pre_pos_scores)
            neg_scores = post_neg_scores*torch.sigmoid(pre_neg_scores)
        else:
            pos_scores = post_pos_scores
            neg_scores = post_neg_scores

        return pos_scores, neg_scores, pre_pos_scores, pre_neg_scores
    
    def bpr_loss(self, user, pos_items, neg_items):
        pos_scores, neg_scores, pre_pos_scores, pre_neg_scores = self.forward(user.cuda().long(), pos_items.cuda().long(), neg_items.cuda().long()) 
        # BPR loss
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores) + eps))
        if self.cf_mode:
            loss_value_pre = -torch.sum(torch.log2(torch.sigmoid(pre_pos_scores - pre_neg_scores) + eps))
            return loss_value + self.alpha * loss_value_pre
        
        return loss_value
        
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
          



