import torch
from data.dataset import Dataset
from torch import nn
import numpy as np
from torch_scatter import scatter
from util.logger import Logger
import torch.nn.functional as F

from util.mlp import MLP

from models import BasicModel

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


class CMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(CMF, self).__init__(dataset,config)
        # self.config = config
        self.num_users  = dataset.num_users
        self.num_items  = dataset.num_items

        self.latent_dim = config['recdim']
        self.temp = config['temp']
        self.logits = self.config["logits"] if 'logits' in self.config else 'cosin' # ['cosin', 'inner_product']
        self.predict_type = self.config["predict_type"] if  'predict_type' in self.config else 'NIE' # ['NIE', 'TIE']
        self.fusion_mode = self.config["fusion_mode"] if  'fusion_mode' in self.config else 'rubi' # ['rubi', 'hm','sum']
        Logger.info('fusion mode: ' + self.fusion_mode)
        self.f = nn.Sigmoid()
        self.__init_weight()
        
        self.all_users = self.all_items = None
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.config.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.config.device)
        self.alpha_i = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=1).to(self.config.device)
        
        # load multi-modal feature
        v_feat = self.dataset.v_feat.to(self.config.device).float()
        self.v_feat = F.normalize(v_feat, dim=1)
        if self.config["data.input.dataset"] != "kwai":
            a_feat = self.dataset.a_feat.to(self.config.device).float()
            self.a_feat = F.normalize(a_feat, dim=1)
        else :
            self.a_feat = None
        
        if self.config["data.input.dataset"] == "tiktok":
            self.words_tensor = self.dataset.words_tensor.to(self.config.device)
            self.word_embedding = torch.nn.Embedding(11574, 128).to(self.config.device)
            torch.nn.init.xavier_normal_(self.word_embedding.weight)
            self.t_feat = scatter(self.word_embedding(self.words_tensor[1]),self.words_tensor[0],reduce='mean',dim=0).to(self.config.device)
        else:
            if self.config["data.input.dataset"] != "kwai":
                t_feat = self.dataset.t_feat.to(self.config.device).float()
                self.t_feat = F.normalize(t_feat, dim=1)
            else :
                self.t_feat = None
        # 
#         feat_stack = [self.v_feat]
        self.v_dense = MLP(self.v_feat.shape[1], [self.latent_dim for _ in range(2)])
        self.v_dense.init_weight('xavier')
#         nn.init.xavier_uniform_(self.v_dense.weight, gain=1)
        if self.a_feat is not None:
#             feat_stack.append(self.a_feat)
            self.a_dense = MLP(self.a_feat.shape[1], [self.latent_dim for _ in range(2)])
            self.a_dense.init_weight('xavier')
        if self.t_feat is not None:
#             feat_stack.append(self.t_feat)
            self.t_dense = MLP(self.t_feat.shape[1], [self.latent_dim for _ in range(2)])
            self.t_dense.init_weight('xavier')
            
        if self.config["data.input.dataset"] != "kwai":
            self.dense_item = torch.nn.Linear(self.latent_dim*4, self.latent_dim)
        else:
            self.dense_item = torch.nn.Linear(self.latent_dim*2, self.latent_dim)
        
        nn.init.xavier_uniform_(self.dense_item.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1) 
        nn.init.xavier_uniform_(self.alpha_i.weight, gain=1)
        
        # cf mode
        if 'rubi' in self.config:
            self.init_cf_mode()
        else:
            Logger.info('use original MF')
    
    def init_cf_mode(self):
        self.v_dense_s = nn.Linear(self.v_feat.shape[1], self.latent_dim)
        nn.init.xavier_uniform_(self.v_dense_s.weight, gain=1)
        if self.config["data.input.dataset"] != "kwai":
            # acoustic feature dense
            self.a_dense_s = nn.Linear(self.a_feat.shape[1], self.latent_dim)
            # textual feature dense
            self.t_dense_s = nn.Linear(self.t_feat.shape[1], self.latent_dim)
            
            nn.init.xavier_uniform_(self.a_dense_s.weight, gain=1)
            nn.init.xavier_uniform_(self.t_dense_s.weight, gain=1)
        
    def multi_modal(self, z_v, z_a, z_t):
        
        feat_stack = [self.embedding_item.weight]
        v_feat = self.v_dense(z_v)
        feat_stack.append(v_feat)
        if self.config["data.input.dataset"] != "kwai":
            a_feat = self.a_dense(z_a)
            feat_stack.append(a_feat)
            t_feat = self.t_dense(z_t)
            feat_stack.append(t_feat)
            
        # concatenate
        feat = torch.cat(feat_stack, dim=1) 
        return self.dense_item(feat)
        
#     def predict(self, user_ids, candidate_items=None):
#         users = torch.tensor(user_ids).long().to(self.config.device)
#         users_emb = self.all_users[users]
#         if candidate_items is None:
#             items_emb = self.all_items
#         else:
#             items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
#         scores = torch.matmul(users_emb, items_emb.t()) + self.alpha_i.weight.T
#         return self.f(scores).detach().cpu()
    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        if candidate_items is None:
            items_emb = self.all_items
        else:
            items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
            
        ui_score = torch.matmul(users_emb, items_emb.t()) + self.alpha_i.weight.T
        if self.predict_type == "TIE":
            s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
            y_u_fixed_i_s = torch.matmul(users_emb, torch.unsqueeze(torch.mean(items_emb,dim=0).t(),dim=1)) + torch.mean(self.alpha_i.weight.T)
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
        
        # NIE
        return self.f(ui_score).detach().cpu()
    
    def compute(self):
        pass
    
    def getEmbedding(self, users, pos_items, neg_items):
        # user preference representation
        self.all_users = self.embedding_user.weight
        # multi-modal representation
        self.all_items = self.multi_modal(self.v_feat, self.a_feat, self.t_feat)
        
        users_emb = self.all_users[users]
        
        users_emb_ego = self.embedding_user(users)
        pos_emb = self.all_items[pos_items.long()]
        pos_emb = pos_emb_ego = self.embedding_item(pos_items)
        
        neg_emb =  self.all_items[neg_items]
        neg_emb_ego = self.embedding_item(neg_items.long()) if neg_items != None else None          
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def original_bpr_loss(self,u_emb,p_emb,n_emb,pos_items,neg_items):
        u_emb=torch.nn.functional.normalize(u_emb,dim=1)
        p_emb=torch.nn.functional.normalize(p_emb,dim=1)
        n_emb=torch.nn.functional.normalize(n_emb,dim=1)
        p_scores = torch.mul(u_emb, p_emb)
        p_scores = torch.sum(p_scores, dim=1)+ self.alpha_i(pos_items).T 
        n_scores = torch.mul(u_emb, n_emb)
        n_scores = torch.sum(n_scores, dim=1)+ self.alpha_i(neg_items).T 
        return torch.mean(torch.nn.functional.softplus(n_scores - p_scores))
    
    def bpr_loss(self,users,pos_items,neg_items):
        users = users.long()
        pos_items =pos_items.long()
        neg_items = neg_items.long()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        pos_emb_s = []
        if neg_items is None:
            neg_emb =None
        else:
            neg_emb = self.all_items[neg_items]
        
        # RUBI mode
        if 'rubi' in self.config and self.config['rubi']:
            self.all_s_embs = []
            if 'v' in self.config['rubi']:
                self.all_s_embs.append(self.v_dense_s(self.v_feat))
            if self.config['data.input.dataset']!="kwai":
                if 'a' in self.config['rubi']:
                    self.all_s_embs.append(self.a_dense_s(self.a_feat))
                if 't' in self.config['rubi']:
                    self.all_s_embs.append(self.t_dense_s(self.t_feat))
        else: 
            return self.original_bpr_loss(users_emb,pos_emb,neg_emb,pos_items, neg_items)
        
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

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)+ self.alpha_i(pos_items).T 
        pos_scores_s = torch.sum(torch.mul(users_emb_s, pos_emb_s),dim=1)
        pos_scores = self.fusion(pos_scores,pos_scores_s)
        
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1) + self.alpha_i(neg_items).T 
        neg_scores_s = torch.sum(torch.mul(users_emb_s, neg_emb_s),dim=1)
        neg_scores = self.fusion(neg_scores,neg_scores_s)
        
        return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))+ \
                torch.mean(torch.nn.functional.softplus(neg_scores_s - pos_scores_s)) #+ \
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
        if 'rubi' in self.config and self.config['rubi']:
            self.all_s_embs = []
            if 'v' in self.config['rubi']:
                self.all_s_embs.append(self.v_dense_s(self.v_feat))
            if self.config['data.input.dataset']!="kwai":
                if 'a' in self.config['rubi']:
                    self.all_s_embs.append(self.a_dense_s(self.a_feat))
                if 't' in self.config['rubi']:
                    self.all_s_embs.append(self.t_dense_s(self.t_feat))
        else: 
            return self.original_infonce(users_emb,pos_emb)
        
        s_emb = torch.mean(torch.stack(self.all_s_embs),dim=0)
        pos_emb_s = s_emb[pos_items]
        pos_emb = self.all_items[pos_items]

        
        if self.logits == "inner_product":
            pass
        elif  self.logits == "cosin":
            users_emb=torch.nn.functional.normalize(users_emb,dim=1)
            pos_emb_s=torch.nn.functional.normalize(pos_emb_s,dim=1)
            pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        
        z_m = torch.mm(users_emb,pos_emb.T)
        z_s = torch.mm(users_emb, pos_emb_s.T)
        logits = self.fusion(z_m, z_s)
        logits /= self.temp
        labels = torch.tensor(list(range(users.shape[0]))).to(self.config.device)
        
        logits_s = torch.mm(users_emb, pos_emb_s.T)
        logits_s /= self.temp

        return self.infonce_criterion(logits, labels) + self.infonce_criterion(logits_s, labels)
    
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
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
