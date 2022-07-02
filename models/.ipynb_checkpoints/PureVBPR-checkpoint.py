import torch
from data.dataset import Dataset
from torch import nn
import numpy as np

from models import BasicModel

class VBPR(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(VBPR, self).__init__(dataset,config)
        # self.config = config
        self.num_users  = dataset.num_users
        self.num_items  = dataset.num_items

        self.latent_dim = config['recdim']
        self.temp = config['temp']
        self.f = nn.Sigmoid()
        self.__init_weight()
        self.all_users = self.all_items = None
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.beta_i = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=1)

        self.v_feat = self.dataset.v_feat.to(self.config.device).float()
        self.E = nn.Linear(self.v_feat.shape[1], self.latent_dim)
        
        self.beta_bias = nn.Embedding(1, self.v_feat.shape[1])
        self.theta = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1) 
        nn.init.xavier_uniform_(self.beta_i.weight, gain=1)
        nn.init.xavier_uniform_(self.E.weight, gain=1)
        nn.init.xavier_uniform_(self.beta_bias.weight, gain=1)
        nn.init.xavier_uniform_(self.theta.weight, gain=1)
        
    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        if candidate_items is None:
            items_emb = self.all_items
            ef = self.E(self.v_feat) # A*D
            theta_ef = torch.matmul(self.theta(users.long()),ef.t()) # B*A
            beta_bias_f = self.beta_bias.weight.mm(self.v_feat.T ) # 1*A
        else:
            items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
            
        # torch.matmul(users_emb, items_emb.t()) B*A
        scores = torch.matmul(users_emb, items_emb.t()) + theta_ef # + beta_bias_f+ self.beta_i.weight.T
        return self.f(scores).detach().cpu()
    
    def compute(self):
        pass
    
    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users = self.embedding_user.weight
        self.all_items = self.embedding_item.weight
        users_emb = users_emb_ego = self.embedding_user(users.long())
        pos_emb = pos_emb_ego = self.embedding_item(pos_items.long())
        
        neg_emb = neg_emb_ego = self.embedding_item(neg_items.long()) if neg_items != None else None          
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # def bpr_loss(self, users, pos, neg)
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        users_emb=torch.nn.functional.normalize(users_emb,dim=1)
        pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        neg_emb=torch.nn.functional.normalize(neg_emb,dim=1)
        
        ef_pos = self.E(self.v_feat[pos.long()]) # B*D
        ef_neg = self.E(self.v_feat[neg.long()]) # B*D
        theta_ef_pos = torch.sum(torch.mul(self.theta(users.long()),ef_pos), dim=1) # B
        theta_ef_neg = torch.sum(torch.mul(self.theta(users.long()),ef_neg), dim=1) # B
        beta_bias_f_pos = self.beta_bias.weight.mm( self.v_feat[pos.long()].T ).squeeze(0) # B
        beta_bias_f_neg = self.beta_bias.weight.mm( self.v_feat[neg.long()].T ).squeeze(0) # B
        # torch.sum(pos_scores, dim=1) B
        pos_scores = torch.mul(users_emb, pos_emb) # B*D 
        pos_scores = torch.sum(pos_scores, dim=1) + theta_ef_pos #+ beta_bias_f_pos+ self.beta_i(pos.long()).T.squeeze(0)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1) + theta_ef_neg #+ beta_bias_f_neg+ self.beta_i(neg.long()).T.squeeze(0)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
