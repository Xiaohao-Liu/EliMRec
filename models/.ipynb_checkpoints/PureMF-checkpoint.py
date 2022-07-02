import torch
from data.dataset import Dataset
from torch import nn
import numpy as np
from torch_scatter import scatter

from models import BasicModel

class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:Dataset):
        super(PureMF, self).__init__(dataset,config)
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
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.config.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.config.device)
        self.alpha_i = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=1).to(self.config.device)
        
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
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1) 
        nn.init.xavier_uniform_(self.alpha_i.weight, gain=1)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def predict(self, user_ids, candidate_items=None):
        users = torch.tensor(user_ids).long().to(self.config.device)
        users_emb = self.all_users[users]
        if candidate_items is None:
            items_emb = self.all_items
        else:
            items_emb = self.all_items[torch.tensor(candidate_items).long().to(self.config.device)]
        scores = torch.matmul(users_emb, items_emb.t()) #+ self.alpha_i.weight.T
        return self.f(scores).detach().cpu()
    
    def compute(self):
        pass
    
    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users = self.embedding_user.weight
        self.all_items = self.dense_item(self.item_embs)
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
 
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1) #+ self.alpha_i(pos.long()).T
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1) #+ self.alpha_i(neg.long()).T
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
