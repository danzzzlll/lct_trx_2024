import torch
from torch import nn as nn
from torch.nn import functional as F

from ptls.frames.coles.sampling_strategies import HardNegativePairSelector


class CARLESLoss(torch.nn.Module):
    def __init__(
            self, 
            margin=0.5,
            pair_selector=HardNegativePairSelector(5),
            lam=0.5,
            numerical_features=[],
            categorical_features=[]
        ):

        super().__init__()

        self.margin = margin
        self.pair_selector = pair_selector
        self.lam = lam

        self.categorical_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.numerical_loss = nn.MSELoss()

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def forward(self, preds_targets, classes):
        preds, targets, seq_embeddings = preds_targets

        ar_loss = self.ar_loss(preds, targets)
        coles_loss = self.coles_loss(seq_embeddings, classes)
    
        return coles_loss + self.lam*ar_loss 
    
    def ar_loss(self, preds, targets):
        
        loss = 0
        
        for feature in self.numerical_features:
            loss += self.numerical_loss(preds[feature], targets[feature])

        for feature in self.categorical_features:
            loss += self.categorical_loss(preds[feature], targets[feature])
        
        return loss
    
    def coles_loss(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()    
