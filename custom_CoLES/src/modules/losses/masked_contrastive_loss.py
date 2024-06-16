import torch
from torch import nn as nn
from torch.nn import functional as F

from ptls.frames.coles.sampling_strategies import HardNegativePairSelector


class MaskedContrastiveLoss(torch.nn.Module):
    def __init__(
            self, 
            margin=0.5,
            pair_selector=HardNegativePairSelector(5),
            temperature=0.05, 
            neg_count=25, 
            lam=0.5
        ):

        super().__init__()

        self.margin = margin
        self.pair_selector = pair_selector
        self.temperature = temperature
        self.neg_count = neg_count
        self.lam = lam

    def forward(self, embeddings, target):
        trx_target, trx_predict, seq_embeddings = embeddings

        mlm_loss = self.mlm_loss(trx_target, trx_predict)
        coles_loss = self.coles_loss(seq_embeddings, target)
    
        return coles_loss + self.lam*mlm_loss 
    
    def mlm_loss(self, trx_target, trx_predict):

        mn = 1 - torch.eye(trx_target.size(0))
        neg_ix = torch.multinomial(mn, self.neg_count)
        all_counterparty = torch.cat([trx_predict.unsqueeze(1), trx_predict[neg_ix]], dim=1)
        
        logits = (trx_target.unsqueeze(1) * all_counterparty).sum(2) / self.temperature
        log_probs = torch.log_softmax(logits, dim=1)

        loss = -log_probs[:, 0]

        return loss.mean()

    def coles_loss(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()
