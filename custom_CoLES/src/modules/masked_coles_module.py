import torch
from torch import nn as nn

from ptls.frames.coles.metric import BatchRecallTopK
from ptls.nn import L2NormEncoder
from ptls.data_load import PaddedBatch

from src.modules.metrics.contrastive_accuracy import ContrastiveAccuracy
from src.modules.abs_module import ABSModuleCustom
from .losses.masked_contrastive_loss import MaskedContrastiveLoss
from .metrics.neg_loss_metric import NegLossMetric


class MaskedCoLESModule(ABSModuleCustom):
    """
    """
    def __init__(self,
                 seq_encoder=None,
                 norm_predict=True,
                 mask_prob=0.1,
                 neg_count=25,
                 temperature=0.05,
                 lam=0.1,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):


        self.save_hyperparameters('norm_predict', 'neg_count', 'temperature', 'mask_prob')

        loss = MaskedContrastiveLoss(temperature=temperature, neg_count=neg_count, lam=lam)
        validation_metric = NegLossMetric(loss)

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)
        
        self.mlm_metric = ContrastiveAccuracy(neg_count)
        self.coles_metric = BatchRecallTopK(4)

        if self.hparams.norm_predict:
            self.fn_norm_predict = L2NormEncoder()

        embedding_size = seq_encoder.embedding_size
        trx_embed_size = seq_encoder.trx_encoder.output_size

        self.proj_head = nn.Linear(embedding_size, trx_embed_size)
        self.head = L2NormEncoder()

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, trx_embed_size), requires_grad=True)

    def get_mask(self, attention_mask):
        return torch.bernoulli(attention_mask.float() * self.hparams.mask_prob).bool()

    def mask_x(self, x, attention_mask, mask):
        shuffled_tokens = x[attention_mask.bool()]
        B, T, H = x.size()
        ix = torch.multinomial(torch.ones(shuffled_tokens.size(0)), B * T, replacement=True)
        shuffled_tokens = shuffled_tokens[ix].view(B, T, H)

        rand = torch.rand(B, T, device=x.device).unsqueeze(2).expand(B, T, H)
        replace_to = torch.where(
            rand < 0.8,
            self.token_mask.expand_as(x),  # [MASK] token 80%
            torch.where(
                rand < 0.9,
                shuffled_tokens,  # random token 10%
                x,  # unchanged 10%
            )
        )
        return torch.where(mask.bool().unsqueeze(2).expand_as(x), replace_to, x)


    def shared_step(self, x, y):
        mask = self.get_mask(x.seq_len_mask)

        trx_embeddings = self.seq_encoder.trx_encoder(x).payload
        trx_embeddings_masked = self.mask_x(trx_embeddings, x.seq_len_mask, mask)

        embeddings = self.seq_encoder.seq_encoder(PaddedBatch(trx_embeddings_masked, x.seq_lens))

        seq_embeddings = self.head(self.seq_encoder.seq_encoder.reducer(embeddings))

        predict = self.proj_head(embeddings.payload[mask])
        target = trx_embeddings[mask]   
        
        if self.hparams.norm_predict:
            predict = self.fn_norm_predict(predict)

        return (target, predict, seq_embeddings), y

    def validation_step(self, batch, _):
        (target, predict, seq_embeddings), y = self.shared_step(*batch)
        self._validation_metric((target, predict, seq_embeddings), y)
        self.mlm_metric((target, predict), y)
        self.coles_metric(seq_embeddings, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self._validation_metric.compute(), prog_bar=True)
        self.log(self.mlm_metric._get_name(), self.mlm_metric.compute(), prog_bar=False)
        self.log(self.coles_metric._get_name(), self.coles_metric.compute(), prog_bar=False)

        self._validation_metric.reset()
        self.mlm_metric.reset()
        self.coles_metric.reset()

    @property
    def is_requires_reduced_sequence(self):
        return False
    
    @property
    def metric_name(self):
        return 'neg_loss'
