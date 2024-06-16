import torch
import torch.nn as nn

from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.normalization import L2NormEncoder

from src.abs_module import ABSModuleCustom
from src.metrics import NegLossMetric
from src.losses.carles_loss import CARLESLoss


class CARLESModule(ABSModuleCustom):
    """
    """

    def __init__(
        self,
        seq_encoder=None,
        lam=0.1,
        optimizer_partial=None,
        lr_scheduler_partial=None,
    ) -> None:
        """
        """
        
        loss = CARLESLoss(lam=lam)
        validation_metric = NegLossMetric(loss)

        super().__init__(
            validation_metric,
            seq_encoder,
            loss,
            optimizer_partial,
            lr_scheduler_partial,
        )

        embedding_dim = seq_encoder.embedding_size

        categorical_features = []
        self.heads = nn.ModuleDict()
        for col_name, noisy_emb in seq_encoder.trx_encoder.embeddings.items():
            self.head[col_name] = nn.Linear(embedding_dim, noisy_emb.num_embeddings)
            categorical_features.append(col_name)

        numerical_features = []
        for col_name, _ in seq_encoder.trx_encoder.custom_embeddings.items():
            self.head[col_name] = nn.Linear(embedding_dim, 1)
            numerical_features.append(col_name)

        self.head = L2NormEncoder()

        self.coles_metric = BatchRecallTopK(4)

    def shared_step(self, x, y):
        mask = self.get_mask(x.seq_len_mask)

        trx_embeddings = self.seq_encoder.trx_encoder(x).payload

        embeddings = self.seq_encoder.seq_encoder(PaddedBatch(trx_embeddings, x.seq_lens))

        seq_embeddings = self.head(self.seq_encoder.seq_encoder.reducer(embeddings))

        preds, targets = {}, {}
        for key, head in self.heads.items():
            preds[key] = head(trx_embeddings[mask])
            targets[key] = x.payload[key][mask]

        return (preds, targets, seq_embeddings), y

    def validation_step(self, batch, _):
        (mcc_pred, mcc_target, amount_pred, amount_target, seq_embeddings), y = self.shared_step(*batch)
        self._validation_metric((mcc_pred, mcc_target, amount_pred, amount_target, seq_embeddings), y)
        self.mcc_metric(mcc_pred, mcc_target)
        self.amount_metric(amount_pred, amount_target)
        self.coles_metric(seq_embeddings, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self._validation_metric.compute(), prog_bar=True)
        # self.log(self.mcc_metric._get_name(), self.mcc_metric.compute(), prog_bar=False)
        # self.log(self.amount_metric._get_name(), self.amount_metric.compute(), prog_bar=False)
        self.log(self.coles_metric._get_name(), self.coles_metric.compute(), prog_bar=False)

        self._validation_metric.reset()
        # self.mcc_metric.reset()
        # self.amount_metric.reset()
        self.coles_metric.reset()

    @property
    def is_requires_reduced_sequence(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return 'neg_loss'
