"""The file with the main logic for the AR model."""

from typing import Literal, Optional, Union

import torch
from omegaconf import DictConfig
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.data_load import PaddedBatch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor


from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn import L2NormEncoder
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion
from torch import Tensor, nn
from torchmetrics import (
    AUROC,
    R2Score,
)

from .abs_module import ABSModuleCustom 
from .metrics.neg_loss_metric import NegLossMetric
from .losses.carles_loss import PrimeNetLoss


class PrimeNetModule(ABSModuleCustom):
    """A module for AR training, just encodes the sequence and predicts its shifted version.

    Logs train/val/test losses:
     - a CrossEntropyLoss on mcc codes
     - an MSELoss on amounts
    and train/val/test metrics:
     - a macro-averaged multiclass f1-score on mcc codes
     - a macro-averaged multiclass auroc score on mcc codes
     - an r2-score on amounts.

    Attributes
    ----------
        amount_loss_weight (float):
            Normalized loss weight for the transaction amount MSE loss.
        mcc_loss_weight (float):
            Normalized loss weight for the transaction mcc code CE loss.
        lr (float):
            The learning rate, extracted from the optimizer_config.

    Notes
    -----
        amount_loss_weight, mcc_loss_weight are normalized so that amount_loss_weight + mcc_loss_weight = 1.
        This is done to remove one hyperparameter. Loss gradient size can be managed separately through lr.

    """

    def __init__(
        self,
        seq_encoder=None,
        num_types=100,
        mask_prob=0.1,
        lam=0.1,
        optimizer_partial=None,
        lr_scheduler_partial=None,
    ) -> None:
        """Initialize GPTModule internal state.

        Args:
        ----
            loss_weights (dict):
                A dictionary with keys "amount" and "mcc", mapping them to the corresponding loss weights
            encoder (SeqEncoderContainer):
                SeqEncoderContainer to be used as an encoder.
            num_types (int):
                Amount of mcc types; clips all input to this value.
            optimizer (DictConfig):
                Optimizer dictconfig, instantiated with params kwarg.
            scheduler (Optional[DictConfig]):
                Optionally, an lr scheduler dictconfig, instantiated with optimizer kwarg
            scheduler_config (Optional[dict]):
                An lr_scheduler config for specifying scheduler-specific params, such as which metric to monitor
                See LightningModule.configure_optimizers docstring for more details.
            encoder_weights (Optional[str], optional):
                Path to encoder weights. If not specified, no weights are loaded by default.
            freeze_enc (Optional[int], optional):
                Whether to freeze the encoder module.
        """
        
        self.save_hyperparameters('num_types', 'mask_prob')

        loss = PrimeNetLoss(lam=lam)
        validation_metric = NegLossMetric(loss)

        super().__init__(
            validation_metric,
            seq_encoder,
            loss,
            optimizer_partial,
            lr_scheduler_partial,
        )

        embedding_dim = seq_encoder.embedding_size
        trx_embed_size = seq_encoder.trx_encoder.output_size

        self.mcc_head = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, seq_encoder.trx_encoder.embeddings.mcc_code.num_embeddings + 1)
        )
        self.amount_head = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1)
        )

        self.head = L2NormEncoder()

        self.mcc_metric = AUROC(
            task="multiclass",
            average="weighted",
            num_classes=seq_encoder.trx_encoder.embeddings.mcc_code.num_embeddings+1,
            ignore_index=0,
        )
        self.amount_metric = R2Score()
        self.coles_metric = BatchRecallTopK(4)

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
        # x.payload["mcc_code"] = torch.clip(
        #     x.payload["mcc_code"], 0, self.hparams.num_types
        # )

        mask = self.get_mask(x.seq_len_mask)

        trx_embeddings = self.seq_encoder.trx_encoder(x).payload
        trx_embeddings_masked = self.mask_x(trx_embeddings, x.seq_len_mask, mask)

        embeddings = self.seq_encoder.seq_encoder(PaddedBatch(trx_embeddings_masked, x.seq_lens))

        seq_embeddings = self.head(self.seq_encoder.seq_encoder.reducer(embeddings))

        mcc_pred = self.mcc_head(embeddings.payload[mask])
        amount_pred = self.amount_head(embeddings.payload[mask]).squeeze(-1)

        mcc_target = x.payload["mcc_code"][mask]
        amount_target = x.payload["amount_log"][mask].float()

        return (mcc_pred, mcc_target, amount_pred, amount_target, seq_embeddings), y

    def validation_step(self, batch, _):
        (mcc_pred, mcc_target, amount_pred, amount_target, seq_embeddings), y = self.shared_step(*batch)
        self._validation_metric((mcc_pred, mcc_target, amount_pred, amount_target, seq_embeddings), y)
        self.mcc_metric(mcc_pred, mcc_target)
        self.amount_metric(amount_pred, amount_target)
        self.coles_metric(seq_embeddings, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self._validation_metric.compute(), prog_bar=True)
        self.log(self.mcc_metric._get_name(), self.mcc_metric.compute(), prog_bar=False)
        self.log(self.amount_metric._get_name(), self.amount_metric.compute(), prog_bar=False)
        self.log(self.coles_metric._get_name(), self.coles_metric.compute(), prog_bar=False)

        self._validation_metric.reset()
        self.mcc_metric.reset()
        self.amount_metric.reset()
        self.coles_metric.reset()

    @property
    def is_requires_reduced_sequence(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return 'neg_loss'
