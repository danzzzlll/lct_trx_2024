import torch

from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.padded_batch import PaddedBatch


from src.modules.abs_module import ABSModuleCustom


class CoLESModule(ABSModuleCustom):
    """Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))

    Subsequences are sampled from original sequence.
    Samples from the same sequence are `positive` examples
    Samples from the different sequences are `negative` examples
    Embeddings for all samples are calculated.
    Paired distances between all embeddings are calculated.
    The loss function tends to make positive distances smaller and negative ones larger.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Model which helps to train. Not used during inference
            Can be normalisation layer which make embedding l2 length equals 1
            Can be MLP as `projection head` like in SymCLR framework.
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        validation_metric:
            Keep None. `ptls.frames.coles.metric.BatchRecallTopK` used by default.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = head

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y


class CoLESMaskedModule(CoLESModule):
    """Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))

    Subsequences are sampled from original sequence.
    Samples from the same sequence are `positive` examples
    Samples from the different sequences are `negative` examples
    Embeddings for all samples are calculated.
    Paired distances between all embeddings are calculated.
    The loss function tends to make positive distances smaller and negative ones larger.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Model which helps to train. Not used during inference
            Can be normalisation layer which make embedding l2 length equals 1
            Can be MLP as `projection head` like in SymCLR framework.
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        validation_metric:
            Keep None. `ptls.frames.coles.metric.BatchRecallTopK` used by default.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 mask_prob=0.1,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):

        super().__init__(seq_encoder,
                         head,
                         loss,
                         validation_metric,
                         optimizer_partial,
                         lr_scheduler_partial)

        trx_embed_size = seq_encoder.trx_encoder.output_size

        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, trx_embed_size), requires_grad=True)
        self.mask_prob = mask_prob

    def get_mask(self, attention_mask):
        return torch.bernoulli(attention_mask.float() * self.mask_prob).bool()

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

        y_h = self.seq_encoder.seq_encoder(PaddedBatch(trx_embeddings_masked, x.seq_lens))
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y