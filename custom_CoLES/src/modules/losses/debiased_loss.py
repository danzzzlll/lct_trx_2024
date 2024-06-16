import numpy as np
import torch


class DebiasedContrastiveLoss(torch.nn.Module):
    """Debiased Contrastive Loss
    All positive pairs as `anchor`-`pos_sample` concatenated with all possible `neg_samples` from batch.
    Softmax is applied over `distances` between `anchor` and pos or neg samples.
    `distances` is a dot product.
    Loss minimizes `-log()` for `anchor-pos` position with debiasing correction.
    
    Params:
        temperature:
            `softmax(distances / temperature)` - scale a sub-exponent expression.
            default 0.05 value is for l2-normalized `embeddings` where dot product distance is in range [-1, 1]
    """
    def __init__(self, tau=0.05, temperature=0.05):
        super().__init__()
        
        self.tau = tau
        self.temperature = temperature
        
    def forward(self, embeddings, classes):
        batch_size, embedding_dim = embeddings.size()
        num_classes = torch.unique(classes).size(0)

        d = torch.einsum('bh,kh->bk', embeddings, embeddings) / self.temperature

        ix_pos = classes.unsqueeze(1) == classes.unsqueeze(0)
        ix_pos.fill_diagonal_(0)
        ix_a, ix_pos = ix_pos.nonzero(as_tuple=True)
        _, ix_neg = (classes[ix_a].unsqueeze(1) != classes.unsqueeze(0)).nonzero(as_tuple=True)
        ix_neg = ix_neg.reshape(ix_a.shape[0], -1)

        pos = torch.exp(d[ix_a, ix_pos])
        neg = torch.sum(torch.exp(d[ix_a.unsqueeze(1).expand_as(ix_neg), ix_neg]), dim=1)

        N = batch_size - batch_size // num_classes

        Ng = torch.clamp((neg - N*pos*self.tau) / (1 - self.tau), min=N*np.exp(-1/self.temperature))
        
        return (torch.log(pos + Ng) - torch.log(pos)).mean()
    

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class OriginalDebiasedContrastiveLoss(torch.nn.Module):
    """Debiased Contrastive Loss
    All positive pairs as `anchor`-`pos_sample` concatenated with all possible `neg_samples` from batch.
    Softmax is applied over `distances` between `anchor` and pos or neg samples.
    `distances` is a dot product.
    Loss minimizes `-log()` for `anchor-pos` position with debiasing correction.
    
    Params:
        temperature:
            `softmax(distances / temperature)` - scale a sub-exponent expression.
            default 0.05 value is for l2-normalized `embeddings` where dot product distance is in range [-1, 1]
    """
    def __init__(self, tau=0.05, temperature=0.05):
        super().__init__()
        
        self.tau = tau
        self.temperature = temperature
        
    def forward(self, embeddings, classes):
        n = len(embeddings)

        ix1 = torch.arange(0, n, 2, device=embeddings.device)
        ix2 = torch.arange(1, n, 2, device=embeddings.device)

        assert n % 2 == 0
        assert (classes[ix1] == classes[ix2]).all(), "Wrong embedding positions"
        
        batch_size = n // 2 

        pos_1 = embeddings[ix1]
        pos_2 = embeddings[ix2]

        # neg score
        out = torch.cat([pos_1, pos_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(pos_1 * pos_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        N = batch_size * 2 - 2
        Ng = (-self.tau * N * pos + neg.sum(dim = -1)) / (1 - self.tau)
        # constrain (optional)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        
        return loss
    

