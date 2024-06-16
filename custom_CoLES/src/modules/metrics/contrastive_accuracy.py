import torch
import torchmetrics


class ContrastiveAccuracy(torchmetrics.MeanMetric):
    def __init__(self, neg_count):
        super().__init__()

        self.neg_count = neg_count

    def update(self, *input):
        (target, predict), _ = input

        mn = 1 - torch.eye(target.size(0))
        neg_ix = torch.multinomial(mn, self.neg_count)
        all_counterparty = torch.cat([predict.unsqueeze(1), predict[neg_ix]], dim=1)

        similarity = torch.cosine_similarity(target.unsqueeze(1), all_counterparty, dim=2)
        accuracy = (similarity.argmax(dim=1) == 0).float().mean()

        super().update(accuracy)


class ContrastiveAccuracy2(torchmetrics.MeanMetric):
    def __init__(self, neg_count):
        super().__init__()

        self.neg_count = neg_count

    def update(self, *input):
        (target, predict), _ = input

        mn = 1 - torch.eye(target.size(0))
        neg_ix = torch.multinomial(mn, self.neg_count)
        all_counterparty = torch.cat([target.unsqueeze(1), target[neg_ix]], dim=1)

        similarity = torch.cosine_similarity(predict.unsqueeze(1), all_counterparty, dim=2)
        accuracy = (similarity.argmax(dim=1) == 0).float().mean()

        super().update(accuracy)
