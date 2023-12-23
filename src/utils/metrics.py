from torch import Tensor
from torchmetrics import Metric
import torch

class WorstGroupAccuracy(Metric):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.add_state("correct", default=torch.zeros(num_groups), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_groups), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, groups: Tensor):
        preds = torch.argmax(preds, dim=-1)
        is_correct = (preds == targets).long()
        for g in range(self.num_groups):
            self.correct[g] += is_correct[groups == g].sum()
            self.total[g] += (groups == g).long().sum()

    def compute(self) -> Tensor:
        worst_group_acc = torch.tensor(1.0)
        for c, t in zip(self.correct, self.total):
            if t > 0:
                worst_group_acc = min(worst_group_acc, c.float() / t)
        return worst_group_acc

    def compute_all_groups(self) -> Tensor:
        ret = self.correct.float() / self.total
        ret[self.total == 0] = float('nan')
        return ret


class WorstGroupLoss(Metric):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.add_state("sum", default=torch.zeros(num_groups, dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_groups),
                       dist_reduce_fx="sum")

    def update(self, losses: Tensor, groups: Tensor):
        for g in range(self.num_groups):
            self.sum[g] += losses[groups == g].sum()
            self.total[g] += (groups == g).long().sum()

    def compute(self) -> Tensor:
        worst_group_loss = torch.tensor(-torch.inf, dtype=torch.float)
        for s, t in zip(self.sum, self.total):
            if t > 0:
                worst_group_loss = max(worst_group_loss, s.float() / t)
        return worst_group_loss

    def compute_all_groups(self) -> Tensor:
        ret = self.sum.float() / self.total
        ret[self.total == 0] = float('nan')
        return ret
