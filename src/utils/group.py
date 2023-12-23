import numpy as np
import torch

def group_idx(y, c):
    num_classes = len(np.unique(y))
    num_confounders = len(np.unique(c))

    idx_range = torch.arange(len(y))

    group_masks = {
        i: {
            j: idx_range[torch.logical_and(y == i, c == j)]
            for j in range(num_confounders)
        }
        for i in range(num_classes)
    }

    return group_masks