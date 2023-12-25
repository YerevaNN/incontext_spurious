import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle

from src.utils import group_indices, sample

log = logging.getLogger(__name__)

class ICLDataSet(Dataset):
    def __init__(
            self,
            context_params: dict,
            cache_path: str,
            split: str,
            num_classes: int,
            num_confounders: int,
            train_dataset = None, # relevant for val and test splits
        ):
        super().__init__()
        self._context_params = context_params
        self._cache_path = cache_path
        self._split = split

        self._num_classes = num_classes
        self._num_confounders = num_confounders

        self._train_dataset = train_dataset

        self._dataset_path =  os.path.join(cache_path, f"{split}.pt")

        self.x, self.y, self.c = torch.load(self._dataset_path)

        if self._split == "train":
            self._groups_idx = group_indices(self.y, self.c)

    def _getsize(self, i, j, context_params):
        size = context_params['size']

        if i == 0:
            remaining_size = int(size * context_params['label_ratio'])
        else:
            remaining_size = size - int(size * context_params['label_ratio'])

        
        if i != j:  # minority
            final_size = int(remaining_size * context_params['minority_ratio'])
        else:
            final_size = remaining_size - int(remaining_size * context_params['minority_ratio'])

        return final_size
    
    @staticmethod
    def _get_label_ratio(mode):
        if mode == "balanced":
            return 0.5 # NOTE: for binary classification
        
        if mode == "random":
            return np.random.uniform(low=0.25, high=0.75)
        
        return mode

    @staticmethod
    def _get_minority_ratio(mode):
        if mode == "balanced":
            return 0.5 # NOTE: for binary classification
        
        if mode == "random":
            return np.random.uniform(low=0.25, high=0.5)

        return mode

    def _getcontext(self, context_params):
        context_params['label_ratio'] = self._get_label_ratio(context_params['label_ratio'])
        context_params['minority_ratio'] = self._get_minority_ratio(context_params['minority_ratio'])

        context_sizes = {
            i: {
                j: self._getsize(i, j, context_params)
                for j in range(self._num_confounders)
            }
            for i in range(self._num_classes)
        }

        context_idx = torch.concat(
            [
                sample(self._groups_idx[i][j], num_samples=context_sizes[i][j])
                for i in range(self._num_classes)
                for j in range(self._num_confounders)
            ]
        )

        x, y, c = shuffle(self.x[context_idx], self.y[context_idx], self.c[context_idx])

        return x, y, c

    def __getitem__(self, idx):
        if self._split == "train":
            x, y, c = self._getcontext(self._context_params)
            return x, y, c
        
        x_context, y_context, c_context = self._train_dataset._getcontext(self._context_params)

        x_query, y_query, c_query = self.x[[idx]], self.y[[idx]], self.c[[idx]]

        return (x_context, y_context), x_query, y_query, c_query

    def __len__(self):
        if self._split == "train":
            return 1280 # TODO: Adjust this part

        return len(self.y)
