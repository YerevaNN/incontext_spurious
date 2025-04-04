from typing import Optional, Tuple
import logging
import os.path

import numpy as np
from torch.utils.data import Dataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class WaterbirdsSubsetForEncodingExtraction(Dataset):
    def __init__(self, wilds_waterbirds_subset: WILDSSubset):
        self._wilds_waterbirds_subset = wilds_waterbirds_subset

        if hasattr(self._wilds_waterbirds_subset, 'collate'): # necessary method for constructing subsets
            setattr(self, 'collate', getattr(self._wilds_waterbirds_subset, 'collate'))

    def __getitem__(self, idx):
        x, y, metadata = self._wilds_waterbirds_subset[idx]
        return x, self._wilds_waterbirds_subset.indices[idx]

    def __len__(self):
        return len(self._wilds_waterbirds_subset)


class WaterbirdsForEncodingExtraction:
    def __init__(self, root_dir: str, download: bool = False):
        self._wilds_waterbirds = WaterbirdsDataset(root_dir=root_dir, download=download)

    def get_subset(self, *args, **kwargs) -> WaterbirdsSubsetForEncodingExtraction:
        return WaterbirdsSubsetForEncodingExtraction(
            self._wilds_waterbirds.get_subset(*args, **kwargs))


class WaterbirdsSubsetExtracted(Dataset):
    def __init__(self,
                 wilds_waterbirds_subset: WILDSSubset,
                 encodings: np.ndarray,
                 index_map: np.ndarray,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):

        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

        # permute rows of `encodings` such that the i-th row corresponds to the i-th example of the subset
        n = len(wilds_waterbirds_subset)
        row_indices = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            idx_within_full_waterbirds = wilds_waterbirds_subset.indices[idx]
            encoding_row_index = index_map[idx_within_full_waterbirds]
            assert encoding_row_index != -1
            row_indices[idx] = encoding_row_index
        encodings = encodings[row_indices]

        self.x = encodings.copy()
        self.y = wilds_waterbirds_subset.y_array.numpy()
        self.c = wilds_waterbirds_subset.metadata_array[:, 0].numpy()

    def __getitem__(self, indices) -> Tuple[np.ndarray, Examples]:
        x = self.x[indices].copy()
        y = self.y[indices].copy()
        c = self.c[indices].copy()

        # add more background information if specified
        if self._sp_vector_to_add is not None:
            x += np.outer(2 * c - 1, self._sp_vector_to_add)

        # reverse the task if specified
        if not self._reverse_task:
            examples = np.stack([indices, c, y], axis=1)
        else:
            examples = np.stack([indices, y, c], axis=1)

        return x, examples

    def __len__(self):
        return len(self.y)


class WaterbirdsExtracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add
        self._wilds_waterbirds = WaterbirdsDataset(root_dir=root_dir)

    def get_subset(self, split, *args, **kwargs) -> WaterbirdsSubsetExtracted:
        encodings_data = np.load(
            os.path.join(self._root_dir, "waterbirds", self._encoding_extractor, split, "combined.npz"))
        return WaterbirdsSubsetExtracted(
            wilds_waterbirds_subset=self._wilds_waterbirds.get_subset(split, *args, **kwargs),
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
            reverse_task=self._reverse_task,
            sp_vector_to_add=self._sp_vector_to_add,
        )
