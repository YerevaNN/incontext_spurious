import logging
import os.path

import numpy as np
from torch.utils.data import Dataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class CivilCommentsSubsetForEncodingExtraction(Dataset):
    def __init__(self, wilds_civilcomments_subset: WILDSSubset):
        self._wilds_civilcomments_subset = wilds_civilcomments_subset

        if hasattr(self._wilds_civilcomments_subset, 'collate'): # necessary method for constructing subsets
            setattr(self, 'collate', getattr(self._wilds_civilcomments_subset, 'collate'))

    def __getitem__(self, idx):
        x, y, metadata = self._wilds_civilcomments_subset[idx]

        if str(x) in ['nan']: # We have exactly two missing values. Fill them with "".
            x = ""

        return x, self._wilds_civilcomments_subset.indices[idx]

    def __len__(self):
        return len(self._wilds_civilcomments_subset)


class CivilCommentsForEncodingExtraction:
    def __init__(self, root_dir: str, download: bool = False):
        self._wilds_civilcomments = CivilCommentsDataset(root_dir=root_dir, download=download)
        
    def get_subset(self, *args, **kwargs) -> CivilCommentsSubsetForEncodingExtraction:
        return CivilCommentsSubsetForEncodingExtraction(
            self._wilds_civilcomments.get_subset(*args, **kwargs))


class CivilCommentsSubsetExtracted(Dataset):
    def __init__(self,
                 wilds_civilcomments_subset: WILDSSubset,
                 encodings: np.ndarray,
                 index_map: np.ndarray,
                 reverse_task: bool = False):
        self._wilds_civilcomments_subset = wilds_civilcomments_subset
        self._reverse_task = reverse_task

        # permute rows of `encodings` such that the i-th row corresponds to the i-th example of the subset
        n = len(wilds_civilcomments_subset)
        row_indices = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            idx_within_full_civilcomments = wilds_civilcomments_subset.indices[idx]
            encoding_row_index = index_map[idx_within_full_civilcomments]
            assert encoding_row_index != -1
            row_indices[idx] = encoding_row_index
        self._encodings = encodings[row_indices]

    def __getitem__(self, indices) -> (np.ndarray, Examples):
        x = self._encodings[indices].copy()
        y = self._wilds_civilcomments_subset.y_array[indices].numpy()
        c_features = self._wilds_civilcomments_subset.metadata_array[indices, :15].numpy()
        c = c_features[:, 6] # 6 indicates "Black" feature

        # reverse the task if specified
        if not self._reverse_task:
            examples = np.stack([indices, c, y], axis=1)
        else:
            examples = np.stack([indices, y, c], axis=1)

        return x, examples

    def __len__(self):
        return len(self._wilds_civilcomments_subset)


class CivilCommentsExtracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._wilds_civilcomments = CivilCommentsDataset(root_dir=root_dir)

    def get_subset(self, split, *args, **kwargs) -> CivilCommentsSubsetExtracted:
        encodings_data = np.load(
            os.path.join(self._root_dir, "civilcomments", self._encoding_extractor, split, "combined.npz"))
        return CivilCommentsSubsetExtracted(
            wilds_civilcomments_subset=self._wilds_civilcomments.get_subset(split, *args, **kwargs),
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
            reverse_task=self._reverse_task,
        )
