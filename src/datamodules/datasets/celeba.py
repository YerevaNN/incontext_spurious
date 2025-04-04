from typing import Optional, Tuple
import logging
import os.path

import numpy as np
from torch.utils.data import Dataset
from wilds.datasets.celebA_dataset import CelebADataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class CelebASubsetForEncodingExtraction(Dataset):
    def __init__(self, wilds_celeba_subset: WILDSSubset):
        self._wilds_celeba_subset = wilds_celeba_subset

        if hasattr(self._wilds_celeba_subset, 'collate'):  # necessary method for constructing subsets
            setattr(self, 'collate', getattr(self._wilds_celeba_subset, 'collate'))

    def __getitem__(self, idx):
        x, y, metadata = self._wilds_celeba_subset[idx]
        return x, self._wilds_celeba_subset.indices[idx]

    def __len__(self):
        return len(self._wilds_celeba_subset)


class CelebAForEncodingExtraction:
    def __init__(self, root_dir: str, download: bool = False):
        # disable SSL certificate verification when using a library, to avoid WILDS's issue
        import ssl, urllib
        
        context = ssl._create_unverified_context()
        original_urlopen = urllib.request.urlopen

        def urlopen_monkeypatch(url, *args, **kwargs):
            return original_urlopen(url, context=context, *args, **kwargs)

        urllib.request.urlopen = urlopen_monkeypatch

        self._wilds_celeba = CelebADataset(root_dir=root_dir, download=download)

    def get_subset(self, *args, **kwargs) -> CelebASubsetForEncodingExtraction:
        return CelebASubsetForEncodingExtraction(
            self._wilds_celeba.get_subset(*args, **kwargs))


class CelebASubsetExtracted(Dataset):
    def __init__(self,
                 wilds_celeba_subset: WILDSSubset,
                 encodings: np.ndarray,
                 index_map: np.ndarray,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):

        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

        # permute rows of `encodings` such that the i-th row corresponds to the i-th example of the subset
        n = len(wilds_celeba_subset)
        row_indices = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            idx_within_full_celeba = wilds_celeba_subset.indices[idx]
            encoding_row_index = index_map[idx_within_full_celeba]
            assert encoding_row_index != -1
            row_indices[idx] = encoding_row_index
        encodings = encodings[row_indices]

        self.x = encodings.copy()
        self.y = wilds_celeba_subset.y_array.numpy()
        self.c = 1 - wilds_celeba_subset.metadata_array[:, 0].numpy()  # Swap spurious labels

    def __getitem__(self, indices) -> Tuple[np.ndarray, Examples]:
        x = self.x[indices].copy()
        y = self.y[indices].copy()
        c = self.c[indices].copy()

        # add more spurious information if specified
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


class CelebAExtracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add
        self._wilds_celeba = CelebADataset(root_dir=root_dir)

    def get_subset(self, split, *args, **kwargs) -> CelebASubsetExtracted:
        encodings_data = np.load(
            os.path.join(self._root_dir, "celeba", self._encoding_extractor, split, "combined.npz"))
        return CelebASubsetExtracted(
            wilds_celeba_subset=self._wilds_celeba.get_subset(split, *args, **kwargs),
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
            reverse_task=self._reverse_task,
            sp_vector_to_add=self._sp_vector_to_add,
        )
