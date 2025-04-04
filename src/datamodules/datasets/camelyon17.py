from typing import Optional, Tuple
import dataclasses
import logging
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class Camelyon17SubsetForEncodingExtraction(Dataset):
    def __init__(self, wilds_camelyon17_subset: WILDSSubset):
        self._wilds_camelyon17_subset = wilds_camelyon17_subset

        if hasattr(self._wilds_camelyon17_subset, 'collate'):  # necessary method for constructing subsets
            setattr(self, 'collate', getattr(self._wilds_camelyon17_subset, 'collate'))

    def __getitem__(self, idx):
        x, y, metadata = self._wilds_camelyon17_subset[idx]
        return x, self._wilds_camelyon17_subset.indices[idx]

    def __len__(self):
        return len(self._wilds_camelyon17_subset)


class Camelyon17ForEncodingExtraction:
    def __init__(self, root_dir: str, download: bool = False):
        self._wilds_camelyon17 = Camelyon17Dataset(root_dir=root_dir, download=download)

    def get_subset(self, *args, **kwargs) -> Camelyon17SubsetForEncodingExtraction:
        return Camelyon17SubsetForEncodingExtraction(
            self._wilds_camelyon17.get_subset(*args, **kwargs))


@dataclasses.dataclass
class CustomExtractedWILDSSubset:
    encodings: np.ndarray
    y_array: torch.Tensor
    metadata_array: torch.Tensor

    def __len__(self):
        return len(self.y_array)


class Camelyon17SubsetExtracted(Dataset):
    def __init__(self,
                 custom_subset: CustomExtractedWILDSSubset,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._custom_subset = custom_subset
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

    def __getitem__(self, indices) -> Tuple[np.ndarray, Examples]:
        x = self._custom_subset.encodings[indices].copy()
        y = self._custom_subset.y_array[indices].numpy()

        # metadata fields: ['hospital', 'slide', 'y', 'from_source_domain']
        hospital_id = self._custom_subset.metadata_array[indices, 0].numpy()
        # Our splits:
        # * train: hospital 0 and hospital 3 of the original train split
        # * val:   hospital 0 and hospital 3 of the original id_val split
        # * test:  hospital 1 of the original val split and hospital 2 of the original test split
        c = (hospital_id < 2)

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
        return len(self._custom_subset)


def _put_encodings_in_right_order(
        indices_within_full_camelyon17: np.ndarray,
        encodings: np.ndarray,
        index_map: np.ndarray,
) -> np.ndarray:
    """Permutes rows of `encodings` such that the i-th row corresponds to the i-th example of the subset."""
    n = len(indices_within_full_camelyon17)
    row_indices = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        idx_within_full_camelyon17 = indices_within_full_camelyon17[idx],
        encoding_row_index = index_map[idx_within_full_camelyon17]
        assert encoding_row_index != -1
        row_indices[idx] = encoding_row_index
    return encodings[row_indices]


def _select_hospitals(
        ds: CustomExtractedWILDSSubset,
        selected_hospital_ids: list[int],
) -> CustomExtractedWILDSSubset:
    hospital_ids = ds.metadata_array[:, 0].numpy()
    mask = np.zeros(hospital_ids.shape, dtype=np.bool_)
    for h in selected_hospital_ids:
        mask |= (hospital_ids == h)
    return CustomExtractedWILDSSubset(
        y_array=ds.y_array[mask],
        metadata_array=ds.metadata_array[mask],
        encodings=ds.encodings[mask],
    )


def _concatenate_custom_wilds_subsets(
        part1: CustomExtractedWILDSSubset,
        part2: CustomExtractedWILDSSubset,
) -> CustomExtractedWILDSSubset:
    return CustomExtractedWILDSSubset(
        y_array=torch.concat(
            [part1.y_array, part2.y_array], dim=0
        ),
        metadata_array=torch.concat(
            [part1.metadata_array, part2.metadata_array], dim=0
        ),
        encodings=np.concatenate(
            [part1.encodings, part2.encodings], axis=0
        )
    )


class Camelyon17Extracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add
        self._wilds_camelyon17 = Camelyon17Dataset(root_dir=root_dir)

    def _select_wilds_subset(self,
                             wilds_split: str) -> CustomExtractedWILDSSubset:
        encodings_data = np.load(
            os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, wilds_split, "combined.npz"))
        wilds_subset = self._wilds_camelyon17.get_subset(wilds_split)

        encodings = _put_encodings_in_right_order(
            indices_within_full_camelyon17=wilds_subset.indices,
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
        )

        return CustomExtractedWILDSSubset(
            y_array=wilds_subset.y_array,
            metadata_array=wilds_subset.metadata_array,
            encodings=encodings,
        )

    def get_subset(self, split, *args, **kwargs) -> Camelyon17SubsetExtracted:
        """Constructs our custom splits.

        The following splits are defined:
            * train: hospital 0 and hospital 3 of the original train split
            * val:   hospital 0 and hospital 3 of the original id_val split
            * test:  hospital 1 of the original val split and hospital 2 of the original test split
        """
        if split in ['train', 'val']:
            if split == 'train':
                wilds_split = 'train'
            else:
                wilds_split = 'id_val'

            ds = self._select_wilds_subset(
                wilds_split=wilds_split,
            )
            ds = _select_hospitals(
               ds, selected_hospital_ids=[0, 3]
            )

            return Camelyon17SubsetExtracted(
                custom_subset=ds,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

        if split == 'test':
            part1 = self._select_wilds_subset(
                wilds_split='val',
            )  # single hospital 1

            part2 = self._select_wilds_subset(
                wilds_split='test',
            )  # single hospital 2

            joint = _concatenate_custom_wilds_subsets(part1, part2)

            return Camelyon17SubsetExtracted(
                custom_subset=joint,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

        raise ValueError(f'Unexpected value {split=}')
