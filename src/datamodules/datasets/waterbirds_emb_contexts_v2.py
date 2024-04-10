from typing import Optional
import logging
import random

import numpy as np

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.datamodules.datasets.waterbirds import WaterbirdsExtracted
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


def _sample(
        dataset,
        dataset_groups,
        num_examples: int,
        minority_group_proportion: float,
        remaining_mask: Optional[np.ndarray] = None,
) -> list[Example]:
    """Samples a subset of examples given a dataset and groups of its examples."""

    if remaining_mask is None:
        remaining_mask = np.ones(len(dataset), dtype=bool)

    minority_proportion = 0.5 * minority_group_proportion

    group_proportions = [0.5 - minority_proportion, minority_proportion,
                         minority_proportion, 0.5 - minority_proportion]

    group_counts = get_group_counts_based_on_proportions(
        num_examples=num_examples,
        group_proportions=group_proportions)

    indices = []
    for i, group_count in enumerate(group_counts):
        all_group_items = np.where(remaining_mask & (dataset_groups == i))[0]
        random_items = np.random.choice(all_group_items, group_count, replace=False)
        indices.extend(random_items)

    labels = [dataset[idx][1] for idx in indices]
    spurious_labels = [dataset[idx][2] for idx in indices]
    examples = list(zip(indices, spurious_labels, labels))
    random.shuffle(examples)
    return examples


class WaterbirdsEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for Waterbirds in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_split: str,
                 query_split: str,
                 context_class_size: int,
                 minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 v1_behavior: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 randomly_swap_labels: bool = False,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_norm_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 saved_data_path: Optional[str] = None):
        """
        Args:

        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_split (str): The split where context examples are selected from ('train' or 'val').
        query_split (str): The split where query examples are selected from ('train' or 'val').
        context_class_size (int): The size of each class in the context.
        minority_group_proportion (float): The proportion of the minority group in the context per class.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'no_spurious'(x), 'sum'(x+c), 'separate_token'(x, c)
                                or 'sum_with_spurious'(x+c, c).
        sp_token_generation_mode (str): Specifies whether the representations of two spurious labels should be
                                        'opposite' or 'random'.
        v1_behavior (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        randomly_swap_labels (bool): Whether to randomly swap labels (0 -> 1 and 1 -> 0) when creating an ILC instance.
        label_noise_ratio_interval (list or None): Interval for the ratio of label noise. 
                                If None, no label noise is added.
        input_noise_norm_interval (list or None): Interval for the norm of Gaussian noise.
                                If None, no Gaussian noise is added to representations.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None). If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(WaterbirdsEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            v1_behavior=v1_behavior,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
            saved_data_path=saved_data_path,
        )

        self._minority_group_proportion = minority_group_proportion
        self._randomly_swap_labels = randomly_swap_labels

        dataset = WaterbirdsExtracted(root_dir,
                                      encoding_extractor=encoding_extractor)

        train_set = dataset.get_subset("train")
        train_groups = np.stack([(2 * y + c) for _, y, c, _ in train_set])

        val_set = dataset.get_subset("val")
        val_groups = np.stack([(2 * y + c) for _, y, c, _ in val_set])

        test_set = dataset.get_subset("test")
        test_groups = np.stack([(2 * y + c) for _, y, c, _ in test_set])

        self._context_split = context_split
        if context_split == 'train':
            self._context_set = train_set
            self._context_groups = train_groups
        elif context_split == 'val':
            self._context_set = val_set
            self._context_groups = val_groups
        else:
            raise ValueError()

        self._query_split = query_split
        if query_split == 'train':
            self._query_set = train_set
            self._query_groups = train_groups
        elif query_split == 'val':
            self._query_set = val_set
            self._query_groups = val_groups
        elif query_split == 'test':
            self._query_set = test_set
            self._query_groups = test_groups
        else:
            raise ValueError()

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are lists of 2 * context_class_size (id, spurious, label) triplets.
        """

        context = _sample(dataset=self._context_set,
                          dataset_groups=self._context_groups,
                          num_examples=2 * self._context_class_size,
                          minority_group_proportion=self._minority_group_proportion)

        if self._context_split == self._query_split:
            remaining_mask = np.ones(len(self._context_set), dtype=bool)
            for idx, _, _ in context:
                remaining_mask[idx] = False
        else:
            remaining_mask = None

        queries = _sample(dataset=self._query_set,
                          dataset_groups=self._query_groups,
                          num_examples=2 * self._context_class_size,
                          minority_group_proportion=0.5,
                          remaining_mask=remaining_mask)

        if self._randomly_swap_labels and np.random.rand() < 0.5:
            context = [(idx, sp, 1 - label) for idx, sp, label in context]
            queries = [(idx, sp, 1 - label) for idx, sp, label in queries]

        return context, queries

    def _prepare_context_image_encodings(
            self,
            context: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing context example encodings."""
        return np.stack(
            [self._context_set[idx][0] for idx, _, _ in context]
        )

    def _prepare_query_image_encodings(
            self,
            queries: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing query example encodings."""
        return np.stack(
            [self._query_set[idx][0] for idx, _, _ in queries]
        )
