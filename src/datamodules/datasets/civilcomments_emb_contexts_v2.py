from typing import Optional
import logging

import numpy as np

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.datamodules.datasets.civilcomments import CivilCommentsExtracted
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


def _sample(
        dataset,
        dataset_groups,
        num_examples: int,
        group_proportions: list[float],
        remaining_mask: Optional[np.ndarray] = None,
        replace: bool = False,
) -> Examples:
    """Samples a subset of examples given a dataset and groups of its examples."""

    if remaining_mask is None:
        remaining_mask = np.ones(len(dataset), dtype=bool)

    group_counts = get_group_counts_based_on_proportions(
        num_examples=num_examples,
        group_proportions=group_proportions)

    indices = []
    for i, group_count in enumerate(group_counts):
        all_group_items = np.where(remaining_mask & (dataset_groups == i))[0]
        random_items = np.random.choice(all_group_items, group_count, replace=replace)
        indices.extend(random_items)

    _, examples = dataset[indices]
    return np.random.permutation(examples)


class CivilCommentsEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for CivilComments in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_split: str,
                 query_split: str,
                 context_class_size: int,
                 context_group_proportions: list[float],
                 query_group_proportions: list[float],
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool = False,
                 reverse_task: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 randomly_swap_labels: bool = False,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None):
        """
        Args:

        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_split (str): The split where context examples are selected from ('train' or 'val').
        query_split (str): The split where query examples are selected from ('train' or 'val').
        context_class_size (int): The size of each class in the context.
        context_group_proportions(list[float]): Proportions for the 4 groups in contexts.
        query_group_proportions (list[float]): Proportions for the 4 groups in queries.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
        sp_token_generation_mode (str): Specifies whether the representations of two spurious labels should be
                                        'opposite' or 'random'.
        use_context_as_intermediate_queries (bool): Whether intermediate queries should be the context examples.
        reverse_task (bool): Whether to predict background instead of foreground.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        randomly_swap_labels (bool): Whether to randomly swap labels (0 -> 1 and 1 -> 0) when creating an ILC instance.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None): If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        """
        assert spurious_setting in ['wb_erm', 'wb_dro', 'swap_erm', 'swap_dro']
        assert not ((spurious_setting in ['swap_erm', 'swap_dro'])
                    and any(context_group_proportions[1:3]) 
                    and any(query_group_proportions[1:3]))

        super(CivilCommentsEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
        )

        self._context_group_proportions = context_group_proportions
        self._query_group_proportions = query_group_proportions
        self._randomly_swap_labels = randomly_swap_labels

        dataset = CivilCommentsExtracted(root_dir,
                                      encoding_extractor=encoding_extractor,
                                      reverse_task=reverse_task)

        train_set = dataset.get_subset("train")
        val_set = dataset.get_subset("val")
        test_set = dataset.get_subset("test")

        def get_groups(ds):
            all_indices = np.arange(len(ds))
            _, examples = ds[all_indices]
            return 2 * examples[:, 2] + examples[:, 1]  # 2 * y + c

        self._context_split = context_split
        if context_split == 'train':
            self._context_set = train_set
            self._context_groups = get_groups(train_set)
        elif context_split == 'val':
            self._context_set = val_set
            self._context_groups = get_groups(val_set)
        else:
            raise ValueError()

        self._query_split = query_split
        if query_split == 'train':
            self._query_set = train_set
            self._query_groups = get_groups(train_set)
        elif query_split == 'val':
            self._query_set = val_set
            self._query_groups = get_groups(val_set)
        elif query_split == 'test':
            self._query_set = test_set
            self._query_groups = get_groups(test_set)
        else:
            raise ValueError()

    def _generate_context_and_queries(
            self,
            num_context_examples: int,
            num_query_examples: int,
    ) -> (Examples, Examples):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are of type Examples.
        """
        context = _sample(dataset=self._context_set,
                          dataset_groups=self._context_groups,
                          num_examples=num_context_examples,
                          group_proportions=self._context_group_proportions,
                          replace=False)

        if self._context_split == self._query_split:
            remaining_mask = np.ones(len(self._context_set), dtype=bool)
            remaining_mask[context[:, 0]] = False
        else:
            remaining_mask = None

        queries = _sample(dataset=self._query_set,
                          dataset_groups=self._query_groups,
                          num_examples=num_query_examples,
                          group_proportions=self._query_group_proportions,
                          remaining_mask=remaining_mask,
                          replace=True)  # NOTE: it is ok to repeat queries. This allows larger context sizes.

        if self._randomly_swap_labels and np.random.rand() < 0.5:
            context[:, 2] = 1 - context[:, 2]
            queries[:, 2] = 1 - queries[:, 2]

        return context, queries

    def _prepare_context_image_encodings(
            self,
            context: Examples,
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing context example encodings."""
        indices = context[:, 0]
        return self._context_set[indices][0]

    def _prepare_query_image_encodings(
            self,
            queries: Examples,
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing query example encodings."""
        indices = queries[:, 0]
        return self._query_set[indices][0]
