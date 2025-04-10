from typing import Optional
import logging
import os

import pandas as pd
import numpy as np

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.utils.dataset_helpers.context_prep_utils import generate_spurious_labels, prepare_context_or_query

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


def _sample(
        df,
        category: str,
        num_context_examples: int,
        num_query_examples: int,
):
    mask = (df["category"] == category)
    context_indices = df.loc[mask, "image_id"].sample(num_context_examples, replace=False)
    mask.loc[context_indices.index] = False
    query_indices = df.loc[mask, "image_id"].sample(num_query_examples, replace=True)  # queries can repeat
    return context_indices.tolist(), query_indices.tolist()


class CUBEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for Caltech-UCSD Birds (CUB) in-context learning instances."""

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 context_minority_group_proportion: float,
                 query_minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_norm_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 simpler_construction: bool = False):
        """
        Additional args:

        dataset_path (str): The path to the dataset directory.
        context_class_size (int): The size of each class in the context.
        context_minority_group_proportion (float): The proportion of the minority group in the context per class.
        query_minority_group_proportion (float): The proportion of the minority group in the context per class.
        """
        super(CUBEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
            simpler_construction=simpler_construction,
        )

        # Prepare encodings and data files
        encodings_data = np.load(os.path.join(dataset_path, encoding_extractor, "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._dataframe = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'), sep=' ',
                                      header=None, names=['image_id', 'category'])

        # Calculate the value counts of categories that appear more than context_class_size times
        valid_categories = self._dataframe['category'].value_counts()[lambda x: x > context_class_size].index

        # Filter the dataframe to only include rows with those categories
        self._dataframe = self._dataframe[self._dataframe['category'].isin(valid_categories)]

        self._categories = self._dataframe["category"].unique()

        self._context_minority_group_proportion = context_minority_group_proportion
        self._query_minority_group_proportion = query_minority_group_proportion

    def _generate_context_and_queries(
            self,
            num_context_examples: int,
            num_query_examples: int,
    ) -> (Examples, Examples):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are of type Examples.
        """
        # Randomly selecting two categories
        category1, category2 = np.random.choice(self._categories, size=2, replace=False)

        # sampling context and query examples for each class
        context_cat1_size = num_context_examples // 2
        context_cat2_size = (num_context_examples + 1) // 2
        query_cat1_size = num_query_examples // 2
        query_cat2_size = (num_query_examples + 1) // 2

        cat1_context_indices, cat1_query_indices = _sample(
            df=self._dataframe,
            category=category1,
            num_context_examples=context_cat1_size,
            num_query_examples=query_cat1_size,
        )
        cat2_context_indices, cat2_query_indices = _sample(
            df=self._dataframe,
            category=category2,
            num_context_examples=context_cat2_size,
            num_query_examples=query_cat2_size,
        )

        # Shuffle the 2 classes to remove any bias for short context sizes
        # Also ensure that (0, 0) and (1, 1) are majority groups
        permutted_labels = np.random.permutation([0, 1])
        cat1_class_label, cat2_class_label = permutted_labels
        spurious_label1, spurious_label2 = permutted_labels

        # Generate spurious labels for context examples
        cat1_context_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, context_cat1_size, self._context_minority_group_proportion)
        cat2_context_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, context_cat2_size, self._context_minority_group_proportion)

        # Generate spurious labels for queries
        cat1_query_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, query_cat1_size, self._query_minority_group_proportion)
        cat2_query_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, query_cat2_size, self._query_minority_group_proportion)

        # Prepare full context information
        context = prepare_context_or_query(
            cat1_indices=cat1_context_indices,
            cat2_indices=cat2_context_indices,
            cat1_spurious_labels=cat1_context_spurious_labels,
            cat2_spurious_labels=cat2_context_spurious_labels,
            cat1_class_label=cat1_class_label,
            cat2_class_label=cat2_class_label)

        queries = prepare_context_or_query(
            cat1_indices=cat1_query_indices,
            cat2_indices=cat2_query_indices,
            cat1_spurious_labels=cat1_query_spurious_labels,
            cat2_spurious_labels=cat2_query_spurious_labels,
            cat1_class_label=cat1_class_label,
            cat2_class_label=cat2_class_label)

        return context, queries

    def _prepare_image_encodings(
            self,
            examples: Examples,
    ) -> np.ndarray:
        """
        Transforms image encodings based on their labels using the appropriate encoding transformer.

        Args:
            examples: List of tuples for the context images containing IDs, spurious labels, and class labels.

        Returns: np.ndarray of shape (num_examples, dim).
        """
        indices = examples[:, 0]
        return self._encodings[self._encodings_indices_map[indices]].copy()

    def _prepare_context_image_encodings(
            self,
            context: Examples,
    ) -> np.ndarray:
        return self._prepare_image_encodings(context)

    def _prepare_query_image_encodings(
            self,
            queries: Examples,
    ) -> np.ndarray:
        return self._prepare_image_encodings(queries)
