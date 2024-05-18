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


class INaturalistEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for iNaturalist in-context learning instances."""

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 class1_split: str,
                 class2_split: str,
                 context_class_size: int,
                 context_minority_group_proportion: float,
                 query_minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 select_classes_from_supercategories: bool = False,
                 use_context_as_intermediate_queries: bool = False,
                 reverse_task: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_norm_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 swapping_minority_proportion_context: Optional[float] = None,
                 swapping_minority_proportion_query: Optional[float] = None,
                 points_to_swap_range: Optional[list] = None,
                 random_task_switching: bool = False,
                 ):
        """
        Arguments:
        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        class1_split (str): The type of data split of class 1 (e.g., 'inner_train', 'inner_val', 'outer').
        class2_split (str): The type of data split of class 2 (e.g., 'inner_train', 'inner_val', 'outer').
        context_class_size (int): The size of each class in the context.
        context_minority_group_proportion (float): The proportion of the minority group in the context per class.
        query_minority_group_proportion (float): The proportion of the minority group in the context per class.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
        sp_token_generation_mode (str): Specifies whether the representations of two spurious labels should be
                                        'opposite' or 'random'.
        select_classes_from_supercategories (bool): Whether to select classes from the same supercategories.
        use_context_as_intermediate_queries (bool): Whether intermediate queries should be the context examples.
        reverse_task (bool): Whether to predict the artificially created spurious variable instead of the object.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        label_noise_ratio_interval (list or None): Interval for the ratio of label noise. 
                                If None, no label noise is added.
        input_noise_norm_interval (list or None): Interval for the norm of Gaussian noise.
                                If None, no Gaussian noise is added to representations.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None). If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        swapping_minority_proportion_context (float): Minority group proportion to create via swapping in the context.
        swapping_minority_proportion_query (float): Minority group proportion to create via swapping in the queries.
        points_to_swap_range (list): The range of the number of points to swap in the selected vectors.
        random_task_switching (bool): Whether to switch the task randomly for each ICL instance.
        """
        assert spurious_setting in ['inat_no_spurious', 'inat_sum_erm', 'inat_sum_dro', 'swap_erm', 'swap_dro']
        assert not ((spurious_setting in ['swap_erm', 'swap_dro'])
                    and context_minority_group_proportion != 0 
                    and query_minority_group_proportion != 0)
        if reverse_task or random_task_switching:
            assert spurious_setting in ['inat_sum_erm', 'inat_sum_dro', 'swap_erm', 'swap_dro']

        super(INaturalistEmbContextsDatasetV2, self).__init__(
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
            swapping_minority_proportion_context=swapping_minority_proportion_context,
            swapping_minority_proportion_query=swapping_minority_proportion_query,
            points_to_swap_range=points_to_swap_range,
        )

        # Prepare encodings and data files
        encodings_data = np.load(os.path.join(dataset_path, encoding_extractor, "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._dataframe = pd.read_csv(os.path.join(dataset_path, "prepared_data.csv"))
        self._dataframe = self._dataframe[self._dataframe['split'].isin([class1_split, class2_split])]

        # Unique categories in the dataset
        self._class1_split = class1_split
        self._class2_split = class2_split
        if class1_split == class2_split:
            if select_classes_from_supercategories:
                self._supercategories = [
                    supercategory for supercategory in self._dataframe["supercategory"].unique() 
                    if len(self._dataframe[self._dataframe['supercategory'] == supercategory]['category'].unique()) >= 2]
                
                self._categories = {
                    supercategory: self._dataframe[self._dataframe['supercategory'] == supercategory]["category"].unique()
                    for supercategory in self._supercategories
                }
            else:
                self._supercategories = None

                self._categories = self._dataframe["category"].unique()
        else:
            if select_classes_from_supercategories:
                self._supercategories = [
                    supercategory for supercategory in self._dataframe["supercategory"].unique() 
                    if len(self._dataframe[(self._dataframe['supercategory'] == supercategory) 
                                           & (self._dataframe['split'] == class1_split)]['category'].unique()) > 0 
                    and len(self._dataframe[(self._dataframe['supercategory'] == supercategory) 
                                            & (self._dataframe['split'] == class2_split)]['category'].unique()) > 0]
            
                self._categories = [{
                        supercategory: self._dataframe[(self._dataframe['supercategory'] == supercategory) 
                                                       & (self._dataframe['split'] == class_split)]["category"].unique()
                        for supercategory in self._supercategories}
                    for class_split in [class1_split, class2_split]]
            else:
                self._supercategories = None
            
                self._categories = (self._dataframe.loc[self._dataframe['split'] == class1_split, "category"].unique(),
                                self._dataframe.loc[self._dataframe['split'] == class2_split, "category"].unique())
                
        if self._supercategories:
            if isinstance(self._categories, list):
                self._supercategories_prop = [
                        len(self._categories[0][supercategory])
                         + len(self._categories[1][supercategory]) 
                    for supercategory in self._supercategories
                    ]
            else:
                self._supercategories_prop = [len(self._categories[supercategory]) for supercategory in self._supercategories]

            self._supercategories_prop = np.array(self._supercategories_prop) / np.sum(self._supercategories_prop)

        self._context_minority_group_proportion = context_minority_group_proportion
        self._query_minority_group_proportion = query_minority_group_proportion
        self._reverse_task = reverse_task
        self._random_task_switching = random_task_switching

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
        if self._class1_split != self._class2_split:
            if self._supercategories:
                supercategory = np.random.choice(self._supercategories, p=self._supercategories_prop, size=1)[0]
                category1, category2 = (np.random.choice(x[supercategory], size=1)[0] for x in self._categories)
            else:
                category1, category2 = (np.random.choice(x, size=1)[0] for x in self._categories)
        else:
            if self._supercategories:
                supercategory = np.random.choice(self._supercategories, p=self._supercategories_prop, size=1)[0]
                category1, category2 = np.random.choice(self._categories[supercategory], size=2, replace=False)
            else:
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

        if self._reverse_task or (self._random_task_switching and np.random.rand() < 0.5):
            tmp = context[:, 1].copy()
            context[:, 1] = context[:, 2]
            context[:, 2] = tmp
            tmp = queries[:, 1].copy()
            queries[:, 1] = queries[:, 2]
            queries[:, 2] = tmp

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
