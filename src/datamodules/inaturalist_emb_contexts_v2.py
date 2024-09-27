import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import INaturalistEmbContextsDatasetV2

log = logging.getLogger(__name__)


class INaturalistEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for iNaturalist in-context learning instances."""

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 train_len: int,
                 eval_len: int,
                 batch_size: int,
                 num_workers: int,
                 context_class_size: int,
                 context_minority_group_proportion: float,
                 query_minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool,
                 reverse_task: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 label_noise_ratio_interval: list,
                 input_noise_norm_interval: list,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 max_n_spurious: int,
                 maximum_minority_prop: float,
                 swapping_minority_proportion_context: float,
                 swapping_minority_proportion_query: float,
                 points_to_swap_range: list[int],
                 random_task_switching: bool,
                 val_sets: list[str],
                 *args, **kwargs):
        super(INaturalistEmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            dataset_path=dataset_path,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            context_minority_group_proportion=context_minority_group_proportion,
            query_minority_group_proportion=query_minority_group_proportion,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            reverse_task=reverse_task,
            max_n_spurious=max_n_spurious,
            maximum_minority_prop=maximum_minority_prop,
            swapping_minority_proportion_context=swapping_minority_proportion_context,
            swapping_minority_proportion_query=swapping_minority_proportion_query,
            points_to_swap_range=points_to_swap_range,
        )

        self._aug_params = dict(
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
            random_task_switching=random_task_switching,
        )

        self._train_len = train_len
        self._eval_len = eval_len
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_sets = val_sets

        # Placeholders for dataset splits
        self._train_set = None
        self._inner_val_set = None
        self._outer_val_set = None
        self._inner_outer_val_set = None

    def setup(self, stage="fit", *args, **kwargs):
        """
        Sets up the dataset splits for training, validation, and testing.
        Initializes the inner train and inner, outer, inner-outer validation datasets.
        """
        # Creating dataset instances for each split using class attributes
        if stage == "fit":
            self._train_set = INaturalistEmbContextsDatasetV2(
                **self._core_params,
                **self._aug_params,
                data_length=self._train_len,
                class1_split="inner_train",
                class2_split="inner_train",
            )

        self._inner_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="inner_val",
            class2_split="inner_val")

        self._outer_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="outer",
            class2_split="outer")

        self._inner_outer_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="inner_val",
            class2_split="outer")

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        """Creates a combined dataloader for all validation datasets."""
        all_loaders = {
            'inner': DataLoader(self._inner_val_set,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers),
            'inner_outer': DataLoader(self._inner_outer_val_set,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers),
            'outer': DataLoader(self._outer_val_set,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)
        }
        selected_loaders = {k: v for k, v in all_loaders.items() if k in self._val_sets}
        return CombinedLoader(selected_loaders, mode="sequential")
