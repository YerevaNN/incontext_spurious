import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader


from src.datamodules.datasets import SpawriousEmbContextsDatasetV2

log = logging.getLogger(__name__)


class SpawriousEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for Spawrious in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 train_len: int,
                 eval_len: int,
                 batch_size: int,
                 num_workers: int,
                 context_class_size: int,
                 context_group_proportions: list[float],
                 query_group_proportions: list[float],
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool,
                 reverse_task: bool,
                 modified: bool,
                 modified_scale: float,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 randomly_swap_labels: bool,
                 label_noise_ratio_interval: list,
                 input_noise_norm_interval: list,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 val_sets: list[str],
                 **kwargs):
        super(SpawriousEmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            root_dir=root_dir,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            context_group_proportions=context_group_proportions,
            query_group_proportions=query_group_proportions,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            reverse_task=reverse_task,
            modified=modified,
            modified_scale=modified_scale,
        )

        self._aug_params = dict(
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            randomly_swap_labels=randomly_swap_labels,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
        )

        # Initializing dataset parameters
        self._train_len = train_len
        self._eval_len = eval_len
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_sets = val_sets

        # Placeholders for dataset splits
        self._train_dataset_for_fit = None
        self._train_dataset_for_eval = None
        self._train_val_dataset = None
        self._test_diff_bg_dataset = None
        self._test_diff_class_dataset = None
        self._test_all_diff_dataset = None

    def setup(self, stage="fit", *args, **kwargs):
        """Sets up the training and validation datasets."""
        if stage == "fit":
            self._train_dataset_for_fit = SpawriousEmbContextsDatasetV2(
                **self._core_params,
                **self._aug_params,
                data_length=self._train_len,
                context_split='train',
                query_split='train',
            )

        # to measure training performance (includes example memorization)
        self._train_dataset_for_eval = SpawriousEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='train',
            query_split='train',
        )

        # to measure ID generalization performance (catches example memorization)
        self._train_val_dataset = SpawriousEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='train',
            query_split='val',
        )

        # generalizing to new backgrounds
        self._test_diff_bg_dataset = SpawriousEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='test_diff_bg',
            query_split='test_diff_bg',
        )

        # generalizing to new classes
        self._test_diff_class_dataset = SpawriousEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='test_diff_class',
            query_split='test_diff_class',
        )

        # generalizing to new backgrounds and classes
        self._test_all_diff_dataset = SpawriousEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='test_all_diff',
            query_split='test_all_diff',
        )


    def train_dataloader(self):
        return DataLoader(self._train_dataset_for_fit, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        """Creates a combined dataloader for all validation datasets."""
        all_loaders = {
            'train': DataLoader(self._train_dataset_for_eval,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers),
            'train_val': DataLoader(self._train_val_dataset,
                                    batch_size=self._batch_size,
                                    num_workers=self._num_workers),
            'test_diff_bg': DataLoader(self._test_diff_bg_dataset,
                                       batch_size=self._batch_size,
                                       num_workers=self._num_workers),
            'test_diff_class': DataLoader(self._test_diff_class_dataset,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers),
            'test_all_diff': DataLoader(self._test_all_diff_dataset,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_workers),
        }
        selected_loaders = {k: v for k, v in all_loaders.items() if k in self._val_sets}
        return CombinedLoader(selected_loaders, mode="sequential")
