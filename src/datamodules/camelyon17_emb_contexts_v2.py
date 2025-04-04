import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader


from src.datamodules.datasets import Camelyon17EmbContextsDatasetV2

log = logging.getLogger(__name__)


class Camelyon17EmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for Camelyon17 in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 train_len: int,
                 eval_len: int,
                 batch_size: int,
                 num_workers: int,
                 context_class_size: int,
                 context_group_proportions: list[float],
                 train_query_group_proportions: list[float],
                 eval_query_group_proportions: list[float],
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
        super(Camelyon17EmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            root_dir=root_dir,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            reverse_task=reverse_task,
            modified=modified,
            modified_scale=modified_scale,
        )

        self.context_group_proportions = context_group_proportions
        self.train_query_group_proportions = train_query_group_proportions
        self.eval_query_group_proportions = eval_query_group_proportions

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
        self._train_test_dataset = None
        self._test_dataset = None

    def setup(self, stage="fit", *args, **kwargs):
        """Sets up the training and validation datasets."""
        if stage == "fit":
            self._train_dataset_for_fit = Camelyon17EmbContextsDatasetV2(
                **self._core_params,
                context_group_proportions=self.context_group_proportions,
                query_group_proportions=self.train_query_group_proportions,
                **self._aug_params,
                data_length=self._train_len,
                context_split='train',
                query_split='train',
            )

        # to measure training performance (includes example memorization)
        self._train_dataset_for_eval = Camelyon17EmbContextsDatasetV2(
            **self._core_params,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            data_length=self._eval_len,
            context_split='train',
            query_split='train',
        )

        # to measure ID generalization performance (catches example memorization)
        self._train_val_dataset = Camelyon17EmbContextsDatasetV2(
            **self._core_params,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            data_length=self._eval_len,
            context_split='train',
            query_split='val',
        )

        # sort of domain generalization
        self._train_test_dataset = Camelyon17EmbContextsDatasetV2(
            **self._core_params,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            data_length=self._eval_len,
            context_split='train',
            query_split='test',
        )

        # sort of domain adaptation
        self._test_dataset = Camelyon17EmbContextsDatasetV2(
            **self._core_params,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            data_length=self._eval_len,
            context_split='test',
            query_split='test',
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset_for_fit, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        """Creates a combined dataloader for all validation datasets."""
        all_loaders = {
            'train': DataLoader(self._train_dataset_for_eval,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers),
            'train_val': DataLoader(self._train_val_dataset,
                                    batch_size=self._batch_size,
                                    num_workers=self._num_workers),
            'train_test': DataLoader(self._train_test_dataset,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers),
            'test': DataLoader(self._test_dataset,
                               batch_size=self._batch_size,
                               num_workers=self._num_workers)
        }
        selected_loaders = {k: v for k, v in all_loaders.items() if k in self._val_sets}
        return CombinedLoader(selected_loaders, mode="sequential")
