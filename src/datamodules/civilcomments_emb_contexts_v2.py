import copy
import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader


from src.datamodules.datasets import CivilCommentsEmbContextsDatasetV2

log = logging.getLogger(__name__)


class CivilCommentsEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for CivilComments in-context learning instances."""

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
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 randomly_swap_labels: bool,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 val_sets: list[str],
                 allow_rotated_eval: bool,
                 simpler_construction: bool,
                 **kwargs):
        super(CivilCommentsEmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            root_dir=root_dir,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            reverse_task=reverse_task,
            simpler_construction=simpler_construction,
        )

        self._core_params_for_eval = copy.deepcopy(self._core_params)
        if allow_rotated_eval:
            self._core_params_for_eval['rotate_encodings'] = rotate_encodings
            self._core_params_for_eval['n_rotation_matrices'] = n_rotation_matrices

        self.context_group_proportions = context_group_proportions
        self.train_query_group_proportions = train_query_group_proportions
        self.eval_query_group_proportions = eval_query_group_proportions
        self.use_context_as_intermediate_queries = use_context_as_intermediate_queries

        self._aug_params = dict(
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            randomly_swap_labels=randomly_swap_labels,
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
        self._val_dataset = None

    def setup(self, stage="fit", *args, **kwargs):
        """Sets up the training and validation datasets."""
        if stage == "fit":
            self._train_dataset_for_fit = CivilCommentsEmbContextsDatasetV2(
                **self._core_params,
                context_group_proportions=self.context_group_proportions,
                query_group_proportions=self.train_query_group_proportions,
                use_context_as_intermediate_queries=self.use_context_as_intermediate_queries,
                **self._aug_params,
                data_length=self._train_len,
                context_split='train',
                query_split='train',
            )

        self._train_dataset_for_eval = CivilCommentsEmbContextsDatasetV2(
            **self._core_params_for_eval,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            use_context_as_intermediate_queries=False,
            data_length=self._eval_len,
            context_split='train',
            query_split='train',
        )

        self._train_val_dataset = CivilCommentsEmbContextsDatasetV2(
            **self._core_params_for_eval,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            use_context_as_intermediate_queries=False,
            data_length=self._eval_len,
            context_split='train',
            query_split='val',
        )

        self._train_test_dataset = CivilCommentsEmbContextsDatasetV2(
            **self._core_params_for_eval,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            use_context_as_intermediate_queries=False,
            data_length=self._eval_len,
            context_split='train',
            query_split='test',
        )

        self._val_dataset = CivilCommentsEmbContextsDatasetV2(
            **self._core_params_for_eval,
            context_group_proportions=self.context_group_proportions,
            query_group_proportions=self.eval_query_group_proportions,
            use_context_as_intermediate_queries=False,
            data_length=self._eval_len,
            context_split='val',
            query_split='val',
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
            'val': DataLoader(self._val_dataset,
                              batch_size=self._batch_size,
                              num_workers=self._num_workers)
        }
        selected_loaders = {k: v for k, v in all_loaders.items() if k in self._val_sets}
        return CombinedLoader(selected_loaders, mode="sequential")
