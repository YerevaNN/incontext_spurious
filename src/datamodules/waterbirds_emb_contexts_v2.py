from typing import Optional
import logging

import pytorch_lightning as pl

from enum import Enum
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader


from src.datamodules.datasets import WaterbirdsEmbContextsDatasetV2

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for Waterbirds in-context learning instances."""
    class ValSets(Enum):
        """Enum of validation splits of WaterbirdsEmbContextsDataModuleV2."""
        TRAIN = "train"
        TRAIN_VAL = "train_val"
        TRAIN_TEST = "train_test"
        VAL = "val"

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 train_len: int,
                 eval_len: int,
                 saved_val_sets_path: Optional[str],
                 batch_size: int,
                 num_workers: Optional[int],
                 context_class_size: int,
                 minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 v1_behavior: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 randomly_swap_labels: bool,
                 label_noise_ratio_interval: list,
                 input_noise_norm_interval: list,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 *args, **kwargs):
        super(WaterbirdsEmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            root_dir=root_dir,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            minority_group_proportion=minority_group_proportion,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            v1_behavior=v1_behavior,
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
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._saved_val_sets_path = saved_val_sets_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._spurious_setting = spurious_setting
        self._sp_token_generation_mode = sp_token_generation_mode
        self._v1_behavior = v1_behavior
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._randomly_swap_labels = randomly_swap_labels
        self._label_noise_ratio_interval = label_noise_ratio_interval
        self._input_noise_norm_interval = input_noise_norm_interval
        self._permute_input_dim = permute_input_dim
        self._ask_context_prob = ask_context_prob

        self._saved_val_sets_path = saved_val_sets_path
        self._train_len = train_len
        self._eval_len = eval_len
        self._batch_size = batch_size
        self._num_workers = num_workers

        # Placeholders for dataset splits
        self._train_dataset_for_fit = None
        self._train_dataset_for_eval = None
        self._train_val_dataset = None
        self._train_test_dataset = None
        self._val_dataset = None

    def setup(self, stage="fit", *args, **kwargs):
        """Sets up the training and validation datasets."""
        if stage == "fit":
            self._train_dataset_for_fit = WaterbirdsEmbContextsDatasetV2(
                **self._core_params,
                **self._aug_params,
                data_length=self._train_len,
                context_split='train',
                query_split='train',
            )

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.TRAIN.value)
        saved_data_path = None
        self._train_dataset_for_eval = WaterbirdsEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='train',
            query_split='train',
            saved_data_path=saved_data_path,
        )

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.TRAIN_VAL.value)
        saved_data_path = None
        self._train_val_dataset = WaterbirdsEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='train',
            query_split='val',
            saved_data_path=saved_data_path,
        )

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.TRAIN_TEST.value)
        saved_data_path = None
        self._train_test_dataset = WaterbirdsEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='train',
            query_split='test',
            saved_data_path=saved_data_path,
        )

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.VAL.value)
        saved_data_path = None
        self._val_dataset = WaterbirdsEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            context_split='val',
            query_split='val',
            saved_data_path=saved_data_path,
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset_for_fit, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        """Creates a combined dataloader for all validation datasets.

        Returns:
            CombinedLoader: A combined DataLoader for "train", "train_val", 
                        "train_test" and "val" validation sets.
        """
        train_dataloader = DataLoader(self._train_dataset_for_eval,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers)
        train_val_dataloader = DataLoader(self._train_val_dataset,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers)
        train_test_dataloader = DataLoader(self._train_test_dataset,
                                           batch_size=self._batch_size,
                                           num_workers=self._num_workers)
        val_dataloader = DataLoader(self._val_dataset,
                                    batch_size=self._batch_size,
                                    num_workers=self._num_workers)

        return CombinedLoader({self.ValSets.TRAIN: train_dataloader,
                               self.ValSets.TRAIN_VAL: train_val_dataloader,
                               self.ValSets.TRAIN_TEST: train_test_dataloader,
                               self.ValSets.VAL: val_dataloader},
                              mode="sequential")
