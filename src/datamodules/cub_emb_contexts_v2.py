import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datamodules.datasets import CUBEmbContextsDatasetV2

log = logging.getLogger(__name__)


class CUBEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for Caltech-UCSD Birds (CUB) in-context learning instances.

    TODO(hrayr): extend to proper training.
    """

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 batch_size: int,
                 num_workers: int,
                 context_class_size: int,
                 context_minority_group_proportion: float,
                 query_minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 label_noise_ratio_interval: list,
                 input_noise_norm_interval: list,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 simpler_construction: bool,
                 *args, **kwargs):
        super(CUBEmbContextsDataModuleV2, self).__init__()

        # Initializing dataset parameters
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._context_minority_group_proportion = context_minority_group_proportion
        self._query_minority_group_proportion = query_minority_group_proportion
        self._spurious_setting = spurious_setting
        self._sp_token_generation_mode = sp_token_generation_mode
        self._use_context_as_intermediate_queries = use_context_as_intermediate_queries
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._label_noise_ratio_interval = label_noise_ratio_interval
        self._input_noise_norm_interval = input_noise_norm_interval
        self._permute_input_dim = permute_input_dim
        self._ask_context_prob = ask_context_prob
        self._simpler_construction = simpler_construction

        self._dataset = None

    def setup(self,  *args, **kwargs):
        """
        Initializes the dataset.
        """
        # Creating dataset instances for each split using class attributes
        self._dataset = CUBEmbContextsDatasetV2(
            dataset_path=self._dataset_path,
            encoding_extractor=self._encoding_extractor,
            data_length=self._data_length,
            context_class_size=self._context_class_size,
            context_minority_group_proportion=self._context_minority_group_proportion,
            query_minority_group_proportion=self._query_minority_group_proportion,
            spurious_setting=self._spurious_setting,
            sp_token_generation_mode=self._sp_token_generation_mode,
            use_context_as_intermediate_queries=self._use_context_as_intermediate_queries,
            rotate_encodings=False,
            label_noise_ratio_interval=None,
            input_noise_norm_interval=None,
            permute_input_dim=False,
            ask_context_prob=None,
            simpler_construction=self._simpler_construction)

    def val_dataloader(self):
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
