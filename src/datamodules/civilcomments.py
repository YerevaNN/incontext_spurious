import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import CivilCommentsForEncodingExtraction

log = logging.getLogger(__name__)


class CivilCommentsDataModule(pl.LightningDataModule):
    """
        A PyTorch Lightning data module for the CivilComments dataset.

        This module is designed to handle the loading and preprocessing of the CivilComments dataset for
        use in training, validation, and testing phases of a machine learning model. It leverages
        PyTorch's DataLoader for efficient data loading and the WILDS library's data loaders for
        handling training and evaluation datasets with a focus on generalization across distribution shifts.

        Attributes:
            name (str): Name of the dataset.
            root_dir (str): Root directory where the dataset is stored.
            resize_size (int): The size to which the smaller edge of the image will be resized.
            batch_size (int): The size of the batch for data loading.
            num_workers (int): The number of worker processes for data loading.
            with_subsets (bool): Indicates if the dataset is split into subsets (train, val, test).
    """
    def __init__(self, name: str, root_dir: str, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        self.with_subsets = True

        self._dataset = None

    def prepare_data(self):
        """Prepares the CivilComments dataset for use, downloading it if necessary."""
        self._dataset = CivilCommentsForEncodingExtraction(root_dir=self._root_dir, download=True)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training subset of the CivilComments dataset."""
        return get_train_loader("standard", self.train_dataset,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation subset of the CivilComments dataset."""
        return get_eval_loader("standard", self.val_dataset,
                               batch_size=self._batch_size,
                               num_workers=self._num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test subset of the CivilComments dataset."""
        return get_eval_loader("standard", self.test_dataset,
                               batch_size=self._batch_size,
                               num_workers=self._num_workers)

    def get_combined_dataloader(self):
        sub_dataloaders = {"train": self.train_dataloader(),
                           "val": self.val_dataloader(),
                           "test": self.test_dataloader()}

        sub_dataloader_names = list(sub_dataloaders.keys())

        return CombinedLoader(sub_dataloaders, mode="sequential"), sub_dataloader_names

    @property
    def train_dataset(self):
        return self._dataset.get_subset("train")

    @property
    def val_dataset(self):
        return self._dataset.get_subset("val")

    @property
    def test_dataset(self):
        return self._dataset.get_subset("test")
