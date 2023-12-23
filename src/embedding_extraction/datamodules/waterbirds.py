import logging
import lightning as L

from typing import NoReturn
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision.transforms import transforms

from src.embedding_extraction.datamodules.datasets import WaterbirdsDataset

log = logging.getLogger(__name__)

class WaterBirdsDataModule(L.LightningDataModule):
    def __init__(
            self, 
            name: str, 
            root_dir: str, 
            input_size: int, 
            batch_size: int, 
            num_workers: int, 
            background: dict, 
            *args, **kwargs
        ):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._background = background

        # https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/data/celebA_dataset.py#L81
        self._transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._dataset = None

    def prepare_data(self) -> NoReturn:
        self._dataset = WaterbirdsDataset(root_dir=self._root_dir, download=True, background=self._background)

    def train_dataloader(self) -> DataLoader: # Add normal data_loader for in_context_learning
        train_data = self._dataset.get_subset("train", transform=self._transform)

        train_loader = get_train_loader("standard", train_data,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_workers)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_data = self._dataset.get_subset("val", transform=self._transform)
        val_loader = get_eval_loader("standard", val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_data = self._dataset.get_subset("test", transform=self._transform)
        test_loader = get_eval_loader("standard", test_data,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers)

        return test_loader

    @property
    def train_dataset(self):
        train_data = self._dataset.get_subset("train", transform=self._transform)

        return train_data

    @property
    def val_dataset(self):
        train_data = self._dataset.get_subset("val", transform=self._transform)

        return train_data

    @property
    def test_dataset(self):
        train_data = self._dataset.get_subset("test", transform=self._transform)

        return train_data
