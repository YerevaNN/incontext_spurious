import logging
import torch
import lightning as L

from typing import NoReturn

from torch.utils.data import DataLoader

from src.incontextlearner.datamodules.datasets import ICLDataSet

log = logging.getLogger(__name__)

class ICLDataModule(L.LightningDataModule):
    def __init__(
            self, 
            train_context: dict,
            val_context: dict,
            test_context: dict,
            batch_size: int,
            num_workers: int,
            num_classes: int,
            num_confounders: int,
            cache_path: str,
            *args, **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        self._train_context = train_context
        self._val_context = val_context
        self._test_context = test_context
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_classes = num_classes
        self._num_confounders = num_confounders
        self._cache_path = cache_path

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def prepare_data(self) -> NoReturn:
        self._train_dataset = ICLDataSet(context_params=self._train_context,
                                         cache_path=self._cache_path,
                                         split="train", 
                                         num_classes=self._num_classes,
                                         num_confounders=self._num_confounders)
        
        self._val_dataset = ICLDataSet(context_params=self._val_context,
                                         cache_path=self._cache_path,
                                         split="val", 
                                         num_classes=self._num_classes,
                                         num_confounders=self._num_confounders,
                                         train_dataset=self._train_dataset)
        
        self._test_dataset = ICLDataSet(context_params=self._test_context,
                                         cache_path=self._cache_path,
                                         split="test", 
                                         num_classes=self._num_classes,
                                         num_confounders=self._num_confounders,
                                         train_dataset=self._train_dataset)
        
    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self._train_dataset,
                                  batch_size=self._batch_size,
                                  num_workers=self._num_workers)
        
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self._val_dataset,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(self._test_dataset,
                                 batch_size=self._batch_size,
                                 num_workers=self._num_workers)

        return test_loader
    
    def custom_test_dataloader(self, custom_context) -> DataLoader:
        custom_dataset = ICLDataSet(context_params=custom_context,
                            cache_path=self._cache_path,
                            split="test", 
                            num_classes=self._num_classes,
                            num_confounders=self._num_confounders,
                            train_dataset=self._train_dataset)
        
        return DataLoader(custom_dataset,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset