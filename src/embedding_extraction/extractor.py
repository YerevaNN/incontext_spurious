import logging
import os
import torch
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm

from src.utils import PipelineAction

torch.multiprocessing.set_sharing_strategy('file_system')

log = logging.getLogger(__name__)

class Extractor(PipelineAction):
    def __init__(self,
                 datamodule: pl.LightningDataModule,
                 network: pl.LightningModule,
                 device: str,
                 cache_path: str,
                 *args, **kwargs):
        super().__init__()

        self._network = network
        self._device = device
        self._dataloaders = self._prepare_dataloaders(datamodule)

        self._network.to(device)
        self._network.eval()

        if hasattr(datamodule, "_background") and datamodule._background['change_bg'] == True: # waterbirds with new backgrounds
            cache_path = os.path.abspath(cache_path)

            networks_path, network_name = os.path.abspath(os.path.dirname(cache_path)), os.path.basename(cache_path)
            datamodules_path, datamodule_name = os.path.dirname(networks_path), os.path.basename(networks_path)

            self._cache_path = os.path.join(datamodules_path, 
                    f"{datamodule_name}_{datamodule._background['water']}_{datamodule._background['land']}",
                        network_name) # .../datamodule_*water*_*land*/network
        else:
            self._cache_path = cache_path

    @staticmethod
    def _prepare_dataloaders(datamodule):
        datamodule.prepare_data()
        dataloaders = {
            "train": datamodule.train_dataloader(),
            "val": datamodule.val_dataloader(),
            "test": datamodule.test_dataloader(),
        }

        return dataloaders

    def _compute_and_save_embeddings(self, cache_path):
        log.info("Computing and saving embeddings in cache...")

        results_dict = dict()
        os.makedirs(cache_path)

        for split, dataloader in self._dataloaders.items():
            embeddings = []
            labels = []
            domains = []
            for x, y, c in tqdm(dataloader):
                if type(x) == torch.Tensor:
                    x = x.to(self._device)

                embedding = self._network(x).detach().cpu()
                y = y.detach().cpu()
                c = c.detach().cpu()

                embeddings.append(embedding)
                labels.append(y)
                domains.append(c)

            embeddings = torch.cat(embeddings)
            labels = torch.cat(labels)
            domains = torch.cat(domains)

            torch.save((embeddings, labels, domains), os.path.join(cache_path, f"{split}.pt"))

            results_dict[split] = (embeddings, labels, domains)

        return results_dict

    def _read_embeddings_from_cache(self, cache_path):
        log.info("Reading embeddings from cache...")

        results_dict = dict()

        for split in self._dataloaders.keys():
            results_dict[split] = torch.load(os.path.join(cache_path, f"{split}.pt"))

        return results_dict

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self._cache_path):
            return self._compute_and_save_embeddings(self._cache_path)
