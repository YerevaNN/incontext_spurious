import logging
import numpy as np
import torch

import lightning as L

from src.utils import PipelineAction
from src.utils import get_early_stopping_callbacks, evaluate_model

log = logging.getLogger(__name__)

class ICLTrainer(PipelineAction):
    def __init__(
            self,
            icl_datamodule: L.LightningDataModule,
            icl_architecture: L.LightningModule,
            model_params: dict,
            metric_show: dict,
            *args, **kwargs
        ):
        self._incontextdatamodule = icl_datamodule
        self._incontextlearner = icl_architecture
        self._model_params = model_params
        self._metric_show = metric_show

        self._callbacks = get_early_stopping_callbacks(self._model_params["early_stopping"],
                                                       chkpt_dirpath="checkpoints",
                                                       chkpt_filename="best_checkpoint")
        
        self.trainer = L.Trainer(accelerator=self._model_params["accelerator"],
                          devices=self._model_params["devices"],
                          max_epochs=self._model_params["max_epochs"],
                          callbacks=self._callbacks,
                          logger=self._model_params["aim"],
                          gradient_clip_val=self._model_params["gradient_clip_val"],
                          log_every_n_steps=self._model_params["log_every_n_steps"])
        
        
    def _create_dataloaders(self):
        size = self._metric_show['size']
        label_ratios = self._metric_show['label_ratios']
        minority_ratios = self._metric_show['minority_ratios']

        if not label_ratios or not minority_ratios:
            return self._incontextdatamodule.val_dataloader()

        val_loader = self._incontextdatamodule.val_dataloader()
        dataloaders = [val_loader]

        for label_ratio in label_ratios:
            for minority_ratio in minority_ratios:
                context_template = dict(
                    size=size,
                    label_ratio=label_ratio,
                    minority_ratio=minority_ratio, 
                )

                dataloaders.append(self._incontextdatamodule.custom_test_dataloader(context_template))

        return dataloaders

    def _start_training(self):
        num_classes = self._incontextlearner._num_classes
        num_confounders = self._incontextlearner._num_confounders
        
        train_loader = self._incontextdatamodule.train_dataloader()

        dataloaders = self._create_dataloaders()

        self.trainer.fit(self._incontextlearner, train_loader, val_dataloaders=dataloaders) 

        val_loader = self._incontextdatamodule.val_dataloader()
        test_loader = self._incontextdatamodule.test_dataloader()
        eval_loaders = dict(val_set=val_loader, test_set=test_loader)

        res = evaluate_model(
            trainer=self.trainer,
            model=self._incontextlearner,
            eval_loaders=eval_loaders,
            num_groups=num_classes * num_confounders,
            overall_acc_metric_name='test_accuracy',
            worst_group_acc_metric_name='test_accuracy_wg',
            group_acc_metric_name_fmt='test_accuracy_g{}',
        )

        return res

    def __call__(self, *args, **kwargs):
        log.info("Start training process...")
        self._incontextdatamodule.prepare_data()

        results = self._start_training()
        log.info(f"Results are ready \n{results}")

        return results