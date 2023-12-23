import copy
import os
import logging
import torch

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

log = logging.getLogger(__name__)


class EarlyStoppingAndRollback(EarlyStopping):
    def __init__(self, *args, rollback_to_best=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollback_to_best = rollback_to_best
        self.best_model_path = 'checkpoints/best_checkpoint.ckpt'

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.should_stop and self.rollback_to_best:
            pl_module.load_state_dict(torch.load(self.best_model_path)['state_dict'])
            log.info('Model rolled back to the best checkpoint.')


def get_early_stopping_callbacks(es_callback, chkpt_dirpath='checkpoints', chkpt_filename='best_checkpoint'):
    es_callback = copy.deepcopy(es_callback)
    callbacks = [es_callback]

    if isinstance(es_callback, EarlyStoppingAndRollback):
        best_checkpoint_filepath = os.path.join(
            chkpt_dirpath,
            f"{chkpt_filename}{ModelCheckpoint.FILE_EXTENSION}")
        if os.path.exists(best_checkpoint_filepath):
            os.remove(best_checkpoint_filepath)

        checkpointer = ModelCheckpoint(dirpath=chkpt_dirpath,
                                       filename=chkpt_filename,
                                       monitor=es_callback.monitor,
                                       mode=es_callback.mode,
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       verbose=True)

        es_callback.best_model_path = best_checkpoint_filepath
        callbacks.append(checkpointer)

    return callbacks