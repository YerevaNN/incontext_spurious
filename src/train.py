import logging
import os

from hydra.utils import instantiate, get_class
from pytorch_lightning.utilities import model_summary
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

import torch.serialization

from src.utils import log_hyperparameters, setup_aim_logger
from src.utils.helper_functions import get_allowed_classes

log = logging.getLogger(__name__)


def train(config: DictConfig):
    log.info(f"Instantiating model <{config.model._target_}>")

    if config.checkpoint_path is None:
        model = instantiate(config.model,
                            optimizer_conf=config.optimizer,
                            scheduler_conf=config.scheduler
                            )
    else:
        # If provided, initialize from a checkpoint
        model_class = get_class(config.model._target_)
        del config.model._target_  # Remove _target_ key before instantiation

        with torch.serialization.safe_globals(get_allowed_classes()):
            model = model_class.load_from_checkpoint(
                config.checkpoint_path,
                **instantiate(config.model),
                optimizer_conf=OmegaConf.to_container(config.optimizer, resolve=True),
                scheduler_conf=OmegaConf.to_container(config.scheduler, resolve=True),
                map_location='cpu',
                weights_only=True
            )

    # Now that the model is initialized, we seed everything again, but with a process-specific seed.
    # This makes sure that with DDP, dataloaders of different processes return different data.
    seed_everything(config.seed + int(os.environ.get('LOCAL_RANK', 0)))

    log.info(repr(model_summary.summarize(model)))
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = instantiate(config.datamodule)

    # Init lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    # Init lightning loggers
    loggers = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = instantiate(lg_conf)
            loggers.append(logger)

            if name == 'aim':
                setup_aim_logger(logger)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial")

    log_hyperparameters(config=config, model=model, trainer=trainer)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)
