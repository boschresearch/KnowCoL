from omegaconf import DictConfig, OmegaConf
import hydra
from typing import List, Union
import logging
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from utils.utils import get_last_checkpoint, print_system_env_info
from pathlib import Path

import pdb
logger = logging.getLogger(__name__)

import torch
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="../conf", config_name="config")
def train(cfg:DictConfig):
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """   
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed, workers=True)  # type: ignore 
    kg = hydra.utils.instantiate(cfg.kg)
    datamodule = hydra.utils.instantiate(cfg.datamodule, kg=kg)
    model = hydra.utils.instantiate(cfg.model, kg=kg)
    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())


    train_logger = setup_logger(cfg, model)
    callbacks = setup_callbacks(cfg.callbacks)
    lr_logger = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_logger)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=datamodule)  # type: ignore
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiate all training callbacks.

    Args:
        callbacks_cfg: DictConfig with all callback params

    Returns:
        List of instantiated callbacks.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks

def setup_logger(cfg: DictConfig, model: LightningModule) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()

    cfg.logger.group = pathlib_cwd.parent.name
    cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
    cfg.logger.id = cfg.logger.name.replace("/", "_")
    train_logger = hydra.utils.instantiate(cfg.logger)

    return train_logger


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)

if __name__ == "__main__":
    train()
