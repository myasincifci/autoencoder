import os

import torch

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from template.data_modules.static_mnist_dm import StaticMNISTDM
from template.data_modules.mnist_dm import MNISTDM
from template.model import VAEModule

@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")

    logger = True
    if cfg.logging:
        wandb.init(
            project=cfg.logger.project,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        logger = WandbLogger()

    L.seed_everything(42, workers=True)

    # Data
    match cfg.data.name:
        case 'mnist':
            data_module = MNISTDM(cfg)
        case 'static_mnist':
            data_module = StaticMNISTDM(cfg)

    # Model
    model = VAEModule(cfg=cfg)

    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )

if __name__ == "__main__":
    main()