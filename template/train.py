import os
import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.pacs_dm import PacsDM
from model import ResnetClf
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms as T
import wandb

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
    data_module = PacsDM(cfg, leave_out=['sketch'])

    # Model
    barlow_twins = ResnetClf(cfg=cfg)

    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(
        model=barlow_twins,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
