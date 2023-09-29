import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from mydpr.model.biencoder import MyEncoder
from mydpr.dataset.cath35 import PdDataModule
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import *

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        for i in range(world_size):
            os.makedirs(os.path.join(self.output_dir, str(i)), exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        torch.save(prediction, os.path.join(self.output_dir, str(rank), "%07d.pt"%batch_idx))

    def write_on_epoch_end(
        self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]
    ):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

def configure_callbacks(cfg: DictConfig):
    return CustomWriter(output_dir='ebd', write_interval='batch')


@hydra.main(config_path="conf", config_name="pred_conf")
def main(cfg: DictConfig):

    os.environ["MASTER_PORT"] = cfg.trainer.master_port
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.trainer.devices

    if cfg.logger.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=cfg.logger.project, log_model=cfg.logger.log_model)
    else:
        logger = True

    pl.seed_everything(cfg.trainer.seed)

    if cfg.model.resume:
        model = MyEncoder.load_from_checkpoint(
        checkpoint_path=cfg.model.ckpt_path,
        )
    else:
        model = MyEncoder()

    dm  = PdDataModule(cfg.trainer.ur90_path, cfg.trainer.batch_size, model.alphabet)

    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        #accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.accelerator,
        accumulate_grad_batches=cfg.trainer.acc_step,
        precision=cfg.trainer.precision,
        #gradient_clip_val=0.5,
        logger=logger,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=configure_callbacks(cfg),
        fast_dev_run=False,
    )

    trainer.predict(model, datamodule=dm)

if __name__ == "__main__":
    main()
