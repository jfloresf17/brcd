"""
Road Segmentation PyTorch Lightning Trainer
Specialized trainer for road detection from satellite imagery.
"""

import torch
import pytorch_lightning as pl
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb

from models.dlink_model import DLinkNetSeg
from utils.losses import bce_dice_loss
from models.rrdbnet import RRDBNet
from torchmetrics import JaccardIndex, F1Score, Precision, Recall
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class RoadTrainer(pl.LightningModule):
    """
    Specialized trainer for road segmentation only.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        
        self.cfg = cfg
        model_cfg = cfg['model']

        # Super-resolution model (optional)
        if 'sr_ckpt' in model_cfg:
            logdir_hr = model_cfg['sr_ckpt']
            net_hr = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
            net_hr.load_state_dict(torch.load(logdir_hr)['net_g_ema'])
            net_hr.eval()
            net_hr.half()
            for p in net_hr.parameters():
                p.requires_grad = False
            self.sr_model = net_hr
        else:
            self.sr_model = None
        
        # ───────── SINGLE MODEL FOR ROADS ─────────
        self.model = DLinkNetSeg(
            in_channels=cfg['model']['in_channels'],  # S1 (4) + S2 (4) bands
            super_mid=cfg['model'].get('super_mid', 3),  # Default 3 for roads
            upscale=cfg['model'].get('upscale', 4),      # Default upscale factor
            num_classes=1                               # Binary segmentation
        )
        
        # Training parameters optimized for roads
        self.lr = cfg['training']['lr']
        self.wd = cfg['training'].get('weight_decay', 0.0)
        self.save_freq = cfg['callbacks'].get('save_freq', 1000)  # Default save frequency
        self.mean = cfg['normalize']['mean']
        self.std = cfg['normalize']['std']
        
        # Metrics        
        self.val_iou = JaccardIndex(num_classes=2, task='binary', threshold=0.5)
        self.val_f1 = F1Score(num_classes=2, task='binary', threshold=0.5)
        self.val_prec = Precision(num_classes=2, task='binary', threshold=0.5)
        self.val_recall = Recall(num_classes=2, task='binary', threshold=0.5)
        
    def forward(self, x):
        """Forward pass for road segmentation only."""
        # Generate super-resolution features if available
        if self.sr_model is not None:
            sr_in = x[:, :3].to(self.device, non_blocking=True)
            with torch.no_grad():
                super_fea = self.sr_model.forward(sr_in)
            super_fea = super_fea.float()
        else:
            B, C, H, W = x.shape
            super_fea = torch.zeros(B, 3, H * 4, W * 4, device=x.device, dtype=x.dtype)
        
        # road prediction only
        road_logits = self.model(x, super_fea)
        return road_logits
    
    def training_step(self, batch, batch_idx):
        """Training step for road segmentation."""
        x, yb, yr = batch  # yb is ignored in road-only training
        
        # Forward pass
        road_logits = self(x)
        
        # Ensure target shape
        yr = yr.unsqueeze(1).float()
        
        # Loss optimized for roads (líneas finas)
        loss = bce_dice_loss(road_logits, yr, 
                           bce_weight=0.5,   # BCE mejor para estructuras finas
                           dice_weight=0.5)
        
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for road segmentation."""
        x, yb, yr = batch
        
        road_logits = self(x)
        yr = yr.unsqueeze(1).float()
        
        loss = bce_dice_loss(road_logits, yr, bce_weight=0.5, dice_weight=0.5)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        road_pred = torch.sigmoid(road_logits)
        
        # Metrics
        self.log('val/iou', self.val_iou(road_pred, yr), prog_bar=True, on_epoch=True)
        self.log('val/f1', self.val_f1(road_pred, yr), prog_bar=True, on_epoch=True)
        self.log('val/precision', self.val_prec(road_pred, yr), prog_bar=True, on_epoch=True)
        self.log('val/recall', self.val_recall(road_pred, yr), prog_bar=True, on_epoch=True)

        # ───────── VISUALIZATION ─────────
        if batch_idx % self.save_freq == 0:
            self._log_road_visualization(x, road_pred, yr, batch_idx)

    def _log_road_visualization(self, x, road_pred, yr, batch_idx):
        """Specialized visualization for roads."""
        N_DBG = 3
        bs = x.size(0)
        
        # Denormalize
        x = x * torch.tensor(self.std).view(1, -1, 1, 1).to(self.device) + \
            torch.tensor(self.mean).view(1, -1, 1, 1).to(self.device)

        for i, idx in enumerate(random.sample(range(bs), k=min(N_DBG, bs))):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # RGB input
            rgb_input = x[idx, :3].cpu().permute(1, 2, 0).numpy()
            rgb_input = np.clip(rgb_input, 0, 1)

            # Road prediction and GT
            pred_np = road_pred[idx, 0].cpu().numpy()
            gt_np = yr[idx, 0].cpu().numpy()

            axes[0].imshow(rgb_input)
            axes[0].set_title("RGB Input")
            axes[0].axis("off")

            axes[1].imshow(pred_np, cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title("Road Prediction")
            axes[1].axis("off")

            axes[2].imshow(gt_np, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title("Road GT")
            axes[2].axis("off")

            plt.suptitle(f"Road Sample {idx} | Step {self.global_step}")
            plt.tight_layout()

            wandb.log({f"road_panel_{i}": wandb.Image(fig)}, commit=False)
            plt.close(fig)

        wandb.log({}, commit=True)

        
    def configure_optimizers(self):
        """Optimizer optimized for road segmentation."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.wd),
            betas=(0.9, 0.99)  # Diferente para roads
        )
        
        return optimizer
    

def build_dlinknet_trainer(cfg: dict) -> pl.Trainer:
    """
    Build PyTorch Lightning trainer specifically configured for DLinkNet road segmentation.
    
    Args:
        cfg (dict): Configuration dictionary with training parameters
        
    Returns:
        pl.Trainer: Configured PyTorch Lightning trainer for road segmentation
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1) WandB logger for road segmentation
    wandb_cfg = cfg.get('wandb', {})
    logger = WandbLogger(
        project=wandb_cfg.get('project', 'road-segmentation'),
        name=wandb_cfg.get('name', 'dlinknet-roads')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Checkpoint callbacks optimized for roads
    callbacks = cfg["callbacks"]

    # Save callback configuration
    save = callbacks["save"]
    save_dir = save["save_dir"]
    filename = save["filename"]
    monitor = save["monitor"]
    mode = save["mode"]

    # Best checkpoint (monitored)
    ckpt_best = ModelCheckpoint(
        dirpath=save_dir,
        filename=filename,
        save_top_k=1,
        monitor=monitor,
        mode=mode
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Early stopping callback
    early_stopping_cfg = callbacks.get('early_stopping', {})
    es_monitor = early_stopping_cfg.get('monitor', 'val/loss')
    es_patience = early_stopping_cfg.get('patience', 10)
    es_mode = early_stopping_cfg.get('mode', 'min')

    es_cb = EarlyStopping(
        monitor=es_monitor,
        patience=es_patience,
        mode=es_mode
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Trainer args optimized for road segmentation
    tr_cfg = cfg.get('training', {})
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    num_gpus = tr_cfg.get('gpu', 1 if torch.cuda.is_available() else 0)
    precision = tr_cfg.get('precision', '32')
    max_epochs = tr_cfg.get('epochs', 50)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_best, es_cb],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=num_gpus,
        precision=precision,
        default_root_dir=save_dir,
        check_val_every_n_epoch=1,  # make sure val/loss is logged every epoch
        gradient_clip_val=1.0,  # Important for road segmentation (thin structures)
    )

    return trainer

