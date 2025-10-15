# Write about the superresolution techniques in segmentation tasks (train  a model from scratch RRSGAN)
# Use Foundational Models (like PhilEO) for segmentation tasks [building and roads] and in a future try 
# to fine-tune them in Spain region
# Add symmetric cross entropy or robust loss
# Try with only highr resolution loss
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from models.br_model import SRSeg  # import the SRSeg from above
from models.rrdbnet import RRDBNet  # import the super-resolution model
from torchmetrics import JaccardIndex, F1Score, Precision, Recall
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.losses import sce_dice_loss, symmetric_cross_entropy_binary

class SRSegLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__() 
        self.save_hyperparameters(ignore=['cfg'])       
        # instantiate your SRSeg
        self.model = SRSeg(
            encoder_name   = cfg['model']['encoder_name'],
            encoder_weights= cfg['model']['encoder_weights'],
            encoder_depth  = cfg['model']['encoder_depth'],
            in_channels    = cfg['model']['in_channels'],
            super_mid      = cfg['model']['super_mid'],
            upscale        = cfg['model']['upscale'],
            num_classes    = cfg['model']['num_classes']
        )

        # save the config for later use
        self.cfg = cfg  # save the config for later use
        
        # uncertainty parameters for high-res losses
        self.save_freq = cfg['callbacks']['save']['save_freq']
        
        # optimizer hyperparams
        self.lr = cfg['training']['lr']
        self.wd = cfg['training'].get('weight_decay', 0.0)
        
        # Superresolution checkpoint
        logdir_hr = cfg['model']['sr_ckpt']
        self.num_in_ch = cfg['model']['sr_in_ch']  # number of input channels for SR
        self.num_out_ch = cfg['model']['sr_out_ch']  # number of output channels for SR
        self.upscale = cfg['model']['upscale']  # super-resolution scale factor

        # Mean and std for normalization
        self.mean = cfg['normalize']['mean']
        self.std = cfg['normalize']['std']
        
        # Pos - weights
        self.pw_building = torch.tensor(float(cfg['weights']['building']), dtype=torch.float32, device=self.device)
        self.pw_road = torch.tensor(float(cfg['weights']['road']), dtype=torch.float32, device=self.device)

        # ───────── METRICS (IoU, F1, Precision, Recall) ─────────
        # High-res 2.5 m building 
        self.val_iou_b_hr = JaccardIndex(num_classes=2, task='binary', threshold=0.5)
        self.val_f1_b_hr = F1Score(num_classes=2, task='binary', threshold=0.5)
        self.val_prec_b_hr = Precision(num_classes=2, task='binary', threshold=0.5)
        self.val_recall_b_hr = Recall(num_classes=2, task='binary', threshold=0.5)

        # High-res 2.5 m road
        self.val_iou_r_hr = JaccardIndex(num_classes=2, task='binary', threshold=0.5)
        self.val_f1_r_hr = F1Score(num_classes=2, task='binary', threshold=0.5)
        self.val_prec_r_hr = Precision(num_classes=2, task='binary', threshold=0.5)
        self.val_recall_r_hr = Recall(num_classes=2, task='binary', threshold=0.5)

        # super‐resolution inference model (frozen)
        net_hr = RRDBNet(num_in_ch=self.num_in_ch, num_out_ch=self.num_out_ch, scale=self.upscale)
        net_hr.load_state_dict(torch.load(logdir_hr)['net_g_ema'])
        net_hr.eval()
        net_hr.half()  # set to eval mode and float precision
        net_hr = net_hr.to('cuda')  # move to GPU if available
        for p in net_hr.parameters():
            p.requires_grad = False
        
        self.sr_model = net_hr
   
    def forward(self, x):
        # x: (B, C, H10, W10)
        # pick the RGB channels for SR
        sr_in = x[:, :self.num_out_ch].to('cuda', non_blocking=True)     # e.g. take channels 3,2,1 for ESRGAN
        with torch.no_grad():
            # Ensure input to RRDBNet is 64x64
            super_fea = self.sr_model.forward(sr_in)  # (B, super_mid, H40, W40) → upsampled HR features
        super_fea = super_fea.float() # convert to float32 for further processing
        # now forward through your segmentation net
        return self.model(x, super_fea)

    def training_step(self, batch, batch_idx):
        x, yb, yr = batch
        bld_hr, rd_hr, bld_aux, rd_aux = self(x)
        # bld_hr, rd_hr = self(x)

        # ensure shapes: (B,1,H,W)
        yb = yb.unsqueeze(1).float()
        yr = yr.unsqueeze(1).float()

        # ——— Pérdidas principales (SCE + Dice) ————————————
        Lb = sce_dice_loss(bld_hr, yb)
        Lr = sce_dice_loss(rd_hr,  yr)

        # ——— Deep supervision ——————————————
        # Opción A (robusta también):
        Laux = (symmetric_cross_entropy_binary(bld_aux, yb) +
                symmetric_cross_entropy_binary(rd_aux,  yr))
        
        # # ——— Main segmentation losses (Eqs. 9–10) ———————————
        # Lb = bce_dice_loss(bld_hr, yb, bce_weight=0.5, dice_weight=0.5) # pos_weight=self.pw_building
        # Lr = bce_dice_loss(rd_hr,  yr, bce_weight=0.5, dice_weight=0.5) # pos_weight=self.pw_road

        # # ——— Deep‐supervision loss (Eq. 11) ——————————————
        # Laux   =( F.binary_cross_entropy_with_logits(bld_aux, yb) +
        #           F.binary_cross_entropy_with_logits(rd_aux, yr) )

        # ——— Total loss (Eq. 12) ———————————————————————
        total_loss = 0.45 * Lb + 0.45* Lr + 0.1 * Laux
        
        # Log each piece
        self.log('train/b_loss', Lb, prog_bar=False)
        self.log('train/r_loss', Lr, prog_bar=False)
        self.log('train/loss', total_loss, prog_bar=True)

        return total_loss
    

    def validation_step(self, batch, batch_idx):
        x, yb, yr = batch

        bld_hr, rd_hr, bld_aux, rd_aux = self(x)
        # bld_hr, rd_hr = self(x)

        # ensure shapes: (B,1,H,W)
        yb = yb.unsqueeze(1).float()
        yr = yr.unsqueeze(1).float()

        # ——— Pérdidas principales (SCE + Dice) ————————————
        Lb = sce_dice_loss(bld_hr, yb)
        Lr = sce_dice_loss(rd_hr,  yr)

        # ——— Deep supervision ——————————————
        # Opción A (robusta también):
        Laux = (symmetric_cross_entropy_binary(bld_aux, yb) +
                symmetric_cross_entropy_binary(rd_aux,  yr))

        # # ——— Main segmentation losses (Eqs. 9–10) ———————————
        # Lb = bce_dice_loss(bld_hr, yb, bce_weight=0.5, dice_weight=0.5) # pos_weight=self.pw_building
        # Lr = bce_dice_loss(rd_hr,  yr, bce_weight=0.5, dice_weight=0.5) # pos_weight=self.pw_road

        # ——— Deep‐supervision loss (Eq. 11) ——————————————
        # Laux   =( F.binary_cross_entropy_with_logits(bld_aux, yb) +
        #           F.binary_cross_entropy_with_logits(rd_aux, yr) )

        # ——— Total loss (Eq. 12) ———————————————————————
        total_loss = 0.45 * Lb + 0.45* Lr + 0.1 * Laux

        # ----- SCALAR LOGGING -----
        # Log each piece
        self.log('val/b_loss', Lb, prog_bar=False)
        self.log('val/r_loss', Lr, prog_bar=False)
        self.log('val/loss', total_loss, prog_bar=True)

        bld_hr_pred = torch.sigmoid(bld_hr)  # (B, 1, H*up, W*up)
        rd_hr_pred = torch.sigmoid(rd_hr)    # (B, 1, H*up, W*up)       
       
        # ───────── HIGH-RES building metrics ─────────
        self.log('val/iou_b_hr', self.val_iou_b_hr(bld_hr_pred, yb),
                 prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)  
        self.log('val/f1_b_hr', self.val_f1_b_hr(bld_hr_pred, yb),
                    prog_bar=True, logger=True,
                    on_step=False, on_epoch=True)
        self.log('val/prec_b_hr', self.val_prec_b_hr(bld_hr_pred, yb),
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True)
        self.log('val/recall_b_hr', self.val_recall_b_hr(bld_hr_pred, yb),
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True)
        
         # ───────── HIGH-RES road metrics ─────────
        self.log('val/iou_r_hr', self.val_iou_r_hr(rd_hr_pred, yr),
                 prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)
        self.log('val/f1_r_hr', self.val_f1_r_hr(rd_hr_pred, yr),
                    prog_bar=True, logger=True,
                    on_step=False, on_epoch=True)
        self.log('val/prec_r_hr', self.val_prec_r_hr(rd_hr_pred, yr),
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True)
        self.log('val/recall_r_hr', self.val_recall_r_hr(rd_hr_pred, yr),
                prog_bar=True, logger=True,
                on_step=False, on_epoch=True)

        # ------------------------------------------------------------------
        # utilities
        # ------------------------------------------------------------------
        def three_class_rgb(build, road):
            """
            Convert 2-channel (building, road) → 3-channel RGB (B,R,BG).
            Output is uint8 in [0,255] ready for imshow().
            """
            if torch.is_tensor(build):
                bg = (1.0 - (build + road)).clamp_(0., 1.)
                rgb = torch.stack([build, road, bg], -1) * 255.0  # (H, W, 3) or (1, H, W, 3)
                rgb = rgb.squeeze(0) if rgb.dim() == 4 else rgb   # <— Fix here
                return rgb.byte().cpu().numpy()
            else:
                bg = np.clip(1.0 - (build + road), 0., 1.)
                rgb = np.stack([build, road, bg], -1) * 255.0
                return rgb.astype(np.uint8)

        def plot_panel(rgb10, sr25,
                    pr_hr, gt_hr,
                    title_suffix=""):
            """Build a 2x3 matplotlib figure with a shared legend."""
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            items = [
                (rgb10, "RGB 10 m"),
                (sr25,  "SR 2 .5 m"),
                (pr_hr, "Pred 2 .5 m"),
                (gt_hr, "GT 2 .5 m"),
            ]
            for ax, (img, ttl) in zip(axes.ravel(), items):
                ax.imshow(img)
                ax.set_title(ttl)
                ax.axis("off")

            # colour legend
            handles = [
                Patch(facecolor=(1, 0, 0), label="Building"),
                Patch(facecolor=(0, 1, 0), label="Road"),
                Patch(facecolor=(0, 0, 1), label="Background"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=3)
            fig.suptitle(f"Sample {title_suffix}", fontsize=14)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            return fig


        # ------------------------------------------------------------------
        # main logging loop  (inside your training/validation step)
        # ------------------------------------------------------------------
        if batch_idx % self.save_freq == 0:
            N_DBG = 5
            bs    = x.size(0)
            x = x.to(self.device, non_blocking=True)  # move to GPU
            # Denormalize input for visualization
            x = x * torch.tensor(self.std).view(1, -1, 1, 1).to(self.device) + \
                torch.tensor(self.mean).view(1, -1, 1, 1).to(self.device)

            # ---------- run SR model for the whole batch (faster) ----------
            for i, idx in enumerate(random.sample(range(bs), k=min(N_DBG, bs))):
                with torch.no_grad():
                    sr_vis = (
                        self.sr_model(x[[idx], :3].to(self.device))
                        .cpu().permute(0, 2, 3, 1).float().numpy().squeeze(0)
                    )                                           # float32, maybe >1
                    
                # ---------- INPUT & SR ----------
                rgb_10m = x[idx, :3].cpu().permute(1, 2, 0).numpy()
                rgb_10m = np.clip(rgb_10m * 2.5, 0, 1)  # ensure [0,1] range

                # ---------- HIGH RES MAPS ----------
                comp_pr_hr = three_class_rgb(bld_hr_pred[idx, 0],
                                            rd_hr_pred[idx, 0])
                comp_gt_hr = three_class_rgb(yb[idx],
                                            yr[idx])

                # ---------- figure & W&B ----------
                fig = plot_panel(rgb10=rgb_10m,
                                sr25=sr_vis,
                                pr_hr=comp_pr_hr,
                                gt_hr=comp_gt_hr,
                                title_suffix=f"idx {idx} | step {self.global_step}")

                wandb.log({f"panel_{i}": wandb.Image(fig)}, commit=False)
                plt.close(fig)                              # free memory

            wandb.log({}, commit=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg['training']['lr'],
            weight_decay=self.cfg['training'].get('weight_decay', 0.0)
        )


def build_trainer(cfg: dict) -> pl.Trainer:
    # ──────────────────────────────────────────────────────────────────────────
    # 1) WandB logger
    wandb_cfg = cfg.get('wandb', {})
    logger = WandbLogger(
        project=wandb_cfg.get('project'),
        name=wandb_cfg.get('name')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Checkpoint callbacks
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
        monitor= es_monitor,
        patience= es_patience,
        mode= es_mode
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Trainer args
    tr_cfg = cfg.get('training', {})
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    num_gpus    = tr_cfg.get('gpu', 1 if torch.cuda.is_available() else 0)
    precision   = tr_cfg.get('precision', '32')
    max_epochs  = tr_cfg.get('epochs', 50)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_best, es_cb],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=num_gpus,
        precision=precision,
        default_root_dir=save_dir,
        check_val_every_n_epoch=1,  # make sure val/loss is logged every epoch
    )

    return trainer


