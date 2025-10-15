import torch.nn as nn
import segmentation_models_pytorch as smp
from models.hrfuse import Upsampler, CBAMFusion

class UNetPPSeg(nn.Module):
    def __init__(
        self,
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet",
        encoder_depth=5,
        in_channels=8,
        super_mid=3,
        upscale=4,
        num_classes=1,
    ):
        super().__init__()

        # 1) HR-fuse
        self.upsampler = Upsampler(scale=upscale, n_feats=in_channels)

        self.fusion = CBAMFusion(
            c_hr=in_channels,
            c_lr=super_mid
        )

        # 2) Encoder-Decoder
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels + super_mid,
            classes=num_classes,
            encoder_depth=encoder_depth
        )
        
    # ------------------------------------------------ #

    def forward(self, x, super_fea):
        # HR fusion
        x_hr = self.upsampler(x)  # (B, super_mid, H, W)
        x_fused = self.fusion(super_fea, x_hr)

        # Forward through the model
        x_out = self.model(x_fused)

        return x_out