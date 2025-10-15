import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from models.hrfuse import Upsampler, CBAMFusion
from models.dlinknet import DLinkNet34

class DLinkNetSeg(nn.Module):
    def __init__(
        self,
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

        # 2) Model
        self.model = DLinkNet34(
            in_channels=in_channels + super_mid,
            num_classes=num_classes
        )
    # ------------------------------------------------ #

    def forward(self, x, super_fea):
        # HR fuse
        x_hr = self.upsampler(x)  # (B, super_mid, H, W)
        x_fused = self.fusion(super_fea, x_hr)

        # Forward through the model
        x_out = self.model(x_fused)

        return x_out