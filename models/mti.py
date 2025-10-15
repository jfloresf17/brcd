import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


class TaskSpecificProjector(nn.Module):
    """Refina las features de UNet++ o LinkNet con 2 x Conv3 x 3."""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNReLU(ch, ch),
            ConvBNReLU(ch, ch)
        )
    def forward(self, x):
        return self.proj(x)            # (B, ch, H, W)
    

class TaskSharedProjector(nn.Module):
    """
    Concat(bld, rd)  → Conv1 x 1 (reduce a ch) → 2 x Conv3 x 3
    Devuelve un mapa con los mismos `ch` canales que los específicos.
    """
    def __init__(self, in_ch_pair, out_ch):
        super().__init__()
        self.reduce = ConvBNReLU(in_ch_pair, out_ch, k=1, p=0)  # 1×1
        self.conv   = nn.Sequential(
            ConvBNReLU(out_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
    def forward(self, bld, rd):
        x = torch.cat([bld, rd], dim=1)   # (B, 2·ch, H, W)
        x = self.reduce(x)                # (B, ch,  H, W)
        return self.conv(x)               # (B, ch,  H, W)