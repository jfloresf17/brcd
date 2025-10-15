import torch
import torch.nn as nn

class CSI(nn.Module):
    def __init__(self, chans: int, reduction: int = 16):
        super().__init__()
        # 1) initial 5×5 depthwise
        self.init = nn.Conv2d(chans, chans, 5, padding=2, groups=chans)
        # 2) large-kernel branches
        self.b7 = nn.Sequential(
            nn.Conv2d(chans, chans, (7,1), padding=(3,0), groups=chans),
            nn.Conv2d(chans, chans, (1,7), padding=(0,3), groups=chans),
        )
        self.b11 = nn.Sequential(
            nn.Conv2d(chans, chans, (11,1), padding=(5,0), groups=chans),
            nn.Conv2d(chans, chans, (1,11), padding=(0,5), groups=chans),
        )
        self.b21 = nn.Sequential(
            nn.Conv2d(chans, chans, (21,1), padding=(10,0), groups=chans),
            nn.Conv2d(chans, chans, (1,21), padding=(0,10), groups=chans),
        )
        # 3) global‐average‐pool (GAP)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 4) MLP to produce 4*C raw weights
        self.mlp = nn.Sequential(
            nn.Conv2d(chans, chans // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans // reduction, 4 * chans, 1, bias=False)
        )
        # 5) normalize over the 4 “scale” branches
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a) branch features
        f0  = self.init(x)     # skip / 5×5
        f7  = self.b7(f0)
        f11 = self.b11(f0)
        f21 = self.b21(f0)

        # b) pointwise fuse
        fuse = f0 + f7 + f11 + f21   # (B, C, H, W)

        # c) GAP to (B, C, 1, 1)
        gap = self.gap(fuse)

        # d) MLP → (B, 4*C, 1, 1)
        w   = self.mlp(gap)

        # e) reshape to (B, 4, C, 1, 1) and softmax over branch dim
        B, _ ,_,_ = w.shape
        w = w.view(B, 4, -1, 1, 1)   # now axis=1 indexes [skip,7,11,21]
        w = self.softmax(w)          # sum_i w[:,i,:,:,:] == 1

        # f) weighted sum of each branch
        out = (w[:,0] * f0
             + w[:,1] * f7
             + w[:,2] * f11
             + w[:,3] * f21)

        return out


# class CSIHead(nn.Module):
#     """
#     CSI + conv1 x 1 ⇒ logits.
#     Retorna (logits, features) para usar deep supervision.
#     """
#     def __init__(self, in_ch, num_classes=1):
#         super().__init__()
#         self.csi  = CSI(in_ch)
#         self.conv = nn.Conv2d(in_ch, num_classes, 1)

#     def forward(self, x):
#         feat = self.csi(x)          # (B, in_ch, H, W)
#         log  = self.conv(feat)      # (B, num_classes, H, W)
#         return log, feat