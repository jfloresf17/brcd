import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.base import SegmentationHead
from models.dlinknet import DLinkNetDecoder
from models.mti import TaskSharedProjector
from models.csi import CSI
from models.hrfuse import Upsampler, CBAMFusion

class SRSeg(nn.Module):
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

        # 2) Encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels + super_mid,
            depth=encoder_depth, weights=encoder_weights
        )
        enc_ch = self.encoder.out_channels        # (64,256,512,1024,2048)

        # 3) Decoders
        dec_ch = (256, 128, 64, 32, 16) # (128, 64, 32, 16, 8)
        ch     = dec_ch[-1]                       # 16

        self.build_decoder = UnetPlusPlusDecoder(
            encoder_channels=enc_ch,  
            decoder_channels=dec_ch[:-1],
            n_blocks=encoder_depth - 1
        )
        self.build_head = SegmentationHead(in_channels=dec_ch[-2], out_channels=ch, upsampling=upscale/2)

        self.road_decoder = DLinkNetDecoder(
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
        )

        # 4) Projectores
        self.proj_s = TaskSharedProjector(2*ch, ch)

        # 5) CSI + heads
        self.csi_sh = CSI(ch)

        # 6) Deep-supervision 1×1
        self.seg_head = SegmentationHead(in_channels=2*ch, out_channels=num_classes, upsampling=1)
        self.seg_aux = SegmentationHead(in_channels=ch, out_channels=num_classes, upsampling=1)

    # ------------------------------------------------ #

    def forward(self, x, super_fea):
        # HR fuse
        x_hr = self.upsampler(x)  # (B, super_mid, H, W)
        x_fused = self.fusion(super_fea, x_hr)

        # Encoder
        feats = self.encoder(x_fused)  # (B, ch, H, W)
        feats = feats[1:] # Always consider the input data as the first feature map

        # Decoders
        bld_dec = self.build_decoder(feats)   # (B, ch, H, W)
        bld_dec = self.build_head(bld_dec)  # (B, ch, H, W)
        
        rd_dec  = self.road_decoder(feats)   # (B, ch, H, W)

        # Projectores
        sh   = self.proj_s(bld_dec, rd_dec)

        # CSI + logits
        feat_sh = self.csi_sh(sh)

        # Concatenate
        feat_b_sh = torch.cat((bld_dec, feat_sh), dim=1)
        feat_r_sh = torch.cat((rd_dec, feat_sh), dim=1)

        # Deep-sup
        aux_b = self.seg_aux(bld_dec)        # in_c = 32 → OK
        aux_r = self.seg_aux(rd_dec)

        # Final logits
        logits_b = self.seg_head(feat_b_sh)    # in_c = 32 → OK
        logits_r = self.seg_head(feat_r_sh)

        return logits_b, logits_r, aux_b, aux_r