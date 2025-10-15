import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchvision import models
# ReLU helper
nonlinearity = partial(F.relu, inplace=True)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)

        # Inicialización adecuada de pesos
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x  # conexión residual

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # suma residual
        out = F.relu(out)  # activación final tras la suma

        return out
    

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        d1 = nonlinearity(self.dilate1(x))
        d2 = nonlinearity(self.dilate2(d1))
        d3 = nonlinearity(self.dilate3(d2))
        d4 = nonlinearity(self.dilate4(d3))
        d5 = nonlinearity(self.dilate5(d4))
        return x + d1 + d2 + d3 + d4 + d5  

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(in_channels//4)
        self.deconv = nn.ConvTranspose2d(in_channels//4, in_channels//4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.bn2   = nn.BatchNorm2d(in_channels//4)
        self.conv3 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.bn3   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = nonlinearity(self.bn1(self.conv1(x)))
        x = nonlinearity(self.bn2(self.deconv(x)))
        x = nonlinearity(self.bn3(self.conv3(x)))
        return x


class DLinkNetDecoder(nn.Module):
    """
    D-LinkNet-like decoder with 5-stage depth to mirror encoder channels:
    encoder_channels: tuple of len 6 (including center): (64, 256, 512, 1024, 2048)
    decoder_channels: tuple of len 5: (256, 128, 64, 32, 16)

    Forward:
    - Bridge: process the deepest feature (center)
    - Four stages of upsampling + skip additions
    - One final stage of upsampling without skip
    """
    def __init__(
        self,
        encoder_channels=(64, 256, 512, 1024, 2048),
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()
        # Number of decoder stages (excluding center) == len(decoder_channels)
        num_stages = len(decoder_channels)

        # ----- Bridge -----
        # Takes the center feature map (last of encoder_channels)
        self.dblock = Dblock(encoder_channels[-1])  # 2048 -> 2048

        # ----- Skip 1x1 projections -----
        # We need one skip per all but the final decoder stage
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                encoder_channels[-2 - i],  # encoder skip from stage -(2+i)
                decoder_channels[i],        # project to match decoder stage channels
                kernel_size=1
            )
            for i in range(num_stages - 1)
        ])

        # ----- Decoder blocks -----
        # Create one DecoderBlock per decoder_channels entry
        dec_blocks = []
        in_ch = encoder_channels[-1]  # start from bridge output channels
        for out_ch in decoder_channels:
            dec_blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoders = nn.ModuleList(dec_blocks)

    def forward(self, features: list[torch.Tensor]):
        # features: [f0, f1, f2, f3, f4, f5]
        # where f5 is center (deepest) and f0..f4 are skips

        # Bridge
        x = self.dblock(features[-1])  # process center feature

        # First (num_stages-1) decoder stages with skip connections:
        # i=0 combines with skip features[-2], i=1 with [-3], etc.
        for i, skip_conv in enumerate(self.skip_convs):  # i from 0 to num_stages-2
            x = self.decoders[i](x)                     # upsample from previous stage
            skip_feat = features[-2 - i]                # corresponding skip map
            # project skip to decoder channels and add
            x = x + skip_conv(skip_feat)

        # Final decoder stage without skip
        x = self.decoders[-1](x)
        return x
    

class DLinkNet34(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, filters=(64, 128, 256, 512)):
        super(DLinkNet34, self).__init__()

        resnet = models.resnet34(pretrained=True) # By default the pretrained model comes from imageNet
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out