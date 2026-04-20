"""
model_siamese.py

Defines a baseline model for a Siamese network. Used to encoder changes
between pre- and post-disaster images. These encoded changes will be
then be used as input to the UNet for segmentation.
"""

import torch
import torch.nn as nn
from src.model_unet import ConvBlock

# Encoder and Decoder design from model_unet.py
# TODO: Siamese design requires separate encoder/decoder. We Can
# TODO: use these classes for all other UNets for consistency,
# TODO: but would of course require refactoring.

class SiameseUnetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_features=32):
        super().__init__()
        f = base_features

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        return e1, e2, e3, e4, b

class SiameseUNetDecoder(nn.Module):
    def __init__(self, num_classes=5, base_features=32):
        super().__init__()
        f = base_features

        self.up4 = nn.ConvTranspose2d(f * 48, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 32, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 16, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 8, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 4, f)

        self.out_conv = nn.Conv2d(f, num_classes, 1)

    def forward(self, e1, e2, e3, e4, b):
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

# Class created with the assistance of LLMs
class SiameseUNet(nn.Module):
    def __init__(self, num_classes, base_features, in_channels=3):
        super().__init__()
        self.encoder = SiameseUnetEncoder(in_channels=in_channels, base_features=base_features)
        self.decoder = SiameseUNetDecoder(num_classes=num_classes, base_features=base_features)

    def forward(self, pre, post):
        # Siamese uses same encoder for pre and post
        e1p, e2p, e3p, e4p, bp = self.encoder(pre)
        e1q, e2q, e3q, e4q, bq = self.encoder(post)

        # Fuse pre- and post- encodings at each scale (Siamese change detection)
        # Use torch.cat to preserve pre-/post- directionality
        e1 = torch.cat([e1p, e1q, torch.abs(e1p - e1q)], dim=1)
        e2 = torch.cat([e2p, e2q, torch.abs(e2p - e2q)], dim=1)
        e3 = torch.cat([e3p, e3q, torch.abs(e3p - e3q)], dim=1)
        e4 = torch.cat([e4p, e4q, torch.abs(e4p - e4q)], dim=1)
        b  = torch.cat([bp, bq,  torch.abs(bp - bq)],   dim=1)

        # Decode fused encoding (segmentation step)
        return self.decoder(e1, e2, e3, e4, b)