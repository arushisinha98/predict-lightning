"""U-NET model architecture for event detection and counting."""

import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self, device):
        super(UNET, self).__init__()
        self.device = device

        # Residual block definition
        def make_residual_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Mish(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Mish(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
            )

        # Encoder Path with Residual Connections
        self.enc1 = make_residual_block(4, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = make_residual_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = make_residual_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deeper Bottleneck with Multiple Residual Blocks
        self.bottleneck = nn.Sequential(
            make_residual_block(128, 256),
            make_residual_block(256, 256),
            make_residual_block(256, 128),
            nn.Conv2d(128, 64, kernel_size=1)
        )

        # Decoder Path with Skip Connections
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3 = make_residual_block(192, 64)  # 192 = 64 + 128 (skip connection)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = make_residual_block(96, 32)   # 96 = 32 + 64 (skip connection)

        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = make_residual_block(64, 32)   # 64 = 32 + 32 (skip connection)

        # Final layer with sigmoid activation
        self.final_heatmap = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add count prediction branch
        self.count_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # ReLU to ensure non-negative count
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Skip Connections
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Split into two outputs
        heatmap = self.final_heatmap(d1).squeeze(1)
        count = self.count_predictor(d1)
        
        return heatmap, count