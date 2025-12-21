"""
Advanced Deep Learning Models for Dermoscopic Feature Detection

Implements:
- U-Net with Attention mechanisms for precise feature segmentation
- Multi-task learning for all dermoscopic features
- ResNet-based feature encoder
- Attention gates for focusing on relevant regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class AttentionGate(nn.Module):
    """
    Attention gate mechanism to focus on relevant image regions.
    Used in U-Net to highlight important features during upsampling.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature maps in gating signal (from decoder)
            F_l: Number of feature maps in skip connection (from encoder)
            F_int: Number of intermediate feature maps
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetWithAttention(nn.Module):
    """
    U-Net architecture with attention gates for dermoscopic feature segmentation.

    This model performs multi-task segmentation to detect:
    - Pigment network
    - Globules
    - Streaks
    - Blue-white veil
    - Vascular patterns
    - Regression structures
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_filters: int = 64):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (6 dermoscopic features)
            base_filters: Number of filters in first layer
        """
        super(UNetWithAttention, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder (Upsampling path with attention)
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Output layer for multi-class segmentation
        self.out = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with attention
        dec4 = self.upconv4(bottleneck)
        enc4_att = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat((enc4_att, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat((enc3_att, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat((enc2_att, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_att = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat((enc1_att, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important spatial locations"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module to focus on important feature channels"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) - combines channel and spatial attention"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DermoscopyFeatureClassifier(nn.Module):
    """
    Multi-task classifier for dermoscopic features.
    Predicts presence and characteristics of each feature type.
    """
    def __init__(self, in_features: int = 512):
        super(DermoscopyFeatureClassifier, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.pigment_network = nn.Linear(128, 4)  # [absent, reticular, atypical, branched]
        self.globules = nn.Linear(128, 3)  # [absent, regular, irregular]
        self.streaks = nn.Linear(128, 3)  # [absent, regular, radial]
        self.blue_white_veil = nn.Linear(128, 2)  # [absent, present]
        self.vascular = nn.Linear(128, 4)  # [absent, dotted, hairpin, irregular]
        self.regression = nn.Linear(128, 2)  # [absent, present]

    def forward(self, x):
        shared_features = self.shared(x)

        return {
            'pigment_network': self.pigment_network(shared_features),
            'globules': self.globules(shared_features),
            'streaks': self.streaks(shared_features),
            'blue_white_veil': self.blue_white_veil(shared_features),
            'vascular': self.vascular(shared_features),
            'regression': self.regression(shared_features)
        }


class DermoscopyNet(nn.Module):
    """
    Complete dermoscopy analysis network combining segmentation and classification.
    Uses U-Net with attention for segmentation + ResNet backbone for classification.
    """
    def __init__(self, num_segmentation_classes=6, pretrained=True):
        super(DermoscopyNet, self).__init__()

        # Segmentation branch (U-Net with attention)
        self.segmentation = UNetWithAttention(
            in_channels=3,
            num_classes=num_segmentation_classes,
            base_filters=64
        )

        # Feature extraction backbone (ResNet-like)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResBlock 1
            ConvBlock(64, 64),
            CBAM(64),

            # ResBlock 2
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            CBAM(128),

            # ResBlock 3
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            CBAM(256),

            # ResBlock 4
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
            CBAM(512),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification head
        self.classifier = DermoscopyFeatureClassifier(in_features=512)

    def forward(self, x, return_segmentation=True):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
            return_segmentation: Whether to return segmentation masks

        Returns:
            Dictionary containing classification predictions and optionally segmentation masks
        """
        results = {}

        # Segmentation
        if return_segmentation:
            seg_out = self.segmentation(x)
            results['segmentation'] = seg_out

        # Feature extraction and classification
        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        classifications = self.classifier(features)
        results['classification'] = classifications

        return results


def get_model(model_type='full', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Factory function to create dermoscopy models.

    Args:
        model_type: Type of model ('unet', 'classifier', 'full')
        device: Device to load model on

    Returns:
        Model instance
    """
    if model_type == 'unet':
        model = UNetWithAttention(in_channels=3, num_classes=6, base_filters=64)
    elif model_type == 'classifier':
        model = DermoscopyFeatureClassifier(in_features=512)
    elif model_type == 'full':
        model = DermoscopyNet(num_segmentation_classes=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing DermoscopyNet architecture...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('full', device=device)

    # Dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(x, return_segmentation=True)

    print(f"\nModel outputs:")
    print(f"Segmentation shape: {outputs['segmentation'].shape}")
    print(f"\nClassification outputs:")
    for key, value in outputs['classification'].items():
        print(f"  {key}: {value.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
