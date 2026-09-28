"""
AI Stethoscope - Custom CNN Baseline Architecture
==================================================
Implements a modular, configurable 2D Convolutional Neural Network baseline
for Log-Mel spectrogram inputs (B, 1, 128, 313).

Features:
  - 3 to 4 Conv2D-BatchNorm2D-ReLU-MaxPool2D blocks with spatial dropout.
  - Temporal-aware masked global feature pooling for ICBHI respiratory cycles.
  - Fully connected classification head with dropout regularization.
  - Factory functions for Heart (2-class), ICBHI (4-class), and Kaggle (3-class) tasks.
  - Parameter reporting and memory estimation utilities.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pooling import masked_temporal_average_pooling


class ConvBlock(nn.Module):
    """Single Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: tuple = (2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class AudioCNNBaseline(nn.Module):
    """
    Configurable 2D CNN baseline for Log-Mel auscultation classification.

    Parameters:
      in_channels: int (default: 1 for mono spectrogram)
      base_channels: int (default: 32)
      num_blocks: int (default: 4, supports 3 or 4)
      dropout: float (default: 0.3)
      classifier_hidden_dim: int (default: 128)
      num_classes: int (e.g. 2 for Heart, 4 for ICBHI, 3 for Kaggle)
      use_masked_pooling: bool (default: True, applies ICBHI temporal mask)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.3,
        classifier_hidden_dim: int = 128,
        num_classes: int = 2,
        use_masked_pooling: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout_rate = dropout
        self.classifier_hidden_dim = classifier_hidden_dim
        self.num_classes = num_classes
        self.use_masked_pooling = use_masked_pooling

        # Build convolutional blocks
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_blocks)]
        # e.g., for num_blocks=4: [1, 32, 64, 128, 256]

        blocks = []
        for i in range(num_blocks):
            # Moderate dropout across early/late layers
            layer_dropout = dropout * 0.5 if i < 2 else dropout
            # Use (2, 2) pool for early layers, (2, 1) or (2, 2) for late
            pool_sz = (2, 2)
            blocks.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    padding=1,
                    pool_size=pool_sz,
                    dropout=layer_dropout,
                )
            )

        self.conv_backbone = nn.Sequential(*blocks)
        final_conv_channels = channels[-1]

        # Fully connected classification head
        self.head = nn.Sequential(
            nn.Linear(final_conv_channels, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract pooled feature vector before the final classification head.
        Shape: (B, final_conv_channels)
        """
        # x: (B, 1, 128, 313)
        feat_map = self.conv_backbone(x)  # (B, C, F, T)

        if self.use_masked_pooling and mask is not None:
            # Masked temporal average pooling (interpolating mask to match feature T)
            pooled = masked_temporal_average_pooling(feat_map, mask)
        else:
            # Standard Global Average Pooling
            pooled = feat_map.mean(dim=[2, 3])

        return pooled

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        x: (B, 1, 128, 313)
        mask: (B, 313) optional temporal mask
        Returns: logits (B, num_classes)
        """
        pooled = self.extract_features(x, mask=mask)
        logits = self.head(pooled)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_model_size_mb(self) -> float:
        """Estimate model parameter size in Megabytes (float32)."""
        total_params = sum(p.numel() for p in self.parameters())
        return (total_params * 4.0) / (1024 * 1024)


# =====================================================================
# Factory Functions for Task-Specific Models
# =====================================================================

def build_heart_cnn(
    base_channels: int = 32,
    num_blocks: int = 4,
    dropout: float = 0.3,
    classifier_hidden_dim: int = 128,
) -> AudioCNNBaseline:
    """Build Heart sound classification CNN: 2 classes (Normal, Abnormal)."""
    return AudioCNNBaseline(
        in_channels=1,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        num_classes=2,
        use_masked_pooling=False,
    )


def build_icbhi_cnn(
    base_channels: int = 32,
    num_blocks: int = 4,
    dropout: float = 0.3,
    classifier_hidden_dim: int = 128,
) -> AudioCNNBaseline:
    """Build ICBHI respiratory cycle CNN: 4 classes (Normal, Crackle, Wheeze, Crackle+Wheeze) with masked pooling."""
    return AudioCNNBaseline(
        in_channels=1,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        num_classes=4,
        use_masked_pooling=True,
    )


def build_kaggle_cnn(
    base_channels: int = 32,
    num_blocks: int = 4,
    dropout: float = 0.3,
    classifier_hidden_dim: int = 128,
) -> AudioCNNBaseline:
    """Build Kaggle respiratory disease CNN: 3 classes (Normal, Asthma, Other)."""
    return AudioCNNBaseline(
        in_channels=1,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        num_classes=3,
        use_masked_pooling=False,
    )
