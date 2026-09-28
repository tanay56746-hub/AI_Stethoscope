"""
AI Stethoscope - Masked Temporal Average Pooling Utilities
===========================================================
Implements temporal masked pooling functions and PyTorch modules to isolate
active acoustic respiratory/cardiac audio frames from zero-padded silence.

Supports:
  - 4D CNN Feature Maps: (B, C, F, T) -> averages over F, masked average over T -> (B, C)
  - 3D Temporal Channels: (B, C, T) -> masked average over T -> (B, C)
  - 3D Transformer Tokens: (B, T, D) -> masked average over T -> (B, D)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_temporal_average_pooling(
    features: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Perform temporal average pooling strictly over active frames specified by mask.

    Parameters:
      features: torch.Tensor
        - 4D shape: (B, C, F, T) e.g., 2D CNN feature map
        - 3D shape: (B, C, T) e.g., 1D temporal convolution
        - 3D shape: (B, T, D) e.g., Transformer sequence embeddings
      mask: torch.Tensor
        - Shape (B, T_in) where 1.0 indicates active frame, 0.0 indicates padded frame.
        - If T_in != T, mask is interpolated via nearest/linear to feature temporal dimension T.
      eps: float
        - Stability constant to prevent division by zero when a mask has zero active frames.

    Returns:
      pooled: torch.Tensor
        - Shape (B, C) for 4D or 3D (B, C, T)
        - Shape (B, D) for 3D (B, T, D)
    """
    if features.ndim not in (3, 4):
        raise ValueError(
            f"Expected features to have 3 or 4 dimensions, got {features.ndim} (shape: {features.shape})"
        )

    # Ensure mask is float on same device
    mask = mask.to(device=features.device, dtype=features.dtype)

    if features.ndim == 4:
        # features: (B, C, F, T)
        B, C, F_dim, T = features.shape
        # If mask length != T, interpolate mask to match feature temporal length T
        if mask.shape[-1] != T:
            # mask: (B, 1, T_in) -> interpolate -> (B, 1, T) -> (B, T)
            mask_interp = F.interpolate(mask.unsqueeze(1), size=T, mode="nearest").squeeze(1)
        else:
            mask_interp = mask

        # Average over frequency bins F first -> (B, C, T)
        time_features = features.mean(dim=2)

        # Expand mask: (B, 1, T)
        mask_expanded = mask_interp.unsqueeze(1)

        # Masked sum over time T -> (B, C)
        weighted_sum = (time_features * mask_expanded).sum(dim=-1)
        active_counts = mask_expanded.sum(dim=-1).clamp(min=eps)

        return weighted_sum / active_counts

    elif features.ndim == 3:
        B, dim1, dim2 = features.shape

        # Distinguish (B, C, T) vs (B, T, D)
        # Check which dimension matches mask temporal length
        if mask.shape[-1] == dim2:
            # (B, C, T) format
            T = dim2
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
            weighted_sum = (features * mask_expanded).sum(dim=-1)
            active_counts = mask_expanded.sum(dim=-1).clamp(min=eps)
            return weighted_sum / active_counts

        elif mask.shape[-1] == dim1:
            # (B, T, D) format
            T = dim1
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            weighted_sum = (features * mask_expanded).sum(dim=1)
            active_counts = mask_expanded.sum(dim=1).clamp(min=eps)
            return weighted_sum / active_counts

        else:
            # Temporal dimension downsampled by backbone (e.g. dim2 is T_down)
            # Default to (B, C, T_down) with interpolated mask
            T = dim2
            mask_interp = F.interpolate(mask.unsqueeze(1), size=T, mode="nearest").squeeze(1)
            mask_expanded = mask_interp.unsqueeze(1)
            weighted_sum = (features * mask_expanded).sum(dim=-1)
            active_counts = mask_expanded.sum(dim=-1).clamp(min=eps)
            return weighted_sum / active_counts


class MaskedTemporalAvgPool2d(nn.Module):
    """
    PyTorch Module wrapper for 2D CNN feature maps (B, C, F, T).
    Averages over F and performs masked temporal average pooling over T.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_temporal_average_pooling(features, mask, eps=self.eps)


class MaskedTemporalAvgPool1d(nn.Module):
    """
    PyTorch Module wrapper for 1D temporal sequence features (B, C, T) or (B, T, D).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_temporal_average_pooling(features, mask, eps=self.eps)
