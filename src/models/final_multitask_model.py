# ==============================================================================
# [LAYER: ML MODEL ARCHITECTURE] | File: src/models/final_multitask_model.py
# Purpose: Verified Tri-Modal Shared Multi-Task Stethoscope Network (CNN + Phys + PANNs)
# ==============================================================================
"""
AI Stethoscope - Final Multi-Task Architecture
==============================================
Implements the unified multi-task architecture specified in
analysis/final_model_architecture_plan.md.

Architecture Overview:
  1. Shared Auscultation CNN Trunk:
     4 Conv2D-BatchNorm2D-ReLU-MaxPool2D-Dropout2d blocks (~1.2M params)
     converting Log-Mel input (B, 1, 128, 313) into shared feature maps (B, 256, F, T).
  
  2. Physical Acoustic Feature Branch:
     Trainable Linear(397 -> 64) + ReLU projecting handcrafted physical features.
  
  3. PANNs Auxiliary Precomputed Feature Branch:
     Trainable Linear(2048 -> 128) + ReLU bottleneck projecting precomputed AudioSet embeddings.
  
  4. Heart Pathology Head:
     256-d pooled CNN + 64-d physical = 320-d -> Linear(320 -> 128) -> ReLU -> Dropout -> Linear(128 -> 2).
     Classes: 0 = Normal, 1 = Abnormal.
  
  5. Lung Acoustic Cycle Head (ICBHI):
     Masked Temporal Average Pooling (MTAP) 256-d + 128-d PANNs = 384-d -> Linear(384 -> 128) -> ReLU -> Dropout -> Linear(128 -> 4).
     Classes: 0 = Normal, 1 = Crackle, 2 = Wheeze, 3 = Crackle+Wheeze.
  
  6. Lung Disease Diagnostic Head (Kaggle):
     256-d pooled CNN + 64-d physical = 320-d -> Linear(320 -> 128) -> ReLU -> Dropout -> Linear(128 -> 3).
     Classes: 0 = Normal, 1 = Asthma, 2 = Other.

Important:
  - PANNs 81.8M-parameter CNN14 is NOT instantiated in this model. Precomputed 2048-d
    features from processed/panns_embeddings/ are supplied at inference/training.
  - Handcrafted features (397-d) and PANNs features (2048-d) are optional in forward();
    if omitted or None, clean zero representations are used gracefully.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pooling import masked_temporal_average_pooling


class ConvBlock(nn.Module):
    """Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: Tuple[int, int] = (2, 2),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SharedAuscultationTrunk(nn.Module):
    """
    Shared 2D CNN Trunk for Auscultation Audio Spectrograms.
    Input:  (B, 1, 128, 313)
    Output: (B, 256, F_out, T_out) where F_out=8, T_out=19 for 4 pool stages.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels, pool_size=(2, 2), dropout=dropout * 0.75)
        self.block2 = ConvBlock(base_channels, base_channels * 2, pool_size=(2, 2), dropout=dropout * 0.75)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4, pool_size=(2, 2), dropout=dropout)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8, pool_size=(2, 2), dropout=dropout)
        self.out_channels = base_channels * 8  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)  # (B, 32, 64, 156)
        x = self.block2(x)  # (B, 64, 32, 78)
        x = self.block3(x)  # (B, 128, 16, 39)
        x = self.block4(x)  # (B, 256, 8, 19)
        return x


class PhysicalFeatureBranch(nn.Module):
    """
    Trainable projection for 397-dimensional physical acoustic features.
    Input:  (B, 397)
    Output: (B, 64)
    """

    def __init__(self, in_features: int = 397, out_features: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"PhysicalFeatureBranch expected input dimension {self.in_features}, "
                f"but received shape {x.shape} (dim={x.shape[-1]})."
            )
        return self.projection(x)


class PANNsAuxiliaryBranch(nn.Module):
    """
    Trainable bottleneck projection for precomputed 2048-dimensional PANNs CNN14 embeddings.
    Input:  (B, 2048)
    Output: (B, 128)
    """

    def __init__(self, in_features: int = 2048, out_features: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"PANNsAuxiliaryBranch expected input dimension {self.in_features}, "
                f"but received shape {x.shape} (dim={x.shape[-1]}). "
                f"Do not supply unprojected or incorrectly shaped vectors."
            )
        return self.projection(x)


class FinalMultiTaskStethoscopeModel(nn.Module):
    """
    Final Multi-Task AI Digital Stethoscope Neural Network.
    
    Integrates:
      - Shared Auscultation CNN Trunk
      - Physical Acoustic Feature Branch (397 -> 64)
      - PANNs Auxiliary Precomputed Feature Branch (2048 -> 128)
      - Heart Pathology Head (320 -> 2)
      - Lung Acoustic Cycle Head (384 -> 4)
      - Lung Disease Diagnostic Head (320 -> 3)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        physical_dim: int = 397,
        physical_proj_dim: int = 64,
        panns_dim: int = 2048,
        panns_proj_dim: int = 128,
        heart_hidden_dim: int = 128,
        acoustic_hidden_dim: int = 128,
        disease_hidden_dim: int = 128,
        dropout_trunk: float = 0.20,
        dropout_heart: float = 0.30,
        dropout_acoustic: float = 0.25,
        dropout_disease: float = 0.35,
    ):
        super().__init__()
        self.physical_dim = physical_dim
        self.physical_proj_dim = physical_proj_dim
        self.panns_dim = panns_dim
        self.panns_proj_dim = panns_proj_dim

        # 1. Shared Auscultation CNN Trunk
        self.shared_trunk = SharedAuscultationTrunk(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout_trunk,
        )
        cnn_dim = self.shared_trunk.out_channels  # 256

        # 2. Auxiliary Feature Branches
        self.physical_branch = PhysicalFeatureBranch(
            in_features=physical_dim,
            out_features=physical_proj_dim,
        )
        self.panns_branch = PANNsAuxiliaryBranch(
            in_features=panns_dim,
            out_features=panns_proj_dim,
        )

        # 3. Heart Head: (256 CNN + 64 Physical = 320) -> 128 -> 2
        heart_in_dim = cnn_dim + physical_proj_dim  # 320
        self.heart_head = nn.Sequential(
            nn.Linear(heart_in_dim, heart_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_heart),
            nn.Linear(heart_hidden_dim, 2),
        )

        # 4. Lung Acoustic Head: (256 MTAP CNN + 128 PANNs = 384) -> 128 -> 4
        acoustic_in_dim = cnn_dim + panns_proj_dim  # 384
        self.acoustic_head = nn.Sequential(
            nn.Linear(acoustic_in_dim, acoustic_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_acoustic),
            nn.Linear(acoustic_hidden_dim, 4),
        )

        # 5. Lung Disease Head: (256 CNN + 64 Physical = 320) -> 128 -> 3
        disease_in_dim = cnn_dim + physical_proj_dim  # 320
        self.disease_head = nn.Sequential(
            nn.Linear(disease_in_dim, disease_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_disease),
            nn.Linear(disease_hidden_dim, 3),
        )

    def extract_cnn_features(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared CNN trunk."""
        return self.shared_trunk(mel)

    def pool_global_spatiotemporal(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """Global average pooling over frequency and time: (B, C, F, T) -> (B, C)."""
        return cnn_feat.mean(dim=[-2, -1])

    def pool_masked_temporal(
        self, cnn_feat: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Masked temporal average pooling for variable duration respiratory cycles.
        If mask is None, falls back to unmasked global average pooling.
        """
        if mask is not None:
            return masked_temporal_average_pooling(cnn_feat, mask)
        return self.pool_global_spatiotemporal(cnn_feat)

    def forward(
        self,
        mel: torch.Tensor,
        acoustic_features: Optional[torch.Tensor] = None,
        panns_embedding: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing predictions for all three task heads.

        Parameters:
          mel: torch.Tensor of shape (B, 1, 128, 313)
          acoustic_features: Optional torch.Tensor of shape (B, 397)
          panns_embedding: Optional torch.Tensor of shape (B, 2048)
          temporal_mask: Optional torch.Tensor of shape (B, T_raw) for active cycles

        Returns:
          Dictionary containing:
            - 'heart_logits': (B, 2)
            - 'acoustic_logits': (B, 4)
            - 'disease_logits': (B, 3)
            - 'shared_cnn_features': (B, 256)
            - 'physical_projected': (B, 64)
            - 'panns_projected': (B, 128)
        """
        B = mel.shape[0]
        device = mel.device
        dtype = mel.dtype

        # 1. Forward through shared auscultation CNN trunk: (B, 256, 8, 19)
        cnn_map = self.extract_cnn_features(mel)

        # 2. Global spatio-temporal pooling for Heart and Disease: (B, 256)
        cnn_global = self.pool_global_spatiotemporal(cnn_map)

        # 3. Masked temporal average pooling for Acoustic cycle head: (B, 256)
        cnn_acoustic = self.pool_masked_temporal(cnn_map, temporal_mask)

        # 4. Process Physical Features (B, 397 -> B, 64)
        if acoustic_features is not None:
            phys_proj = self.physical_branch(acoustic_features)
        else:
            # Clean zero vector fallback if task has no physical features
            phys_proj = torch.zeros(B, self.physical_proj_dim, device=device, dtype=dtype)

        # 5. Process PANNs Auxiliary Features (B, 2048 -> B, 128)
        if panns_embedding is not None:
            panns_proj = self.panns_branch(panns_embedding)
        else:
            # Clean zero vector fallback if task has no PANNs embedding
            panns_proj = torch.zeros(B, self.panns_proj_dim, device=device, dtype=dtype)

        # 6. Heart Head: (B, 256 + 64 = 320) -> (B, 2)
        heart_rep = torch.cat([cnn_global, phys_proj], dim=-1)
        heart_logits = self.heart_head(heart_rep)

        # 7. Lung Acoustic Head: (B, 256 + 128 = 384) -> (B, 4)
        acoustic_rep = torch.cat([cnn_acoustic, panns_proj], dim=-1)
        acoustic_logits = self.acoustic_head(acoustic_rep)

        # 8. Lung Disease Head: (B, 256 + 64 = 320) -> (B, 3)
        disease_rep = torch.cat([cnn_global, phys_proj], dim=-1)
        disease_logits = self.disease_head(disease_rep)

        return {
            "heart_logits": heart_logits,
            "acoustic_logits": acoustic_logits,
            "disease_logits": disease_logits,
            "shared_cnn_features": cnn_global,
            "physical_projected": phys_proj,
            "panns_projected": panns_proj,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown by major architectural branch."""
        counts = {
            "shared_trunk": sum(p.numel() for p in self.shared_trunk.parameters()),
            "physical_branch": sum(p.numel() for p in self.physical_branch.parameters()),
            "panns_branch": sum(p.numel() for p in self.panns_branch.parameters()),
            "heart_head": sum(p.numel() for p in self.heart_head.parameters()),
            "acoustic_head": sum(p.numel() for p in self.acoustic_head.parameters()),
            "disease_head": sum(p.numel() for p in self.disease_head.parameters()),
        }
        counts["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
