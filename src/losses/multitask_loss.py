"""
AI Stethoscope - Masked Multi-Task Cross-Entropy Loss
=====================================================
Implements masked multi-task loss for training a shared audio backbone across
datasets with non-overlapping label spaces (PhysioNet Heart, ICBHI, Kaggle).

Key Design Principles:
  1. Strict Task Isolation:
     Samples lacking ground-truth labels for a task (mask=0) are NEVER treated as
     negative examples (Class 0). Their contribution to that task's loss is strictly 0.0.
  
  2. Per-Task Normalization:
     Losses for each task are normalized solely by the number of active samples
     for that task in the batch:
       L_task = (sum_i m_i * loss_i) / (sum_i m_i + eps) if active else 0.0
     This ensures missing tasks do not distort or dilute gradient magnitudes.
  
  3. Class Weighting:
     Supports inverse-frequency class weights for each task to address extreme
     imbalance (e.g., Abnormal heart sounds, Crackle+Wheeze combination).
  
  4. Configurable Task Weights:
     lambda_heart, lambda_acoustic, lambda_disease balance task contributions.
     These are initialized with sensible defaults (1.0) and are tunable hyperparameters
     that should be selected using validation split performance.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked Multi-Task Loss supporting Heart, ICBHI Acoustic, and Kaggle Disease tasks.
    """

    def __init__(
        self,
        lambda_heart: float = 1.0,
        lambda_acoustic: float = 1.0,
        lambda_disease: float = 1.0,
        heart_class_weights: Optional[torch.Tensor] = None,
        acoustic_class_weights: Optional[torch.Tensor] = None,
        disease_class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_heart = float(lambda_heart)
        self.lambda_acoustic = float(lambda_acoustic)
        self.lambda_disease = float(lambda_disease)
        self.eps = eps

        # Store class weight buffers (or None)
        if heart_class_weights is not None:
            self.register_buffer("heart_class_weights", heart_class_weights.float())
        else:
            self.heart_class_weights = None

        if acoustic_class_weights is not None:
            self.register_buffer("acoustic_class_weights", acoustic_class_weights.float())
        else:
            self.acoustic_class_weights = None

        if disease_class_weights is not None:
            self.register_buffer("disease_class_weights", disease_class_weights.float())
        else:
            self.disease_class_weights = None

    def compute_single_task_loss(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute mean cross-entropy loss over active masked samples for a single task.

        Parameters:
          logits: (B, num_classes)
          targets: (B,) integer class labels, or None
          mask: (B,) binary/boolean mask where 1/True indicates label is valid, or None
          weight: Optional class weight tensor (num_classes,)

        Returns:
          task_loss: scalar tensor (0.0 if no active samples)
          num_active: integer count of active samples
        """
        if targets is None or mask is None:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0

        # Cast mask to boolean
        mask_bool = mask.bool()
        num_active = int(mask_bool.sum().item())

        if num_active == 0:
            # Zero active samples: return zero with attached computation graph
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), 0

        # Filter strictly active samples
        active_logits = logits[mask_bool]
        active_targets = targets[mask_bool].long()

        # Compute unreduced cross entropy or weighted cross entropy
        loss = F.cross_entropy(
            active_logits,
            active_targets,
            weight=weight,
            reduction="mean",
        )
        return loss, num_active

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate total multi-task loss and per-task losses.

        Parameters:
          predictions: Dictionary containing:
            - 'heart_logits': (B, 2)
            - 'acoustic_logits': (B, 4)
            - 'disease_logits': (B, 3)
          targets: Dictionary containing:
            - 'heart_targets': (B,) optional
            - 'acoustic_targets': (B,) optional
            - 'disease_targets': (B,) optional
          masks: Dictionary containing:
            - 'heart_mask': (B,) boolean/binary
            - 'acoustic_mask': (B,) boolean/binary
            - 'disease_mask': (B,) boolean/binary

        Returns:
          Dictionary containing:
            - 'total_loss': scalar total loss
            - 'heart_loss': scalar heart loss (0 if inactive)
            - 'acoustic_loss': scalar acoustic loss (0 if inactive)
            - 'disease_loss': scalar disease loss (0 if inactive)
            - 'num_active_heart': int
            - 'num_active_acoustic': int
            - 'num_active_disease': int
        """
        device = predictions["heart_logits"].device

        # 1. Heart Task Loss
        heart_loss, n_heart = self.compute_single_task_loss(
            logits=predictions["heart_logits"],
            targets=targets.get("heart_targets"),
            mask=masks.get("heart_mask"),
            weight=self.heart_class_weights,
        )

        # 2. Lung Acoustic Task Loss
        acoustic_loss, n_acoustic = self.compute_single_task_loss(
            logits=predictions["acoustic_logits"],
            targets=targets.get("acoustic_targets"),
            mask=masks.get("acoustic_mask"),
            weight=self.acoustic_class_weights,
        )

        # 3. Lung Disease Task Loss
        disease_loss, n_disease = self.compute_single_task_loss(
            logits=predictions["disease_logits"],
            targets=targets.get("disease_targets"),
            mask=masks.get("disease_mask"),
            weight=self.disease_class_weights,
        )

        # 4. Total Multi-Task Loss
        total_loss = (
            (self.lambda_heart * heart_loss if n_heart > 0 else torch.tensor(0.0, device=device))
            + (self.lambda_acoustic * acoustic_loss if n_acoustic > 0 else torch.tensor(0.0, device=device))
            + (self.lambda_disease * disease_loss if n_disease > 0 else torch.tensor(0.0, device=device))
        )

        return {
            "total_loss": total_loss,
            "heart_loss": heart_loss,
            "acoustic_loss": acoustic_loss,
            "disease_loss": disease_loss,
            "num_active_heart": torch.tensor(n_heart, device=device),
            "num_active_acoustic": torch.tensor(n_acoustic, device=device),
            "num_active_disease": torch.tensor(n_disease, device=device),
        }
