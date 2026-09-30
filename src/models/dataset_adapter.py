# ==============================================================================
# [LAYER: ML MODEL ARCHITECTURE] | File: src/models/dataset_adapter.py
# Purpose: Dataset-Specific Input Stem & Feature Adapter Layers
# ==============================================================================
"""
AI Stethoscope - Multi-Task Dataset Compatibility Adapter
==========================================================
Demonstrates how the disparate auscultation datasets (PhysioNet Heart, ICBHI, Kaggle)
are formatted into unified multi-task tensors WITHOUT fabricating non-existent labels.

Key Principle:
  - If a label does not exist in the source dataset, the target entry is omitted or None,
    and the corresponding task mask is explicitly set to 0.
  - Zero-imputation is strictly forbidden: missing labels are NEVER labeled as Class 0.
"""

from typing import Dict, Any, Optional, List
import torch


class AuscultationMultiTaskAdapter:
    """
    Standardized adapter converting task-specific sample dictionaries into
    the unified dictionary structure required by FinalMultiTaskStethoscopeModel
    and MaskedMultiTaskLoss.
    """

    @staticmethod
    def format_heart_sample(
        mel: torch.Tensor,
        heart_label: int,
        acoustic_features: Optional[torch.Tensor] = None,
        panns_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Format a single Heart auscultation sample (PhysioNet 2016).
        Heart label is present (0=Normal, 1=Abnormal).
        Acoustic and Disease labels are ABSENT (masks set to 0).
        """
        return {
            "mel": mel,  # (1, 128, 313)
            "acoustic_features": acoustic_features,  # (397,) or None
            "panns_embedding": panns_embedding,  # (2048,) or None
            "temporal_mask": None,  # Not applicable to heart sounds
            "targets": {
                "heart_targets": torch.tensor(heart_label, dtype=torch.long),
            },
            "masks": {
                "heart_mask": torch.tensor(1.0, dtype=torch.float32),
                "acoustic_mask": torch.tensor(0.0, dtype=torch.float32),
                "disease_mask": torch.tensor(0.0, dtype=torch.float32),
            },
            "source_dataset": "heart_physionet_2016",
        }

    @staticmethod
    def format_icbhi_sample(
        mel: torch.Tensor,
        acoustic_label: int,
        temporal_mask: Optional[torch.Tensor] = None,
        acoustic_features: Optional[torch.Tensor] = None,
        panns_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Format an ICBHI 2017 respiratory cycle sample.
        Acoustic cycle label is present (0=Normal, 1=Crackle, 2=Wheeze, 3=Both).
        Heart and Disease labels are ABSENT (masks set to 0).
        """
        return {
            "mel": mel,  # (1, 128, 313)
            "acoustic_features": acoustic_features,  # (397,) or None
            "panns_embedding": panns_embedding,  # (2048,) or None
            "temporal_mask": temporal_mask,  # (T,) active cycle mask
            "targets": {
                "acoustic_targets": torch.tensor(acoustic_label, dtype=torch.long),
            },
            "masks": {
                "heart_mask": torch.tensor(0.0, dtype=torch.float32),
                "acoustic_mask": torch.tensor(1.0, dtype=torch.float32),
                "disease_mask": torch.tensor(0.0, dtype=torch.float32),
            },
            "source_dataset": "icbhi_2017",
        }

    @staticmethod
    def format_kaggle_sample(
        mel: torch.Tensor,
        disease_label: int,
        acoustic_features: Optional[torch.Tensor] = None,
        panns_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Format a Kaggle Respiratory Disease sample.
        Disease label is present (0=Normal, 1=Asthma, 2=Other).
        Heart and Acoustic cycle labels are ABSENT (masks set to 0).
        """
        return {
            "mel": mel,  # (1, 128, 313)
            "acoustic_features": acoustic_features,  # (397,) or None
            "panns_embedding": panns_embedding,  # (2048,) or None
            "temporal_mask": None,
            "targets": {
                "disease_targets": torch.tensor(disease_label, dtype=torch.long),
            },
            "masks": {
                "heart_mask": torch.tensor(0.0, dtype=torch.float32),
                "acoustic_mask": torch.tensor(0.0, dtype=torch.float32),
                "disease_mask": torch.tensor(1.0, dtype=torch.float32),
            },
            "source_dataset": "kaggle_respiratory_disease",
        }

    @staticmethod
    def collate_multitask_batch(batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of sample dictionaries into batched tensors.
        Handles mixed batches containing any combination of tasks.
        """
        B = len(batch_samples)
        mel_batch = torch.stack([s["mel"] for s in batch_samples], dim=0)

        # Collate physical features if present
        phys_list = [s["acoustic_features"] for s in batch_samples]
        if all(p is not None for p in phys_list):
            acoustic_features = torch.stack(phys_list, dim=0)
        else:
            acoustic_features = None

        # Collate PANNs features if present
        panns_list = [s["panns_embedding"] for s in batch_samples]
        if any(p is not None for p in panns_list):
            # Fill missing with zero tensors if mixed
            first_valid = next(p for p in panns_list if p is not None)
            dim_panns = first_valid.shape[-1]
            panns_batch = []
            for p in panns_list:
                if p is not None:
                    panns_batch.append(p)
                else:
                    panns_batch.append(torch.zeros(dim_panns, dtype=first_valid.dtype))
            panns_embedding = torch.stack(panns_batch, dim=0)
        else:
            panns_embedding = None

        # Collate temporal masks if present
        mask_list = [s["temporal_mask"] for s in batch_samples]
        if any(m is not None for m in mask_list):
            first_m = next(m for m in mask_list if m is not None)
            T_len = first_m.shape[-1]
            t_masks = []
            for m in mask_list:
                if m is not None:
                    t_masks.append(m)
                else:
                    t_masks.append(torch.ones(T_len, dtype=torch.float32))
            temporal_mask = torch.stack(t_masks, dim=0)
        else:
            temporal_mask = None

        # Collate masks
        heart_mask = torch.stack([s["masks"]["heart_mask"] for s in batch_samples], dim=0)
        acoustic_mask = torch.stack([s["masks"]["acoustic_mask"] for s in batch_samples], dim=0)
        disease_mask = torch.stack([s["masks"]["disease_mask"] for s in batch_samples], dim=0)

        # Collate targets safely using default placeholder values where mask is 0
        # Placeholder values are NEVER evaluated because loss selects where mask == 1.
        heart_targets = torch.tensor(
            [s["targets"].get("heart_targets", 0) for s in batch_samples], dtype=torch.long
        )
        acoustic_targets = torch.tensor(
            [s["targets"].get("acoustic_targets", 0) for s in batch_samples], dtype=torch.long
        )
        disease_targets = torch.tensor(
            [s["targets"].get("disease_targets", 0) for s in batch_samples], dtype=torch.long
        )

        return {
            "inputs": {
                "mel": mel_batch,
                "acoustic_features": acoustic_features,
                "panns_embedding": panns_embedding,
                "temporal_mask": temporal_mask,
            },
            "targets": {
                "heart_targets": heart_targets,
                "acoustic_targets": acoustic_targets,
                "disease_targets": disease_targets,
            },
            "masks": {
                "heart_mask": heart_mask,
                "acoustic_mask": acoustic_mask,
                "disease_mask": disease_mask,
            },
        }
