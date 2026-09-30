# ==============================================================================
# [LAYER: ML DATASETS] | File: src/datasets.py
# Purpose: PyTorch Dataset Loaders for PhysioNet Heart, ICBHI Lung, and Kaggle Lung
# ==============================================================================
"""
AI Stethoscope - PyTorch Dataset Classes & DataLoader Utilities
================================================================
Implements robust PyTorch Dataset classes for:
  - Heart Sound Dataset (PhysioNet 2016): Normal (0), Abnormal (1)
  - ICBHI 2017 Lung Sound Dataset: Normal (0), Crackle (1), Wheeze (2), Crackle+Wheeze (3)
  - Kaggle Respiratory Dataset: Normal (0), Asthma (1), Other (2)

Features:
  - Strict tensor dimension enforcement: (1, 128, 313), float32.
  - No silent transposition.
  - Zero NaN/Inf validation on every loaded tensor.
  - ICBHI Temporal Mask generation: (313,) float32 mask isolating active acoustic duration from zero-padding.
  - Full traceability: every batch retains patient_id, recording_id, class_label, and metadata.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EXPECTED_SPECTROGRAM_SHAPE = (128, 313)
EXPECTED_TENSOR_SHAPE = (1, 128, 313)
TIME_FRAMES = 313
AUDIO_DURATION_SEC = 5.0


class BaseStethoscopeDataset(Dataset):
    """Base class providing common loading, validation, and metadata management."""

    def __init__(
        self,
        index_csv_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        transform=None,
    ):
        self.index_csv_path = Path(index_csv_path)
        if base_dir is None:
            # Workspace root is parent of processed/
            self.base_dir = self.index_csv_path.resolve().parents[2]
        else:
            self.base_dir = Path(base_dir)

        self.transform = transform
        self.records: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        if not self.index_csv_path.exists():
            raise FileNotFoundError(f"Index CSV not found: {self.index_csv_path}")
        with open(self.index_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

    def __len__(self) -> int:
        return len(self.records)

    def _load_and_validate_tensor(self, rel_path: str) -> torch.Tensor:
        full_path = self.base_dir / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Tensor file missing: {full_path}")

        arr = np.load(full_path)

        if arr.shape != EXPECTED_SPECTROGRAM_SHAPE:
            raise ValueError(
                f"Unexpected tensor shape {arr.shape} in {rel_path}; expected {EXPECTED_SPECTROGRAM_SHAPE}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite value (NaN or Inf) detected in tensor: {rel_path}")

        tensor = torch.from_numpy(arr).float()

        # Add single channel dimension at dim 0 -> (1, 128, 313)
        tensor = tensor.unsqueeze(0)
        assert tensor.shape == EXPECTED_TENSOR_SHAPE, f"Shape mismatch: {tensor.shape} != {EXPECTED_TENSOR_SHAPE}"
        return tensor


class HeartDataset(BaseStethoscopeDataset):
    """
    PyTorch Dataset for PhysioNet 2016 Heart Sound Tensors.
    Task: heart
    Classes: Normal (0), Abnormal (1)
    """

    def __init__(self, index_csv_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, transform=None):
        super().__init__(index_csv_path, base_dir, transform)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        tensor = self._load_and_validate_tensor(row["tensor_path"])

        if self.transform:
            tensor = self.transform(tensor)

        # Heart recordings use continuous windows; default temporal mask is all 1s
        mask = torch.ones(TIME_FRAMES, dtype=torch.float32)

        return {
            "tensor": tensor,
            "mask": mask,
            "dataset": "heart",
            "split": self.index_csv_path.stem.split("_")[1],  # train, val, test
            "patient_id": row["patient_id"],
            "recording_id": row["recording_id"],
            "class_label": row["label"],
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "task": "heart",
            "tensor_path": row["tensor_path"],
            "metadata": {
                "window_idx": int(row.get("window_idx", 0)),
            },
        }


class ICBHIDataset(BaseStethoscopeDataset):
    """
    PyTorch Dataset for ICBHI 2017 Respiratory Cycle Tensors.
    Task: lung_acoustic
    Classes: Normal (0), Crackle (1), Wheeze (2), Crackle+Wheeze (3)
    Includes exact Temporal Mask isolating active respiratory cycle from zero-padding.
    """

    def __init__(self, index_csv_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, transform=None):
        super().__init__(index_csv_path, base_dir, transform)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        tensor = self._load_and_validate_tensor(row["tensor_path"])

        if self.transform:
            tensor = self.transform(tensor)

        t_start = float(row["start_time"])
        t_end = float(row["end_time"])
        duration = max(0.0, t_end - t_start)

        # Calculate exact active temporal frames
        active_frames = min(TIME_FRAMES, max(1, int(round((duration / AUDIO_DURATION_SEC) * TIME_FRAMES))))

        # Active region = 1.0, Padded region = 0.0
        mask = torch.zeros(TIME_FRAMES, dtype=torch.float32)
        mask[:active_frames] = 1.0

        return {
            "tensor": tensor,
            "mask": mask,
            "dataset": "icbhi",
            "split": self.index_csv_path.stem.split("_")[1],
            "patient_id": row["patient_id"],
            "recording_id": row["recording_id"],
            "class_label": row["label"],
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "task": "lung_acoustic",
            "tensor_path": row["tensor_path"],
            "metadata": {
                "cycle_idx": int(row.get("cycle_idx", 0)),
                "chest_location": row.get("chest_location", ""),
                "start_time": t_start,
                "end_time": t_end,
                "duration": duration,
                "active_frames": active_frames,
            },
        }


class KaggleDataset(BaseStethoscopeDataset):
    """
    PyTorch Dataset for Kaggle Respiratory Disease Tensors.
    Task: lung_disease
    Classes: Normal (0), Asthma (1), Other (2)
    Preserves patient_id, recording_id, mode, sound_type, location, age, gender.
    """

    def __init__(self, index_csv_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None, transform=None):
        super().__init__(index_csv_path, base_dir, transform)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        tensor = self._load_and_validate_tensor(row["tensor_path"])

        if self.transform:
            tensor = self.transform(tensor)

        mask = torch.ones(TIME_FRAMES, dtype=torch.float32)

        return {
            "tensor": tensor,
            "mask": mask,
            "dataset": "kaggle",
            "split": self.index_csv_path.stem.split("_")[1],
            "patient_id": row["patient_id"],
            "recording_id": row["recording_id"],
            "class_label": row["label"],
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "task": "lung_disease",
            "tensor_path": row["tensor_path"],
            "metadata": {
                "window_idx": int(row.get("window_idx", 0)),
                "diagnosis": row.get("diagnosis", ""),
                "mode": row.get("mode", ""),
                "sound_type": row.get("sound_type", ""),
                "location": row.get("location", ""),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            },
        }


def collate_audio_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function stacking tensors, masks, and label_ids into uniform batches.
    Preserves sample-level metadata traceability in lists.
    """
    tensors = torch.stack([item["tensor"] for item in batch], dim=0)  # (B, 1, 128, 313)
    masks = torch.stack([item["mask"] for item in batch], dim=0)      # (B, 313)
    label_ids = torch.stack([item["label_id"] for item in batch], dim=0)  # (B,)

    return {
        "tensor": tensors,
        "mask": masks,
        "label_id": label_ids,
        "dataset": [item["dataset"] for item in batch],
        "split": [item["split"] for item in batch],
        "task": [item["task"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "recording_id": [item["recording_id"] for item in batch],
        "class_label": [item["class_label"] for item in batch],
        "tensor_path": [item["tensor_path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def get_dataset(
    dataset_name: str,
    split: str,
    base_dir: Optional[Union[str, Path]] = None,
    transform=None,
) -> BaseStethoscopeDataset:
    """Factory function to instantiate appropriate dataset."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    else:
        base_dir = Path(base_dir)

    dataset_name = dataset_name.lower().strip()
    split = split.lower().strip()

    index_file = base_dir / "processed" / "full" / f"{dataset_name}_{split}_index.csv"

    if dataset_name == "heart":
        return HeartDataset(index_file, base_dir=base_dir, transform=transform)
    elif dataset_name == "icbhi":
        return ICBHIDataset(index_file, base_dir=base_dir, transform=transform)
    elif dataset_name == "kaggle":
        return KaggleDataset(index_file, base_dir=base_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Choose from: heart, icbhi, kaggle.")


def get_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    base_dir: Optional[Union[str, Path]] = None,
    transform=None,
) -> DataLoader:
    """Factory function to build DataLoader with standardized batch collation."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = get_dataset(dataset_name, split, base_dir=base_dir, transform=transform)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_audio_batch,
    )
