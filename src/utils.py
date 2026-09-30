# ==============================================================================
# [LAYER: ML UTILITIES] | File: src/utils.py
# Purpose: Reproducibility Seeds, Device Selection, Logging & Config Helpers
# ==============================================================================
"""
AI Stethoscope - Training Utilities & Class Weights Calculation
================================================================
Implements:
  - Deterministic random seed initialization.
  - Dynamic inverse-frequency class weight calculation from training manifests/indices.
  - Class weights export to processed/config/class_weights.json.
  - YAML training configuration loaders.
"""

import os
import csv
import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Union, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for complete reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_and_save_class_weights(
    base_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Calculate inverse-frequency class weights exclusively from the training split index files:
      weight_i = N / (K * count_i)
    where:
      N = total training samples
      K = number of distinct classes
      count_i = count of class i in training split

    Saves results to processed/config/class_weights.json.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    else:
        base_dir = Path(base_dir)

    if output_path is None:
        output_path = base_dir / "processed" / "config" / "class_weights.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks_info = {
        "heart": {
            "index_file": base_dir / "processed" / "full" / "heart_train_index.csv",
            "classes": ["Normal", "Abnormal"],
            "label_col": "label",
            "id_col": "label_id",
        },
        "lung_acoustic": {
            "index_file": base_dir / "processed" / "full" / "icbhi_train_index.csv",
            "classes": ["Normal", "Crackle", "Wheeze", "Crackle+Wheeze"],
            "label_col": "label",
            "id_col": "label_id",
        },
        "lung_disease": {
            "index_file": base_dir / "processed" / "full" / "kaggle_train_index.csv",
            "classes": ["Normal", "Asthma", "Other"],
            "label_col": "label",
            "id_col": "label_id",
        },
    }

    result = {
        "calculation_method": "inverse_frequency (N / (K * count_i))",
        "source": "Training split index files only",
        "tasks": {},
    }

    for task_name, info in tasks_info.items():
        fpath = info["index_file"]
        if not fpath.exists():
            raise FileNotFoundError(f"Required training index file missing: {fpath}")

        with open(fpath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        N = len(records)
        K = len(info["classes"])
        counts = Counter(r[info["label_col"]] for r in records)

        # Build label to ID map from records
        label_to_id = {}
        for r in records:
            lbl = r[info["label_col"]]
            lid = int(r[info["id_col"]])
            label_to_id[lbl] = lid

        weights_by_class = {}
        weights_by_id = {}
        ordered_weights = [0.0] * K

        for cls_name in info["classes"]:
            c_cnt = counts.get(cls_name, 0)
            if c_cnt == 0:
                w = 0.0
            else:
                w = float(N / (K * c_cnt))

            cls_id = label_to_id.get(cls_name, info["classes"].index(cls_name))
            weights_by_class[cls_name] = round(w, 5)
            weights_by_id[str(cls_id)] = round(w, 5)
            ordered_weights[cls_id] = round(w, 5)

        result["tasks"][task_name] = {
            "total_train_samples": N,
            "num_classes": K,
            "class_counts": {k: counts.get(k, 0) for k in info["classes"]},
            "class_weights": weights_by_class,
            "weights_by_id": weights_by_id,
            "ordered_tensor_weights": ordered_weights,
        }

    with open(output_path, mode="w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2)

    return result


def load_training_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML training configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f)
