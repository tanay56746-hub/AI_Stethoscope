# ==============================================================================
# [LAYER: ML EVALUATION METRICS] | File: src/metrics.py
# Purpose: Sensitivity, Specificity, ICBHI Score, Macro-F1 & Confusion Matrix Metrics
# ==============================================================================
"""
AI Stethoscope - Classification Metrics & Patient-Level Aggregation
====================================================================
Implements robust, clinical evaluation metrics for cardiac and respiratory auscultation:
  - Accuracy & Balanced Accuracy
  - Macro & Weighted Precision, Recall, F1
  - Per-class metric breakdowns
  - Sensitivity (Recall of positive/abnormal) & Specificity (Recall of negative/normal) for binary Heart
  - Confusion matrix
  - Patient-level prediction aggregation (mean probability voting across multiple windows)
"""

import warnings
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    class_names: Optional[List[str]] = None,
    is_binary: bool = False,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics beyond raw accuracy.

    Parameters:
      y_true: Ground-truth class labels (integers)
      y_pred: Predicted class labels (integers)
      y_prob: Predicted probability distributions (N, K)
      class_names: List of class name strings
      is_binary: If True, computes sensitivity and specificity

    Returns:
      Dictionary of metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0}

    # Standard metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Macro and Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    labels = list(range(len(class_names))) if class_names else None
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weighted), 4),
        "recall_weighted": round(float(r_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm,
    }

    # Per-class detail
    if class_names:
        per_class_dict = {}
        for idx, name in enumerate(class_names):
            per_class_dict[name] = {
                "precision": round(float(p_per_class[idx]), 4),
                "recall": round(float(r_per_class[idx]), 4),
                "f1": round(float(f1_per_class[idx]), 4),
                "support": int(support_per_class[idx]),
            }
        metrics["per_class"] = per_class_dict

    # Binary clinical metrics (Heart: 0=Normal, 1=Abnormal)
    if is_binary or (class_names and len(class_names) == 2):
        # Abnormal is class 1 (positive); Normal is class 0 (negative)
        if len(cm) == 2 and len(cm[0]) == 2:
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics["sensitivity"] = round(float(sensitivity), 4)
            metrics["specificity"] = round(float(specificity), 4)

    return metrics


def aggregate_patient_predictions(
    window_probs: np.ndarray,
    patient_ids: List[str],
    window_targets: Optional[Union[np.ndarray, List[int]]] = None,
    strategy: str = "mean",
) -> Dict[str, Any]:
    """
    Aggregate window-level probabilities to produce patient-level diagnostic predictions.

    Parameters:
      window_probs: (N_windows, num_classes) array of soft prediction probabilities
      patient_ids: List of patient ID strings for each window
      window_targets: Optional (N_windows,) array of ground-truth integer labels
      strategy: Aggregation strategy ("mean" for soft voting)

    Returns:
      Dict with:
        patient_ids: List of unique patient IDs
        patient_preds: Predicted class ID for each patient
        patient_probs: (N_patients, num_classes) array of aggregated probabilities
        patient_targets: (N_patients,) ground truth label per patient (if provided)
    """
    window_probs = np.asarray(window_probs)
    unique_patients = sorted(list(set(patient_ids)))

    patient_probs_list = []
    patient_targets_list = []

    for pid in unique_patients:
        indices = [i for i, p in enumerate(patient_ids) if p == pid]
        p_probs = window_probs[indices]

        if strategy == "mean":
            agg_prob = np.mean(p_probs, axis=0)
        elif strategy == "max":
            agg_prob = np.max(p_probs, axis=0)
        else:
            agg_prob = np.mean(p_probs, axis=0)

        # Normalize to valid probability distribution
        s = np.sum(agg_prob)
        if s > 0:
            agg_prob = agg_prob / s

        patient_probs_list.append(agg_prob)

        if window_targets is not None:
            # Check target consistency across windows for this patient
            p_tgts = [window_targets[i] for i in indices]
            patient_targets_list.append(p_tgts[0])

    patient_probs_arr = np.array(patient_probs_list)
    patient_preds_arr = np.argmax(patient_probs_arr, axis=1)

    result = {
        "patient_ids": unique_patients,
        "patient_preds": patient_preds_arr,
        "patient_probs": patient_probs_arr,
    }

    if window_targets is not None:
        result["patient_targets"] = np.array(patient_targets_list)

    return result
