#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — TRAINING] | File: scripts/train_classical_ml.py
# Purpose: Trains Classical ML Baselines (Random Forest, SVM, MLP, XGBoost)
# ==============================================================================
"""
AI Stethoscope - Classical Machine Learning Benchmark Runner
============================================================
Trains and evaluates 4 classical ML models across Heart, ICBHI, and Kaggle:
  1. Logistic Regression
  2. SVM with RBF Kernel (on PCA features for tractability)
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  * XGBoost: Documented as unavailable in current Python environment.

Strict Leakage Prevention Protocol:
  - Scaler and PCA fitted EXCLUSIVELY on Train split.
  - Applied to Validation and Test splits.
  - Model selection based strictly on Validation Macro F1.
  - Held-out Test split evaluated ONCE per task on selected best model.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_CONFIG = {
    "heart": {
        "classes": ["Normal", "Abnormal"],
        "is_binary": True,
        "task_name": "heart",
    },
    "icbhi": {
        "classes": ["Normal", "Crackle", "Wheeze", "Crackle+Wheeze"],
        "is_binary": False,
        "task_name": "lung_acoustic",
    },
    "kaggle": {
        "classes": ["Normal", "Asthma", "Other"],
        "is_binary": False,
        "task_name": "lung_disease",
    },
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    is_binary: bool = False,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Per-class metrics
    per_class = {}
    for i, cname in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        supp = int(cm[i, :].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cname] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": supp,
        }

    res = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_weighted": round(r_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    if is_binary and len(class_names) == 2:
        # Abnormal = class 1, Normal = class 0
        sensitivity = per_class[class_names[1]]["recall"]
        specificity = per_class[class_names[0]]["recall"]
        res["sensitivity"] = sensitivity
        res["specificity"] = specificity

    return res


def compute_patient_metrics(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    is_binary: bool = False,
) -> Dict[str, Any]:
    """Aggregate window probabilities per patient and evaluate patient metrics."""
    unique_patients = np.unique(patient_ids)
    pat_y_true = []
    pat_y_pred = []

    for pid in unique_patients:
        idx = np.where(patient_ids == pid)[0]
        mean_prob = np.mean(probs[idx], axis=0)
        pred_class = int(np.argmax(mean_prob))
        # True class is consistent across windows of the same patient
        true_class = int(y_true[idx[0]])
        pat_y_true.append(true_class)
        pat_y_pred.append(pred_class)

    return compute_metrics(
        np.array(pat_y_true),
        np.array(pat_y_pred),
        class_names=class_names,
        is_binary=is_binary,
    )


def plot_and_save_cm(
    cm: List[List[int]],
    class_names: List[str],
    title: str,
    save_path: Path,
):
    """Render and save confusion matrix figure."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.ylabel("Actual True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_and_evaluate_task(dataset_name: str, base_dir: Path = PROJECT_ROOT):
    """Train and evaluate all classical ML models on a specific dataset."""
    print("\n" + "=" * 80)
    print(f"  STARTING CLASSICAL ML BENCHMARK: {dataset_name.upper()}")
    print("=" * 80)

    config = TASK_CONFIG[dataset_name]
    class_names = config["classes"]
    is_binary = config["is_binary"]
    n_classes = len(class_names)

    feat_dir = base_dir / "processed" / "classical_features"
    train_data = np.load(feat_dir / f"{dataset_name}_train.npz", allow_pickle=True)
    val_data = np.load(feat_dir / f"{dataset_name}_val.npz", allow_pickle=True)
    test_data = np.load(feat_dir / f"{dataset_name}_test.npz", allow_pickle=True)

    X_train_raw, y_train = train_data["X"], train_data["y"]
    X_val_raw, y_val = val_data["X"], val_data["y"]
    X_test_raw, y_test = test_data["X"], test_data["y"]

    pat_train = train_data["patient_ids"]
    pat_val = val_data["patient_ids"]
    pat_test = test_data["patient_ids"]

    print(f"Loaded feature splits:")
    print(f"  * Train: {X_train_raw.shape[0]} samples, {len(np.unique(pat_train))} patients")
    print(f"  * Val:   {X_val_raw.shape[0]} samples, {len(np.unique(pat_val))} patients")
    print(f"  * Test:  {X_test_raw.shape[0]} samples, {len(np.unique(pat_test))} patients")

    # 1. Scaler fitting on Train ONLY
    print("\n[Step 1] Fitting StandardScaler on Train split only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 2. PCA fitting on Train ONLY (95% variance)
    print("[Step 2] Fitting PCA (95% variance) on Train split only...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    n_pca = X_train_pca.shape[1]
    print(f"  * PCA reduced {X_train_raw.shape[1]} -> {n_pca} components (explained var: {pca.explained_variance_ratio_.sum():.4f})")

    # 3. Define candidate model suites
    # We evaluate models on both Scaled and PCA features to find the optimal representation
    models_to_test = {
        "Logistic Regression (Scaled)": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        ),
        "Logistic Regression (PCA)": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"),
            X_train_pca, X_val_pca, X_test_pca, "pca"
        ),
        "SVM RBF (PCA)": (
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
            X_train_pca, X_val_pca, X_test_pca, "pca"
        ),
        "Random Forest (Scaled)": (
            RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        ),
        "Random Forest (PCA)": (
            RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1),
            X_train_pca, X_val_pca, X_test_pca, "pca"
        ),
        "KNN (PCA)": (
            KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
            X_train_pca, X_val_pca, X_test_pca, "pca"
        ),
    }

    # For small datasets like Kaggle/ICBHI, we also evaluate SVM on full scaled features
    if dataset_name in ["kaggle", "icbhi"]:
        models_to_test["SVM RBF (Scaled)"] = (
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )
        models_to_test["KNN (Scaled)"] = (
            KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )

    val_benchmark_results = []
    trained_clfs = {}

    print("\n[Step 3] Training and Validating candidate models...")
    print("-" * 80)
    print(f"{'Model Name':32s} | {'Val Acc':8s} | {'Val BalAcc':10s} | {'Val Macro F1':12s} | {'Train Time':10s}")
    print("-" * 80)

    for mname, (clf, X_tr, X_v, X_te, rep) in models_to_test.items():
        t0 = time.time()
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0

        val_preds = clf.predict(X_v)
        val_probs = clf.predict_proba(X_v)

        val_metrics = compute_metrics(y_val, val_preds, class_names=class_names, is_binary=is_binary)
        val_pat_metrics = compute_patient_metrics(pat_val, y_val, val_probs, class_names=class_names, is_binary=is_binary)

        entry = {
            "model_name": mname,
            "feature_representation": rep,
            "training_time_sec": round(fit_time, 2),
            "window_metrics": val_metrics,
            "patient_metrics": val_pat_metrics,
        }
        val_benchmark_results.append(entry)
        trained_clfs[mname] = clf

        print(
            f"{mname:32s} | {val_metrics['accuracy']:8.4f} | {val_metrics['balanced_accuracy']:10.4f} | "
            f"{val_metrics['f1_macro']:12.4f} | {fit_time:8.2f}s"
        )

    # 4. Model Selection: Select model with highest Validation Macro F1
    val_benchmark_results.sort(key=lambda x: x["window_metrics"]["f1_macro"], reverse=True)
    best_entry = val_benchmark_results[0]
    best_model_name = best_entry["model_name"]
    best_clf = trained_clfs[best_model_name]
    best_rep = best_entry["feature_representation"]
    best_val_f1 = best_entry["window_metrics"]["f1_macro"]

    print("\n" + "=" * 80)
    print(f"  MODEL SELECTION WINNER: {best_model_name}")
    print(f"  Validation Window Macro F1: {best_val_f1:.4f}")
    print("=" * 80)

    # 5. Held-out Test Evaluation: Evaluate the selected model ONCE on the test set
    print("\n[Step 4] Evaluating SELECTED model ONCE on Held-Out Test split...")
    X_test_use = X_test_pca if best_rep == "pca" else X_test_scaled

    t_test_start = time.time()
    test_preds = best_clf.predict(X_test_use)
    test_probs = best_clf.predict_proba(X_test_use)
    test_dur = time.time() - t_test_start

    test_window_metrics = compute_metrics(y_test, test_preds, class_names=class_names, is_binary=is_binary)
    test_patient_metrics = compute_patient_metrics(pat_test, y_test, test_probs, class_names=class_names, is_binary=is_binary)

    print("\nHELD-OUT TEST RESULTS:")
    print(f"  * Window Accuracy:          {test_window_metrics['accuracy']:.4f}")
    print(f"  * Window Balanced Accuracy: {test_window_metrics['balanced_accuracy']:.4f}")
    print(f"  * Window Macro F1:          {test_window_metrics['f1_macro']:.4f}")
    print(f"  * Window Weighted F1:       {test_window_metrics['f1_weighted']:.4f}")
    print(f"  * Patient-Level Accuracy:   {test_patient_metrics['accuracy']:.4f}")
    print(f"  * Patient-Level Macro F1:   {test_patient_metrics['f1_macro']:.4f}")
    if is_binary:
        print(f"  * Window Sensitivity:       {test_window_metrics.get('sensitivity', 0.0):.4f}")
        print(f"  * Window Specificity:       {test_window_metrics.get('specificity', 0.0):.4f}")
        print(f"  * Patient Sensitivity:      {test_patient_metrics.get('sensitivity', 0.0):.4f}")
        print(f"  * Patient Specificity:      {test_patient_metrics.get('specificity', 0.0):.4f}")

    # 6. Save artifacts
    exp_dir = base_dir / "experiments" / "classical_ml" / dataset_name
    models_dir = exp_dir / "models"
    metrics_dir = exp_dir / "metrics"
    figures_dir = exp_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save model and transformers
    joblib.dump(best_clf, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(pca, models_dir / "pca.joblib")

    # Feature configuration
    feature_config = {
        "dataset": dataset_name,
        "raw_feature_dim": X_train_raw.shape[1],
        "pca_dim": n_pca,
        "pca_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "selected_model": best_model_name,
        "selected_representation": best_rep,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(models_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    # Save validation benchmark comparison
    with open(metrics_dir / "val_comparison.json", "w", encoding="utf-8") as f:
        json.dump(val_benchmark_results, f, indent=2)

    # Save final test metrics
    final_test_output = {
        "dataset": dataset_name,
        "selected_model": best_model_name,
        "feature_representation": best_rep,
        "test_duration_sec": round(test_dur, 3),
        "window_metrics": test_window_metrics,
        "patient_metrics": test_patient_metrics,
        "num_windows": len(y_test),
        "num_patients": len(np.unique(pat_test)),
        "xgboost_available": False,
    }
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_output, f, indent=2)

    # Plot confusion matrices
    plot_and_save_cm(
        best_entry["window_metrics"]["confusion_matrix"],
        class_names,
        f"{dataset_name.upper()} Classical ML ({best_model_name}) — Val Confusion Matrix",
        figures_dir / "val_confusion_matrix.png",
    )
    plot_and_save_cm(
        test_window_metrics["confusion_matrix"],
        class_names,
        f"{dataset_name.upper()} Classical ML ({best_model_name}) — Test Confusion Matrix",
        figures_dir / "test_confusion_matrix.png",
    )

    print(f"\n[ARTIFACTS SAVED] Experiment directory: {exp_dir}")
    return final_test_output


def main():
    parser = argparse.ArgumentParser(description="Run classical ML baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to run (heart, icbhi, kaggle)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["kaggle", "icbhi", "heart"]

    summary = {}
    for d in datasets:
        res = train_and_evaluate_task(d)
        summary[d] = res

    print("\n" + "=" * 80)
    print("ALL CLASSICAL ML BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
