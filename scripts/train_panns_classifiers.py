#!/usr/bin/env python3
"""
AI Stethoscope - PANNs CNN14 Frozen Embeddings Classifier Benchmark
===================================================================
Evaluates 5 classifier families on frozen 2048-dimensional PANNs CNN14
AudioSet embeddings across three auscultation tasks:
  1. Heart (PhysioNet 2016)
  2. ICBHI 2017 (Respiratory Cycles)
  3. Kaggle Respiratory Disease

Candidate Models:
  1. Logistic Regression (L-BFGS, balanced weights, max_iter=1000)
  2. Linear SVM (LinearSVC / SGDClassifier with margin sigmoid/softmax mapping)
  3. RBF SVM (Exact SVC for ICBHI/Kaggle; Nystroem + SGDClassifier for Heart)
  4. Random Forest (100 estimators, max_depth=15, balanced weights)
  5. Lightweight MLP (128 hidden units, ReLU, balanced sample weights, early stopping)

Strict Leakage Prevention Protocol:
  - StandardScaler fitted EXCLUSIVELY on Train split; transforms Val & Test.
  - Model selection based strictly on Validation Macro F1.
  - Held-out Test split evaluated ONCE per task on selected best model.
  - Patient/Entity level metrics computed via soft-voting probability aggregation.
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


class DecisionFunctionSoftmaxClassifier:
    """Wrapper that provides predict_proba via sigmoid/softmax on decision_function margins."""
    def __init__(self, base_estimator, is_binary: bool = False):
        self.base_estimator = base_estimator
        self.is_binary = is_binary
        self.classes_ = None

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = getattr(self.base_estimator, "classes_", np.unique(y))
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        df = self.base_estimator.decision_function(X)
        if self.is_binary:
            if df.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-np.clip(df, -30.0, 30.0)))
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
            else:
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)


class NystroemSVMClassifier:
    """Scalable RBF SVM via Nystroem kernel approximation + SGD hinge loss."""
    def __init__(self, n_components: int = 500, is_binary: bool = True, random_state: int = 42):
        self.n_components = n_components
        self.is_binary = is_binary
        self.random_state = random_state
        self.nystroem = Nystroem(kernel="rbf", n_components=n_components, random_state=random_state)
        self.clf = SGDClassifier(loss="hinge", class_weight="balanced", max_iter=2000, random_state=random_state)
        self.classes_ = None

    def fit(self, X, y):
        X_trans = self.nystroem.fit_transform(X)
        self.clf.fit(X_trans, y)
        self.classes_ = getattr(self.clf, "classes_", np.unique(y))
        return self

    def predict(self, X):
        X_trans = self.nystroem.transform(X)
        return self.clf.predict(X_trans)

    def predict_proba(self, X):
        X_trans = self.nystroem.transform(X)
        df = self.clf.decision_function(X_trans)
        if self.is_binary:
            if df.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-np.clip(df, -30.0, 30.0)))
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
            else:
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)


class WeightedMLPClassifier:
    """Wrapper around MLPClassifier that computes balanced sample weights on fit."""
    def __init__(self, hidden_layer_sizes=(128,), max_iter=100, alpha=1e-4, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=random_state,
        )
        self.classes_ = None

    def fit(self, X, y):
        sample_weight = compute_sample_weight("balanced", y)
        self.mlp.fit(X, y, sample_weight=sample_weight)
        self.classes_ = getattr(self.mlp, "classes_", np.unique(y))
        return self

    def predict(self, X):
        return self.mlp.predict(X)

    def predict_proba(self, X):
        return self.mlp.predict_proba(X)


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
    plt.title(title, fontsize=11, fontweight="bold", pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
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
    """Train and evaluate 5 candidate classifiers on frozen PANNs embeddings."""
    print("\n" + "=" * 80)
    print(f"  STARTING PANNS CLASSIFIER BENCHMARK: {dataset_name.upper()}")
    print("=" * 80)

    config = TASK_CONFIG[dataset_name]
    class_names = config["classes"]
    is_binary = config["is_binary"]

    emb_dir = base_dir / "processed" / "panns_embeddings" / dataset_name
    train_data = np.load(emb_dir / "train.npz", allow_pickle=True)
    val_data = np.load(emb_dir / "val.npz", allow_pickle=True)
    test_data = np.load(emb_dir / "test.npz", allow_pickle=True)

    X_train_raw = train_data["embeddings"] if "embeddings" in train_data else train_data["X"]
    y_train = train_data["y"]
    pat_train = train_data["patient_ids"]

    X_val_raw = val_data["embeddings"] if "embeddings" in val_data else val_data["X"]
    y_val = val_data["y"]
    pat_val = val_data["patient_ids"]

    X_test_raw = test_data["embeddings"] if "embeddings" in test_data else test_data["X"]
    y_test = test_data["y"]
    pat_test = test_data["patient_ids"]

    print(f"Loaded frozen PANNs embeddings splits:")
    print(f"  * Train: {X_train_raw.shape[0]} samples, {len(np.unique(pat_train))} patients, dim={X_train_raw.shape[1]}")
    print(f"  * Val:   {X_val_raw.shape[0]} samples, {len(np.unique(pat_val))} patients, dim={X_val_raw.shape[1]}")
    print(f"  * Test:  {X_test_raw.shape[0]} samples, {len(np.unique(pat_test))} patients, dim={X_test_raw.shape[1]}")

    # Patient isolation check
    train_patients_set = set(np.unique(pat_train))
    val_patients_set = set(np.unique(pat_val))
    test_patients_set = set(np.unique(pat_test))
    assert len(train_patients_set.intersection(val_patients_set)) == 0, "Patient leakage between train and val!"
    assert len(train_patients_set.intersection(test_patients_set)) == 0, "Patient leakage between train and test!"
    assert len(val_patients_set.intersection(test_patients_set)) == 0, "Patient leakage between val and test!"
    print("  [Patient Isolation] Verified zero overlap across Train, Val, and Test patient sets.")

    # 1. Scaler fitting on Train ONLY
    print("\n[Step 1] Fitting StandardScaler on Train split only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 2. Define 5 candidate models
    models_to_test = {}

    # Model 1: Logistic Regression
    models_to_test["Logistic Regression"] = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"),
        X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
    )

    # Model 2: Linear SVM
    if dataset_name == "heart":
        models_to_test["Linear SVM"] = (
            DecisionFunctionSoftmaxClassifier(
                SGDClassifier(loss="hinge", class_weight="balanced", max_iter=2000, random_state=42),
                is_binary=True
            ),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )
    else:
        models_to_test["Linear SVM"] = (
            DecisionFunctionSoftmaxClassifier(
                LinearSVC(dual=False, C=0.01, class_weight="balanced", random_state=42),
                is_binary=is_binary
            ),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )

    # Model 3: RBF SVM
    if dataset_name == "heart":
        models_to_test["RBF SVM"] = (
            NystroemSVMClassifier(n_components=500, is_binary=True, random_state=42),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )
    else:
        models_to_test["RBF SVM"] = (
            DecisionFunctionSoftmaxClassifier(
                SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42),
                is_binary=is_binary
            ),
            X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
        )

    # Model 4: Random Forest
    models_to_test["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        X_train_raw, X_val_raw, X_test_raw, "raw"
    )

    # Model 5: Lightweight MLP
    models_to_test["Lightweight MLP"] = (
        WeightedMLPClassifier(hidden_layer_sizes=(128,), max_iter=100, alpha=1e-4, random_state=42),
        X_train_scaled, X_val_scaled, X_test_scaled, "scaled"
    )

    val_benchmark_results = []
    trained_clfs = {}

    print("\n[Step 2] Training and Validating candidate models...")
    print("-" * 85)
    print(f"{'Model Name':22s} | {'Val Acc':8s} | {'Val BalAcc':10s} | {'Val Macro F1':12s} | {'Val W-F1':8s} | {'Train Time':10s}")
    print("-" * 85)

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
        trained_clfs[mname] = (clf, X_te, rep)

        print(
            f"{mname:22s} | {val_metrics['accuracy']:8.4f} | {val_metrics['balanced_accuracy']:10.4f} | "
            f"{val_metrics['f1_macro']:12.4f} | {val_metrics['f1_weighted']:8.4f} | {fit_time:8.2f}s"
        )

    # 3. Model Selection: Select model with highest Validation Macro F1
    val_benchmark_results.sort(key=lambda x: x["window_metrics"]["f1_macro"], reverse=True)
    best_entry = val_benchmark_results[0]
    best_model_name = best_entry["model_name"]
    best_clf, X_test_use, best_rep = trained_clfs[best_model_name]
    best_val_f1 = best_entry["window_metrics"]["f1_macro"]

    print("\n" + "=" * 85)
    print(f"  MODEL SELECTION WINNER FOR {dataset_name.upper()}: {best_model_name}")
    print(f"  Validation Window Macro F1: {best_val_f1:.4f}")
    print("=" * 85)

    # 4. Held-out Test Evaluation: Evaluate the selected winner ONCE on Held-out Test
    print("\n[Step 3] Evaluating SELECTED model ONCE on Held-Out Test split...")
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
    print(f"  * Patient-Level Balanced Acc: {test_patient_metrics['balanced_accuracy']:.4f}")
    print(f"  * Patient-Level Macro F1:   {test_patient_metrics['f1_macro']:.4f}")
    print(f"  * Patient-Level Weighted F1:{test_patient_metrics['f1_weighted']:.4f}")
    if is_binary:
        print(f"  * Window Sensitivity:       {test_window_metrics.get('sensitivity', 0.0):.4f}")
        print(f"  * Window Specificity:       {test_window_metrics.get('specificity', 0.0):.4f}")
        print(f"  * Patient Sensitivity:      {test_patient_metrics.get('sensitivity', 0.0):.4f}")
        print(f"  * Patient Specificity:      {test_patient_metrics.get('specificity', 0.0):.4f}")

    # 5. Save artifacts in experiments/panns_classifiers/{dataset}/
    exp_dir = base_dir / "experiments" / "panns_classifiers" / dataset_name
    models_dir = exp_dir / "models"
    metrics_dir = exp_dir / "metrics"
    plots_dir = exp_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save model and scaler
    joblib.dump(best_clf, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    # Feature configuration
    feature_config = {
        "dataset": dataset_name,
        "embedding_type": "panns_cnn14",
        "embedding_dim": int(X_train_raw.shape[1]),
        "selected_model": best_model_name,
        "selected_representation": best_rep,
        "num_train_samples": int(X_train_raw.shape[0]),
        "num_val_samples": int(X_val_raw.shape[0]),
        "num_test_samples": int(X_test_raw.shape[0]),
        "num_train_patients": int(len(train_patients_set)),
        "num_val_patients": int(len(val_patients_set)),
        "num_test_patients": int(len(test_patients_set)),
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
    }
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_output, f, indent=2)

    # Plot confusion matrices
    plot_and_save_cm(
        best_entry["window_metrics"]["confusion_matrix"],
        class_names,
        f"{dataset_name.upper()} PANNs ({best_model_name}) — Val Confusion Matrix",
        plots_dir / "val_confusion_matrix.png",
    )
    plot_and_save_cm(
        test_window_metrics["confusion_matrix"],
        class_names,
        f"{dataset_name.upper()} PANNs ({best_model_name}) — Test Confusion Matrix",
        plots_dir / "confusion_matrix.png",
    )
    plot_and_save_cm(
        test_patient_metrics["confusion_matrix"],
        class_names,
        f"{dataset_name.upper()} PANNs ({best_model_name}) — Patient Test Confusion Matrix",
        plots_dir / "patient_confusion_matrix.png",
    )

    print(f"\n[ARTIFACTS SAVED] Experiment directory: {exp_dir}")
    return final_test_output


def main():
    parser = argparse.ArgumentParser(description="Run PANNs classifier benchmarks")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to run (heart, icbhi, kaggle)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["kaggle", "icbhi", "heart"]

    summary = {}
    for d in datasets:
        res = train_and_evaluate_task(d)
        summary[d] = res

    print("\n" + "=" * 80)
    print("ALL PANNS CLASSIFIER BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
