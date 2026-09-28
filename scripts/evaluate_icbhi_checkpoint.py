#!/usr/bin/env python3
"""
AI Stethoscope - Step 1: ICBHI Checkpoint Evaluation
===================================================
Loads existing experiments/icbhi_cnn/checkpoints/best_model.pt,
evaluates on held-out test split (manifests/icbhi_test.csv),
and generates test metrics and confusion matrix figures.
Also plots the Epoch 2 validation confusion matrix stored in checkpoint.
"""

import os
import sys
import json
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_baseline import build_icbhi_cnn
from src.datasets import get_dataloader
from src.trainer import StethoscopeTrainer, plot_confusion_matrix
from src.utils import set_seed, load_training_config


def main():
    print("=" * 80)
    print("STEP 1: ICBHI BEST CHECKPOINT EVALUATION (NO RETRAINING)")
    print("=" * 80)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "experiments" / "icbhi_cnn" / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    best_epoch = checkpoint.get("epoch", "Unknown")
    val_metrics = checkpoint.get("val_metrics", {})
    class_weights = checkpoint.get("class_weights", None)
    config = checkpoint.get("config", None)

    print(f"Loaded checkpoint metadata: Best Epoch = {best_epoch}")
    print(f"Stored Validation Metrics (Epoch {best_epoch}):")
    print(f"  * Val Loss: {val_metrics.get('loss')}")
    print(f"  * Val Macro F1: {val_metrics.get('window_metrics', {}).get('f1_macro')}")
    print(f"  * Val Accuracy: {val_metrics.get('window_metrics', {}).get('accuracy')}")

    # Build model with exact parameters
    model = build_icbhi_cnn(
        base_channels=32,
        num_blocks=4,
        dropout=0.3,
        classifier_hidden_dim=128,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    print("Successfully loaded model state with strict=True!")

    # Load validation dataloader (dummy/unused for training, just for trainer init) and test dataloader
    val_loader = get_dataloader(
        dataset_name="icbhi",
        split="val",
        batch_size=32,
        shuffle=False,
        num_workers=0,
        base_dir=PROJECT_ROOT,
    )
    test_loader = get_dataloader(
        dataset_name="icbhi",
        split="test",
        batch_size=32,
        shuffle=False,
        num_workers=0,
        base_dir=PROJECT_ROOT,
    )

    exp_dir = PROJECT_ROOT / "experiments" / "icbhi_cnn"
    figures_dir = exp_dir / "figures"
    metrics_dir = exp_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Normal", "Crackle", "Wheeze", "Crackle+Wheeze"]

    # 1. Validation confusion matrix from stored checkpoint
    if "window_metrics" in val_metrics and "confusion_matrix" in val_metrics["window_metrics"]:
        stored_val_cm = val_metrics["window_metrics"]["confusion_matrix"]
        val_cm_path = figures_dir / "val_confusion_matrix.png"
        plot_confusion_matrix(
            cm=stored_val_cm,
            class_names=class_names,
            title=f"ICBHI CNN — Validation Confusion Matrix (Best Epoch {best_epoch})",
            save_path=val_cm_path,
        )
        print(f"Saved validation confusion matrix to: {val_cm_path}")
    else:
        print("Validation confusion matrix not found in checkpoint metadata.")

    # 2. Instantiate trainer for evaluation
    trainer = StethoscopeTrainer(
        model=model,
        task_name="icbhi",
        train_loader=val_loader,  # Not used for evaluate()
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        class_weights=class_weights,
        experiment_dir=exp_dir,
        device=device,
    )

    # 3. Evaluate held-out test split
    print("\nRunning evaluation on held-out test set (manifests/icbhi_test.csv)...")
    test_res = trainer.evaluate(test_loader, is_test=True)

    test_metrics_path = metrics_dir / "test_metrics.json"
    with open(test_metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(test_res, f, indent=2)
    print(f"Saved test metrics to: {test_metrics_path}")

    # Plot test confusion matrix
    test_cm_path = figures_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(
        cm=test_res["window_metrics"]["confusion_matrix"],
        class_names=class_names,
        title="ICBHI CNN — Test Confusion Matrix (Held-out Split)",
        save_path=test_cm_path,
    )
    print(f"Saved test confusion matrix to: {test_cm_path}")

    # Print summary of results
    t_win = test_res["window_metrics"]
    t_pat = test_res["patient_metrics"]
    print("\n" + "=" * 80)
    print("ICBHI TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Test Windows: {test_res['num_windows']} across {test_res['num_patients']} patients")
    print(f"Test Loss: {test_res['loss']:.4f}")
    print(f"Window Accuracy:          {t_win['accuracy']:.4f}")
    print(f"Window Balanced Accuracy: {t_win['balanced_accuracy']:.4f}")
    print(f"Window Macro Precision:   {t_win['precision_macro']:.4f}")
    print(f"Window Macro Recall:      {t_win['recall_macro']:.4f}")
    print(f"Window Macro F1:          {t_win['f1_macro']:.4f}")
    print(f"Window Weighted F1:       {t_win['f1_weighted']:.4f}")
    print("\nClass-wise Window Performance:")
    if "per_class" in t_win:
        for c_name, c_m in t_win["per_class"].items():
            print(f"  * {c_name:<16}: Precision={c_m['precision']:.4f} | Recall={c_m['recall']:.4f} | F1={c_m['f1']:.4f} (Support={c_m['support']})")
    print(f"\nPatient Accuracy:         {t_pat['accuracy']:.4f}")
    print(f"Patient Macro F1:         {t_pat['f1_macro']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
