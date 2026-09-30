#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — TRAINING] | File: scripts/train_heart_baseline.py
# Purpose: Trains PhysioNet 2016 Heart Sound Classification Baseline
# ==============================================================================
"""
AI Stethoscope - Step 2: Heart CNN Baseline Training
===================================================
Trains ONLY the Heart CNN baseline (PhysioNet 2016):
  - Model: AudioCNNBaseline (2 classes: Normal, Abnormal)
  - Preprocessing: uniform (1, 128, 313) mel-spectrograms
  - Dynamic class weights from processed/config/class_weights.json: [0.65545, 2.10818]
  - Optimizer: AdamW, lr=3e-4, weight_decay=0.01
  - Scheduler: CosineAnnealingLR with 2 warmup epochs
  - Selection: Best checkpoint selected by validation Macro F1
  - Tracking: Accuracy, Balanced Accuracy, Macro F1, Sensitivity, Specificity
  - Target directory: experiments/heart_cnn/
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_baseline import build_heart_cnn
from src.datasets import get_dataloader
from src.trainer import StethoscopeTrainer
from src.utils import set_seed, load_training_config


def main():
    print("\n" + "=" * 95)
    print("  AI STETHOSCOPE - STEP 2: HEART CNN BASELINE TRAINING ONLY")
    print("  (PhysioNet 2016 Cardiac Sound Classification: Normal vs Abnormal)")
    print("=" * 95, flush=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} (PyTorch {torch.__version__})", flush=True)

    config_path = PROJECT_ROOT / "configs" / "training_config.yaml"
    weights_path = PROJECT_ROOT / "processed" / "config" / "class_weights.json"

    cfg = load_training_config(config_path)
    with open(weights_path, mode="r", encoding="utf-8") as f:
        weights_data = json.load(f)

    heart_weights = weights_data["tasks"]["heart"]["ordered_tensor_weights"]
    print(f"Loaded dynamic class weights for Heart: {heart_weights}", flush=True)

    exp_dir = PROJECT_ROOT / "experiments" / "heart_cnn"
    for sub in ["checkpoints", "logs", "metrics", "figures"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    # Build Heart Model
    model = build_heart_cnn(
        base_channels=32,
        num_blocks=4,
        dropout=0.3,
        classifier_hidden_dim=128,
    )
    p_info = model.count_parameters()
    print(f"Architecture: 4-Block AudioCNNBaseline | Total Params: {p_info['total']:,} ({model.get_model_size_mb():.2f} MB)", flush=True)

    # DataLoaders
    train_loader = get_dataloader(
        dataset_name="heart",
        split="train",
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=0,
        base_dir=PROJECT_ROOT,
    )
    val_loader = get_dataloader(
        dataset_name="heart",
        split="val",
        batch_size=cfg["data"]["val_batch_size"],
        shuffle=False,
        num_workers=0,
        base_dir=PROJECT_ROOT,
    )
    test_loader = get_dataloader(
        dataset_name="heart",
        split="test",
        batch_size=cfg["data"]["val_batch_size"],
        shuffle=False,
        num_workers=0,
        base_dir=PROJECT_ROOT,
    )

    print(f"DataLoaders initialized: Train={len(train_loader)} batches | Val={len(val_loader)} batches | Test={len(test_loader)} batches", flush=True)

    # Trainer
    trainer = StethoscopeTrainer(
        model=model,
        task_name="heart",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        class_weights=heart_weights,
        experiment_dir=exp_dir,
        device=device,
    )

    t_start = time.time()
    res = trainer.train()
    total_dur = time.time() - t_start

    print("\n" + "=" * 95)
    print(f"HEART CNN TRAINING AND TEST EVALUATION COMPLETED IN {total_dur/60.0:.2f} MINUTES ({total_dur:.1f}s)")
    print("=" * 95, flush=True)

    # Verification of generated artifacts
    best_ckpt_path = exp_dir / "checkpoints" / "best_model.pt"
    last_ckpt_path = exp_dir / "checkpoints" / "last_checkpoint.pt"
    test_metrics_path = exp_dir / "metrics" / "test_metrics.json"

    assert best_ckpt_path.exists(), f"Missing {best_ckpt_path}"
    assert last_ckpt_path.exists(), f"Missing {last_ckpt_path}"
    assert test_metrics_path.exists(), f"Missing {test_metrics_path}"

    # Verify best_model loads with strict=True
    verify_model = build_heart_cnn()
    ckpt_data = torch.load(best_ckpt_path, map_location="cpu")
    verify_model.load_state_dict(ckpt_data["model_state"], strict=True)
    print(f"[VERIFIED] best_model.pt (Epoch {ckpt_data['epoch']}) loaded with strict=True successfully!", flush=True)

    with open(test_metrics_path, mode="r", encoding="utf-8") as f:
        t_metrics = json.load(f)

    w_m = t_metrics["window_metrics"]
    print("\nHEART CNN HELD-OUT TEST RESULTS:")
    print(f"  * Test Loss:          {t_metrics['loss']:.4f}")
    print(f"  * Accuracy:           {w_m['accuracy']:.4f}")
    print(f"  * Balanced Accuracy:  {w_m['balanced_accuracy']:.4f}")
    print(f"  * Macro F1:           {w_m['f1_macro']:.4f}")
    print(f"  * Sensitivity:        {w_m.get('sensitivity', 0.0):.4f}")
    print(f"  * Specificity:        {w_m.get('specificity', 0.0):.4f}")
    print(f"  * Confusion Matrix:   {w_m['confusion_matrix']}", flush=True)


if __name__ == "__main__":
    main()
