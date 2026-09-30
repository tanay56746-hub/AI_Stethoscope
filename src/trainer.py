# ==============================================================================
# [LAYER: ML TRAINER ENGINE] | File: src/trainer.py
# Purpose: Multi-Task & Single-Task PyTorch Training Loop, Checkpointing & Early Stopping
# ==============================================================================
"""
AI Stethoscope - Model Training & Evaluation Engine
===================================================
Implements the training loop, validation loop, weighted loss computation,
warmup + cosine-annealing scheduling, checkpoint management, metric logging,
figure generation, and patient-level evaluation.

Configurable via configs/training_config.yaml.
Supports:
  - Task-specific weighted Cross-Entropy
  - Temporal Masked Pooling for ICBHI respiratory cycles
  - Patient-level metric aggregation for Kaggle disease classification
  - Best validation checkpoint selection & evaluation on held-out test split
  - Figure generation (loss curves, F1 curves, confusion matrices)
"""

import os
import time
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import compute_classification_metrics, aggregate_patient_predictions
from src.utils import set_seed, load_training_config


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], title: str, save_path: Path):
    """Generate and save publication-quality confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, Any]], task_name: str, out_dir: Path):
    """Plot Loss and Macro F1 curves over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h.get("train_f1_macro", 0.0) for h in history]
    val_f1 = [h.get("val_f1_macro", 0.0) for h in history]

    # Best epoch based on val_f1_macro
    best_f1_idx = int(np.argmax(val_f1))
    best_epoch = epochs[best_f1_idx]

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
    ax.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.0, marker="s", markersize=4)
    ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch})")
    ax.set_title(f"{task_name.upper()} CNN — Training vs Validation Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Weighted Cross-Entropy Loss", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)

    # 2. Macro F1 Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(epochs, train_f1, label="Train Macro F1", color="#2ca02c", linewidth=2.0, marker="o", markersize=4)
    ax.plot(epochs, val_f1, label="Validation Macro F1", color="#d62728", linewidth=2.0, marker="s", markersize=4)
    ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch})")
    ax.set_title(f"{task_name.upper()} CNN — Training vs Validation Macro F1", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Macro F1 Score", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_curve.png")
    plt.close(fig)


class StethoscopeTrainer:
    """
    Standardized trainer for single-task or multi-task CNN baselines.
    """

    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        class_weights: Optional[List[float]] = None,
        experiment_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.task_name = task_name.lower().strip()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Load or use provided config
        if config is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml"
            self.config = load_training_config(config_path)
        else:
            self.config = config

        # Set seed
        self.seed = self.config.get("experiment", {}).get("random_seed", 42)
        set_seed(self.seed)

        # Device selection
        if device is None:
            dev_cfg = self.config.get("hardware", {}).get("device", "auto")
            if dev_cfg == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif dev_cfg == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Setup loss function with class weights
        self.class_weights = class_weights
        if class_weights is not None:
            w_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=w_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        opt_cfg = self.config.get("optimization", {})
        self.lr = float(opt_cfg.get("learning_rate", 3e-4))
        self.wd = float(opt_cfg.get("weight_decay", 1e-2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        # Learning Rate Scheduler with Warmup + Cosine Annealing
        self.epochs = int(self.config.get("training", {}).get("epochs", 15))
        self.warmup_epochs = int(opt_cfg.get("scheduler", {}).get("warmup_epochs", 2))
        self.eta_min = float(opt_cfg.get("scheduler", {}).get("eta_min", 1e-6))
        self.patience = int(self.config.get("training", {}).get("early_stopping_patience", 5))

        # Mixed precision
        use_amp = (
            self.device.type == "cuda"
            and self.config.get("hardware", {}).get("mixed_precision", "no") in ("fp16", "bf16")
        )
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Experiment directories
        if experiment_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            self.exp_dir = base_dir / "experiments" / f"{self.task_name}_cnn"
        else:
            self.exp_dir = Path(experiment_dir)

        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.metrics_dir = self.exp_dir / "metrics"
        self.figures_dir = self.exp_dir / "figures"

        for d in [self.ckpt_dir, self.logs_dir, self.metrics_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Class names for reporting
        self.class_names = self._get_class_names()
        self.is_binary = (len(self.class_names) == 2)

    def _get_class_names(self) -> List[str]:
        task_info = self.config.get("tasks", {}).get(self.task_name, {})
        return task_info.get("classes", [])

    def _get_current_lr(self, epoch: int) -> float:
        """Calculate learning rate with linear warmup and cosine decay."""
        if epoch < self.warmup_epochs:
            # Linear warmup from lr*0.1 to lr
            factor = (epoch + 1) / float(max(1, self.warmup_epochs))
            return max(self.eta_min, self.lr * factor)
        else:
            # Cosine decay from lr to eta_min
            progress = (epoch - self.warmup_epochs) / float(max(1, self.epochs - self.warmup_epochs))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.eta_min + (self.lr - self.eta_min) * float(cosine_decay)

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Execute one training epoch."""
        self.model.train()
        current_lr = self._get_current_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        running_loss = 0.0
        all_preds = []
        all_targets = []

        total_batches = len(self.train_loader)
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            tensors = batch["tensor"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            targets = batch["label_id"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(tensors, mask=masks)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tensors, mask=masks)
                loss = self.criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

            if (step + 1) % 50 == 0 or (step + 1) == total_batches:
                elapsed = time.time() - t0
                pct = (step + 1) / total_batches * 100.0
                print(f"    [Epoch {epoch} Step {step + 1:>3}/{total_batches} ({pct:>5.1f}%)] Running Loss: {running_loss / (step + 1):.4f} ({elapsed:.1f}s)", flush=True)

        dur = time.time() - t0
        avg_loss = running_loss / max(1, total_batches)

        metrics = compute_classification_metrics(
            y_true=all_targets,
            y_pred=all_preds,
            class_names=self.class_names,
            is_binary=self.is_binary,
        )

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "lr": current_lr,
            "epoch_duration": dur,
        }

    def evaluate(self, loader: DataLoader, is_test: bool = False) -> Dict[str, Any]:
        """Run complete evaluation over validation or test split."""
        self.model.eval()
        running_loss = 0.0
        all_logits = []
        all_probs = []
        all_targets = []
        all_pids = []
        all_rids = []

        with torch.no_grad():
            for batch in loader:
                tensors = batch["tensor"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                targets = batch["label_id"].to(self.device, non_blocking=True)

                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(tensors, mask=masks)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(tensors, mask=masks)
                    loss = self.criterion(logits, targets)

                running_loss += loss.item()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs)
                all_targets.extend(targets.cpu().numpy())
                all_pids.extend(batch["patient_id"])
                all_rids.extend(batch["recording_id"])

        avg_loss = running_loss / max(1, len(loader))
        all_probs_arr = np.concatenate(all_probs, axis=0)
        window_preds = np.argmax(all_probs_arr, axis=1)

        # Window-level metrics
        window_metrics = compute_classification_metrics(
            y_true=all_targets,
            y_pred=window_preds,
            class_names=self.class_names,
            is_binary=self.is_binary,
        )

        result = {
            "loss": round(avg_loss, 4),
            "window_metrics": window_metrics,
        }

        # Patient-level aggregation (especially critical for Kaggle and Heart)
        patient_agg = aggregate_patient_predictions(
            window_probs=all_probs_arr,
            patient_ids=all_pids,
            window_targets=all_targets,
            strategy="mean",
        )

        patient_metrics = compute_classification_metrics(
            y_true=patient_agg["patient_targets"],
            y_pred=patient_agg["patient_preds"],
            class_names=self.class_names,
            is_binary=self.is_binary,
        )

        result["patient_metrics"] = patient_metrics
        result["num_windows"] = len(all_targets)
        result["num_patients"] = len(patient_agg["patient_ids"])

        return result

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint with full reproducibility metadata."""
        state = {
            "epoch": epoch,
            "task_name": self.task_name,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": self.config,
            "class_weights": self.class_weights,
            "seed": self.seed,
        }

        # Save last checkpoint
        last_ckpt_path = self.ckpt_dir / "last_checkpoint.pt"
        torch.save(state, last_ckpt_path)

        # Save best checkpoint
        if is_best:
            best_ckpt_path = self.ckpt_dir / "best_model.pt"
            torch.save(state, best_ckpt_path)

    def train(self) -> Dict[str, Any]:
        """Full training loop with early stopping, checkpointing, and evaluation."""
        print(f"\n" + "=" * 80)
        print(f"  STARTING FULL TRAINING: {self.task_name.upper()} CNN BASELINE")
        print(f"  (Max Epochs: {self.epochs}, Patience: {self.patience}, Device: {self.device})")
        print("=" * 80)

        history: List[Dict[str, Any]] = []
        best_val_f1 = -1.0
        best_epoch = -1
        patience_counter = 0

        t_start_total = time.time()

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.time()

            # Train one epoch
            train_res = self.train_epoch(epoch)

            # Evaluate on validation set
            val_res = self.evaluate(self.val_loader)
            v_metrics = val_res["window_metrics"]
            val_loss = val_res["loss"]
            val_f1 = v_metrics["f1_macro"]
            val_acc = v_metrics["accuracy"]
            val_bal_acc = v_metrics["balanced_accuracy"]

            # For Kaggle, check patient F1
            pat_f1 = val_res["patient_metrics"]["f1_macro"]

            epoch_dur = time.time() - t_epoch_start

            # Log record
            record = {
                "epoch": epoch,
                "train_loss": round(train_res["loss"], 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_res["accuracy"], 4),
                "val_acc": round(val_acc, 4),
                "train_bal_acc": round(train_res["balanced_accuracy"], 4),
                "val_bal_acc": round(val_bal_acc, 4),
                "train_f1_macro": round(train_res["f1_macro"], 4),
                "val_f1_macro": round(val_f1, 4),
                "val_patient_f1_macro": round(pat_f1, 4),
                "learning_rate": round(train_res["lr"], 7),
                "epoch_duration_sec": round(epoch_dur, 2),
            }
            history.append(record)

            # Print concise progress line
            print(
                f"Epoch [{epoch:>2}/{self.epochs}] "
                f"Train Loss: {train_res['loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} (BalAcc: {val_bal_acc:.4f}) | "
                f"Pat F1: {pat_f1:.4f} | "
                f"LR: {train_res['lr']:.1e} ({epoch_dur:.1f}s)",
                flush=True
            )

            # Checkpoint selection based on validation Macro F1
            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, val_res, is_best=True)
                print(f"   --> [*] Best model updated! Val Macro F1: {val_f1:.4f} (Saved to best_model.pt)", flush=True)
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, val_res, is_best=False)

            # Early stopping check
            if patience_counter >= self.patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs).")
                break

        total_train_time = time.time() - t_start_total

        # Save history log files
        with open(self.logs_dir / "history.json", mode="w", encoding="utf-8") as jf:
            json.dump(history, jf, indent=2)

        with open(self.logs_dir / "history.csv", mode="w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        # Generate training curves
        plot_training_curves(history, self.task_name, self.figures_dir)

        # ---------------------------------------------------------
        # FINAL EVALUATION ON HELD-OUT TEST SET (BEST MODEL ONLY)
        # ---------------------------------------------------------
        print(f"\n" + "-" * 80)
        print(f"  EVALUATING BEST CHECKPOINT (Epoch {best_epoch}) ON HELD-OUT TEST SPLIT")
        print("-" * 80)

        best_ckpt = torch.load(self.ckpt_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(best_ckpt["model_state"])

        # Final Validation evaluation for confusion matrix
        final_val_res = self.evaluate(self.val_loader)
        plot_confusion_matrix(
            cm=final_val_res["window_metrics"]["confusion_matrix"],
            class_names=self.class_names,
            title=f"{self.task_name.upper()} CNN — Validation Confusion Matrix (Best Epoch {best_epoch})",
            save_path=self.figures_dir / "val_confusion_matrix.png",
        )

        # Final Test evaluation
        test_res = None
        if self.test_loader is not None:
            test_res = self.evaluate(self.test_loader, is_test=True)
            plot_confusion_matrix(
                cm=test_res["window_metrics"]["confusion_matrix"],
                class_names=self.class_names,
                title=f"{self.task_name.upper()} CNN — Test Confusion Matrix (Held-out Split)",
                save_path=self.figures_dir / "test_confusion_matrix.png",
            )

            # Save test metrics
            with open(self.metrics_dir / "test_metrics.json", mode="w", encoding="utf-8") as tf:
                json.dump(test_res, tf, indent=2)

            t_win = test_res["window_metrics"]
            t_pat = test_res["patient_metrics"]
            print(f"  * Held-Out Test Window Accuracy: {t_win['accuracy']:.4f}")
            print(f"  * Held-Out Test Window BalAcc  : {t_win['balanced_accuracy']:.4f}")
            print(f"  * Held-Out Test Window Macro F1: {t_win['f1_macro']:.4f}")
            print(f"  * Held-Out Test Patient F1     : {t_pat['f1_macro']:.4f} (across {test_res['num_patients']} patients)")
            if self.is_binary:
                print(f"  * Held-Out Test Sensitivity    : {t_win.get('sensitivity', 0.0):.4f}")
                print(f"  * Held-Out Test Specificity    : {t_win.get('specificity', 0.0):.4f}")

        return {
            "task_name": self.task_name,
            "device": str(self.device),
            "total_duration_sec": round(total_train_time, 2),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": round(final_val_res["loss"], 4),
            "best_val_window_metrics": final_val_res["window_metrics"],
            "best_val_patient_metrics": final_val_res["patient_metrics"],
            "test_metrics": test_res,
            "history": history,
            "best_ckpt_path": str((self.ckpt_dir / "best_model.pt").relative_to(self.exp_dir.parent.parent)),
            "last_ckpt_path": str((self.ckpt_dir / "last_checkpoint.pt").relative_to(self.exp_dir.parent.parent)),
        }

    def run_smoke_test(self) -> Dict[str, Any]:
        """Execute a complete single-step smoke test without running full training."""
        self.model.train()
        train_iter = iter(self.train_loader)
        batch = next(train_iter)

        tensors = batch["tensor"].to(self.device)
        masks = batch["mask"].to(self.device)
        targets = batch["label_id"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(tensors, mask=masks)
        loss = self.criterion(logits, targets)
        loss_val = float(loss.item())
        loss.backward()

        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(float(param.grad.norm().item()))
        total_grad_norm = float(np.sqrt(sum(g ** 2 for g in grad_norms)))

        self.optimizer.step()

        self.model.eval()
        val_iter = iter(self.val_loader)
        val_batch = next(val_iter)

        v_tensors = val_batch["tensor"].to(self.device)
        v_masks = val_batch["mask"].to(self.device)
        v_targets = val_batch["label_id"].to(self.device)
        v_pids = val_batch["patient_id"]

        with torch.no_grad():
            v_logits = self.model(v_tensors, mask=v_masks)
            v_loss = self.criterion(v_logits, v_targets).item()
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            v_preds = np.argmax(v_probs, axis=1)

        metrics = compute_classification_metrics(
            y_true=v_targets.cpu().numpy(),
            y_pred=v_preds,
            class_names=self.class_names,
            is_binary=self.is_binary,
        )

        patient_agg = aggregate_patient_predictions(
            window_probs=v_probs,
            patient_ids=v_pids,
            window_targets=v_targets.cpu().numpy(),
        )

        return {
            "task": self.task_name,
            "device": str(self.device),
            "train_batch_size": len(tensors),
            "val_batch_size": len(v_tensors),
            "train_loss": round(loss_val, 4),
            "val_loss": round(v_loss, 4),
            "total_grad_norm": round(total_grad_norm, 4),
            "logits_shape": list(logits.shape),
            "classes_in_batch": list(Counter(batch["class_label"]).items()),
            "val_metrics": metrics,
            "patient_aggregation_checked": bool(len(patient_agg["patient_ids"]) > 0),
            "status": "PASS",
        }
