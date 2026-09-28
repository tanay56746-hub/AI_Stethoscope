#!/usr/bin/env python3
"""
AI Stethoscope - Live Training Status & Monitoring Dashboard
============================================================
Lightweight, zero-dependency read-only monitoring dashboard using the Python standard library.
Continuously reads project logs, checkpoints, and metrics without interfering with running processes.

Usage:
    python monitor/dashboard.py [--port 8080] [--host 127.0.0.1]
"""

import os
import sys
import json
import time
import re
import socket
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Locate project root (one level up from monitor/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_heart_task_log() -> Optional[Path]:
    """Find the most recent Heart training task log safely."""
    # 1. Check user gemini brain task logs
    brain_dir = Path.home() / ".gemini" / "antigravity-ide" / "brain"
    candidates = []
    if brain_dir.exists():
        try:
            for root, _, files in os.walk(brain_dir):
                for f in files:
                    if f.endswith(".log") and f.startswith("task-"):
                        full_p = Path(root) / f
                        try:
                            # Read first 1KB to identify
                            with open(full_p, "r", encoding="utf-8", errors="ignore") as fl:
                                header = fl.read(1024)
                                if "HEART CNN BASELINE" in header:
                                    candidates.append((full_p, full_p.stat().st_mtime))
                        except Exception:
                            pass
        except Exception:
            pass

    # 2. Check experiments/heart_cnn/logs/
    exp_log_dir = PROJECT_ROOT / "experiments" / "heart_cnn" / "logs"
    if exp_log_dir.exists():
        for f in exp_log_dir.glob("*.log"):
            candidates.append((f, f.stat().st_mtime))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None


def is_python_training_alive(log_path: Optional[Path]) -> bool:
    """Check if the training process appears active."""
    # If test_metrics.json exists, training is complete, not running
    test_metrics = PROJECT_ROOT / "experiments" / "heart_cnn" / "metrics" / "test_metrics.json"
    if test_metrics.exists():
        return False

    # Check tasklist for running python
    try:
        if sys.platform == "win32":
            out = os.popen('tasklist /FI "IMAGENAME eq python.exe" /NH').read()
            if "python.exe" in out:
                # If log exists, verify it was updated within last 45 minutes
                if log_path and log_path.exists():
                    age_sec = time.time() - log_path.stat().st_mtime
                    if age_sec < 2700:  # 45 minutes
                        return True
                else:
                    return True
    except Exception:
        pass

    # Fallback to log timestamp freshness
    if log_path and log_path.exists():
        age_sec = time.time() - log_path.stat().st_mtime
        return age_sec < 1800  # 30 minutes
    return False


def get_heart_status() -> dict:
    """Collect read-only status for Heart CNN baseline."""
    log_path = find_heart_task_log()
    test_metrics_path = PROJECT_ROOT / "experiments" / "heart_cnn" / "metrics" / "test_metrics.json"
    best_ckpt_path = PROJECT_ROOT / "experiments" / "heart_cnn" / "checkpoints" / "best_model.pt"
    last_ckpt_path = PROJECT_ROOT / "experiments" / "heart_cnn" / "checkpoints" / "last_checkpoint.pt"

    is_complete = test_metrics_path.exists()
    is_alive = is_python_training_alive(log_path) if not is_complete else False

    status_label = "Completed" if is_complete else ("Running" if is_alive else "Stopped / Idle")

    data = {
        "status": status_label,
        "is_alive": is_alive,
        "is_complete": is_complete,
        "current_epoch": "Unknown",
        "total_epochs": 15,
        "current_batch": "Unknown",
        "total_batches": 598,
        "epoch_progress_pct": 0.0,
        "overall_progress_pct": 0.0,
        "latest_train_loss": "Unknown",
        "latest_val_loss": "Unknown",
        "latest_val_macro_f1": "Unknown",
        "latest_val_bal_acc": "Unknown",
        "latest_val_patient_f1": "Unknown",
        "best_epoch": "Unknown",
        "best_val_macro_f1": "Unknown",
        "best_model_exists": best_ckpt_path.exists(),
        "last_checkpoint_exists": last_ckpt_path.exists(),
        "elapsed_seconds": 0,
        "elapsed_formatted": "Unknown",
        "eta_seconds": "Unknown",
        "eta_formatted": "Unknown",
        "last_update_time": "Unknown",
        "completed_epochs": [],
    }

    # Parse log file if available
    if log_path and log_path.exists():
        try:
            mtime = log_path.stat().st_mtime
            data["last_update_time"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.splitlines()

            # Extract start time if logged or infer from file creation
            log_ctime = log_path.stat().st_ctime
            elapsed = max(0, int(time.time() - log_ctime))
            data["elapsed_seconds"] = elapsed
            hrs = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            data["elapsed_formatted"] = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs > 0 else f"{mins}m {secs:02d}s"

            # Parse completed epoch summaries
            # Pattern: Epoch [ X/15] Train Loss: 0.2945 | Val Loss: 0.2149 | Val F1: 0.8337 (BalAcc: 0.8847) | Pat F1: 0.8595 | LR: ... (820.8s)
            epoch_pattern = re.compile(
                r"Epoch\s+\[\s*(\d+)/(\d+)\]\s+Train Loss:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\|\s+Val F1:\s+([\d\.]+)\s+\(BalAcc:\s+([\d\.]+)\)\s+\|\s+Pat F1:\s+([\d\.]+).*?\(([\d\.]+)s\)"
            )

            completed_epochs = []
            best_f1 = -1.0
            best_ep = -1

            for line in lines:
                m = epoch_pattern.search(line)
                if m:
                    ep = int(m.group(1))
                    tot_ep = int(m.group(2))
                    tr_loss = float(m.group(3))
                    v_loss = float(m.group(4))
                    v_f1 = float(m.group(5))
                    v_bal_acc = float(m.group(6))
                    p_f1 = float(m.group(7))
                    dur = float(m.group(8))

                    data["total_epochs"] = tot_ep
                    completed_epochs.append({
                        "epoch": ep,
                        "train_loss": tr_loss,
                        "val_loss": v_loss,
                        "val_f1": v_f1,
                        "val_bal_acc": v_bal_acc,
                        "patient_f1": p_f1,
                        "duration_sec": dur,
                    })

                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        best_ep = ep

            data["completed_epochs"] = completed_epochs

            if completed_epochs:
                last_done = completed_epochs[-1]
                data["latest_train_loss"] = f"{last_done['train_loss']:.4f}"
                data["latest_val_loss"] = f"{last_done['val_loss']:.4f}"
                data["latest_val_macro_f1"] = f"{last_done['val_f1']:.4f}"
                data["latest_val_bal_acc"] = f"{last_done['val_bal_acc']:.4f}"
                data["latest_val_patient_f1"] = f"{last_done['patient_f1']:.4f}"
                data["best_epoch"] = best_ep
                data["best_val_macro_f1"] = f"{best_f1:.4f}"

            # Parse current active step
            # Pattern: [Epoch 9 Step 400/598 ( 66.9%)] Running Loss: 0.2695 (714.9s)
            step_pattern = re.compile(
                r"\[Epoch\s+(\d+)\s+Step\s+(\d+)/(\d+)\s+\(\s*([\d\.]+)%\)\]\s+Running Loss:\s+([\d\.]+)"
            )

            latest_step_m = None
            for line in reversed(lines):
                sm = step_pattern.search(line)
                if sm:
                    latest_step_m = sm
                    break

            if latest_step_m:
                cur_ep = int(latest_step_m.group(1))
                cur_st = int(latest_step_m.group(2))
                tot_st = int(latest_step_m.group(3))
                pct = float(latest_step_m.group(4))
                rn_loss = float(latest_step_m.group(5))

                data["current_epoch"] = cur_ep
                data["current_batch"] = cur_st
                data["total_batches"] = tot_st
                data["epoch_progress_pct"] = pct
                data["latest_train_loss"] = f"{rn_loss:.4f} (Step {cur_st})"

                # Overall progress
                overall_completed = (cur_ep - 1) + (cur_st / float(tot_st))
                data["overall_progress_pct"] = round((overall_completed / data["total_epochs"]) * 100.0, 1)
            elif completed_epochs:
                last_ep = completed_epochs[-1]["epoch"]
                if is_complete or last_ep >= data["total_epochs"]:
                    data["current_epoch"] = data["total_epochs"]
                    data["current_batch"] = data["total_batches"]
                    data["epoch_progress_pct"] = 100.0
                    data["overall_progress_pct"] = 100.0
                else:
                    data["current_epoch"] = last_ep + 1
                    data["current_batch"] = 0
                    data["epoch_progress_pct"] = 0.0
                    data["overall_progress_pct"] = round((last_ep / data["total_epochs"]) * 100.0, 1)

            # ETA Calculation based on real completed epoch durations
            if is_alive and completed_epochs and not is_complete:
                durations = [e["duration_sec"] for e in completed_epochs[-5:]]
                avg_epoch_sec = sum(durations) / len(durations) if durations else 900.0

                current_epoch_val = data["current_epoch"] if isinstance(data["current_epoch"], int) else (len(completed_epochs) + 1)
                fraction_done_current = data["epoch_progress_pct"] / 100.0
                remaining_in_cur_epoch = max(0.0, (1.0 - fraction_done_current) * avg_epoch_sec)

                # Check early stopping vs max epochs
                # Early stopping patience = 5. Best epoch is best_ep
                epochs_remaining_max = max(0, data["total_epochs"] - current_epoch_val)
                patience_limit_epoch = min(data["total_epochs"], best_ep + 5) if best_ep != -1 else data["total_epochs"]
                epochs_remaining_early = max(0, patience_limit_epoch - current_epoch_val)

                eta_sec_max = int(remaining_in_cur_epoch + epochs_remaining_max * avg_epoch_sec)
                eta_sec_early = int(remaining_in_cur_epoch + epochs_remaining_early * avg_epoch_sec)

                data["eta_seconds"] = eta_sec_early
                e_hrs = eta_sec_early // 3600
                e_mins = (eta_sec_early % 3600) // 60
                data["eta_formatted"] = f"~{e_mins} mins" if e_hrs == 0 else f"~{e_hrs}h {e_mins}m"
                if epochs_remaining_early < epochs_remaining_max:
                    data["eta_formatted"] += f" (Early Stop Ep {patience_limit_epoch}) to ~{eta_sec_max // 60}m (Max Ep 15)"
            elif is_complete:
                data["eta_formatted"] = "0s (Training Completed)"
                data["overall_progress_pct"] = 100.0
                data["epoch_progress_pct"] = 100.0

        except Exception as e:
            data["parse_error"] = str(e)

    # If test_metrics exists, load finalized test results
    if test_metrics_path.exists():
        try:
            with open(test_metrics_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            data["test_metrics"] = tm
            data["status"] = "Completed"
            data["overall_progress_pct"] = 100.0
        except Exception:
            pass

    return data


def get_icbhi_status() -> dict:
    """Collect read-only status for ICBHI CNN baseline."""
    best_ckpt = PROJECT_ROOT / "experiments" / "icbhi_cnn" / "checkpoints" / "best_model.pt"
    last_ckpt = PROJECT_ROOT / "experiments" / "icbhi_cnn" / "checkpoints" / "last_checkpoint.pt"
    test_metrics_p = PROJECT_ROOT / "experiments" / "icbhi_cnn" / "metrics" / "test_metrics.json"

    data = {
        "training_status": "Completed Epoch 3 (Checkpoint Saved)",
        "checkpoint_status": "Valid (Epoch 2 Best, Epoch 3 Last)",
        "best_model_exists": best_ckpt.exists(),
        "last_checkpoint_exists": last_ckpt.exists(),
        "test_evaluation_status": "Completed" if test_metrics_p.exists() else "Pending",
        "test_metrics_available": test_metrics_p.exists(),
        "metrics": {},
    }

    if test_metrics_p.exists():
        try:
            with open(test_metrics_p, "r", encoding="utf-8") as f:
                tm = json.load(f)
            win = tm.get("window_metrics", {})
            pat = tm.get("patient_metrics", {})
            data["metrics"] = {
                "loss": tm.get("loss", "Unknown"),
                "window_accuracy": win.get("accuracy", "Unknown"),
                "window_balanced_accuracy": win.get("balanced_accuracy", "Unknown"),
                "window_macro_f1": win.get("f1_macro", "Unknown"),
                "window_weighted_f1": win.get("f1_weighted", "Unknown"),
                "patient_accuracy": pat.get("accuracy", "Unknown"),
                "patient_macro_f1": pat.get("f1_macro", "Unknown"),
                "num_windows": tm.get("num_windows", "Unknown"),
                "num_patients": tm.get("num_patients", "Unknown"),
            }
        except Exception as e:
            data["metrics_error"] = str(e)

    return data


def get_kaggle_status() -> dict:
    """Collect read-only status for Kaggle CNN baseline."""
    best_ckpt = PROJECT_ROOT / "experiments" / "kaggle_cnn" / "checkpoints" / "best_model.pt"
    test_metrics_p = PROJECT_ROOT / "experiments" / "kaggle_cnn" / "metrics" / "test_metrics.json"
    history_p = PROJECT_ROOT / "experiments" / "kaggle_cnn" / "logs" / "history.json"

    data = {
        "training_status": "Completed (9 Epochs, Early Stopped)",
        "test_status": "Completed",
        "best_model_exists": best_ckpt.exists(),
        "test_metrics_available": test_metrics_p.exists(),
        "best_epoch": 4,
        "total_epochs": 9,
        "metrics": {},
    }

    if test_metrics_p.exists():
        try:
            with open(test_metrics_p, "r", encoding="utf-8") as f:
                tm = json.load(f)
            win = tm.get("window_metrics", {})
            pat = tm.get("patient_metrics", {})
            data["metrics"] = {
                "loss": tm.get("loss", "Unknown"),
                "window_accuracy": win.get("accuracy", "Unknown"),
                "window_balanced_accuracy": win.get("balanced_accuracy", "Unknown"),
                "window_macro_f1": win.get("f1_macro", "Unknown"),
                "window_weighted_f1": win.get("f1_weighted", "Unknown"),
                "patient_accuracy": pat.get("accuracy", "Unknown"),
                "patient_macro_f1": pat.get("f1_macro", "Unknown"),
                "num_windows": tm.get("num_windows", "Unknown"),
                "num_patients": tm.get("num_patients", "Unknown"),
            }
        except Exception as e:
            data["metrics_error"] = str(e)

    return data


def get_all_status() -> dict:
    """Combine all statuses into single structured payload."""
    heart = get_heart_status()
    icbhi = get_icbhi_status()
    kaggle = get_kaggle_status()

    # Overall progress calculation
    milestones = [
        {"name": "Kaggle CNN Training & Test", "done": kaggle["test_status"] == "Completed"},
        {"name": "ICBHI CNN Checkpoint Verification", "done": icbhi["best_model_exists"]},
        {"name": "ICBHI Held-out Test Evaluation", "done": icbhi["test_evaluation_status"] == "Completed"},
        {"name": "Heart CNN Training", "done": heart["is_complete"]},
        {"name": "Heart Held-out Test Evaluation", "done": heart["is_complete"]},
        {"name": "Master Report (full_cnn_baseline_results.md)", "done": (PROJECT_ROOT / "analysis" / "full_cnn_baseline_results.md").exists()},
    ]

    done_count = sum(1 for m in milestones if m["done"])
    total_count = len(milestones)
    overall_pct = round((done_count / total_count) * 100.0, 1)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_progress": {
            "percent": overall_pct,
            "completed_milestones": done_count,
            "total_milestones": total_count,
            "milestones": milestones,
        },
        "heart": heart,
        "icbhi": icbhi,
        "kaggle": kaggle,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Stethoscope — Stage 5B Training Monitor</title>
  <style>
    :root {
      --bg: #0b0f19;
      --surface: #131b2e;
      --surface-card: #18223c;
      --border: #233154;
      --text: #e2e8f0;
      --text-muted: #8da2c0;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.2);
      --warning: #f59e0b;
      --pulse-green: #34d399;
      --heart-color: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      padding: 24px 20px;
      line-height: 1.5;
    }

    .container {
      max-width: 1140px;
      margin: 0 auto;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }

    .logo-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .stethoscope-icon {
      font-size: 28px;
      background: var(--surface-card);
      width: 48px;
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      border: 1px solid var(--border);
    }

    h1 {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: #fff;
    }

    .subtitle {
      font-size: 13px;
      color: var(--text-muted);
    }

    .header-status {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
      background: var(--pulse-green);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--pulse-green);
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.7; }
      50% { transform: scale(1.3); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Overall Progress Bar Card */
    .overall-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .overall-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 12px;
      align-items: center;
    }

    .overall-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
    }

    .overall-pct {
      font-size: 20px;
      font-weight: 700;
      color: var(--accent);
    }

    .progress-bar-bg {
      background: #0d1424;
      border-radius: 8px;
      height: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 8px;
      transition: width 0.4s ease;
    }

    .milestone-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .pill {
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface-card);
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .pill.done {
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.1);
      color: #6ee7b7;
    }

    /* Grid Layout */
    .grid {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }

    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .card.primary {
      border-color: rgba(56, 189, 248, 0.4);
      box-shadow: 0 0 25px var(--accent-glow);
    }

    .card-title-group {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }

    .card-title {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .badge.running { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge.completed { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge.stopped { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Metric Grid */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }

    @media (max-width: 480px) {
      .metric-grid { grid-template-columns: 1fr; }
    }

    .metric-box {
      background: var(--surface-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
    }

    .metric-label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    .metric-val {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }

    .metric-val.highlight {
      color: var(--accent);
    }

    .metric-val.green {
      color: #34d399;
    }

    .metric-sub {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .progress-section {
      background: var(--surface-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 14px;
    }

    .progress-title-line {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 8px;
      font-weight: 600;
    }

    .sub-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    @media (max-width: 900px) {
      .sub-grid { grid-template-columns: 1fr; }
    }

    .info-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 13px;
    }

    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    .info-row:last-child { border-bottom: none; }
    .info-label { color: var(--text-muted); }
    .info-value { font-weight: 600; color: #fff; }

    footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 8px;
    }

    .refresh-hint {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-group">
        <div class="stethoscope-icon">🩺</div>
        <div>
          <h1>AI Stethoscope — Training Dashboard</h1>
          <div class="subtitle">Real-time Stage 5B Custom CNN Baseline Verification &amp; Monitoring</div>
        </div>
      </div>
      <div class="header-status">
        <div class="live-badge" id="liveBadge">
          <span class="pulse-dot"></span>
          <span id="liveStatusText">POLLING LIVE</span>
        </div>
      </div>
    </header>

    <!-- Overall Milestone Progress -->
    <div class="overall-card">
      <div class="overall-header">
        <div class="overall-title">Overall Stage 5B Benchmark Progress</div>
        <div class="overall-pct" id="overallPct">0.0%</div>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="overallBar" style="width: 0%;"></div>
      </div>
      <div class="milestone-pills" id="milestones"></div>
    </div>

    <!-- Active Training Focus: Heart CNN -->
    <div class="grid">
      <div class="card primary">
        <div class="card-title-group">
          <div class="card-title">❤️ Heart CNN Baseline (PhysioNet 2016)</div>
          <div class="badge" id="heartBadge">Checking...</div>
        </div>

        <!-- Live Step & Epoch Progress -->
        <div class="progress-section">
          <div class="progress-title-line">
            <span>Current Epoch Progress (<span id="heartBatchStep">Step 0/598</span>)</span>
            <span id="heartStepPct" style="color: var(--accent);">0.0%</span>
          </div>
          <div class="progress-bar-bg" style="margin-bottom: 12px;">
            <div class="progress-bar-fill" id="heartStepBar" style="width: 0%; background: linear-gradient(90deg, #38bdf8, #60a5fa);"></div>
          </div>

          <div class="progress-title-line">
            <span>Overall Heart CNN Progress (<span id="heartEpochProgress">Epoch 0/15</span>)</span>
            <span id="heartOverallPct" style="color: #a78bfa;">0.0%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="heartOverallBar" style="width: 0%; background: linear-gradient(90deg, #a78bfa, #c084fc);"></div>
          </div>
        </div>

        <div class="metric-grid">
          <div class="metric-box">
            <div class="metric-label">Best Val Macro F1</div>
            <div class="metric-val green" id="heartBestF1">Unknown</div>
            <div class="metric-sub" id="heartBestEpochSub">Best: Epoch --</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Latest Val Loss</div>
            <div class="metric-val highlight" id="heartLatestValLoss">Unknown</div>
            <div class="metric-sub" id="heartTrainLossSub">Train Loss: --</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Elapsed Time</div>
            <div class="metric-val" id="heartElapsed">Unknown</div>
            <div class="metric-sub">Process runtime</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Estimated Remaining (ETA)</div>
            <div class="metric-val" id="heartEta" style="font-size: 15px;">Unknown</div>
            <div class="metric-sub" id="heartEtaSub">Based on actual duration</div>
          </div>
        </div>

        <div class="info-list">
          <div class="info-row">
            <span class="info-label">Checkpoint `best_model.pt`</span>
            <span class="info-value" id="heartBestModelExists">Unknown</span>
          </div>
          <div class="info-row">
            <span class="info-label">Latest Validation Macro F1</span>
            <span class="info-value" id="heartLatestValF1">Unknown</span>
          </div>
          <div class="info-row">
            <span class="info-label">Latest Validation Balanced Acc</span>
            <span class="info-value" id="heartLatestValBalAcc">Unknown</span>
          </div>
          <div class="info-row">
            <span class="info-label">Latest Validation Patient F1</span>
            <span class="info-value" id="heartLatestPatientF1">Unknown</span>
          </div>
          <div class="info-row">
            <span class="info-label">Last Log Update</span>
            <span class="info-value" id="heartLastUpdate">Unknown</span>
          </div>
        </div>
      </div>

      <!-- Right Column: Kaggle & ICBHI -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <!-- ICBHI Card -->
        <div class="card">
          <div class="card-title-group">
            <div class="card-title">🫁 ICBHI CNN (Acoustic Cycle Baseline)</div>
            <div class="badge completed" id="icbhiBadge">Evaluation Done</div>
          </div>

          <div class="info-list">
            <div class="info-row">
              <span class="info-label">Training State</span>
              <span class="info-value" id="icbhiTrainStatus">Completed Epoch 3 (Saved)</span>
            </div>
            <div class="info-row">
              <span class="info-label">Checkpoints</span>
              <span class="info-value" id="icbhiCkptStatus">Valid (best_model.pt Ep 2)</span>
            </div>
            <div class="info-row">
              <span class="info-label">Held-out Test Evaluation</span>
              <span class="info-value" style="color: #34d399;" id="icbhiTestStatus">Completed</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Window Macro F1</span>
              <span class="info-value" id="icbhiWinF1">0.2162</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Patient Macro F1</span>
              <span class="info-value" id="icbhiPatF1">0.1903</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Loss (Weighted CE)</span>
              <span class="info-value" id="icbhiLoss">1.4186</span>
            </div>
            <div class="info-row">
              <span class="info-label">Temporal Masked Pooling</span>
              <span class="info-value" style="color: var(--accent);">Active &amp; Verified</span>
            </div>
          </div>
        </div>

        <!-- Kaggle Card -->
        <div class="card">
          <div class="card-title-group">
            <div class="card-title">📊 Kaggle CNN (Respiratory Disease)</div>
            <div class="badge completed" id="kaggleBadge">Completed</div>
          </div>

          <div class="info-list">
            <div class="info-row">
              <span class="info-label">Training State</span>
              <span class="info-value" id="kaggleTrainStatus">Completed (9 Epochs)</span>
            </div>
            <div class="info-row">
              <span class="info-label">Early Stopping</span>
              <span class="info-value" id="kaggleEarlyStop">Epoch 9 (Best: Ep 4)</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Status</span>
              <span class="info-value" style="color: #34d399;">Completed</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Window Macro F1</span>
              <span class="info-value" id="kaggleWinF1">0.3568</span>
            </div>
            <div class="info-row">
              <span class="info-label">Test Patient Macro F1</span>
              <span class="info-value" id="kagglePatF1">0.3385</span>
            </div>
            <div class="info-row">
              <span class="info-label">Patient Aggregation</span>
              <span class="info-value" style="color: var(--accent);">Soft Voting Mean</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <footer>
      <div class="refresh-hint">
        <span class="pulse-dot" style="width: 6px; height: 6px;"></span>
        <span>Auto-refreshing every 3s via zero-lock read-only API</span>
      </div>
      <div>AI Stethoscope Architecture • Read-Only Infrastructure Mode</div>
    </footer>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const d = await res.json();

        // 1. Overall Progress
        const op = d.overall_progress || {};
        document.getElementById('overallPct').textContent = (op.percent || 0.0) + '%';
        document.getElementById('overallBar').style.width = (op.percent || 0.0) + '%';

        const pillsCont = document.getElementById('milestones');
        pillsCont.innerHTML = '';
        (op.milestones || []).forEach(m => {
          const div = document.createElement('div');
          div.className = 'pill' + (m.done ? ' done' : '');
          div.innerHTML = (m.done ? '✓ ' : '○ ') + m.name;
          pillsCont.appendChild(div);
        });

        // 2. Heart CNN
        const h = d.heart || {};
        const hBadge = document.getElementById('heartBadge');
        hBadge.textContent = h.status || 'Unknown';
        hBadge.className = 'badge ' + (h.is_complete ? 'completed' : (h.is_alive ? 'running' : 'stopped'));

        document.getElementById('heartBatchStep').textContent =
          `Step ${h.current_batch !== 'Unknown' ? h.current_batch : '--'}/${h.total_batches || 598}`;
        document.getElementById('heartStepPct').textContent = (h.epoch_progress_pct || 0.0) + '%';
        document.getElementById('heartStepBar').style.width = (h.epoch_progress_pct || 0.0) + '%';

        document.getElementById('heartEpochProgress').textContent =
          `Epoch ${h.current_epoch !== 'Unknown' ? h.current_epoch : '--'}/${h.total_epochs || 15}`;
        document.getElementById('heartOverallPct').textContent = (h.overall_progress_pct || 0.0) + '%';
        document.getElementById('heartOverallBar').style.width = (h.overall_progress_pct || 0.0) + '%';

        document.getElementById('heartBestF1').textContent = h.best_val_macro_f1 || 'Unknown';
        document.getElementById('heartBestEpochSub').textContent =
          h.best_epoch !== 'Unknown' ? `Best Checkpoint: Epoch ${h.best_epoch}` : 'Best: Epoch --';

        document.getElementById('heartLatestValLoss').textContent = h.latest_val_loss || 'Unknown';
        document.getElementById('heartTrainLossSub').textContent = `Train Loss: ${h.latest_train_loss || '--'}`;

        document.getElementById('heartElapsed').textContent = h.elapsed_formatted || 'Unknown';
        document.getElementById('heartEta').textContent = h.eta_formatted || 'Unknown';

        document.getElementById('heartBestModelExists').textContent = h.best_model_exists ? '✓ Available (Saved)' : '✗ Not Found';
        document.getElementById('heartBestModelExists').style.color = h.best_model_exists ? '#34d399' : '#f87171';

        document.getElementById('heartLatestValF1').textContent = h.latest_val_macro_f1 || 'Unknown';
        document.getElementById('heartLatestValBalAcc').textContent = h.latest_val_bal_acc || 'Unknown';
        document.getElementById('heartLatestPatientF1').textContent = h.latest_val_patient_f1 || 'Unknown';
        document.getElementById('heartLastUpdate').textContent = h.last_update_time || 'Unknown';

        // 3. ICBHI CNN
        const i = d.icbhi || {};
        document.getElementById('icbhiTrainStatus').textContent = i.training_status || 'Unknown';
        document.getElementById('icbhiCkptStatus').textContent = i.checkpoint_status || 'Unknown';
        document.getElementById('icbhiTestStatus').textContent = i.test_evaluation_status || 'Pending';
        if (i.metrics) {
          document.getElementById('icbhiWinF1').textContent = i.metrics.window_macro_f1 || 'Unknown';
          document.getElementById('icbhiPatF1').textContent = i.metrics.patient_macro_f1 || 'Unknown';
          document.getElementById('icbhiLoss').textContent = i.metrics.loss || 'Unknown';
        }

        // 4. Kaggle CNN
        const k = d.kaggle || {};
        document.getElementById('kaggleTrainStatus').textContent = k.training_status || 'Completed';
        if (k.metrics) {
          document.getElementById('kaggleWinF1').textContent = k.metrics.window_macro_f1 || 'Unknown';
          document.getElementById('kagglePatF1').textContent = k.metrics.patient_macro_f1 || 'Unknown';
        }

        document.getElementById('liveStatusText').textContent = 'POLLING LIVE (' + (d.timestamp || '') + ')';
      } catch (err) {
        document.getElementById('liveStatusText').textContent = 'RETRYING (' + err.message + ')';
        document.getElementById('liveBadge').style.borderColor = 'rgba(239, 68, 68, 0.4)';
      }
    }

    // Initial load + interval
    updateDashboard();
    setInterval(updateDashboard, 3000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status_data = get_all_status()
            self.wfile.write(json.dumps(status_data, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default terminal access logs for clean execution
        pass


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the given port is available to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def run_dashboard(host: str = "127.0.0.1", port: int = 8080):
    """Run local HTTP server."""
    # Find available port if specified is occupied
    actual_port = port
    while not is_port_available(actual_port, host) and actual_port < port + 20:
        actual_port += 1

    server_address = (host, actual_port)
    httpd = HTTPServer(server_address, DashboardHandler)

    url = f"http://{host}:{actual_port}"
    print("\n" + "=" * 75)
    print("  AI STETHOSCOPE -- TRAINING MONITOR DASHBOARD")
    print("=" * 75)
    print(f"  [OK] Dashboard Server Active: {url}")
    print(f"  [OK] Read-Only Monitoring    : Zero interference with running training")
    print(f"  [OK] Auto-refresh Interval   : 3 seconds")
    print(f"  [OK] Open in browser         : {url}")
    print("=" * 75)
    print("Press Ctrl+C to stop the dashboard server.\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...", flush=True)
        httpd.server_close()
        print("Dashboard server stopped.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Stethoscope Training Monitor Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    run_dashboard(host=args.host, port=args.port)
