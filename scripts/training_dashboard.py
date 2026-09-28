#!/usr/bin/env python3
"""
AI Stethoscope - Live Terminal Training Dashboard
=================================================
Real-time, PowerShell-compatible terminal dashboard for monitoring AI Stethoscope
controlled multi-task ablation experiments.

Features:
  - 100% READ-ONLY monitor: never alters training or model states.
  - Automatically refreshes every 1.0s.
  - Unicode progress bars with graceful ASCII fallback for legacy Windows consoles.
  - State handling:
      A. WAITING FOR TRAINING TO START (missing/pending status)
      B. TRAINING RUNNING (live epoch/batch/task progress & metrics)
      C. VALIDATING... (validation loop active)
      D. EXPERIMENT COMPLETED ([✓] COMPLETED)
      E. TRAINING INTERRUPTED ([!] INTERRUPTED)
      F. ALL ABLATIONS COMPLETE (with master summary comparison table)
  - Safe, non-crashing status file reader with atomic write tolerance.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_FILE = PROJECT_ROOT / "experiments" / "multitask_ablations" / "training_status.json"

CONFIG_DISPLAY_NAMES = {
    "cnn_only": "CNN Only",
    "cnn_physical": "CNN + Physical Features",
    "cnn_panns": "CNN + PANNs",
    "cnn_physical_panns": "CNN + Physical + PANNs",
}

CONFIG_PARAMETERS = {
    "cnn_only": "520,937",
    "cnn_physical": "546,409",
    "cnn_panns": "783,209",
    "cnn_physical_panns": "808,681",
}


def supports_unicode() -> bool:
    """Determine whether the current stdout encoding supports Unicode block characters."""
    try:
        encoding = sys.stdout.encoding or "ascii"
        "\u2588\u2591\u2713\u25b6".encode(encoding)
        return True
    except Exception:
        return False


USE_UNICODE = supports_unicode()


def format_seconds(seconds: Optional[float]) -> str:
    """Format seconds into HH:MM:SS or 'Calculating...'."""
    if seconds is None:
        return "Calculating..."
    try:
        s = int(max(0, seconds))
        hrs = s // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        return "Calculating..."


def make_progress_bar(percent: float, width: int = 24) -> str:
    """Generate an aesthetic progress bar with Unicode or ASCII characters."""
    pct = max(0.0, min(100.0, percent))
    filled_len = int(round(width * (pct / 100.0)))
    empty_len = max(0, width - filled_len)

    if USE_UNICODE:
        # Unicode block characters
        fill_char = "\u2588"  # █
        empty_char = "\u2591"  # ░
    else:
        # Standard ASCII fallback
        fill_char = "#"
        empty_char = "-"

    bar = fill_char * filled_len + empty_char * empty_len
    return f"[{bar}] {pct:5.1f}%"


def read_status_file(path: Path) -> Optional[Dict[str, Any]]:
    """Safely read and parse the central training_status.json file."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, PermissionError, OSError):
        # File might be in the middle of an atomic swap
        time.sleep(0.05)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def clear_terminal():
    """Clear terminal screen in a Windows PowerShell compatible manner."""
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()


def render_dashboard(status: Optional[Dict[str, Any]], status_path: Path):
    """Render the dashboard UI according to the current training state."""
    bar_sep = "=" * 60
    sub_sep = "-" * 60

    lines = []
    lines.append(bar_sep)
    lines.append("              AI STETHOSCOPE ML TRAINING MONITOR")
    lines.append(bar_sep)

    # State A: Training not started / missing status file
    if status is None:
        lines.append("")
        lines.append(">>> WAITING FOR TRAINING TO START <<<")
        lines.append("")
        lines.append("Status: No active training process detected.")
        lines.append(f"Target: {status_path.relative_to(PROJECT_ROOT) if status_path.is_relative_to(PROJECT_ROOT) else status_path}")
        lines.append("")
        lines.append("Run the ablation training in a separate window:")
        lines.append("  python scripts/train_multitask_ablations.py --config all")
        lines.append("")
        lines.append("Auto-refreshing every 1.0s (Press Ctrl+C to exit)")
        lines.append(bar_sep)
        print("\n".join(lines))
        return

    st = status.get("status", "waiting").lower()
    cur_cfg = status.get("current_configuration", "none")
    cur_cfg_name = CONFIG_DISPLAY_NAMES.get(cur_cfg, cur_cfg)
    cfg_idx = status.get("configuration_index", 0)
    total_cfgs = status.get("total_configurations", 4)

    # State F: All experiments completed
    if st == "completed":
        lines.append("")
        lines.append("                  ALL ABLATIONS COMPLETE")
        lines.append(bar_sep)
        lines.append("")
        lines.append(f"{'Configuration':26s} | {'Best Val F1':11s} | {'Test F1':9s} | {'Status':10s}")
        lines.append("-" * 65)

        exps = status.get("experiments", {})
        for cfg_k in ["cnn_only", "cnn_physical", "cnn_panns", "cnn_physical_panns"]:
            c_disp = CONFIG_DISPLAY_NAMES.get(cfg_k, cfg_k)
            exp_data = exps.get(cfg_k, {})
            val_f1 = exp_data.get("best_val_f1")
            test_f1 = exp_data.get("test_f1")
            exp_st = exp_data.get("status", "completed").capitalize()
            val_str = f"{val_f1:.4f}" if val_f1 is not None else "N/A"
            test_str = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
            lines.append(f"{c_disp:26s} | {val_str:11s} | {test_str:9s} | {exp_st:10s}")

        lines.append("-" * 65)
        lines.append(f"Total Elapsed Time: {format_seconds(status.get('elapsed_seconds', 0))}")
        lines.append("All artifacts and model checkpoints preserved in experiments/multitask_ablations/.")
        lines.append(bar_sep)
        print("\n".join(lines))
        return

    # State E: Training interrupted
    if st == "interrupted":
        lines.append("")
        lines.append("                 [!] TRAINING INTERRUPTED")
        lines.append(bar_sep)
        lines.append("")
        lines.append(f"Last Configuration : {cur_cfg_name} ({cfg_idx} / {total_cfgs})")
        lines.append(f"Last Epoch / Batch : Epoch {status.get('current_epoch', 0)} / Batch {status.get('current_batch', 0)}")
        lines.append(f"Elapsed Time       : {format_seconds(status.get('elapsed_seconds', 0))}")
        lines.append(f"Last Recorded Loss : {status.get('train_loss', 0.0):.4f}")
        lines.append("")
        lines.append("The training process was stopped early or encountered an interruption.")
        lines.append(bar_sep)
        print("\n".join(lines))
        return

    # States B, C, D: Running / Validating
    overall_pct = float(status.get("overall_progress_percent", 0.0))
    lines.append("OVERALL PROGRESS")
    lines.append("")
    lines.append(make_progress_bar(overall_pct, width=24))
    lines.append("")
    lines.append(f"Configuration : {cfg_idx} / {total_cfgs}")
    lines.append(f"Current       : {cur_cfg_name}")
    lines.append(sub_sep)

    lines.append("CURRENT TRAINING")
    lines.append("")
    if st == "validating":
        lines.append("Status:")
        lines.append(">>> VALIDATING VALIDATION SPLITS... <<<")
    else:
        lines.append("Task:")
        lines.append(str(status.get("current_task", "Multi-Task")))
    lines.append("")
    lines.append("Epoch:")
    lines.append(f"{status.get('current_epoch', 1)} / {status.get('total_epochs', 5)}")
    lines.append("")
    lines.append("Epoch Progress:")
    epoch_pct = float(status.get("epoch_progress_percent", 0.0))
    lines.append(make_progress_bar(epoch_pct, width=24))
    lines.append("")
    lines.append("Batch / Round:")
    lines.append(f"{status.get('current_batch', 0)} / {status.get('total_batches', 50)}")
    lines.append(sub_sep)

    lines.append("METRICS")
    lines.append("")
    tr_loss = status.get("train_loss")
    val_loss = status.get("validation_loss")
    val_f1 = status.get("validation_macro_f1")
    best_f1 = status.get("best_validation_macro_f1")
    best_ep = status.get("best_epoch")

    tr_loss_str = f"{tr_loss:.4f}" if tr_loss is not None else "--"
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "--"
    val_f1_str = f"{val_f1:.4f}" if val_f1 is not None else "--"
    best_f1_str = f"{best_f1:.4f}" if best_f1 is not None else "--"
    best_ep_str = str(best_ep) if best_ep is not None else "--"

    lines.append(f"Training Loss:            {tr_loss_str}")
    lines.append(f"Validation Loss:          {val_loss_str}")
    lines.append("")
    lines.append(f"Validation Macro F1:      {val_f1_str}")
    lines.append(f"Best Validation Macro F1: {best_f1_str}")
    lines.append(f"Best Epoch:               {best_ep_str}")
    lines.append(sub_sep)

    lines.append("TIME")
    lines.append("")
    lines.append(f"Elapsed:                  {format_seconds(status.get('elapsed_seconds', 0))}")
    lines.append(f"Estimated Remaining:      {format_seconds(status.get('estimated_remaining_seconds'))}")
    lines.append(sub_sep)

    lines.append("EXPERIMENTS")
    lines.append("")
    exps = status.get("experiments", {})
    check_icon = "\u2713" if USE_UNICODE else "X"
    arrow_icon = "\u25b6" if USE_UNICODE else ">"

    for cfg_k in ["cnn_only", "cnn_physical", "cnn_panns", "cnn_physical_panns"]:
        c_disp = CONFIG_DISPLAY_NAMES.get(cfg_k, cfg_k)
        exp_data = exps.get(cfg_k, {})
        exp_st = exp_data.get("status", "pending")
        if exp_st == "completed":
            icon = f"[{check_icon}]"
        elif exp_st == "running":
            icon = f"[{arrow_icon}]"
        elif exp_st == "interrupted":
            icon = "[!]"
        else:
            icon = "[ ]"
        lines.append(f"{icon} {c_disp}")
    lines.append(sub_sep)

    lines.append("PARAMETERS")
    lines.append("")
    lines.append(f"CNN Only:                       {CONFIG_PARAMETERS['cnn_only']:>10s}")
    lines.append(f"CNN + Physical:                {CONFIG_PARAMETERS['cnn_physical']:>10s}")
    lines.append(f"CNN + PANNs:                   {CONFIG_PARAMETERS['cnn_panns']:>10s}")
    lines.append(f"CNN + Physical + PANNs:        {CONFIG_PARAMETERS['cnn_physical_panns']:>10s}")
    lines.append(bar_sep)

    # Print dashboard
    print("\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Training Dashboard for AI Stethoscope")
    parser.add_argument("--status-file", type=str, default=str(DEFAULT_STATUS_FILE), help="Path to training_status.json")
    parser.add_argument("--refresh-rate", type=float, default=1.0, help="Refresh interval in seconds (default 1.0s)")
    args = parser.parse_args()

    status_path = Path(args.status_file)

    try:
        while True:
            status = read_status_file(status_path)
            clear_terminal()
            render_dashboard(status, status_path)
            time.sleep(args.refresh_rate)
    except KeyboardInterrupt:
        print("\n[Dashboard stopped by user.]")
        sys.exit(0)


if __name__ == "__main__":
    main()
