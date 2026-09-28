#!/usr/bin/env python3
"""
AI Stethoscope - Batch Feature Extraction for Classical ML
==========================================================
Extracts the 397-dimensional acoustic feature matrix for all splits across:
  - Heart (PhysioNet 2016)
  - ICBHI 2017
  - Kaggle Respiratory Disease

Saves extracted feature caches into `processed/classical_features/{dataset}_{split}.npz`.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_features import extract_audio_features, FEATURE_DIM, TIME_FRAMES


def extract_split_features(
    dataset_name: str,
    split: str,
    output_dir: Path,
    base_dir: Path = PROJECT_ROOT,
    force_recompute: bool = False,
) -> Path:
    """Extract and cache features for a specific dataset split."""
    output_path = output_dir / f"{dataset_name}_{split}.npz"
    if output_path.exists() and not force_recompute:
        print(f"[{dataset_name.upper()} {split.upper()}] Cache already exists at: {output_path}")
        return output_path

    index_file = base_dir / "processed" / "full" / f"{dataset_name}_{split}_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_file}")

    df = pd.read_csv(index_file)
    n_samples = len(df)
    print(f"[{dataset_name.upper()} {split.upper()}] Extracting features for {n_samples} samples...")

    X = np.zeros((n_samples, FEATURE_DIM), dtype=np.float32)
    y = df["label_id"].to_numpy(dtype=np.int64)
    patient_ids = df["patient_id"].astype(str).to_numpy()
    recording_ids = df["recording_id"].astype(str).to_numpy()
    labels = df["label"].astype(str).to_numpy()

    t_start = time.time()

    for i, row in df.iterrows():
        tensor_rel_path = row["tensor_path"]
        tensor_full_path = base_dir / tensor_rel_path
        if not tensor_full_path.exists():
            raise FileNotFoundError(f"Missing tensor: {tensor_full_path}")

        spectrogram = np.load(tensor_full_path)

        # Handle mask
        mask = None
        if dataset_name == "icbhi" and "start_time" in row and "end_time" in row:
            t_s = float(row["start_time"])
            t_e = float(row["end_time"])
            duration = max(0.0, t_e - t_s)
            act_frames = min(TIME_FRAMES, max(1, int(round((duration / 5.0) * TIME_FRAMES))))
            mask = np.zeros(TIME_FRAMES, dtype=np.float32)
            mask[:act_frames] = 1.0

        feat = extract_audio_features(spectrogram, mask=mask)
        X[i] = feat

        if (i + 1) % 2000 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(0.001, elapsed)
            print(f"   Processed {i+1}/{n_samples} ({rate:.1f} samples/s, elapsed: {elapsed:.1f}s)")

    duration = time.time() - t_start
    print(f"   -> Finished in {duration:.2f}s ({n_samples/max(0.001, duration):.1f} samples/s)")

    # Validation
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    assert nan_count == 0, f"Found {nan_count} NaNs in {dataset_name}_{split}"
    assert inf_count == 0, f"Found {inf_count} Infs in {dataset_name}_{split}"

    # Save compressed
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        patient_ids=patient_ids,
        recording_ids=recording_ids,
        labels=labels,
    )
    print(f"   -> Saved: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract 397-d classical features")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    parser.add_argument("--dataset", type=str, default=None, help="Optional specific dataset (heart, icbhi, kaggle)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "processed" / "classical_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["kaggle", "icbhi", "heart"]
    splits = ["train", "val", "test"]

    for d in datasets:
        print("\n" + "=" * 70)
        print(f"FEATURE EXTRACTION: {d.upper()}")
        print("=" * 70)
        for s in splits:
            extract_split_features(d, s, output_dir=out_dir, force_recompute=args.force)

    print("\n[ALL FEATURE EXTRACTION COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
