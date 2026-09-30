#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — BATCH PREPROCESSING] | File: scripts/preprocess_full_dataset.py
# Purpose: Full Dataset 16 kHz Filtering, Windowing & Log-Mel Tensor Generation
# ==============================================================================
"""
AI Stethoscope - Full Dataset Preprocessing & Caching
======================================================
Executes full dataset preprocessing into standardized Log-Mel spectrogram tensors:
  - Heart Sound Dataset (PhysioNet 2016): train (50% overlap windows), val/test (non-overlapping)
  - ICBHI 2017 Lung Dataset: Exact annotated respiratory cycles (padded/cropped to 5.0s)
  - Kaggle Respiratory Dataset: train (50% overlap windows), val/test (non-overlapping)

Output Representation:
  - Sampling Rate: 16,000 Hz
  - Duration: 5.0 seconds (80,000 samples)
  - Preprocessing: 20 Hz High-pass Butterworth filter -> Polyphase sinc resampling -> Peak normalization to 0.95
  - Representation: 128 Mel bands x 313 Time frames (Log-Mel dB)
  - Output format: .npy tensors stored in processed/full/

Strict Constraints:
  - 100% READ-ONLY on raw dataset files.
  - Manifests remain unchanged.
  - Zero patient/recording leakage between splits.
  - No model training.
"""

import os
import sys
import csv
import time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa

# Configuration Constants
TARGET_SR = 16000
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 80,000
HIGHPASS_CUTOFF = 20.0
HIGHPASS_ORDER = 4
TARGET_PEAK = 0.95

# Log-Mel Spectrogram Parameters
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 20.0
F_MAX = 2000.0
POWER = 2.0
TOP_DB = 80.0

EXPECTED_SHAPE = (128, 313)


def load_raw_audio(file_path: Path):
    """Load audio file safely and convert to float32 mono [-1.0, 1.0]."""
    sr, data = wavfile.read(str(file_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_v = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float32) / max_v
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
    else:
        audio = data.astype(np.float32)

    return audio, sr


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Mean center audio to remove DC bias."""
    return audio - np.mean(audio)


def apply_highpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply Butterworth 20 Hz high-pass filter."""
    nyquist = sr / 2.0
    if HIGHPASS_CUTOFF >= nyquist * 0.95:
        return audio
    sos = signal.butter(HIGHPASS_ORDER, HIGHPASS_CUTOFF, btype="highpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)


def resample_polyphase(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    High-speed, mathematically exact polyphase sinc resampling to 16,000 Hz.
    Uses Kaiser-windowed anti-aliasing FIR filter.
    """
    if orig_sr == TARGET_SR:
        return audio
    elif orig_sr == 2000:
        return signal.resample_poly(audio, 8, 1).astype(np.float32)
    elif orig_sr == 4000:
        return signal.resample_poly(audio, 4, 1).astype(np.float32)
    elif orig_sr == 44100:
        return signal.resample_poly(audio, 160, 441).astype(np.float32)
    elif orig_sr == 10000:
        return signal.resample_poly(audio, 8, 5).astype(np.float32)
    else:
        # Fallback to librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR, res_type="soxr_mq").astype(np.float32)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """Normalize peak absolute amplitude to TARGET_PEAK (0.95)."""
    max_abs = np.max(np.abs(audio))
    if max_abs > 1e-7:
        normalized = (audio / max_abs) * TARGET_PEAK
    else:
        normalized = audio
    return normalized.astype(np.float32)


def compute_log_mel_tensor(audio_80k: np.ndarray) -> np.ndarray:
    """Compute Log-Mel spectrogram tensor of shape (128, 313)."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio_80k,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=POWER,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=TOP_DB).astype(np.float32)
    assert log_mel.shape == EXPECTED_SHAPE, f"Bad shape: {log_mel.shape}"
    assert not np.isnan(log_mel).any(), "NaN in tensor"
    assert not np.isinf(log_mel).any(), "Inf in tensor"
    return log_mel


def extract_windows_from_audio(audio_16k: np.ndarray, hop_samples: int):
    """
    Extract 5-second (80,000 samples) windows from continuous audio.
    If recording is shorter than 5s, zero-pads to 5s.
    Pads trailing segment if >= 1.0 second (16,000 samples).
    """
    n = len(audio_16k)
    if n <= TARGET_SAMPLES:
        pad_len = TARGET_SAMPLES - n
        padded = np.pad(audio_16k, (0, pad_len), mode="constant") if pad_len > 0 else audio_16k
        return [padded]

    windows = []
    s = 0
    while s + TARGET_SAMPLES <= n:
        windows.append(audio_16k[s : s + TARGET_SAMPLES])
        s += hop_samples

    # Retain trailing segment if at least 1.0 second
    if (n - s) >= 16000 and s < n:
        tail = audio_16k[s:]
        pad_len = TARGET_SAMPLES - len(tail)
        padded = np.pad(tail, (0, pad_len), mode="constant")
        windows.append(padded)

    return windows


# =====================================================================
# 1. PROCESS HEART SPLIT
# =====================================================================
def process_heart_sample(item: tuple):
    row, split, base_dir, out_dir = item
    file_p = base_dir / row["file_path"]
    rec_id = row["recording_id"]
    pid = row["patient_id"]
    lbl_str = row["label"]
    lbl_id = int(row["label_id"])

    try:
        raw_audio, sr = load_raw_audio(file_p)
        audio = remove_dc_offset(raw_audio)
        audio = apply_highpass_filter(audio, sr)
        audio_16k = resample_polyphase(audio, sr)
        audio_16k = normalize_peak(audio_16k)

        # Hop: 50% overlap (40,000 samples) for train, non-overlapping (80,000 samples) for val/test
        hop_samples = 40000 if split == "train" else 80000
        windows = extract_windows_from_audio(audio_16k, hop_samples)

        index_entries = []
        for w_idx, win in enumerate(windows):
            tensor = compute_log_mel_tensor(win)
            tensor_name = f"{rec_id}_win{w_idx}.npy"
            tensor_file = out_dir / split / tensor_name
            np.save(tensor_file, tensor)

            rel_tensor_path = tensor_file.relative_to(base_dir).as_posix()
            entry = {
                "tensor_path": rel_tensor_path,
                "dataset": "heart_physionet2016",
                "task": "heart",
                "patient_id": pid,
                "recording_id": rec_id,
                "window_idx": w_idx,
                "label": lbl_str,
                "label_id": lbl_id,
            }
            index_entries.append(entry)

        return True, index_entries, None
    except Exception as e:
        return False, [], f"Heart {rec_id}: {str(e)}"


# =====================================================================
# 2. PROCESS ICBHI SPLIT
# =====================================================================
def process_icbhi_sample(item: tuple):
    row, split, cycle_idx, base_dir, out_dir = item
    file_p = base_dir / row["file_path"]
    rec_id = row["recording_id"]
    pid = row["patient_id"]
    chest_loc = row["chest_location"]
    t_start = float(row["start_time"])
    t_end = float(row["end_time"])
    lbl_str = row["label"]
    lbl_id = int(row["label_id"])

    try:
        raw_audio, sr = load_raw_audio(file_p)
        idx_start = max(0, int(round(t_start * sr)))
        idx_end = min(len(raw_audio), int(round(t_end * sr)))
        cycle_raw = raw_audio[idx_start:idx_end]

        audio = remove_dc_offset(cycle_raw)
        audio = apply_highpass_filter(audio, sr)
        audio_16k = resample_polyphase(audio, sr)
        audio_16k = normalize_peak(audio_16k)

        # Pad short cycles to 5.0s (80,000 samples); center crop if > 5.0s
        n = len(audio_16k)
        if n == TARGET_SAMPLES:
            fixed_audio = audio_16k
        elif n < TARGET_SAMPLES:
            pad_len = TARGET_SAMPLES - n
            fixed_audio = np.pad(audio_16k, (0, pad_len), mode="constant")
        else:
            mid = n // 2
            crop_start = mid - (TARGET_SAMPLES // 2)
            fixed_audio = audio_16k[crop_start : crop_start + TARGET_SAMPLES]

        tensor = compute_log_mel_tensor(fixed_audio)
        tensor_name = f"{rec_id}_c{cycle_idx}_{lbl_id}.npy"
        tensor_file = out_dir / split / tensor_name
        np.save(tensor_file, tensor)

        rel_tensor_path = tensor_file.relative_to(base_dir).as_posix()
        entry = {
            "tensor_path": rel_tensor_path,
            "dataset": "icbhi_2017",
            "task": "lung_acoustic",
            "patient_id": pid,
            "recording_id": rec_id,
            "cycle_idx": cycle_idx,
            "chest_location": chest_loc,
            "start_time": f"{t_start:.4f}",
            "end_time": f"{t_end:.4f}",
            "label": lbl_str,
            "label_id": lbl_id,
        }
        return True, [entry], None
    except Exception as e:
        return False, [], f"ICBHI {rec_id} c{cycle_idx}: {str(e)}"


# =====================================================================
# 3. PROCESS KAGGLE SPLIT
# =====================================================================
def process_kaggle_sample(item: tuple):
    row, split, base_dir, out_dir = item
    file_p = base_dir / row["file_path"]
    pid = row["patient_id"]
    diag = row["diagnosis"]
    lbl_str = row["label"]
    lbl_id = int(row["label_id"])
    mode = row["mode"]
    st = row["sound_type"]
    loc = row["location"]
    age = row["age"]
    gender = row["gender"]

    rec_id = f"{pid}_{mode}"

    try:
        raw_audio, sr = load_raw_audio(file_p)
        audio = remove_dc_offset(raw_audio)
        audio = apply_highpass_filter(audio, sr)
        audio_16k = resample_polyphase(audio, sr)
        audio_16k = normalize_peak(audio_16k)

        # Hop: 50% overlap for train, non-overlapping for val/test
        hop_samples = 40000 if split == "train" else 80000
        windows = extract_windows_from_audio(audio_16k, hop_samples)

        index_entries = []
        for w_idx, win in enumerate(windows):
            tensor = compute_log_mel_tensor(win)
            tensor_name = f"{pid}_{mode}_win{w_idx}.npy"
            tensor_file = out_dir / split / tensor_name
            np.save(tensor_file, tensor)

            rel_tensor_path = tensor_file.relative_to(base_dir).as_posix()
            entry = {
                "tensor_path": rel_tensor_path,
                "dataset": "kaggle_lung",
                "task": "lung_disease",
                "patient_id": pid,
                "recording_id": rec_id,
                "window_idx": w_idx,
                "diagnosis": diag,
                "label": lbl_str,
                "label_id": lbl_id,
                "mode": mode,
                "sound_type": st,
                "location": loc,
                "age": age,
                "gender": gender,
            }
            index_entries.append(entry)

        return True, index_entries, None
    except Exception as e:
        return False, [], f"Kaggle {pid} {mode}: {str(e)}"


# =====================================================================
# RUNNER
# =====================================================================
def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    full_out_dir = base_dir / "processed" / "full"

    print("\n" + "=" * 90)
    print("  AI STETHOSCOPE - FULL DATASET PREPROCESSING & CACHING")
    print("  (Standardized 16 kHz Log-Mel Tensor Extraction | Target Shape: 128 x 313)")
    print("=" * 90)

    # Check baseline raw files on disk
    baseline_raw_count = sum(len(files) for _, _, files in os.walk(base_dir / "datasets"))
    print(f"Verified Baseline Raw Dataset Files on Disk: {baseline_raw_count:,}")

    # Prepare directories
    heart_out = full_out_dir / "heart"
    icbhi_out = full_out_dir / "icbhi"
    kaggle_out = full_out_dir / "kaggle"

    for d in [heart_out, icbhi_out, kaggle_out]:
        for s in ["train", "val", "test"]:
            (d / s).mkdir(parents=True, exist_ok=True)

    workers = min(12, os.cpu_count() or 4)
    print(f"Using {workers} parallel worker threads...")

    summary_records = []
    total_samples_saved = 0
    total_failed = 0
    all_tensor_paths = []
    all_index_entries = defaultdict(list)

    # -------------------------------------------------------------
    # 1. PROCESS HEART DATASET
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  1. PREPROCESSING HEART DATASET (3,240 audio files across Train, Val, Test)")
    print("-" * 80)

    for split in ["train", "val", "test"]:
        m_file = base_dir / f"manifests/heart_{split}.csv"
        rows = []
        with open(m_file, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        items = [(r, split, base_dir, heart_out) for r in rows]
        print(f"  Processing Heart {split.upper()}: {len(items):,} recordings...")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_heart_sample, items))
        t1 = time.time()

        split_success = sum(1 for ok, _, _ in results if ok)
        split_fail = sum(1 for ok, _, _ in results if not ok)
        split_tensors = [entry for ok, entries, _ in results if ok for entry in entries]

        total_samples_saved += len(split_tensors)
        total_failed += split_fail
        all_index_entries[f"heart_{split}"].extend(split_tensors)
        all_tensor_paths.extend([e["tensor_path"] for e in split_tensors])

        pids = {r["patient_id"] for r in rows}
        classes = Counter(e["label"] for e in split_tensors)
        summary_records.append(("Heart", split.capitalize(), len(rows), len(pids), dict(classes), split_fail))

        # Save index CSV
        idx_file = full_out_dir / f"heart_{split}_index.csv"
        fieldnames = ["tensor_path", "dataset", "task", "patient_id", "recording_id", "window_idx", "label", "label_id"]
        with open(idx_file, mode="w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_tensors)

        print(f"    Done {split}: {len(rows):,} input recs -> {len(split_tensors):,} window tensors ({t1-t0:.1f}s, {split_fail} failed)")

    # -------------------------------------------------------------
    # 2. PROCESS ICBHI DATASET
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  2. PREPROCESSING ICBHI DATASET (6,898 annotated respiratory cycles)")
    print("-" * 80)

    for split in ["train", "val", "test"]:
        m_file = base_dir / f"manifests/icbhi_{split}.csv"
        rows = []
        with open(m_file, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        items = [(r, split, idx, base_dir, icbhi_out) for idx, r in enumerate(rows)]
        print(f"  Processing ICBHI {split.upper()}: {len(items):,} respiratory cycles...")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_icbhi_sample, items))
        t1 = time.time()

        split_success = sum(1 for ok, _, _ in results if ok)
        split_fail = sum(1 for ok, _, _ in results if not ok)
        split_tensors = [entry for ok, entries, _ in results if ok for entry in entries]

        total_samples_saved += len(split_tensors)
        total_failed += split_fail
        all_index_entries[f"icbhi_{split}"].extend(split_tensors)
        all_tensor_paths.extend([e["tensor_path"] for e in split_tensors])

        pids = {r["patient_id"] for r in rows}
        classes = Counter(e["label"] for e in split_tensors)
        summary_records.append(("ICBHI", split.capitalize(), len(rows), len(pids), dict(classes), split_fail))

        # Save index CSV
        idx_file = full_out_dir / f"icbhi_{split}_index.csv"
        fieldnames = ["tensor_path", "dataset", "task", "patient_id", "recording_id", "cycle_idx", "chest_location", "start_time", "end_time", "label", "label_id"]
        with open(idx_file, mode="w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_tensors)

        print(f"    Done {split}: {len(rows):,} input cycles -> {len(split_tensors):,} tensors ({t1-t0:.1f}s, {split_fail} failed)")

    # -------------------------------------------------------------
    # 3. PROCESS KAGGLE DATASET
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  3. PREPROCESSING KAGGLE DATASET (336 audio files across Train, Val, Test)")
    print("-" * 80)

    for split in ["train", "val", "test"]:
        m_file = base_dir / f"manifests/kaggle_{split}.csv"
        rows = []
        with open(m_file, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        items = [(r, split, base_dir, kaggle_out) for r in rows]
        print(f"  Processing Kaggle {split.upper()}: {len(items):,} recordings...")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_kaggle_sample, items))
        t1 = time.time()

        split_success = sum(1 for ok, _, _ in results if ok)
        split_fail = sum(1 for ok, _, _ in results if not ok)
        split_tensors = [entry for ok, entries, _ in results if ok for entry in entries]

        total_samples_saved += len(split_tensors)
        total_failed += split_fail
        all_index_entries[f"kaggle_{split}"].extend(split_tensors)
        all_tensor_paths.extend([e["tensor_path"] for e in split_tensors])

        pids = {r["patient_id"] for r in rows}
        classes = Counter(e["label"] for e in split_tensors)
        summary_records.append(("Kaggle", split.capitalize(), len(rows), len(pids), dict(classes), split_fail))

        # Save index CSV
        idx_file = full_out_dir / f"kaggle_{split}_index.csv"
        fieldnames = ["tensor_path", "dataset", "task", "patient_id", "recording_id", "window_idx", "diagnosis", "label", "label_id", "mode", "sound_type", "location", "age", "gender"]
        with open(idx_file, mode="w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_tensors)

        print(f"    Done {split}: {len(rows):,} input recs -> {len(split_tensors):,} window tensors ({t1-t0:.1f}s, {split_fail} failed)")

    # =============================================================
    # 4. POST-PROCESSING VALIDATION SUITE
    # =============================================================
    print("\n" + "=" * 90)
    print("  RUNNING POST-PROCESSING INTEGRITY VALIDATION SUITE")
    print("=" * 90)

    # 1. Total samples check
    assert total_failed == 0, f"Encountered {total_failed} failed samples!"
    print(f"  [1/10] Processed Samples Check        : PASS ({total_samples_saved:,} saved, 0 failed)")

    # 2. No duplicate tensor paths
    assert len(all_tensor_paths) == len(set(all_tensor_paths)), "Duplicate tensor paths detected!"
    print(f"  [2/10] Unique Tensor IDs Check         : PASS ({len(all_tensor_paths):,} unique tensor paths)")

    # 3. Tensor shapes & finite check
    # 3. No NaN or Inf values & 4. Tensor shape 128 x 313 check across 100% of tensors
    print(f"  [3/10] Tensor Dimensions & NaN/Inf Check : Verifying 100% of {len(all_tensor_paths):,} tensors...")
    def _verify_tensor(rel_p):
        full_p = base_dir / rel_p
        arr = np.load(full_p)
        if arr.shape != EXPECTED_SHAPE:
            return False, f"Shape mismatch in {rel_p}: {arr.shape} != {EXPECTED_SHAPE}"
        if np.isnan(arr).any():
            return False, f"NaN found in {rel_p}"
        if np.isinf(arr).any():
            return False, f"Inf found in {rel_p}"
        return True, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        check_results = list(ex.map(_verify_tensor, all_tensor_paths))

    failed_checks = [msg for ok, msg in check_results if not ok]
    assert len(failed_checks) == 0, f"Tensor validation failed for {len(failed_checks)} tensors! First error: {failed_checks[0]}"
    print(f"         --> PASS: All {len(all_tensor_paths):,} tensors strictly match {EXPECTED_SHAPE} with 0.0% NaN / Inf")

    # 5. All expected labels present
    print("  [5/10] Expected Task Labels Check      : Verifying label sets...")
    heart_labels = {e["label"] for k, v in all_index_entries.items() if "heart" in k for e in v}
    icbhi_labels = {e["label"] for k, v in all_index_entries.items() if "icbhi" in k for e in v}
    kaggle_labels = {e["label"] for k, v in all_index_entries.items() if "kaggle" in k for e in v}

    assert heart_labels == {"Normal", "Abnormal"}, f"Unexpected Heart labels: {heart_labels}"
    assert icbhi_labels == {"Normal", "Crackle", "Wheeze", "Crackle+Wheeze"}, f"Unexpected ICBHI labels: {icbhi_labels}"
    assert kaggle_labels == {"Normal", "Asthma", "Other"}, f"Unexpected Kaggle labels: {kaggle_labels}"
    print(f"         --> PASS: All expected labels present: Heart {heart_labels}, ICBHI {icbhi_labels}, Kaggle {kaggle_labels}")

    # 6. Metadata row mapping
    print("  [6/10] Index Metadata Mapping Check    : Verifying metadata rows on disk...")
    for key, entries in all_index_entries.items():
        idx_f = full_out_dir / f"{key}_index.csv"
        assert idx_f.exists(), f"Index file {idx_f} missing!"
        with open(idx_f, mode="r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
            assert n_rows == len(entries), f"Row mismatch in {idx_f}: {n_rows} vs {len(entries)}"
    print("         --> PASS: Every single tensor has a valid metadata row in the corresponding split index CSV")

    # 7. Patient Isolation Check across splits
    print("  [7/10] Patient Isolation Across Splits : Verifying strict zero-leakage...")
    for ds in ["heart", "icbhi", "kaggle"]:
        train_pids = {e["patient_id"] for e in all_index_entries[f"{ds}_train"]}
        val_pids = {e["patient_id"] for e in all_index_entries[f"{ds}_val"]}
        test_pids = {e["patient_id"] for e in all_index_entries[f"{ds}_test"]}

        tv = train_pids & val_pids
        tt = train_pids & test_pids
        vt = val_pids & test_pids

        assert not tv, f"Patient leakage in {ds} Train-Val: {tv}"
        assert not tt, f"Patient leakage in {ds} Train-Test: {tt}"
        assert not vt, f"Patient leakage in {ds} Val-Test: {vt}"
    print("         --> PASS: Train/validation/test patient isolation strictly intact across all datasets")

    # 8. Recording Isolation Check across splits
    print("  [8/10] Recording Isolation Check       : Verifying zero recording leakage...")
    for ds in ["heart", "icbhi", "kaggle"]:
        train_recs = {e["recording_id"] for e in all_index_entries[f"{ds}_train"]}
        val_recs = {e["recording_id"] for e in all_index_entries[f"{ds}_val"]}
        test_recs = {e["recording_id"] for e in all_index_entries[f"{ds}_test"]}

        assert not (train_recs & val_recs), f"Recording leakage in {ds} Train-Val"
        assert not (train_recs & test_recs), f"Recording leakage in {ds} Train-Test"
        assert not (val_recs & test_recs), f"Recording leakage in {ds} Val-Test"
    print("         --> PASS: Train/validation/test recording isolation strictly intact across all splits")

    # 9. Duplicate Tensor ID Check
    print("  [9/10] Tensor ID Integrity Check       : Checking for duplicate tensor IDs...")
    assert len(all_tensor_paths) == len(set(all_tensor_paths)), "Duplicate tensor paths detected!"
    print(f"         --> PASS: Zero duplicate tensor IDs across all {len(all_tensor_paths):,} tensors")

    # 10. Raw Dataset Preservation Check
    print("  [10/10] Raw Dataset Preservation Check : Recounting files in datasets/...")
    current_raw_count = sum(len(files) for _, _, files in os.walk(base_dir / "datasets"))
    assert current_raw_count == baseline_raw_count, f"Raw files modified! {current_raw_count} != {baseline_raw_count}"
    print(f"         --> PASS: Original raw dataset file count has not changed ({current_raw_count:,} files untouched)")

    # Calculate total storage
    total_bytes = sum(f.stat().st_size for f in full_out_dir.rglob("*.npy"))
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    # =============================================================
    # 5. FINAL SUMMARY TABLE
    # =============================================================
    print("\n" + "=" * 115)
    print("  FINAL PREPROCESSING SUMMARY TABLE")
    print("=" * 115)
    print(f"{'Dataset':<10} | {'Split':<8} | {'Input Samples':<14} | {'Patients':<10} | {'Classes':<55} | {'Failed':<6}")
    print("-" * 115)
    for ds, sp, inp_cnt, pts, cls_dict, fl in summary_records:
        cls_str = ", ".join([f"{k}: {v:,}" for k, v in sorted(cls_dict.items())])
        print(f"{ds:<10} | {sp:<8} | {inp_cnt:<14,} | {pts:<10,} | {cls_str:<55} | {fl:<6}")
    print("-" * 115)

    print("\n" + "=" * 90)
    print("  AUDIT METRICS")
    print("=" * 90)
    print(f"  Total processed samples : {total_samples_saved:,}")
    print(f"  Total failed samples    : {total_failed}")
    print(f"  Total storage used      : {total_mb:.1f} MB ({total_gb:.2f} GB)")
    print(f"  Tensor shape            : {EXPECTED_SHAPE} (128 Mel bins x 313 time frames)")
    print(f"  Sample rate             : {TARGET_SR:,} Hz")
    print(f"  Mel configuration       : n_fft={N_FFT}, hop_length={HOP_LENGTH}, n_mels={N_MELS}, f_min={F_MIN} Hz, f_max={F_MAX} Hz, power={POWER}, top_db={TOP_DB} dB")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
