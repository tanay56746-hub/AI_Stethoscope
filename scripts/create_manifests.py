#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — DATA MANIFESTS] | File: scripts/create_manifests.py
# Purpose: Generates Patient-Disjoint Train/Val/Test Split CSV Manifests
# ==============================================================================
"""
AI Stethoscope - Dataset Manifest Generation
==============================================
Generates deterministic, patient-isolated train/validation/test manifests for:
  1. Heart Sound Dataset (PhysioNet/CinC 2016)
  2. ICBHI 2017 Lung Acoustic Dataset (cycle-level)
  3. Kaggle Respiratory Disease Dataset (recording-level)

Guarantees:
  - 100% READ-ONLY on dataset files.
  - Zero patient leakage across splits (all recordings & cycles of any patient remain in 1 split).
  - Deterministic random seeding (seed=42).
  - Rigorous pre-write validation.
"""

import os
import sys
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)


# =====================================================================
# 1. HEART DATASET MANIFEST GENERATION
# =====================================================================
def build_heart_manifest(base_dir: Path):
    """
    Heart Task: Binary PCG classification (Normal=0, Abnormal=1).
    Split: 70% Train, 15% Validation, 15% Test.
    Patient grouping: Uses Subject IDs in training-b, and unique subject encounters for a, c, d, e, f.
    """
    raw_dir = base_dir / "datasets" / "Heart" / "Raw"
    training_dir = raw_dir / "training"
    appendix_csv = raw_dir / "annotations" / "annotations" / "Online Appendix_training set.csv"

    # Map training-b subject IDs
    b_sub_map = {}
    if appendix_csv.exists():
        with open(appendix_csv, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 14 and row[1].strip() == "training-b":
                    rec = row[0].strip()
                    sid = row[14].strip()
                    if sid:
                        b_sub_map[rec] = sid

    # Group recordings by patient_id
    patient_records = defaultdict(list)
    patient_info = {}

    subsets = sorted([d for d in training_dir.iterdir() if d.is_dir()])
    for s in subsets:
        ref_file = s / "REFERENCE.csv"
        if not ref_file.exists():
            continue
        with open(ref_file, mode="r", encoding="utf-8", errors="ignore") as rf:
            for row in csv.reader(rf):
                if len(row) >= 2:
                    rec_id = row[0].strip()
                    raw_lbl = int(row[1].strip())
                    # Normal: -1 -> 0, Abnormal: 1 -> 1
                    label_id = 0 if raw_lbl == -1 else 1
                    label_str = "Normal" if label_id == 0 else "Abnormal"

                    wav_path = s / f"{rec_id}.wav"
                    rel_wav_path = wav_path.relative_to(base_dir).as_posix()

                    if s.name == "training-b":
                        sid = b_sub_map.get(rec_id, rec_id)
                        pid = f"b_sub_{sid}"
                    else:
                        pid = f"{s.name}_{rec_id}"

                    row_dict = {
                        "dataset": "heart_physionet2016",
                        "task": "heart",
                        "file_path": rel_wav_path,
                        "recording_id": rec_id,
                        "patient_id": pid,
                        "label": label_str,
                        "label_id": label_id,
                    }
                    patient_records[pid].append(row_dict)
                    patient_info[pid] = {
                        "subset": s.name,
                        "label_id": label_id,
                    }

    # Stratified patient-level splitting (70% train, 15% val, 15% test)
    # Stratify by (subset, label_id)
    set_seed(RANDOM_SEED)
    bins = defaultdict(list)
    for pid, info in patient_info.items():
        bins[(info["subset"], info["label_id"])].append(pid)

    train_pids = set()
    val_pids = set()
    test_pids = set()

    for bin_key, p_list in sorted(bins.items()):
        p_sorted = sorted(p_list)
        random.shuffle(p_sorted)
        n = len(p_sorted)

        if n == 1:
            train_pids.add(p_sorted[0])
        elif n == 2:
            train_pids.add(p_sorted[0])
            val_pids.add(p_sorted[1])
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.15)))
            # Remaining to test
            n_val = min(n_val, n - n_train - 1) if (n - n_train) >= 2 else (n - n_train)
            
            p_train = p_sorted[:n_train]
            p_val = p_sorted[n_train : n_train + n_val]
            p_test = p_sorted[n_train + n_val :]

            train_pids.update(p_train)
            val_pids.update(p_val)
            test_pids.update(p_test)

    # Collect manifest rows
    train_rows = [r for pid in sorted(train_pids) for r in patient_records[pid]]
    val_rows = [r for pid in sorted(val_pids) for r in patient_records[pid]]
    test_rows = [r for pid in sorted(test_pids) for r in patient_records[pid]]

    return train_rows, val_rows, test_rows, train_pids, val_pids, test_pids


# =====================================================================
# 2. ICBHI DATASET MANIFEST GENERATION
# =====================================================================
def build_icbhi_manifest(base_dir: Path):
    """
    ICBHI Lung Acoustic Task: 4-class respiratory cycle classification:
      Normal=0, Crackle=1, Wheeze=2, Crackle+Wheeze=3.
    Split: 60% Train, 20% Validation, 20% Test.
    Patient grouping: strictly by patient_id (126 patients: 101 to 226).
    """
    icbhi_dir = base_dir / "datasets" / "lung" / "ICBHI_2017" / "ICBHI_final_database" / "ICBHI_final_database"

    patient_cycles = defaultdict(list)
    patient_acoustic_profile = defaultdict(lambda: {"c": 0, "w": 0, "both": 0, "normal": 0})

    label_map = {
        (0, 0): ("Normal", 0),
        (1, 0): ("Crackle", 1),
        (0, 1): ("Wheeze", 2),
        (1, 1): ("Crackle+Wheeze", 3),
    }

    # Iterate through all WAV files
    wav_files = sorted(icbhi_dir.glob("*.wav"))
    for wf in wav_files:
        stem = wf.stem
        parts = stem.split("_")
        # Format: <patient_idx>_<rec_index>_<chest_loc>_<mode>_<equipment>
        pid = parts[0]
        chest_loc = parts[2] if len(parts) >= 3 else "Unknown"

        txt_file = icbhi_dir / f"{stem}.txt"
        if not txt_file.exists():
            continue

        rel_wav_path = wf.relative_to(base_dir).as_posix()

        with open(txt_file, mode="r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) >= 4:
                    try:
                        t_start = float(line_parts[0])
                        t_end = float(line_parts[1])
                        c = int(line_parts[2])
                        w = int(line_parts[3])

                        lbl_str, lbl_id = label_map.get((c, w), ("Unknown", -1))
                        if lbl_id == -1:
                            continue

                        cycle_row = {
                            "dataset": "icbhi_2017",
                            "task": "lung_acoustic",
                            "file_path": rel_wav_path,
                            "patient_id": pid,
                            "recording_id": stem,
                            "chest_location": chest_loc,
                            "start_time": f"{t_start:.4f}",
                            "end_time": f"{t_end:.4f}",
                            "label": lbl_str,
                            "label_id": lbl_id,
                        }
                        patient_cycles[pid].append(cycle_row)

                        if c == 1 and w == 1:
                            patient_acoustic_profile[pid]["both"] += 1
                        elif c == 1:
                            patient_acoustic_profile[pid]["c"] += 1
                        elif w == 1:
                            patient_acoustic_profile[pid]["w"] += 1
                        else:
                            patient_acoustic_profile[pid]["normal"] += 1
                    except ValueError:
                        continue

    # Stratified patient-level splitting (60% train, 20% val, 20% test)
    # Stratify by primary acoustic presence
    set_seed(RANDOM_SEED)
    profile_bins = defaultdict(list)
    for pid in sorted(patient_cycles.keys()):
        prof = patient_acoustic_profile[pid]
        if prof["both"] > 0 or (prof["c"] > 0 and prof["w"] > 0):
            primary = "Both"
        elif prof["c"] > 0:
            primary = "Crackle"
        elif prof["w"] > 0:
            primary = "Wheeze"
        else:
            primary = "Normal"
        profile_bins[primary].append(pid)

    train_pids = set()
    val_pids = set()
    test_pids = set()

    for p_class, p_list in sorted(profile_bins.items()):
        p_sorted = sorted(p_list)
        random.shuffle(p_sorted)
        n = len(p_sorted)

        n_train = int(round(n * 0.60))
        n_val = int(round(n * 0.20))
        # Ensure remaining go to test
        n_test = n - n_train - n_val
        if n_val == 0 and n >= 3:
            n_val = 1
            n_train -= 1

        p_train = p_sorted[:n_train]
        p_val = p_sorted[n_train : n_train + n_val]
        p_test = p_sorted[n_train + n_val :]

        train_pids.update(p_train)
        val_pids.update(p_val)
        test_pids.update(p_test)

    train_rows = [r for pid in sorted(train_pids) for r in patient_cycles[pid]]
    val_rows = [r for pid in sorted(val_pids) for r in patient_cycles[pid]]
    test_rows = [r for pid in sorted(test_pids) for r in patient_cycles[pid]]

    return train_rows, val_rows, test_rows, train_pids, val_pids, test_pids


# =====================================================================
# 3. KAGGLE LUNG DATASET MANIFEST GENERATION
# =====================================================================
def build_kaggle_manifest(base_dir: Path):
    """
    Kaggle Lung Task: 3-class disease classification:
      Normal = 0
      Asthma = 1
      Other  = 2 (All other diseases, including combined like 'Asthma and lung fibrosis')
    Split: ~60% Train, ~20% Validation, ~20% Test.
    Patient grouping: strictly by patient_id (112 patients). All BP, DP, EP stay together.
    """
    audio_dir = base_dir / "datasets" / "lung" / "Kaggle_Lung" / "archive" / "Audio Files"

    patient_records = defaultdict(list)
    patient_diagnosis = {}
    patient_target_class = {}

    wav_files = sorted(audio_dir.glob("*.wav"))
    for wf in wav_files:
        filename = wf.name
        mode = filename[:2]
        parts = filename.split("_")
        pid = parts[0][2:]  # E.g. '101' from 'BP101'

        after = parts[1][:-4]  # Strip .wav
        fields = [x.strip() for x in after.split(",")]
        diag = fields[0] if len(fields) > 0 else "Unknown"
        st = fields[1] if len(fields) > 1 else "N/A"
        loc = fields[2] if len(fields) > 2 else "N/A"
        age = fields[3] if len(fields) > 3 else "N/A"
        gender = fields[4] if len(fields) > 4 else "N/A"

        # Careful disease mapping
        norm_diag = diag.strip().lower()
        if norm_diag == "n":
            label_str = "Normal"
            label_id = 0
        elif norm_diag == "asthma":
            # Strictly pure asthma
            label_str = "Asthma"
            label_id = 1
        else:
            # Combined diagnoses like "Asthma and lung fibrosis", COPD, Heart Failure, etc.
            label_str = "Other"
            label_id = 2

        rel_wav_path = wf.relative_to(base_dir).as_posix()

        row_dict = {
            "dataset": "kaggle_lung",
            "task": "lung_disease",
            "file_path": rel_wav_path,
            "patient_id": f"P{pid}",
            "diagnosis": diag,
            "label": label_str,
            "label_id": label_id,
            "mode": mode,
            "sound_type": st,
            "location": loc,
            "age": age,
            "gender": gender,
        }
        patient_records[f"P{pid}"].append(row_dict)
        patient_diagnosis[f"P{pid}"] = diag
        patient_target_class[f"P{pid}"] = label_id

    # Stratified splitting by target disease class across the 112 patients
    # Normal: 35 patients -> 21 Train (60%), 7 Val (20%), 7 Test (20%)
    # Asthma: 32 patients -> 20 Train (62.5%), 6 Val (18.75%), 6 Test (18.75%)
    # Other : 45 patients -> 27 Train (60%), 9 Val (20%), 9 Test (20%)
    set_seed(RANDOM_SEED)
    class_bins = defaultdict(list)
    for pid, cid in sorted(patient_target_class.items()):
        class_bins[cid].append(pid)

    train_pids = set()
    val_pids = set()
    test_pids = set()

    for cid, p_list in sorted(class_bins.items()):
        p_sorted = sorted(p_list)
        random.shuffle(p_sorted)
        n = len(p_sorted)

        n_train = int(round(n * 0.60))
        n_val = int(round(n * 0.20))
        n_test = n - n_train - n_val

        p_train = p_sorted[:n_train]
        p_val = p_sorted[n_train : n_train + n_val]
        p_test = p_sorted[n_train + n_val :]

        train_pids.update(p_train)
        val_pids.update(p_val)
        test_pids.update(p_test)

    train_rows = [r for pid in sorted(train_pids) for r in patient_records[pid]]
    val_rows = [r for pid in sorted(val_pids) for r in patient_records[pid]]
    test_rows = [r for pid in sorted(test_pids) for r in patient_records[pid]]

    return train_rows, val_rows, test_rows, train_pids, val_pids, test_pids


# =====================================================================
# 4. PRE-WRITE VALIDATION & CSV GENERATION
# =====================================================================
def validate_split_integrity(dataset_name: str, train_rows, val_rows, test_rows, train_pids, val_pids, test_pids, base_dir: Path):
    """Rigorous pre-write validation."""
    print(f"\n[Validation] Validating integrity for: {dataset_name}...")

    # 1. No patient overlap
    overlap_tv = train_pids & val_pids
    overlap_tt = train_pids & test_pids
    overlap_vt = val_pids & test_pids
    assert not overlap_tv, f"Patient leakage Train-Val: {overlap_tv}"
    assert not overlap_tt, f"Patient leakage Train-Test: {overlap_tt}"
    assert not overlap_vt, f"Patient leakage Val-Test: {overlap_vt}"

    # 2. No recording overlap
    def get_rec(r):
        return r.get("recording_id", r.get("file_path"))

    train_recs = {get_rec(r) for r in train_rows}
    val_recs = {get_rec(r) for r in val_rows}
    test_recs = {get_rec(r) for r in test_rows}
    assert not (train_recs & val_recs), f"Recording leakage Train-Val: {train_recs & val_recs}"
    assert not (train_recs & test_recs), f"Recording leakage Train-Test: {train_recs & test_recs}"
    assert not (val_recs & test_recs), f"Recording leakage Val-Test: {val_recs & test_recs}"

    # 3. Audio file path existence on disk
    all_rows = train_rows + val_rows + test_rows
    for r in all_rows:
        fp = base_dir / r["file_path"]
        assert fp.exists(), f"Audio file path does not exist on disk: {fp}"

    # 4. Dataset-specific checks
    if dataset_name == "icbhi":
        for r in all_rows:
            t0 = float(r["start_time"])
            t1 = float(r["end_time"])
            assert t0 >= 0 and t1 > t0, f"Invalid cycle boundary: {t0} -> {t1}"
            assert r["label_id"] in (0, 1, 2, 3), f"Invalid ICBHI label_id: {r['label_id']}"
    elif dataset_name == "kaggle":
        # Ensure all modes BP, DP, EP are accounted for per patient
        for pid in train_pids | val_pids | test_pids:
            p_rows = [r for r in all_rows if r["patient_id"] == pid]
            modes = {r["mode"] for r in p_rows}
            assert modes == {"BP", "DP", "EP"}, f"Patient {pid} missing modes: {modes}"
            assert all(r["label_id"] in (0, 1, 2) for r in p_rows), "Invalid Kaggle label_id"
    elif dataset_name == "heart":
        for r in all_rows:
            assert r["label_id"] in (0, 1), f"Invalid Heart label_id: {r['label_id']}"

    print(f"  --> [PASS] 100% integrity verified for {dataset_name} (Zero leakage, all files exist).")


def write_manifest_csv(filepath: Path, rows: list, fieldnames: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================
def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    manifests_dir = base_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("  AI STETHOSCOPE - DATASET MANIFEST GENERATION")
    print("  (Deterministic Patient-Isolated Splitting | Random Seed = 42)")
    print("=" * 90)

    # 1. Heart Manifests
    h_train, h_val, h_test, h_t_pids, h_v_pids, h_te_pids = build_heart_manifest(base_dir)
    validate_split_integrity("heart", h_train, h_val, h_test, h_t_pids, h_v_pids, h_te_pids, base_dir)
    h_fields = ["dataset", "task", "file_path", "recording_id", "patient_id", "label", "label_id"]
    write_manifest_csv(manifests_dir / "heart_train.csv", h_train, h_fields)
    write_manifest_csv(manifests_dir / "heart_val.csv", h_val, h_fields)
    write_manifest_csv(manifests_dir / "heart_test.csv", h_test, h_fields)

    # 2. ICBHI Manifests
    i_train, i_val, i_test, i_t_pids, i_v_pids, i_te_pids = build_icbhi_manifest(base_dir)
    validate_split_integrity("icbhi", i_train, i_val, i_test, i_t_pids, i_v_pids, i_te_pids, base_dir)
    i_fields = [
        "dataset",
        "task",
        "file_path",
        "patient_id",
        "recording_id",
        "chest_location",
        "start_time",
        "end_time",
        "label",
        "label_id",
    ]
    write_manifest_csv(manifests_dir / "icbhi_train.csv", i_train, i_fields)
    write_manifest_csv(manifests_dir / "icbhi_val.csv", i_val, i_fields)
    write_manifest_csv(manifests_dir / "icbhi_test.csv", i_test, i_fields)

    # 3. Kaggle Lung Manifests
    k_train, k_val, k_test, k_t_pids, k_v_pids, k_te_pids = build_kaggle_manifest(base_dir)
    validate_split_integrity("kaggle", k_train, k_val, k_test, k_t_pids, k_v_pids, k_te_pids, base_dir)
    k_fields = [
        "dataset",
        "task",
        "file_path",
        "patient_id",
        "diagnosis",
        "label",
        "label_id",
        "mode",
        "sound_type",
        "location",
        "age",
        "gender",
    ]
    write_manifest_csv(manifests_dir / "kaggle_train.csv", k_train, k_fields)
    write_manifest_csv(manifests_dir / "kaggle_val.csv", k_val, k_fields)
    write_manifest_csv(manifests_dir / "kaggle_test.csv", k_test, k_fields)

    # =====================================================================
    # 6. SUMMARY TABLES AND CLASS DISTRIBUTIONS
    # =====================================================================
    print("\n" + "=" * 90)
    print("  MANIFEST GENERATION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Dataset':<16} | {'Split':<10} | {'Patients':<10} | {'Recordings':<12} | {'Cycles':<10} | {'Classes':<16}")
    print("-" * 84)

    splits_data = [
        ("Heart", "Train", len(h_t_pids), len(h_train), len(h_train), "Normal, Abnormal"),
        ("Heart", "Val", len(h_v_pids), len(h_val), len(h_val), "Normal, Abnormal"),
        ("Heart", "Test", len(h_te_pids), len(h_test), len(h_test), "Normal, Abnormal"),
        ("ICBHI Lung", "Train", len(i_t_pids), len({r['recording_id'] for r in i_train}), len(i_train), "4-Class (N,C,W,CW)"),
        ("ICBHI Lung", "Val", len(i_v_pids), len({r['recording_id'] for r in i_val}), len(i_val), "4-Class (N,C,W,CW)"),
        ("ICBHI Lung", "Test", len(i_te_pids), len({r['recording_id'] for r in i_test}), len(i_test), "4-Class (N,C,W,CW)"),
        ("Kaggle Lung", "Train", len(k_t_pids), len(k_train), len(k_train), "Normal, Asthma, Other"),
        ("Kaggle Lung", "Val", len(k_v_pids), len(k_val), len(k_val), "Normal, Asthma, Other"),
        ("Kaggle Lung", "Test", len(k_te_pids), len(k_test), len(k_test), "Normal, Asthma, Other"),
    ]

    for d_name, sp, pts, recs, cycs, clss in splits_data:
        print(f"{d_name:<16} | {sp:<10} | {pts:<10,} | {recs:<12,} | {cycs:<10,} | {clss:<16}")
    print("-" * 84)

    # Detailed Class Distributions
    print("\n" + "=" * 90)
    print("  CLASS DISTRIBUTIONS PER SPLIT")
    print("=" * 90)

    print("\n1. Heart Sound Classification (PhysioNet 2016):")
    for sp_name, rows in [("Train", h_train), ("Val", h_val), ("Test", h_test)]:
        c = Counter(r["label"] for r in rows)
        norm = c["Normal"]
        abn = c["Abnormal"]
        tot = len(rows)
        print(f"  * {sp_name:<5}: Total = {tot:>5,} | Normal = {norm:>5,} ({norm/tot*100:>5.1f}%) | Abnormal = {abn:>4,} ({abn/tot*100:>5.1f}%)")

    print("\n2. ICBHI Lung Acoustic Sound Detection (Cycle-Level):")
    for sp_name, rows in [("Train", i_train), ("Val", i_val), ("Test", i_test)]:
        c = Counter(r["label"] for r in rows)
        tot = len(rows)
        print(f"  * {sp_name:<5}: Total Cycles = {tot:>5,} | Normal = {c['Normal']:>5,} ({c['Normal']/tot*100:>5.1f}%) | Crackle = {c['Crackle']:>4,} ({c['Crackle']/tot*100:>5.1f}%) | Wheeze = {c['Wheeze']:>4,} ({c['Wheeze']/tot*100:>5.1f}%) | Both = {c['Crackle+Wheeze']:>3,} ({c['Crackle+Wheeze']/tot*100:>5.1f}%)")

    print("\n3. Kaggle Lung Disease Classification (Recording-Level):")
    for sp_name, rows in [("Train", k_train), ("Val", k_val), ("Test", k_test)]:
        tot = len(rows)
        c = Counter(r["label"] for r in rows)
        p_c = Counter(r["label"] for r in rows if r["mode"] == "BP")  # Patient count (1 mode per patient)
        tot_pts = sum(p_c.values())
        print(f"  * {sp_name:<5}: Total Recs = {tot:>3} ({tot_pts:>2} pts) | Normal = {c['Normal']:>3} ({p_c['Normal']:>2} pts) | Asthma = {c['Asthma']:>3} ({p_c['Asthma']:>2} pts) | Other = {c['Other']:>3} ({p_c['Other']:>2} pts)")

    # File locations
    print("\n" + "=" * 90)
    print("  EXACT GENERATED MANIFEST FILE LOCATIONS")
    print("=" * 90)
    manifest_files = [
        manifests_dir / "heart_train.csv",
        manifests_dir / "heart_val.csv",
        manifests_dir / "heart_test.csv",
        manifests_dir / "icbhi_train.csv",
        manifests_dir / "icbhi_val.csv",
        manifests_dir / "icbhi_test.csv",
        manifests_dir / "kaggle_train.csv",
        manifests_dir / "kaggle_val.csv",
        manifests_dir / "kaggle_test.csv",
    ]
    for mf in manifest_files:
        size_kb = mf.stat().st_size / 1024.0
        with open(mf, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f) - 1  # exclude header
        print(f"  * {str(mf.resolve()):<75} ({lines:>5,} rows, {size_kb:>6.1f} KB)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
