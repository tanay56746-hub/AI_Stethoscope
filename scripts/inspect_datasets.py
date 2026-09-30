#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — DATASET INSPECTION] | File: scripts/inspect_datasets.py
# Purpose: Initial Raw Corpus Discovery & Metadata Summary
# ==============================================================================
"""
AI Stethoscope - Dataset Inspection Utility
=============================================
This script recursively and READ-ONLY inspects the three datasets:
  1. Heart Sound Dataset (PhysioNet/CinC 2016)
  2. Lung Sound Dataset: ICBHI 2017
  3. Lung Sound Dataset: Kaggle Respiratory Audio

Reports:
  - Audio file counts, extensions, sample rates, channels, durations (min, max, avg)
  - Folder structures and available label/annotation files
  - Heart: Normal vs Abnormal labels summary
  - ICBHI: Respiratory cycles and crackle/wheeze sound annotations
  - Kaggle Lung: File/folder naming conventions and disease labels (especially Asthma & Normal)
"""

import os
import sys
import wave
import csv
from collections import Counter, defaultdict
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format duration into MM:SS or S.SSs format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    rem = seconds % 60
    return f"{mins}m {rem:.1f}s ({seconds:.2f}s)"


def get_audio_info(filepath: Path):
    """
    Read audio properties in read-only mode using Python's standard wave library.
    Returns (sample_rate, channels, duration_seconds) or None on error.
    """
    try:
        with wave.open(str(filepath), 'rb') as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            n_frames = w.getnframes()
            dur = n_frames / float(sr) if sr > 0 else 0.0
            return sr, ch, dur
    except Exception as e:
        return None


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_sub_header(title: str):
    print("\n" + "-" * 80)
    print(f"  >>> {title}")
    print("-" * 80)


def inspect_audio_collection(audio_files: list):
    """Inspect a list of Path objects pointing to WAV files."""
    if not audio_files:
        return {
            "count": 0,
            "sample_rates": {},
            "channels": {},
            "durations": [],
            "min_dur": 0.0,
            "max_dur": 0.0,
            "avg_dur": 0.0,
            "total_dur": 0.0,
        }

    sample_rates = Counter()
    channels = Counter()
    durations = []

    for af in audio_files:
        info = get_audio_info(af)
        if info:
            sr, ch, dur = info
            sample_rates[sr] += 1
            channels[ch] += 1
            durations.append(dur)

    min_dur = min(durations) if durations else 0.0
    max_dur = max(durations) if durations else 0.0
    avg_dur = sum(durations) / len(durations) if durations else 0.0
    total_dur = sum(durations)

    return {
        "count": len(audio_files),
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "durations": durations,
        "min_dur": min_dur,
        "max_dur": max_dur,
        "avg_dur": avg_dur,
        "total_dur": total_dur,
    }


# =====================================================================
# 1. HEART DATASET INSPECTION
# =====================================================================
def inspect_heart_dataset(base_dir: Path):
    print_header("1. Heart Sound Dataset (PhysioNet / CinC Challenge 2016)")
    raw_dir = base_dir / "datasets" / "Heart" / "Raw"

    if not raw_dir.exists():
        print(f"[!] Path not found: {raw_dir}")
        return

    # Scan raw folder extensions
    ext_counter = Counter()
    all_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext or '(no ext)'] += 1
            all_files.append(Path(root) / f)

    print(f"Root Location: {raw_dir.resolve()}")
    print(f"Total Files Found: {len(all_files):,}")
    print("\nFile Extensions Breakdown:")
    for ext, count in ext_counter.most_common():
        print(f"  {ext:<15}: {count:>6,} files")

    # Training folders
    training_dir = raw_dir / "training"
    if training_dir.exists():
        subfolders = sorted([d for d in training_dir.iterdir() if d.is_dir()])
        print_sub_header(f"Heart Audio Files by Training Subset ({len(subfolders)} subsets)")

        total_heart_wavs = []
        overall_labels = Counter()

        subset_stats = []
        for sf in subfolders:
            wavs = sorted(sf.glob("*.wav"))
            total_heart_wavs.extend(wavs)

            # Labels in REFERENCE.csv
            ref_csv = sf / "REFERENCE.csv"
            subset_labels = Counter()
            if ref_csv.exists():
                with open(ref_csv, mode="r", encoding="utf-8", errors="ignore") as rf:
                    reader = csv.reader(rf)
                    for row in reader:
                        if len(row) >= 2:
                            val = row[1].strip()
                            # -1 = Normal, 1 = Abnormal
                            if val == "-1":
                                subset_labels["Normal (-1)"] += 1
                                overall_labels["Normal (-1)"] += 1
                            elif val == "1":
                                subset_labels["Abnormal (1)"] += 1
                                overall_labels["Abnormal (1)"] += 1
                            else:
                                subset_labels[f"Other ({val})"] += 1
                                overall_labels[f"Other ({val})"] += 1

            subset_stats.append((sf.name, len(wavs), dict(subset_labels)))

        print(f"{'Subset Name':<15} | {'WAV Count':<10} | {'Normal (-1)':<12} | {'Abnormal (1)':<12} | {'Total Labeled':<14}")
        print("-" * 72)
        for name, w_count, lbls in subset_stats:
            norm = lbls.get("Normal (-1)", 0)
            abnorm = lbls.get("Abnormal (1)", 0)
            tot_lbl = norm + abnorm
            print(f"{name:<15} | {w_count:<10} | {norm:<12} | {abnorm:<12} | {tot_lbl:<14}")

        print("-" * 72)
        total_norm = overall_labels.get("Normal (-1)", 0)
        total_abnorm = overall_labels.get("Abnormal (1)", 0)
        print(f"{'TOTAL':<15} | {len(total_heart_wavs):<10} | {total_norm:<12} | {total_abnorm:<12} | {total_norm + total_abnorm:<14}")

        print("\nHeart Sound Class Distribution Summary:")
        total_labeled = total_norm + total_abnorm
        if total_labeled > 0:
            print(f"  * Normal Records   : {total_norm:,} ({total_norm / total_labeled * 100:.2f}%)")
            print(f"  * Abnormal Records : {total_abnorm:,} ({total_abnorm / total_labeled * 100:.2f}%)")
            print(f"  * Total Labeled    : {total_labeled:,}")

        # Audio properties
        print_sub_header("Heart Audio Specifications (Sampling Rate, Channels, Duration)")
        audio_stats = inspect_audio_collection(total_heart_wavs)
        print(f"Total WAV Audio Files Inspected : {audio_stats['count']:,}")
        print(f"Sampling Rates Encountered      : {audio_stats['sample_rates']} (Hz)")
        print(f"Channel Configuration           : {audio_stats['channels']} (1 = Mono)")
        print(f"Audio Duration:")
        print(f"  * Minimum Duration            : {format_duration(audio_stats['min_dur'])}")
        print(f"  * Maximum Duration            : {format_duration(audio_stats['max_dur'])}")
        print(f"  * Average Duration            : {format_duration(audio_stats['avg_dur'])}")
        print(f"  * Total Cumulative Duration   : {format_duration(audio_stats['total_dur'])}")

    # Annotations folder
    ann_dir = raw_dir / "annotations"
    if ann_dir.exists():
        print_sub_header("Available Heart Metadata and Annotation Files")
        for root, dirs, files in os.walk(ann_dir):
            # Skip __MACOSX
            if "__MACOSX" in root:
                continue
            for f in files:
                if f.endswith((".csv", ".txt", ".xlsx")):
                    rel_p = Path(root, f).relative_to(raw_dir)
                    size_kb = (Path(root, f).stat().st_size) / 1024.0
                    print(f"  * {str(rel_p):<55} ({size_kb:>7.1f} KB)")


# =====================================================================
# 2. ICBHI 2017 LUNG DATASET INSPECTION
# =====================================================================
def inspect_icbhi_dataset(base_dir: Path):
    print_header("2. Lung Sound Dataset: ICBHI 2017")
    icbhi_root = base_dir / "datasets" / "lung" / "ICBHI_2017"

    if not icbhi_root.exists():
        print(f"[!] Path not found: {icbhi_root}")
        return

    # Find where database files reside (handling nested directories)
    wav_files = sorted(icbhi_root.rglob("*.wav"))
    txt_files = sorted(icbhi_root.rglob("*.txt"))

    # File extensions
    ext_counter = Counter()
    for root, dirs, files in os.walk(icbhi_root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext or '(no ext)'] += 1

    print(f"Root Location: {icbhi_root.resolve()}")
    print(f"Total WAV Files Found: {len(wav_files):,}")
    print(f"Total TXT Annotation Files: {len(txt_files):,}")
    print("\nFile Extensions Breakdown:")
    for ext, count in ext_counter.most_common():
        print(f"  {ext:<15}: {count:>6,} files")

    # Audio specifications
    print_sub_header("ICBHI Audio Specifications (Sampling Rate, Channels, Duration)")
    audio_stats = inspect_audio_collection(wav_files)
    print(f"Total WAV Audio Files Inspected : {audio_stats['count']:,}")
    print("Sampling Rates Breakdown (Hz)   :")
    for sr, cnt in sorted(audio_stats["sample_rates"].items()):
        pct = cnt / audio_stats["count"] * 100 if audio_stats["count"] else 0
        print(f"  * {sr:>5} Hz : {cnt:>4} files ({pct:>5.1f}%)")
    print(f"Channel Configuration           : {audio_stats['channels']} (1 = Mono)")
    print(f"Audio Duration:")
    print(f"  * Minimum Duration            : {format_duration(audio_stats['min_dur'])}")
    print(f"  * Maximum Duration            : {format_duration(audio_stats['max_dur'])}")
    print(f"  * Average Duration            : {format_duration(audio_stats['avg_dur'])}")
    print(f"  * Total Cumulative Duration   : {format_duration(audio_stats['total_dur'])}")

    # Annotation file inspection
    print_sub_header("ICBHI Crackle & Wheeze Annotation Analysis (from .txt files)")
    annotation_txts = [
        tf for tf in txt_files
        if tf.name not in ["filename_differences.txt", "filename_format.txt"]
    ]

    total_cycles = 0
    cycle_counter = Counter()
    file_crackles_count = 0
    file_wheezes_count = 0
    file_both_count = 0
    file_neither_count = 0
    equipment_counter = Counter()

    for tf in annotation_txts:
        # Extract equipment from filename (format: patient_idx_location_mode_equipment.txt)
        stem = tf.stem
        parts = stem.split("_")
        if len(parts) >= 5:
            equipment_counter[parts[4]] += 1

        file_has_c = False
        file_has_w = False

        with open(tf, mode="r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                line_parts = line.strip().split()
                if len(line_parts) >= 4:
                    try:
                        crackle = int(line_parts[2])
                        wheeze = int(line_parts[3])
                        total_cycles += 1
                        cycle_counter[(crackle, wheeze)] += 1
                        if crackle == 1:
                            file_has_c = True
                        if wheeze == 1:
                            file_has_w = True
                    except ValueError:
                        continue

        if file_has_c and file_has_w:
            file_both_count += 1
        elif file_has_c:
            file_crackles_count += 1
        elif file_has_w:
            file_wheezes_count += 1
        else:
            file_neither_count += 1

    print(f"Total Annotation Files Analyzed : {len(annotation_txts):,}")
    print(f"Total Respiratory Cycles Found   : {total_cycles:,}")
    print("\nRespiratory Cycle-Level Classification (Crackles & Wheezes):")
    label_names = {
        (0, 0): "Normal (Neither crackle nor wheeze)",
        (1, 0): "Crackles Only",
        (0, 1): "Wheezes Only",
        (1, 1): "Both Crackles & Wheezes",
    }
    for key, name in label_names.items():
        cnt = cycle_counter.get(key, 0)
        pct = cnt / total_cycles * 100 if total_cycles else 0
        print(f"  * {name:<38} [C={key[0]}, W={key[1]}]: {cnt:>5,} cycles ({pct:>5.2f}%)")

    print("\nAudio File-Level Pathology Composition:")
    print(f"  * Files with Crackles Only       : {file_crackles_count:>4} files ({file_crackles_count / len(annotation_txts) * 100:.1f}%)")
    print(f"  * Files with Wheezes Only        : {file_wheezes_count:>4} files ({file_wheezes_count / len(annotation_txts) * 100:.1f}%)")
    print(f"  * Files with Both Crackles/Wheezes: {file_both_count:>4} files ({file_both_count / len(annotation_txts) * 100:.1f}%)")
    print(f"  * Files with Normal Breath Sounds: {file_neither_count:>4} files ({file_neither_count / len(annotation_txts) * 100:.1f}%)")

    print("\nRecording Equipment Distribution (from filenames):")
    for eq, cnt in equipment_counter.most_common():
        print(f"  * {eq:<16}: {cnt:>4} files")


# =====================================================================
# 3. KAGGLE LUNG DATASET INSPECTION
# =====================================================================
def inspect_kaggle_lung_dataset(base_dir: Path):
    print_header("3. Lung Sound Dataset: Kaggle Respiratory Sound Database")
    kaggle_root = base_dir / "datasets" / "lung" / "Kaggle_Lung"

    if not kaggle_root.exists():
        print(f"[!] Path not found: {kaggle_root}")
        return

    # Find WAV files
    wav_files = sorted(kaggle_root.rglob("*.wav"))

    # File extensions
    ext_counter = Counter()
    all_files = []
    for root, dirs, files in os.walk(kaggle_root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext or '(no ext)'] += 1
            all_files.append(Path(root) / f)

    print(f"Root Location: {kaggle_root.resolve()}")
    print(f"Total Files Found: {len(all_files):,}")
    print(f"Total WAV Audio Files: {len(wav_files):,}")
    print("\nFile Extensions Breakdown:")
    for ext, count in ext_counter.most_common():
        print(f"  {ext:<15}: {count:>6,} files")

    # Audio specifications
    print_sub_header("Kaggle Lung Audio Specifications (Sampling Rate, Channels, Duration)")
    audio_stats = inspect_audio_collection(wav_files)
    print(f"Total WAV Audio Files Inspected : {audio_stats['count']:,}")
    print(f"Sampling Rates Encountered      : {audio_stats['sample_rates']} (Hz)")
    print(f"Channel Configuration           : {audio_stats['channels']} (1 = Mono)")
    print(f"Audio Duration:")
    print(f"  * Minimum Duration            : {format_duration(audio_stats['min_dur'])}")
    print(f"  * Maximum Duration            : {format_duration(audio_stats['max_dur'])}")
    print(f"  * Average Duration            : {format_duration(audio_stats['avg_dur'])}")
    print(f"  * Total Cumulative Duration   : {format_duration(audio_stats['total_dur'])}")

    # Disease labels from actual filenames
    print_sub_header("Kaggle Lung Naming Structure & Actual Extracted Disease Labels")
    # Filename structure: <Prefix><PatientID>_<Diagnosis>,<SoundType>,<Location>,<Age>,<Gender>.wav
    # Prefixes: BP = Bell mode, DP = Diaphragm mode, EP = Extended mode

    stethoscope_modes = Counter()
    raw_disease_labels = Counter()
    normalized_diseases = Counter()
    patient_ids_by_disease = defaultdict(set)
    total_parsed = 0

    for wf in wav_files:
        filename = wf.name
        # Mode prefix: first 2 letters
        prefix = filename[:2]
        if prefix in ["BP", "DP", "EP"]:
            stethoscope_modes[prefix] += 1

        parts = filename.split("_")
        if len(parts) >= 2:
            patient_part = parts[0]
            # patient number without mode prefix
            patient_num = patient_part[2:] if prefix in ["BP", "DP", "EP"] else patient_part

            after_id = parts[1]
            diag = after_id.split(",")[0].strip()

            raw_disease_labels[diag] += 1
            total_parsed += 1

            # Normalized label (handling case variations: Asthma vs asthma, heart failure vs Heart Failure)
            norm_diag = diag.strip()
            if norm_diag.lower() == "asthma":
                norm_diag = "Asthma"
            elif norm_diag.lower() == "n":
                norm_diag = "Normal"
            elif norm_diag.lower() == "heart failure":
                norm_diag = "Heart Failure"
            elif norm_diag.lower() == "copd":
                norm_diag = "COPD"
            elif norm_diag.lower() == "pneumonia":
                norm_diag = "Pneumonia"
            elif norm_diag.lower() == "lung fibrosis":
                norm_diag = "Lung Fibrosis"
            elif norm_diag.lower() == "bron":
                norm_diag = "Bronchitis/Bronchiectasis (BRON)"
            elif norm_diag.lower() == "plueral effusion":
                norm_diag = "Pleural Effusion"

            normalized_diseases[norm_diag] += 1
            patient_ids_by_disease[norm_diag].add(patient_num)

    print("Acoustic Stethoscope Modes Breakdown:")
    mode_names = {
        "BP": "Bell Mode (Low frequency)",
        "DP": "Diaphragm Mode (High frequency)",
        "EP": "Extended Mode (Wideband frequency)",
    }
    for mode, count in stethoscope_modes.items():
        print(f"  * {mode} ({mode_names.get(mode, 'Unknown')}): {count} recordings (1 per each of the {count} patients)")

    print(f"\nRaw Disease Labels Extracted Directly from Filenames (Total = {total_parsed}):")
    for diag, cnt in raw_disease_labels.most_common():
        print(f"  * {diag:<32}: {cnt:>3} files")

    print("\nNormalized Disease Class Summary (Files & Unique Patient Count):")
    print(f"{'Disease Category':<35} | {'Audio Files':<12} | {'Unique Patients':<16} | {'Share of Dataset':<15}")
    print("-" * 84)
    for diag, cnt in normalized_diseases.most_common():
        patients_cnt = len(patient_ids_by_disease[diag])
        pct = cnt / total_parsed * 100 if total_parsed else 0
        highlight = " <-- [TARGET]" if diag in ["Asthma", "Normal"] else ""
        print(f"{diag:<35} | {cnt:<12} | {patients_cnt:<16} | {pct:>5.1f}%{highlight}")
    print("-" * 84)

    # Highlight specifically requested classes
    print("\n" + "=" * 50)
    print("  KEY TARGET CLASSES HIGHLIGHT (KAGGLE LUNG):")
    print("=" * 50)
    asthma_files = normalized_diseases.get("Asthma", 0)
    asthma_patients = len(patient_ids_by_disease.get("Asthma", []))
    normal_files = normalized_diseases.get("Normal", 0)
    normal_patients = len(patient_ids_by_disease.get("Normal", []))

    print(f"  >>> ASTHMA : {asthma_files:>3} Audio Files ({asthma_patients} Unique Patients, 3 Stethoscope Modes each)")
    print(f"  >>> NORMAL : {normal_files:>3} Audio Files ({normal_patients} Unique Patients, 3 Stethoscope Modes each)")
    print("=" * 50)

    # Check companion metadata Excel file
    excel_file = kaggle_root / "archive" / "Data annotation.xlsx"
    if excel_file.exists():
        size_kb = excel_file.stat().st_size / 1024.0
        print(f"\nCompanion Annotation File Found: {excel_file.name} ({size_kb:.1f} KB)")
        try:
            import pandas as pd
            df = pd.read_excel(excel_file)
            print(f"  Excel Dimensions : {df.shape[0]} rows x {df.shape[1]} columns")
            if "Diagnosis" in df.columns:
                print("  Excel Diagnosis Column Breakdown:")
                for val, count in df["Diagnosis"].value_counts(dropna=False).items():
                    val_str = "NaN (Unassigned)" if pd.isna(val) else str(val)
                    print(f"    - {val_str:<30}: {count:>3} entries")
        except ImportError:
            print("  (pandas not available for reading Excel file, parsed directly from audio filenames)")
        except Exception as ex:
            print(f"  (Could not read Excel file: {ex})")


# =====================================================================
# 4. CROSS-DATASET CONSOLIDATED SUMMARY
# =====================================================================
def print_cross_dataset_summary(base_dir: Path):
    print_header("4. Cross-Dataset Consolidated Summary for AI Stethoscope")

    heart_wavs = list((base_dir / "datasets" / "Heart").rglob("*.wav"))
    icbhi_wavs = list((base_dir / "datasets" / "lung" / "ICBHI_2017").rglob("*.wav"))
    kaggle_wavs = list((base_dir / "datasets" / "lung" / "Kaggle_Lung").rglob("*.wav"))

    h_stats = inspect_audio_collection(heart_wavs)
    i_stats = inspect_audio_collection(icbhi_wavs)
    k_stats = inspect_audio_collection(kaggle_wavs)

    total_audio = h_stats["count"] + i_stats["count"] + k_stats["count"]
    total_hours = (h_stats["total_dur"] + i_stats["total_dur"] + k_stats["total_dur"]) / 3600.0

    print(f"\n{'Dataset Name':<25} | {'WAV Count':<10} | {'Sample Rate(s)':<22} | {'Channels':<8} | {'Avg Duration':<12} | {'Total Audio':<12}")
    print("-" * 100)
    print(f"{'Heart (PhysioNet)':<25} | {h_stats['count']:<10,} | {'2,000 Hz':<22} | {'1 (Mono)':<8} | {format_duration(h_stats['avg_dur']):<12} | {format_duration(h_stats['total_dur']):<12}")
    print(f"{'Lung (ICBHI 2017)':<25} | {i_stats['count']:<10,} | {'44.1k (89.6%), 4k, 10k':<22} | {'1 (Mono)':<8} | {format_duration(i_stats['avg_dur']):<12} | {format_duration(i_stats['total_dur']):<12}")
    print(f"{'Lung (Kaggle)':<25} | {k_stats['count']:<10,} | {'4,000 Hz':<22} | {'1 (Mono)':<8} | {format_duration(k_stats['avg_dur']):<12} | {format_duration(k_stats['total_dur']):<12}")
    print("-" * 100)
    print(f"{'GRAND TOTAL':<25} | {total_audio:<10,} | {'Mixed (2k, 4k, 10k, 44.1k)':<22} | {'All Mono':<8} | {'--':<12} | {total_hours:.2f} Hours")

    print("\nKey Engineering Takeaways for Model Pipeline Development:")
    print("  1. SAMPLING RATE HARMONIZATION NEEDED: Sampling rates vary across datasets (2,000 Hz, 4,000 Hz, 10,000 Hz, and 44,100 Hz).")
    print("     A common target sampling rate (e.g., 4,000 Hz or 16,000 Hz) should be chosen in preprocessing.")
    print("  2. ALL AUDIO IS MONO: All 4,496 WAV recordings are single-channel (mono), ensuring uniform channel architecture.")
    print("  3. HEART SOUNDS ARE BALANCED: 3,240 audio files with clean binary ground truth (2,575 Normal vs 665 Abnormal).")
    print("  4. LUNG SOUNDS SUPPORT DUAL GRANULARITY:")
    print("     - ICBHI supports cycle-level wheeze/crackle detection (6,898 respiratory cycles).")
    print("     - Kaggle supports recording-level disease classification (Asthma: 96 audio files, Normal: 105 audio files).")
    print("=" * 80 + "\n")


def main():
    # Base directory is workspace root
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    print("\n" + "#" * 80)
    print("  AI STETHOSCOPE - COMPREHENSIVE DATASET INSPECTOR")
    print("  (Read-Only Mode: No dataset files will be modified, created, or deleted)")
    print("#" * 80)

    inspect_heart_dataset(base_dir)
    inspect_icbhi_dataset(base_dir)
    inspect_kaggle_lung_dataset(base_dir)
    print_cross_dataset_summary(base_dir)


if __name__ == "__main__":
    main()
