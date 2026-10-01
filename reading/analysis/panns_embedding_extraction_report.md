# PANNs CNN14 Pretrained Frozen Embedding Extraction Report

**Generated:** 2026-09-16 17:21:20 +0530
**Overall Status:** PASS

---

## 1. Executive Summary

This report documents the end-to-end extraction and caching of 2048-dimensional frozen audio embeddings using the official pretrained **PANNs CNN14** model across all three target stethoscope diagnostic datasets.

All embeddings were extracted with the CNN14 backbone in strict `eval()` mode with zero gradient tracking and zero weight updates. Split assignments, patient-level isolation (0 overlap), and original task labels were strictly preserved.

---

## 2. PANNs CNN14 Model & Frontend Configuration

| Parameter | Specification | Verification |
| :--- | :--- | :--- |
| **Architecture** | Official PANNs Cnn14 | Verified (scripts/test_panns_cnn14.py) |
| **Pretrained Weights** | Official AudioSet checkpoint (`Cnn14_mAP=0.431.pth`) | 100% Parameter Match (strict=True) |
| **Total Parameters** | 81,837,071 (~81.8M) | Intact (SHA256 verified before/after) |
| **Backbone Mode** | Frozen / eval (`torch.inference_mode()`) | No gradients / No weight modifications |
| **Sampling Rate** | 32,000 Hz | Resampled via librosa `soxr_mq` |
| **Window Duration** | 5.0 seconds | 160,000 samples per window |
| **Frontend** | Pure PyTorch STFT + 64-band Log-Mel filterbank | In-graph STFT (n_fft=1024, hop=320) |
| **Embedding Dimension** | 2048-d penultimate representation | `(B, 2048)` float32 vector per window |

---

## 3. Preprocessing Specifications per Dataset

### 3.1 Heart Sound (PhysioNet 2016)
- **Task:** Binary Classification (`Normal` vs `Abnormal`)
- **Audio Handling:** Resampled to 32,000 Hz, DC offset removed, peak normalized to 0.95.
- **Windowing:** 5.0-second windows (160,000 samples).
  * Train: 50% overlap (hop = 80,000 samples / 2.5s).
  * Val / Test: Non-overlapping (hop = 160,000 samples / 5.0s).
  * Trailing segments >= 1.0s padded with zeros to 160,000 samples.

### 3.2 ICBHI 2017 Respiratory Sound
- **Task:** 4-class Respiratory Sound (`Normal`, `Crackle`, `Wheeze`, `Crackle+Wheeze`).
- **Cycle Boundaries:** Exact start/end timestamps from expert annotations.
- **Windowing:** Resampled to 32,000 Hz, zero-padded if < 5.0s (160,000 samples), center cropped if > 5.0s.
- **Normalization:** Peak normalized to 0.95 per cycle.

### 3.3 Kaggle Respiratory Disease
- **Task:** 3-class Patient-Level Diagnosis (`Normal`, `Asthma`, `Other`).
- **Audio Handling:** Resampled to 32,000 Hz, DC offset removed, peak normalized to 0.95.
- **Windowing:** 5.0-second windows (160,000 samples).
  * Train: 50% overlap (hop = 80,000 samples).
  * Val / Test: Non-overlapping (hop = 160,000 samples).

---

## 4. Extraction & Validation Statistics

| Dataset | Split | Examples / Windows | Embedding Shape | Dtype | NaNs | Infs | Mean | Std | Cache Size | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KAGGLE** | `train` | 1,296 | `[1296, 2048]` | `float32` | 0 | 0 | 0.0908 | 0.2173 | 7.40 MB | **PASS** |
| **KAGGLE** | `val` | 218 | `[218, 2048]` | `float32` | 0 | 0 | 0.0900 | 0.2176 | 1.20 MB | **PASS** |
| **KAGGLE** | `test` | 288 | `[288, 2048]` | `float32` | 0 | 0 | 0.0909 | 0.2181 | 1.64 MB | **PASS** |
| **ICBHI** | `train` | 3,804 | `[3804, 2048]` | `float32` | 0 | 0 | 0.0953 | 0.2770 | 14.39 MB | **PASS** |
| **ICBHI** | `val` | 1,549 | `[1549, 2048]` | `float32` | 0 | 0 | 0.0956 | 0.2855 | 5.33 MB | **PASS** |
| **ICBHI** | `test` | 1,545 | `[1545, 2048]` | `float32` | 0 | 0 | 0.0942 | 0.2771 | 5.70 MB | **PASS** |
| **HEART** | `train` | 19,117 | `[19117, 2048]` | `float32` | 0 | 0 | 0.0918 | 0.2187 | 116.50 MB | **PASS** |
| **HEART** | `val` | 2,358 | `[2358, 2048]` | `float32` | 0 | 0 | 0.0923 | 0.2198 | 14.54 MB | **PASS** |
| **HEART** | `test` | 2,427 | `[2427, 2048]` | `float32` | 0 | 0 | 0.0922 | 0.2198 | 14.91 MB | **PASS** |

---

## 5. Patient Isolation Audit (Zero-Leakage Guarantee)

| Dataset | Train / Val Overlap | Train / Test Overlap | Val / Test Overlap | Isolation Status |
| :--- | :---: | :---: | :---: | :---: |
| **KAGGLE** | 0 | 0 | 0 | **PASS (0 Overlap)** |
| **ICBHI** | 0 | 0 | 0 | **PASS (0 Overlap)** |
| **HEART** | 0 | 0 | 0 | **PASS (0 Overlap)** |

---

## 6. Audit of 11 Critical Integrity Criteria

1. **Split Membership Unchanged:** PASS. Preserved exact manifest split assignments.
2. **Example/Window Counts per Split:** PASS. Matched exact project indices.
3. **Embedding Dimensionality:** PASS. Exactly 2048 float32 values per embedding.
4. **Data Type:** PASS. float32 throughout all NPZ caches.
5. **Zero NaN Values:** PASS. Verified 0 NaN across all tensors.
6. **Zero Inf Values:** PASS. Verified 0 Inf across all tensors.
7. **Patient Isolation:** PASS. Exactly 0 patient overlap between train, val, and test splits.
8. **Manifest Labels Intact:** PASS. All labels correspond exactly to source annotations.
9. **1-to-1 Source Mapping:** PASS. Every source window/cycle maps to exactly one embedding vector.
10. **Zero Duplicate Keys:** PASS. Verified composite recording/window keys are unique within every split.
11. **Frozen Parameters:** PASS. SHA256 parameter checksum identical before and after inference.

---

## 7. Next Stage Readiness

Frozen PANNs CNN14 audio embeddings are cached and ready for downstream classifier evaluation (Logistic Regression, Linear probe, Multi-layer Perceptron, SVM, etc.).