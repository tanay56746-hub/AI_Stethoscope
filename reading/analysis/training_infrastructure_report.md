# AI Stethoscope — Training Infrastructure Validation Report

**Execution Mode**: STAGE 5A TRAINING INFRASTRUCTURE ONLY (No models built or trained).

## 1. Class Weights (Calculated on Training Splits)

Method: `weight_i = N / (K * count_i)` saved to [`processed/config/class_weights.json`](file:///C:/Users/Tanay/Downloads/AI_Stethoscope/processed/config/class_weights.json)

| Task | Class | Train Count | Percentage | Dynamic Weight |
| :--- | :--- | :---: | :---: | :---: |
| **heart** | Normal | 14,583 | 76.3% | `0.65545` |
| **heart** | Abnormal | 4,534 | 23.7% | `2.10818` |
| **lung_acoustic** | Normal | 2,120 | 55.7% | `0.44858` |
| **lung_acoustic** | Crackle | 1,044 | 27.4% | `0.91092` |
| **lung_acoustic** | Wheeze | 394 | 10.4% | `2.41371` |
| **lung_acoustic** | Crackle+Wheeze | 246 | 6.5% | `3.86585` |
| **lung_disease** | Normal | 417 | 32.2% | `1.03597` |
| **lung_disease** | Asthma | 339 | 26.2% | `1.27434` |
| **lung_disease** | Other | 540 | 41.7% | `0.80000` |

## 2. Configuration Parameters

Centralized YAML: [`configs/training_config.yaml`](file:///C:/Users/Tanay/Downloads/AI_Stethoscope/configs/training_config.yaml)

- **Batch Size**: `32`
- **Learning Rate**: `0.0003` (`AdamW` with CosineAnnealing)
- **Weight Decay**: `0.01`
- **Epochs / Early Stopping**: `50` / patience `10`
- **Reproducibility Seed**: `42`

## 3. Batch Sanity Test (All 9 DataLoaders)

| Dataset | Split | Batch Size | Tensor Shape | Dtype | Mask Shape | Active Frames (min/max/mean) | Class Dist | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **heart** | train | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Abnormal: 12, Normal: 4 | **PASS** |
| **heart** | val | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Normal: 16 | **PASS** |
| **heart** | test | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Normal: 16 | **PASS** |
| **icbhi** | train | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 34 / 135 / 106.8 | Normal: 16 | **PASS** |
| **icbhi** | val | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 66 / 171 / 142.6 | Crackle: 8, Crackle+Wheeze: 7, Normal: 1 | **PASS** |
| **icbhi** | test | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 47 / 160 / 143.3 | Crackle: 8, Normal: 2, Wheeze: 6 | **PASS** |
| **kaggle** | train | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Asthma: 16 | **PASS** |
| **kaggle** | val | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Asthma: 9, Normal: 7 | **PASS** |
| **kaggle** | test | 16 | `[16, 1, 128, 313]` | `torch.float32` | `[16, 313]` | 313 / 313 / 313.0 | Asthma: 4, Normal: 12 | **PASS** |

## 4. Masked Temporal Average Pooling

Validated [`src/pooling.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/src/pooling.py):

- **4D CNN Feature Map**: `(B, 64, 4, 313)` with mask `(B, 313)` $\rightarrow$ `(B, 64)` [PASS]
- **Downsampled Receptive Fields**: `(B, 64, 2, 79)` with mask `(B, 313)` $\rightarrow$ `(B, 64)` via interpolation [PASS]
- **Division by Zero Protection**: `clamp(min=1e-8)` prevents non-finite values even with single-frame or empty masks [PASS]

## 5. Cross-Split Zero-Leakage Audit

| Dataset | Total Patients | Patient Overlap | Recording Overlap | Window / Cycle Containment | Triad Completeness | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **heart** | 2856 | 0 | 0 | 0 multi-split items | N/A | **PASS** |
| **icbhi** | 126 | 0 | 0 | 0 multi-split items | N/A | **PASS** |
| **kaggle** | 112 | 0 | 0 | 0 multi-split items | 100% Intact | **PASS** |

## 6. Patient-Level Evaluation Readiness

- **Kaggle**: DataLoader preserves `patient_id` and `recording_id` in every batch dictionary, enabling patient-level soft voting / probability averaging during validation.

- **Heart**: Multiple windows per patient/recording are cleanly bundled with exact metadata for recording-level and patient-level metric aggregation.


## 7. Final Verdict

# `>>> VERDICT: READY FOR MODEL TRAINING <<<`

- All 9 DataLoaders validated with uniform `(B, 1, 128, 313)` tensor batching.

- 0.0% NaN or Inf.

- Zero patient or recording leakage across splits.

- Dynamic inverse-frequency class weights computed from training split.

- ICBHI temporal masking functional and verified.

- Centralized YAML configuration loaded and verified.
