# AI Stethoscope — Custom CNN Baseline Smoke Test Report

**Execution Stage**: STAGE 5B CUSTOM CNN BASELINE (Smoke Test Only — No Full Training).

**Host Device**: `cpu` | **PyTorch Version**: `2.14.0+cpu`

## 1. Model Architecture & Parameter Summary

| Task | Model Class | Input Tensor | Total Params | Trainable | Size (MB) | Classes | Masked Pooling |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **heart** | `AudioCNNBaseline` | `(B, 1, 128, 313)` | 421,730 | 421,730 | 1.61 MB | 2 (Normal, Abnormal) | No |
| **lung_acoustic** | `AudioCNNBaseline` | `(B, 1, 128, 313)` | 421,988 | 421,988 | 1.61 MB | 4 (Normal, Crackle, Wheeze, Crackle+Wheeze) | Yes |
| **lung_disease** | `AudioCNNBaseline` | `(B, 1, 128, 313)` | 421,859 | 421,859 | 1.61 MB | 3 (Normal, Asthma, Other) | No |

## 2. Smoke Test Execution Results

| Task | Train Batch | Train Loss (Weighted CE) | Total Grad Norm | Val Batch | Val Loss | Output Shape | Patient Aggregation | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **heart** | 16 | `0.7403` | `7.7430` | 16 | `0.5922` | `[16, 2]` | Verified | **PASS** |
| **lung_acoustic** | 16 | `1.4108` | `8.0419` | 16 | `1.3032` | `[16, 4]` | Verified | **PASS** |
| **lung_disease** | 16 | `1.1530` | `7.0961` | 16 | `1.0897` | `[16, 3]` | Verified | **PASS** |

## 3. Detailed Component Verifications

- **Heart CNN**: Output `(16, 2)`, weighted loss computed cleanly, sensitivity and specificity metrics calculated.
- **ICBHI CNN**: Output `(16, 4)`, temporal masks verified to pool only active acoustic frames, zero-padding silence ignored.
- **Kaggle CNN**: Output `(16, 3)`, window probabilities successfully aggregated to patient-level diagnostic predictions.
- **Gradients**: 100% of parameter tensors have non-zero, finite gradients after backward pass (`0 NaN`, `0 Inf`).
- **Optimizer Step**: Weights updated successfully without errors.

## 4. Experiment Folder Structure Initialized

```text
experiments/
├── heart_cnn/        [checkpoints/, logs/, metrics/, figures/]
├── icbhi_cnn/        [checkpoints/, logs/, metrics/, figures/]
└── kaggle_cnn/       [checkpoints/, logs/, metrics/, figures/]
```

## 5. Final Verdict

# `>>> VERDICT: READY FOR FULL CNN TRAINING <<<`

All 3 models, losses, backward passes, and validation paths have been verified end-to-end.
