# Multi-Task Architecture Ablation & Synthesis Benchmark Report
**AI Stethoscope Research Project**  
**Date:** September 28, 2026  
**Status:** Evaluation Completed — Final Benchmark Synthesis  
**Scope:** Controlled Four-Configuration Ablation Study of the Unified Multi-Task Architecture Across Cardiac and Respiratory Auscultation (PhysioNet 2016 Heart, ICBHI 2017 Lung Acoustics, Kaggle Lung Disease)

> [!IMPORTANT]
> **Clinical Disclaimer:** This benchmark report, its associated multi-task deep neural network, and baseline implementations are designed strictly for scientific machine learning research and computational acoustic modeling. None of these models have undergone prospective clinical trials, medical regulatory evaluation, or clinical diagnostic certification. They must NOT be utilized for clinical diagnostic decisions, patient monitoring, triage, or clinical treatment selection.

---

## 1. Objective

The objective of this investigation is to evaluate a unified multi-task deep neural network architecture for simultaneous auscultation pattern recognition across three clinically distinct acoustic domains:
1. **Cardiac Auscultation (PhysioNet/CinC 2016):** Binary pathology detection (`Normal` vs `Abnormal`).
2. **Respiratory Cycle Auscultation (ICBHI 2017):** 4-class adventitious breath sound detection (`Normal`, `Crackle`, `Wheeze`, `Crackle+Wheeze`).
3. **Respiratory Disease Diagnosis (Kaggle):** 3-class patient disease classification (`Normal`, `Asthma`, `Other`).

Specifically, this study isolates and quantifies the incremental contribution of two auxiliary representations when integrated with a shared 2D convolutional audio trunk:
- **Domain-Engineered Physical Features (397 dims):** Frame-level spectral centroid, bandwidth, rolloff, zero-crossing dynamics, log-mel energy statistics, and MFCC velocity/acceleration envelopes.
- **Pretrained Foundation Audio Embeddings (2,048 dims):** Frozen high-capacity representations extracted from the AudioSet-pretrained PANNs CNN14 backbone.

The experimental goal is a controlled empirical comparison across four structured ablation configurations, evaluated strictly against task-specific single-task baselines, classical ML baselines, and frozen PANNs linear classifiers under identical patient-isolated splits.

---

## 2. Experimental Setup

The multi-task training framework operates under strict data leakage prevention protocols:

```
                               Shared Auscultation CNN Trunk
                          (4 Conv Blocks: 1 -> 32 -> 64 -> 128 -> 256)
                                               │
                                               ▼
                                 Feature Map (B, 256, 8, 19)
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
       [Global Pooling]             [Masked Temporal Avg]               [Global Pooling]
               │                               │                               │
        CNN Vec (B, 256)                CNN Vec (B, 256)                CNN Vec (B, 256)
               │                               │                               │
               ├──────────────┐                ├──────────────┐                ├──────────────┐
               │  Physical    │                │  PANNs Aux   │                │  Physical    │
               │  (397 -> 64) │                │ (2048 -> 128)│                │  (397 -> 64) │
               ▼              ▼                ▼              ▼                ▼              ▼
          CNN (256)      Phys (64)        CNN (256)     PANNs (128)       CNN (256)      Phys (64)
               │              │                │              │                │              │
               └──────┬───────┘                └──────┬───────┘                └──────┬───────┘
                      ▼                               ▼                               ▼
                Concat (B, 320)                 Concat (B, 384)                 Concat (B, 320)
                      │                               │                               │
                      ▼                               ▼                               ▼
              Heart MLP Head                  Acoustic MLP Head               Disease MLP Head
             (320 -> 128 -> 2)               (384 -> 128 -> 4)               (320 -> 128 -> 3)
                      │                               │                               │
                      ▼                               ▼                               ▼
             Heart Logits (B, 2)             Acoustic Logits (B, 4)          Disease Logits (B, 3)
```

### 2.1 Optimization & Masked Loss Formulation
- **Optimizer:** AdamW with initial learning rate $\eta = 3 \times 10^{-4}$, weight decay $\lambda = 10^{-2}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- **Learning Rate Schedule:** Cosine annealing over 15 epochs with a 2-epoch linear warmup phase.
- **Masked Multi-Task Objective:** Each input batch contains interleaved samples from Heart, ICBHI, and Kaggle. Task-specific masks $m_H, m_A, m_D \in \{0, 1\}$ gate cross-entropy computation:
  $$\mathcal{L}_{\text{total}} = \lambda_H \mathcal{L}_H + \lambda_A \mathcal{L}_A + \lambda_D \mathcal{L}_D$$
  where for task $T$, $\mathcal{L}_T = \frac{1}{\sum_i m_{T, i}} \sum_i m_{T, i} \cdot \ell_{\text{CE}}(\hat{y}_{T, i}, y_{T, i})$ if $\sum_i m_{T, i} > 0$, else $0.0$.
  Task loss weights were set equally to $\lambda_H = 1.0, \lambda_A = 1.0, \lambda_D = 1.0$.
- **Class Imbalance Mitigation:** Inverse-frequency class weights $w_c = \frac{N}{K \cdot N_c}$ applied within each task's cross-entropy loss.
- **Model Selection Protocol:** Checkpoints were recorded at each epoch; the selected model was chosen based strictly on the unweighted arithmetic mean of **Validation Macro F1** across the three tasks:
  $$\text{Macro F1}_{\text{selection}} = \frac{1}{3} \left( \text{F1}_{\text{Heart}}^{\text{val}} + \text{F1}_{\text{ICBHI}}^{\text{val}} + \text{F1}_{\text{Kaggle}}^{\text{val}} \right)$$
- **Held-Out Test Protocol:** The selected checkpoint per configuration was evaluated **exactly once** on the held-out test splits. Test data was never used for model selection, learning rate adjustment, or early stopping.

---

## 3. Dataset & Split Information

Strict patient/subject isolation was enforced across all datasets to ensure no recordings or windows from the same subject appear across partitions.

| Dataset | Clinical Domain | Classes | Train Windows (Subjects) | Val Windows (Subjects) | Test Windows (Subjects) | Total Windows (Subjects) | Patient Leakage Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhysioNet 2016** | Cardiac Sound | 2 | 19,117 (2,000) | 2,358 (429) | 2,427 (427) | 23,902 (2,856) | **0 Overlap (Passed)** |
| **ICBHI 2017** | Adventitious Lung Cycles | 4 | 3,804 (76) | 1,549 (26) | 1,545 (24) | 6,898 (126) | **0 Overlap (Passed)** |
| **Kaggle** | Respiratory Disease | 3 | 1,296 (67) | 218 (22) | 288 (23) | 1,802 (112) | **0 Overlap (Passed)** |
| **Combined** | Multi-Task Suite | 9 total | **24,217 (2,143)** | **4,125 (477)** | **4,260 (474)** | **32,602 (3,094)** | **0 Overlap (Passed)** |

---

## 4. Four Ablation Configurations

Four configurations were trained and evaluated to isolate component contributions:

1. **`cnn_only` (Baseline Multi-Task):**
   Shared 4-block 2D CNN trunk with global pooling (Heart, Kaggle) and masked temporal average pooling (ICBHI) connected directly to task classification heads. Physical and PANNs branches disabled (zero-filled).
2. **`cnn_physical` (CNN + Handcrafted Physical Features):**
   Shared CNN trunk augmented with the 397-dimensional physical feature projection branch (`Linear(397, 64) -> ReLU`) concatenated to Heart and Kaggle representations. PANNs branch disabled.
3. **`cnn_panns` (CNN + Frozen PANNs Embeddings):**
   Shared CNN trunk augmented with the 2048-dimensional PANNs CNN14 projection branch (`Linear(2048, 128) -> ReLU`) concatenated to the ICBHI representation. Physical feature branch disabled.
4. **`cnn_physical_panns` (Full Tri-Modal Multi-Task Network):**
   Full unified architecture integrating the shared CNN trunk, 397-d physical feature branch (Heart, Kaggle), and 2048-d PANNs auxiliary branch (ICBHI).

---

## 5. Parameter Counts & 6. Training Times

All models were trained on CPU in the local environment.

| Configuration ID | Trainable Parameters | Auxiliary Frozen Parameters | Total Parameter Footprint | Best Epoch | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`cnn_only`** | 520,937 | 0 | 520,937 (~2.08 MB) | Epoch 5 | 1,138.00 s (~19.0 min) |
| **`cnn_physical`** | 546,409 | 0 | 546,409 (~2.19 MB) | Epoch 5 | 1,141.45 s (~19.0 min) |
| **`cnn_panns`** | 783,209 | 0 (Embeddings cached) | 783,209 (~3.13 MB) | Epoch 5 | 1,133.73 s (~18.9 min) |
| **`cnn_physical_panns`** | **808,681** | 0 (Embeddings cached) | **808,681 (~3.23 MB)** | **Epoch 6** | **3,211.75 s (~53.5 min)** |

> [!NOTE]
> The auxiliary branches add minimal computational overhead: `cnn_physical` adds 25,472 parameters (+4.9%), while `cnn_panns` adds 262,272 linear projection parameters (+50.3%). In all configurations, the model footprint remains under 3.5 MB, maintaining CPU compatibility.

---

## 7. Validation Macro F1 & 8. Held-Out Test Macro F1

*Model selection was determined solely by Validation Macro F1 Mean. The selected epoch was evaluated once on the held-out test splits.*

| Configuration | Val Heart F1 | Val ICBHI F1 | Val Kaggle F1 | **Val Macro F1 Mean** | Test Heart F1 | Test ICBHI F1 | Test Kaggle F1 | **Test Macro F1 Mean** | Absolute Gain (Test vs CNN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_only` | 0.5693 | 0.2000 | 0.4634 | **0.4109** | 0.6129 | 0.1615 | 0.3403 | **0.3716** | Baseline |
| `cnn_physical` | 0.6977 | 0.2164 | 0.4227 | **0.4456** | 0.7395 | 0.1635 | 0.3401 | **0.4144** | +4.28% |
| `cnn_panns` | 0.5771 | 0.2159 | 0.4810 | **0.4247** | 0.6294 | 0.2631 | 0.3386 | **0.4104** | +3.88% |
| **`cnn_physical_panns`** | **0.7692** | **0.2692** | 0.3731 | **0.4705** 🏆 | **0.7795** | **0.2929** | **0.3862** | **0.4862** 🏆 | **+11.46%** |

---

## 9. Heart Sound Classification Results (PhysioNet 2016)

*Evaluated on 2,427 held-out test windows across 427 unseen subjects.*

### 9.1 Overall Performance Across Ablations
| Configuration | Window Accuracy | Window Bal Acc | Window Macro F1 | Window Sensitivity | Window Specificity | Patient Accuracy | Patient Macro F1 | Patient Sensitivity | Patient Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_only` | 72.97% | 61.24% | 0.6129 | 39.75% | 82.73% | 75.64% | 0.6147 | 36.78% | 85.59% |
| `cnn_physical` | 79.98% | 76.54% | 0.7395 | 70.24% | 82.84% | 83.37% | 0.7690 | 74.71% | 85.59% |
| `cnn_panns` | 73.67% | 63.10% | 0.6294 | 43.74% | 82.46% | 76.11% | 0.6254 | 39.08% | 85.59% |
| **`cnn_physical_panns`** | **82.16%** | **83.08%** | **0.7795** | **84.75%** | **81.40%** | **83.84%** | **0.7894** | **87.36%** | **82.94%** |

### 9.2 Per-Class Breakdown (Final Model: `cnn_physical_panns`)
| Evaluation Level | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Window Level** | **Normal** | 0.9479 | 0.8140 | 0.8758 | 1,876 |
| | **Abnormal** | 0.5723 | 0.8475 | 0.6832 | 551 |
| **Patient Level** | **Normal** | 0.9625 | 0.8294 | 0.8910 | 340 |
| | **Abnormal** | 0.5672 | 0.8736 | 0.6878 | 87 |

---

## 10. ICBHI 2017 Respiratory Sound Classification Results

*Evaluated on 1,545 held-out respiratory cycles across 24 unseen patients.*

### 10.1 Overall Performance Across Ablations
| Configuration | Cycle Accuracy | Cycle Bal Acc | Cycle Macro F1 | Normal F1 | Crackle F1 | Wheeze F1 | Both F1 | Patient Accuracy | Patient Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_only` | 24.40% | 22.21% | 0.1615 | 0.0470 | 0.4248 | 0.1368 | 0.0376 | 8.33% | 0.0588 |
| `cnn_physical` | 24.47% | 21.99% | 0.1635 | 0.0707 | 0.4167 | 0.1369 | 0.0296 | 12.50% | 0.1042 |
| `cnn_panns` | 37.73% | 28.62% | 0.2631 | 0.4614 | 0.4703 | 0.0990 | 0.0215 | 29.17% | 0.1754 |
| **`cnn_physical_panns`** | **42.39%** | **30.87%** | **0.2929** | **0.5502** | **0.4709** | **0.0802** | **0.0704** | **54.17%** | **0.2666** |

### 10.2 Per-Class Breakdown (Final Model: `cnn_physical_panns`)
| Class | Cycle Precision | Cycle Recall | Cycle F1-Score | Support | Patient Precision | Patient Recall | Patient F1-Score | Patient Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0.5668 | 0.5346 | **0.5502** | 651 | 0.9167 | 0.6471 | **0.7586** | 17 |
| **Crackle** | 0.3920 | 0.5896 | **0.4709** | 480 | 0.2000 | 0.6667 | **0.3077** | 3 |
| **Wheeze** | 0.1012 | 0.0664 | **0.0802** | 256 | 0.0000 | 0.0000 | **0.0000** | 1 |
| **Crackle+Wheeze** | 0.1707 | 0.0443 | **0.0704** | 158 | 0.0000 | 0.0000 | **0.0000** | 3 |

---

## 11. Kaggle Respiratory Disease Classification Results

*Evaluated on 288 held-out test windows across 23 unseen patients.*

### 11.1 Overall Performance Across Ablations
| Configuration | Window Accuracy | Window Bal Acc | Window Macro F1 | Normal F1 | Asthma F1 | Other F1 | Patient Accuracy | Patient Bal Acc | Patient Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_only` | 33.68% | 33.44% | 0.3403 | 0.3750 | 0.2871 | 0.3587 | 30.43% | 28.04% | 0.2727 |
| `cnn_physical` | 36.11% | 37.69% | 0.3401 | 0.4316 | 0.4232 | 0.1655 | 52.17% | 53.97% | **0.5256** |
| `cnn_panns` | 36.46% | 35.64% | 0.3386 | 0.4186 | 0.1560 | 0.4411 | 34.78% | 32.80% | 0.3054 |
| **`cnn_physical_panns`** | **40.97%** | **42.76%** | **0.3862** | **0.4947** | **0.4653** | **0.1986** | **52.17%** | **53.97%** | **0.5175** |

### 11.2 Per-Class Breakdown (Final Model: `cnn_physical_panns`)
| Class | Window Precision | Window Recall | Window F1-Score | Support | Patient Precision | Patient Recall | Patient F1-Score | Patient Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0.4563 | 0.5402 | **0.4947** | 87 | 0.7500 | 0.4286 | **0.5455** | 7 |
| **Asthma** | 0.3750 | 0.6129 | **0.4653** | 93 | 0.4000 | 0.8571 | **0.5455** | 7 |
| **Other** | 0.4242 | 0.1296 | **0.1986** | 108 | 0.7500 | 0.3333 | **0.4615** | 9 |

---

## 12. Patient / Entity-Level Evaluation Summary

Window-level softmax probabilities were aggregated across all windows of a subject using uniform mean soft-voting:
$$\bar{\mathbf{p}}^{(i)} = \frac{1}{|W_i|} \sum_{w \in W_i} \mathbf{p}(w), \quad \hat{y}^{(i)} = \arg\max_k \bar{p}_k^{(i)}$$

| Task | Configuration | Patient Accuracy | Patient Balanced Acc | Patient Macro F1 | Clinical Specificity / Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Heart Sound** | `cnn_only` | 75.64% (323/427) | 61.18% | 0.6147 | Sens: 36.78% \| Spec: 85.59% |
| *(427 Subjects)* | `cnn_physical` | 83.37% (356/427) | 80.15% | 0.7690 | Sens: 74.71% \| Spec: 85.59% |
| | `cnn_panns` | 76.11% (325/427) | 62.33% | 0.6254 | Sens: 39.08% \| Spec: 85.59% |
| | **`cnn_physical_panns`** | **83.84% (358/427)** | **85.15%** | **0.7894** | **Sens: 87.36% \| Spec: 82.94%** |
| **ICBHI 2017** | `cnn_only` | 8.33% (2/24) | 16.67% | 0.0588 | Crackle F1: 0.2353 \| Normal F1: 0.0000 |
| *(24 Patients)* | `cnn_physical` | 12.50% (3/24) | 41.67% | 0.1042 | Crackle F1: 0.2500 \| Wheeze F1: 0.1667 |
| | `cnn_panns` | 29.17% (7/24) | 24.02% | 0.1754 | Crackle F1: 0.2667 \| Normal F1: 0.4348 |
| | **`cnn_physical_panns`** | **54.17% (13/24)** | **32.84%** | **0.2666** | **Crackle F1: 0.3077 \| Normal F1: 0.7586** |
| **Kaggle** | `cnn_only` | 30.43% (7/23) | 28.04% | 0.2727 | Normal F1: 0.3636 \| Asthma F1: 0.0000 |
| *(23 Patients)* | `cnn_physical` | 52.17% (12/23) | 53.97% | **0.5256** | Normal F1: 0.6154 \| Asthma F1: 0.5000 |
| | `cnn_panns` | 34.78% (8/23) | 32.80% | 0.3054 | Normal F1: 0.4615 \| Asthma F1: 0.0000 |
| | **`cnn_physical_panns`** | **52.17% (12/23)** | **53.97%** | **0.5175** | **Normal F1: 0.5455 \| Asthma F1: 0.5455** |

---

## 13. Confusion Matrices (Final Model: `cnn_physical_panns`)

### 13.1 Heart Sound (PhysioNet 2016)
```
Window-Level (N=2,427):                 Patient-Level (N=427):
                 Pred Norm   Pred Abn                    Pred Norm   Pred Abn
Actual Normal      1,527       349       Actual Normal      282         58
Actual Abnormal       84       467       Actual Abnormal     11         76
```

### 13.2 ICBHI 2017 Respiratory Cycles
```
Cycle-Level (N=1,545):
                        Pred Normal   Pred Crackle   Pred Wheeze   Pred Both
Actual Normal               348           221            69           13
Actual Crackle              129           283            63            5
Actual Wheeze                90           133            17           16
Actual Crackle+Wheeze        47            85            19            7

Patient-Level (N=24):
                        Pred Normal   Pred Crackle   Pred Wheeze   Pred Both
Actual Normal                11             5             1            0
Actual Crackle                0             2             1            0
Actual Wheeze                 0             1             0            0
Actual Crackle+Wheeze         1             2             0            0
```

### 13.3 Kaggle Respiratory Disease
```
Window-Level (N=288):                   Patient-Level (N=23):
                 Pred Norm  Pred Ast  Pred Oth                   Pred Norm  Pred Ast  Pred Oth
Actual Normal       47        31         9       Actual Normal       3         3         1
Actual Asthma       26        57        10       Actual Asthma       1         6         0
Actual Other        30        64        14       Actual Other        0         6         3
```

---

## 14. Comparison Against Existing Baselines

*All numbers represent held-out test split evaluations extracted from respective finalized reports.*

| Dataset / Task | Metric | Custom CNN Baseline | Classical ML Baseline | PANNs Frozen Classifier | Full Multi-Task Model (`cnn_physical_panns`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Heart Sound** | **Window Accuracy** | 89.04% | **90.11%** *(RF)* | 67.99% *(RBF SVM)* | 82.16% |
| *(PhysioNet 2016)* | **Window Bal Acc** | **90.15%** | 89.50% *(RF)* | 60.64% *(RBF SVM)* | 83.08% |
| | **Window Macro F1** | 85.90% | **86.82%** *(RF)* | 59.12% *(RBF SVM)* | 77.95% |
| | **Window Sensitivity** | **92.20%** | 88.38% *(RF)* | 47.19% *(RBF SVM)* | 84.75% |
| | **Window Specificity**| 88.11% | **90.62%** *(RF)* | 74.09% *(RBF SVM)* | 81.40% |
| | **Patient Accuracy** | 91.80% | **93.44%** *(RF)* | 75.18% *(RBF SVM)* | 83.84% |
| | **Patient Macro F1** | 88.69% | **90.59%** *(RF)* | 62.99% *(RBF SVM)* | 78.94% |
| | **Patient Sensitivity**| **96.55%** | 94.25% *(RF)* | 43.68% *(RBF SVM)* | 87.36% |
| | **Patient Specificity**| 90.59% | **93.24%** *(RF)* | 83.24% *(RBF SVM)* | 82.94% |
| **ICBHI 2017** | **Cycle Accuracy** | 23.82% | 38.90% *(RBF SVM)* | **42.91%** *(MLP)* | 42.39% |
| *(4-Class Acoustic)*| **Cycle Bal Acc** | 24.52% | **34.15%** *(RBF SVM)* | 33.93% *(MLP)* | 30.87% |
| | **Cycle Macro F1** | 21.62% | 33.46% *(RBF SVM)* | **33.91%** *(MLP)* | 29.29% |
| | **Normal F1** | 17.56% | 52.13% *(RBF SVM)* | **56.82%** *(MLP)* | 55.02% |
| | **Crackle F1** | 39.54% | 34.76% *(RBF SVM)* | 41.15% *(MLP)* | **47.09%** |
| | **Wheeze F1** | 14.88% | **23.84%** *(RBF SVM)* | 22.71% *(MLP)* | 8.02% |
| | **Crackle+Wheeze F1**| 14.50% | **22.51%** *(RBF SVM)* | 14.97% *(MLP)* | 7.04% |
| | **Patient Accuracy** | 16.67% | **70.83%** *(RBF SVM)* | 66.67% *(MLP)* | 54.17% |
| | **Patient Macro F1** | 19.03% | 21.79% *(RBF SVM)* | **32.24%** *(MLP)* | 26.66% |
| **Kaggle** | **Window Accuracy** | 35.76% | **42.01%** *(RBF SVM)* | 33.33% *(Linear SVM)* | 40.97% |
| *(3-Class Disease)* | **Window Bal Acc** | 36.46% | 41.72% *(RBF SVM)* | 32.51% *(Linear SVM)* | **42.76%** |
| | **Window Macro F1** | 35.68% | **39.83%** *(RBF SVM)* | 32.23% *(Linear SVM)* | 38.62% |
| | **Normal F1** | 43.18% | 43.16% *(RBF SVM)* | 30.97% *(Linear SVM)* | **49.47%** |
| | **Asthma F1** | 26.83% | 35.44% *(RBF SVM)* | 26.35% *(Linear SVM)* | **46.53%** |
| | **Other F1** | 37.04% | 40.89% *(RBF SVM)* | 39.37% *(Linear SVM)* | 19.86% |
| | **Patient Accuracy** | 34.78% | **56.52%** *(RBF SVM)* | 30.43% *(Linear SVM)* | 52.17% |
| | **Patient Macro F1** | 33.85% | 50.84% *(RBF SVM)* | 22.06% *(Linear SVM)* | **51.75%** |

---

## 15. Final Model Candidate & Architecture Description

The empirical ablation results designate **`cnn_physical_panns`** as the final candidate model for the AI Stethoscope project:

### 15.1 Component Specifications
- **Shared Auscultation CNN Trunk (388,320 params):**
  - Input: Standardized Log-Mel spectrogram $(B, 1, 128, 313)$.
  - 4 convolutional blocks ($1 \to 32 \to 64 \to 128 \to 256$), each with $3 \times 3$ convolutions, Batch Normalization, ReLU, $2 \times 2$ Max Pooling, and spatial Dropout ($0.11$ to $0.20$).
  - Output feature map: $(B, 256, 8, 19)$.
- **Pooling Mechanisms:**
  - Standard Global Average Pooling over time and frequency $\to (B, 256)$ for Heart and Kaggle.
  - Masked Temporal Average Pooling (MTAP) across frequency and unpadded time frames $\to (B, 256)$ for ICBHI cycles, preventing zero-padded silent frames from degrading the acoustic embedding.
- **Physical Feature Branch (25,472 params):**
  - Input: 397-dimensional physical feature vector.
  - Linear projection: `Linear(397, 64) -> ReLU` $\to (B, 64)$.
  - Concatenated with the CNN vector for Heart and Kaggle heads $\to (B, 320)$.
- **PANNs Auxiliary Branch (262,272 params):**
  - Input: 2048-dimensional frozen AudioSet embedding vector.
  - Linear projection: `Linear(2048, 128) -> ReLU` $\to (B, 128)$.
  - Concatenated with the MTAP CNN vector for the ICBHI head $\to (B, 384)$.
- **Task Classification Heads (132,617 params total):**
  - Heart Head: `Linear(320, 128) -> ReLU -> Dropout(0.30) -> Linear(128, 2)` (41,346 params).
  - ICBHI Head: `Linear(384, 128) -> ReLU -> Dropout(0.25) -> Linear(128, 4)` (49,796 params).
  - Kaggle Head: `Linear(320, 128) -> ReLU -> Dropout(0.35) -> Linear(128, 3)` (41,475 params).

---

## 16. Final Checkpoint Path & Verification

- **Final Checkpoint File:**
  [`experiments/multitask_ablations/cnn_physical_panns/checkpoints/best_checkpoint.pt`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/experiments/multitask_ablations/cnn_physical_panns/checkpoints/best_checkpoint.pt)
- **Filesize:** `9,750,359` bytes (~9.75 MB)
- **Total Trainable Parameters:** `808,681`
- **Total State Dict Keys:** 44 tensor entries (matching `AudioMultiTaskModel` architecture with strict loading).
- **Associated Metric Logs:**
  - Validation: [`experiments/multitask_ablations/cnn_physical_panns/metrics/validation_metrics.json`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/experiments/multitask_ablations/cnn_physical_panns/metrics/validation_metrics.json)
  - Test: [`experiments/multitask_ablations/cnn_physical_panns/metrics/test_metrics.json`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/experiments/multitask_ablations/cnn_physical_panns/metrics/test_metrics.json)
  - Config: [`experiments/multitask_ablations/cnn_physical_panns/config.json`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/experiments/multitask_ablations/cnn_physical_panns/config.json)

---

## 17. Limitations

1. **Domain Transfer Gap in AudioSet Pretraining:** PANNs CNN14 was trained on generic environmental sounds and music. While its high-level representations transfer to respiratory cycle detection, raw PANNs features without domain fine-tuning perform poorly when evaluated in isolation on low-frequency cardiac phonocardiograms (Heart window Macro F1 = 59.12%).
2. **Extreme Class Imbalance in Rare Adventitious Sounds:** Continuous wheezing combined with crackles (`Crackle+Wheeze`) represents only 3.8% of ICBHI training cycles. In the multi-task model, held-out test recall on this class remains low (4.43%), as the shared representation prioritizes dominant classes.
3. **Small Sample Sizes in Clinical Datasets:** The Kaggle dataset contains only 23 unseen patients in the test split (7 Normal, 7 Asthma, 9 Other), and ICBHI contains only 24 test patients. Small sample sizes introduce wider variance in patient-level metric estimates.
4. **Hardware Environment Constraints:** All training was performed on CPU. Larger convolutional backbones or full end-to-end PANNs fine-tuning were not computationally viable under this setup.
5. **No Prospective or Regulatory Validation:** All experiments were conducted on retrospective, publicly available benchmark databases. Results do not constitute clinical validation.

---

## 18. Reproducibility Information

To reproduce the exact numerical metrics and audit the results from source:

1. **Environment:** Python 3.14.6, PyTorch 2.14.0+cpu, scikit-learn 1.7.0, numpy 2.3.0, scipy 1.16.1.
2. **Fixed Random Seeds:** Global seeds fixed to `42` (`torch.manual_seed(42)`, `np.random.seed(42)`) across batch sampling, layer initialization, and dropout.
3. **Data Splits:** Partitions are deterministically loaded from `processed/manifests/{heart,icbhi,kaggle}/{train,val,test}.csv`.
4. **Cached Features:**
   - Log-Mel spectrograms: `processed/{heart,icbhi,kaggle}/*.pt`
   - Physical features: `processed/classical_features/{heart,icbhi,kaggle}_{train,val,test}.npz`
   - PANNs embeddings: `processed/panns_embeddings/{heart,icbhi,kaggle}/{train,val,test}.npz`
5. **Ablation Training Script:** [`scripts/train_multitask_ablations.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/scripts/train_multitask_ablations.py)
6. **Multi-Task Model Architecture:** [`src/models/final_multitask_model.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/src/models/final_multitask_model.py)
7. **Loss Function:** [`src/losses/multitask_loss.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/src/losses/multitask_loss.py)

---

## 19. Factual Conclusions Supported by Experiments

1. **Ablation Progression:**
   Adding auxiliary inputs produced progressive increases in overall multi-task performance:
   - `cnn_only`: Test Macro F1 Mean = **0.3716**
   - `cnn_panns`: Test Macro F1 Mean = **0.4104** (+3.88% over CNN-only)
   - `cnn_physical`: Test Macro F1 Mean = **0.4144** (+4.28% over CNN-only)
   - `cnn_physical_panns`: Test Macro F1 Mean = **0.4862** (+11.46% over CNN-only)
2. **Task-Specific Impact of Physical Features:**
   Adding physical acoustic features (`cnn_physical`) yielded substantial improvements on Heart sound detection (window Macro F1 rose from 0.6129 to 0.7395; patient sensitivity increased from 36.78% to 74.71%). On Kaggle disease classification, patient Macro F1 rose from 0.2727 to 0.5256.
3. **Task-Specific Impact of PANNs Embeddings:**
   Adding PANNs embeddings (`cnn_panns`) produced the primary gain on ICBHI respiratory acoustic cycle classification (cycle Macro F1 rose from 0.1615 to 0.2631, and reached 0.2929 in the combined model).
4. **Synergy in the Full Tri-Modal Model:**
   The full configuration (`cnn_physical_panns`) achieved the highest test macro F1 across all three tasks simultaneously, demonstrating that physical moments and generic pretrained audio features provide complementary information to the shared CNN trunk.
5. **Comparison to Single-Task Baselines:**
   - On Heart, the single-task Random Forest baseline (0.8682 window F1) and single-task CNN (0.8590 window F1) outperformed the multi-task model (0.7795 window F1), indicating task interference or capacity sharing trade-offs for cardiac sounds.
   - On ICBHI cycles, the multi-task model (0.2929 cycle F1) substantially exceeded the single-task CNN (0.2162 cycle F1), while performing within 4.17% of the best classical ML model (0.3346).
   - On Kaggle disease classification, the multi-task model achieved patient-level Macro F1 of **0.5175**, surpassing both the single-task CNN baseline (0.3385) and the PANNs frozen linear classifier (0.2206), while matching the classical ML SVM baseline (0.5084).
