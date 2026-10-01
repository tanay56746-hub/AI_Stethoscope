# PANNs CNN14 Frozen Embeddings Classifier Benchmark Report
**AI Stethoscope Research Project**  
**Date:** September 23, 2026  
**Status:** Evaluation Completed — Benchmark Synthesis  
**Scope:** Multi-Classifier Evaluation on Frozen 2048-Dimensional PANNs CNN14 AudioSet Embeddings (Heart, ICBHI, Kaggle)

> [!IMPORTANT]
> **Clinical Disclaimer:** This benchmark report and its associated machine learning models are designed strictly for computational research and acoustic representation comparison. None of these models have undergone clinical trials, medical device certification, or prospective clinical evaluation. They must NOT be used for clinical diagnosis, patient triage, or medical decision-making.

---

## 1. Objective

The primary objective of this benchmark is to rigorously assess the transferability and clinical utility of **frozen 2048-dimensional PANNs CNN14 representations** pretrained on AudioSet ($2.1\times 10^6$ environmental and musical sound clips, 527 ontology classes). We evaluate five distinct classifier families across three core stethoscope auscultation tasks:
1. **Heart Sound Classification (PhysioNet 2016 / CinC Challenge):** Binary cardiac pathology detection (`Normal` vs `Abnormal`).
2. **ICBHI 2017 Respiratory Sound Database:** 4-class adventitious lung cycle classification (`Normal`, `Crackle`, `Wheeze`, `Crackle+Wheeze`).
3. **Kaggle Respiratory Disease Database:** 3-class patient-level respiratory pathology diagnosis (`Normal`, `Asthma`, `Other`).

This investigation directly answers whether high-capacity, general-purpose audio foundation features can substitute for or surpass:
- End-to-end task-specific CNN backbones trained from scratch.
- Compact, domain-engineered psychoacoustic feature representations (397 physical moments, MFCC trajectories, and spectral envelopes).

---

## 2. Experimental Setup & Leakage-Free Protocol

To ensure 100% scientific validity and reproducibility, the evaluation strictly adheres to the following leakage-free experimental protocol:

```
[Frozen AudioSet PANNs CNN14] 
            │ (Extract 2048-d Global Embeddings)
            ▼
  ┌───────────────────┐
  │ Train Split (N_tr)│ ──> Fit StandardScaler ──> Fit Candidate Classifiers
  └───────────────────┘             │
                                    │ (Fixed Transformation)
  ┌───────────────────┐             ▼
  │  Val Split (N_v)  │ ──> Transform Scaler   ──> Evaluate All 5 Classifiers ──> Select Winner (Val Macro F1)
  └───────────────────┘                                                                  │
                                                                                         ▼
  ┌───────────────────┐                                                       Evaluate Best Model ONCE
  │ Test Split (N_te) │ ──> Transform Scaler   ──────────────────────────────────────────┘
  └───────────────────┘                                                                  │
                                                                                         ▼
                                                                  [Window & Soft-Voting Patient Metrics]
```

### 2.1 Frozen Embedding Guarantee
All 2048-dimensional embedding vectors were loaded directly from `processed/panns_embeddings/{heart,icbhi,kaggle}/{train,val,test}.npz`. No backpropagation into PANNs weights was performed, no audio segments were reprocessed, and no test data influenced feature scaling or parameter selection.

### 2.2 Five Candidate Classifier Families
1. **Logistic Regression:** Multinomial / binary logistic regression optimized with L-BFGS, balanced class weights ($w_c = N / (K \cdot N_c)$), and `max_iter=1000`.
2. **Linear Support Vector Machine (Linear SVM):** 
   - Kaggle & ICBHI: `LinearSVC(dual=False, C=0.01, class_weight='balanced')`.
   - Heart ($N=19,117$): `SGDClassifier(loss='hinge', class_weight='balanced', max_iter=2000)`.
   - Margin-to-probability mapping via sigmoid/softmax on decision function outputs for soft-voting patient aggregation.
3. **Nonlinear Support Vector Machine (RBF SVM):**
   - Kaggle ($N=1,296$) & ICBHI ($N=3,804$): Exact kernel `SVC(kernel='rbf', C=1.0, class_weight='balanced')` with margin softmax/sigmoid mapping.
   - Heart ($N=19,117$): Scalable RBF kernel approximation via `Nystroem(kernel='rbf', n_components=500)` coupled to `SGDClassifier(loss='hinge', class_weight='balanced')` to ensure CPU tractability.
4. **Random Forest:** Ensemble of 100 trees, `max_depth=15`, `min_samples_split=4`, balanced class weighting, evaluated directly on raw embeddings.
5. **Lightweight Multi-Layer Perceptron (MLP):** Compact architecture (`Linear(2048, 128) -> ReLU -> Linear(128, K)`), L2 weight regularization ($\alpha=10^{-4}$), balanced sample weighting, and early stopping with 5 epochs patience monitoring validation performance.

### 2.3 Model Selection Rule
Model selection was governed **exclusively by Validation Window Macro F1**. The highest-performing model per task was evaluated **exactly once** on the held-out test split.

---

## 3. Dataset Statistics & Patient Isolation Audit

All splits maintain strict patient/subject isolation. Zero recording or window overlap exists across partitions.

| Dataset / Task | Classes | Train Samples (Patients) | Val Samples (Patients) | Test Samples (Patients) | Total Samples (Patients) | Patient Leakage Audit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle Respiratory** | 3 (`Normal`, `Asthma`, `Other`) | 1,296 (67) | 218 (22) | 288 (23) | 1,802 (112) | **0 Overlap (Passed)** |
| **ICBHI 2017 Cycles** | 4 (`Normal`, `Crackle`, `Wheeze`, `Both`) | 3,804 (76) | 1,549 (26) | 1,545 (24) | 6,898 (126) | **0 Overlap (Passed)** |
| **Heart PhysioNet 2016**| 2 (`Normal`, `Abnormal`) | 19,117 (2,000) | 2,358 (429) | 2,427 (427) | 23,902 (2,856)| **0 Overlap (Passed)** |

---

## 4. Validation Benchmark & Model Selection

Every candidate classifier was trained exclusively on the training split and benchmarked across the validation split.

### 4.1 Kaggle Respiratory Disease Validation Suite
*Train: 1,296 samples \| Validation: 218 samples (22 patients)*

| Classifier | Input Rep | Val Accuracy | Val Balanced Acc | Val Macro F1 | Val Weighted F1 | Training Time | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | Scaled | **36.24%** | **35.87%** | **0.3566** | **35.96%** | 7.00s | **WINNER (Selected)** 🏆 |
| **Logistic Regression** | Scaled | 34.86% | 34.90% | 0.3482 | 34.88% | 1.57s | Candidate |
| **Random Forest** | Raw | 35.78% | 33.60% | 0.3256 | 34.19% | 0.25s | Candidate |
| **RBF SVM** | Scaled | 31.19% | 30.97% | 0.3106 | 31.20% | 1.13s | Candidate |
| **Lightweight MLP** | Scaled | 33.94% | 31.97% | 0.3088 | 32.22% | 0.40s | Candidate |

### 4.2 ICBHI 2017 Respiratory Cycle Validation Suite
*Train: 3,804 samples \| Validation: 1,549 samples (26 patients)*

| Classifier | Input Rep | Val Accuracy | Val Balanced Acc | Val Macro F1 | Val Weighted F1 | Training Time | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lightweight MLP** | Scaled | 45.06% | **35.40%** | **0.3504** | **46.72%** | 2.97s | **WINNER (Selected)** 🏆 |
| **RBF SVM** | Scaled | 42.09% | 33.38% | 0.3317 | 44.09% | 13.12s | Candidate |
| **Random Forest** | Raw | **46.55%** | 32.73% | 0.3272 | 45.89% | 0.43s | Candidate |
| **Linear SVM** | Scaled | 41.58% | 33.32% | 0.3267 | 43.33% | 58.30s | Candidate |
| **Logistic Regression** | Scaled | 39.64% | 32.72% | 0.3136 | 41.55% | 5.64s | Candidate |

### 4.3 Heart PhysioNet 2016 Validation Suite
*Train: 19,117 samples \| Validation: 2,358 samples (429 patients)*

| Classifier | Input Rep | Val Accuracy | Val Balanced Acc | Val Macro F1 | Val Weighted F1 | Training Time | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF SVM (Nystroem)** | Scaled | 67.85% | 61.45% | **0.5956** | **69.56%** | 1.18s | **WINNER (Selected)** 🏆 |
| **Lightweight MLP** | Scaled | 67.68% | 60.81% | 0.5911 | 69.34% | 12.06s | Candidate |
| **Logistic Regression** | Scaled | 65.86% | **61.74%** | 0.5886 | 68.13% | 14.03s | Candidate |
| **Random Forest** | Raw | 67.98% | 59.75% | 0.5855 | 69.35% | 4.50s | Candidate |
| **Linear SVM (SGD)** | Scaled | **68.28%** | 55.98% | 0.5582 | 68.64% | 6.92s | Candidate |

---

## 5. Final Held-Out Test Results (Window Level)

The winning model selected from validation was evaluated **once** on the held-out test split.

| Task / Dataset | Winning Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle** ($N=288$) | **Linear SVM** | 33.33% | 32.51% | 0.3309 | 0.3251 | **0.3223** | 32.63% | 0.004s |
| **ICBHI** ($N=1,545$) | **Lightweight MLP** | 42.91% | 33.93% | 0.3577 | 0.3393 | **0.3391** | 42.02% | 0.008s |
| **Heart** ($N=2,427$) | **RBF SVM** | 67.99% | 60.64% | 0.5877 | 0.6064 | **0.5912** | 69.51% | 0.223s |

### Per-Class Detailed Window Breakdown

#### Kaggle Respiratory Disease (Linear SVM)
- **Normal (Support: 87):** Precision: 0.3529 | Recall: 0.2759 | **F1: 0.3097**
- **Asthma (Support: 93):** Precision: 0.2973 | Recall: 0.2366 | **F1: 0.2635**
- **Other (Support: 108):** Precision: 0.3425 | Recall: 0.4630 | **F1: 0.3937**

#### ICBHI 2017 Respiratory Cycles (Lightweight MLP)
- **Normal (Support: 651):** Precision: 0.5066 | Recall: 0.6467 | **F1: 0.5682**
- **Crackle (Support: 480):** Precision: 0.5486 | Recall: 0.3292 | **F1: 0.4115**
- **Wheeze (Support: 256):** Precision: 0.2138 | Recall: 0.2422 | **F1: 0.2271**
- **Crackle+Wheeze (Support: 158):** Precision: 0.1618 | Recall: 0.1392 | **F1: 0.1497**

#### Heart PhysioNet 2016 (RBF SVM)
- **Normal (Support: 1,876):** Precision: 0.8269 | Recall: 0.7409 | **F1: 0.7816** | **Specificity: 74.09%**
- **Abnormal (Support: 551):** Precision: 0.3485 | Recall: 0.4719 | **F1: 0.4009** | **Sensitivity: 47.19%**

---

## 6. Patient/Entity-Level Results (Soft-Voting Aggregation)

Predicted probabilities across all windows/cycles belonging to the same subject were aggregated via mean soft-voting:
$$\bar{p}_k^{(i)} = \frac{1}{|W_i|} \sum_{w \in W_i} p_k(w), \quad \hat{y}^{(i)} = \arg\max_k \bar{p}_k^{(i)}$$

| Task / Dataset | Subject Count | Patient Accuracy | Patient Balanced Acc | Patient Macro F1 | Patient Weighted F1 | Specificity / Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle** | 23 patients | 30.43% (7/23) | 26.98% | **0.2206** | 24.32% | — |
| **ICBHI** | 24 patients | 66.67% (16/24) | 30.39% | **0.3224** | 62.17% | — |
| **Heart** | 427 recordings | 75.18% (321/427) | 63.46% | **0.6299** | 75.57% | Sens: 43.68% \| Spec: 83.24% |

### Patient-Level Per-Class Breakdown

* **Kaggle (23 Test Patients):**
  - `Normal` (7 patients): Precision: 0.2500, Recall: 0.1429 (1/7), F1: 0.1818
  - `Asthma` (7 patients): Precision: 0.0000, Recall: 0.0000 (0/7), F1: 0.0000
  - `Other` (9 patients): Precision: 0.3750, Recall: 0.6667 (6/9), F1: 0.4800
* **ICBHI (24 Test Patients):**
  - `Normal` (17 patients): Precision: 0.7143, Recall: 0.8824 (15/17), F1: 0.7895
  - `Crackle` (3 patients): Precision: 1.0000, Recall: 0.3333 (1/3), F1: 0.5000
  - `Wheeze` (1 patient): Precision: 0.0000, Recall: 0.0000 (0/1), F1: 0.0000
  - `Crackle+Wheeze` (3 patients): Precision: 0.0000, Recall: 0.0000 (0/3), F1: 0.0000
* **Heart (427 Test Recordings):**
  - `Normal` (340 subjects): Precision: 0.8524, Recall: 0.8324 (283/340), F1: 0.8423 (**Specificity: 83.24%**)
  - `Abnormal` (87 subjects): Precision: 0.4000, Recall: 0.4368 (38/87), F1: 0.4176 (**Sensitivity: 43.68%**)

---

## 7. Confusion Matrices

### 7.1 Kaggle Respiratory Disease (Linear SVM)

**Window Confusion Matrix ($N=288$):**
```
                Predicted Normal   Predicted Asthma   Predicted Other
Actual Normal          24                 14                 49
Actual Asthma          24                 22                 47
Actual Other           20                 38                 50
```

**Patient Confusion Matrix ($N=23$):**
```
                Predicted Normal   Predicted Asthma   Predicted Other
Actual Normal           1                  1                  5
Actual Asthma           2                  0                  5
Actual Other            1                  2                  6
```

### 7.2 ICBHI 2017 Respiratory Cycles (Lightweight MLP)

**Cycle Confusion Matrix ($N=1,545$):**
```
                    Predicted Normal   Predicted Crackle   Predicted Wheeze   Predicted Both
Actual Normal             421                 73                 132                25
Actual Crackle            227                158                  54                41
Actual Wheeze             118                 28                  62                48
Actual Both                65                 29                  42                22
```

**Patient Confusion Matrix ($N=24$):**
```
                    Predicted Normal   Predicted Crackle   Predicted Wheeze   Predicted Both
Actual Normal              15                  0                   2                 0
Actual Crackle              2                  1                   0                 0
Actual Wheeze               1                  0                   0                 0
Actual Both                 3                  0                   0                 0
```

### 7.3 Heart PhysioNet 2016 (RBF SVM)

**Window Confusion Matrix ($N=2,427$):**
```
                Predicted Normal   Predicted Abnormal
Actual Normal         1,390               486
Actual Abnormal         291               260
```

**Patient/Recording Confusion Matrix ($N=427$):**
```
                Predicted Normal   Predicted Abnormal
Actual Normal          283                 57
Actual Abnormal         49                 38
```

---

## 8. Cross-Paradigm Comparison: CNN Baseline vs Classical ML vs Frozen PANNs

This section presents the primary scientific synthesis: benchmarking the three acoustic representation paradigms on identical held-out test splits.

| Auscultation Task | Evaluation Level | Metric | Custom CNN Baseline (from Scratch) | Classical ML Baseline (397-d Handcrafted) | Frozen PANNs CNN14 (2048-d AudioSet) | Paradigm Winner & Empirical Margin |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Heart Sound**<br>*(PhysioNet 2016)* | **Window** | **Accuracy** | 89.04% | **90.11%** *(RF)* | 67.99% *(RBF SVM)* | **Classical ML (+22.12% over PANNs)** |
| | | **Balanced Accuracy** | **90.15%** | 89.50% *(RF)* | 60.64% *(RBF SVM)* | **Custom CNN (+29.51% over PANNs)** |
| | | **Macro F1** | 85.90% | **86.82%** *(RF)* | 59.12% *(RBF SVM)* | **Classical ML (+27.70% over PANNs)** |
| | | **Abnormal Sensitivity**| **92.20%** | 88.38% *(RF)* | 47.19% *(RBF SVM)* | **Custom CNN (+45.01% over PANNs)** |
| | | **Normal Specificity** | 88.11% | **90.62%** *(RF)* | 74.09% *(RBF SVM)* | **Classical ML (+16.53% over PANNs)** |
| | **Patient** | **Accuracy** | 91.80% | **93.44%** *(RF)* | 75.18% *(RBF SVM)* | **Classical ML (+18.26% over PANNs)** |
| | | **Macro F1** | 88.69% | **90.59%** *(RF)* | 62.99% *(RBF SVM)* | **Classical ML (+27.60% over PANNs)** |
| | | **Abnormal Sensitivity**| **96.55%** | 94.25% *(RF)* | 43.68% *(RBF SVM)* | **Custom CNN (+52.87% over PANNs)** |
| | | **Normal Specificity** | 90.59% | **93.24%** *(RF)* | 83.24% *(RBF SVM)* | **Classical ML (+10.00% over PANNs)** |
| **ICBHI 2017**<br>*(4-Class Acoustic)*| **Window** | **Accuracy** | 23.82% | 38.90% *(SVM)* | **42.91%** *(MLP)* | **PANNs Embeddings (+4.01% over ClassML)** 🏆 |
| | | **Balanced Accuracy** | 24.52% | **34.15%** *(SVM)* | 33.93% *(MLP)* | **Classical ML (+0.22% over PANNs)** |
| | | **Macro F1** | 21.62% | 33.46% *(SVM)* | **33.91%** *(MLP)* | **PANNs Embeddings (+0.45% over ClassML)** 🏆 |
| | | **Normal F1** | 17.56% | 52.13% *(SVM)* | **56.82%** *(MLP)* | **PANNs Embeddings (+4.69% over ClassML)** 🏆 |
| | | **Crackle F1** | 39.54% | 34.76% *(SVM)* | **41.15%** *(MLP)* | **PANNs Embeddings (+1.61% over CNN)** 🏆 |
| | | **Wheeze F1** | 14.88% | **23.84%** *(SVM)* | 22.71% *(MLP)* | **Classical ML (+1.13% over PANNs)** |
| | | **Crackle+Wheeze F1** | 14.50% | **22.51%** *(SVM)* | 14.97% *(MLP)* | **Classical ML (+7.54% over PANNs)** |
| | **Patient** | **Accuracy** | 16.67% | **70.83%** *(SVM)* | 66.67% *(MLP)* | **Classical ML (+4.16% over PANNs)** |
| | | **Macro F1** | 19.03% | 21.79% *(SVM)* | **32.24%** *(MLP)* | **PANNs Embeddings (+10.45% over ClassML)** 🏆 |
| **Kaggle**<br>*(3-Class Disease)* | **Window** | **Accuracy** | 35.76% | **42.01%** *(SVM)* | 33.33% *(LinSVM)* | **Classical ML (+8.68% over PANNs)** |
| | | **Balanced Accuracy** | 36.46% | **41.72%** *(SVM)* | 32.51% *(LinSVM)* | **Classical ML (+9.21% over PANNs)** |
| | | **Macro F1** | 35.68% | **39.83%** *(SVM)* | 32.23% *(LinSVM)* | **Classical ML (+7.60% over PANNs)** |
| | **Patient** | **Accuracy** | 34.78% | **56.52%** *(SVM)* | 30.43% *(LinSVM)* | **Classical ML (+26.09% over PANNs)** |
| | | **Macro F1** | 33.85% | **50.84%** *(SVM)* | 22.06% *(LinSVM)* | **Classical ML (+28.78% over PANNs)** |

---

## 9. Factual Scientific Observations

1. **Acoustic Domain Mismatch on Cardiac Signals (Heart Task):**
   - Frozen AudioSet PANNs exhibits severe performance degradation on heart sound auscultations: test Macro F1 is **59.12%** vs **86.82%** for Classical ML and **85.90%** for Custom CNN.
   - PANNs detects abnormal cardiac sounds with only **47.19% window sensitivity** (vs **92.20%** for Custom CNN).
   - *Acoustic Mechanism:* AudioSet pretraining focuses on broad-spectrum, heterogeneous environmental sounds (engines, animal vocalizations, musical instruments) with energy spanning up to 8 kHz. In contrast, heart auscultation signals are quasi-periodic low-frequency waveforms with critical diagnostic energy concentrated below 500 Hz (valve closure transience and murmurs). Without fine-tuning, PANNs embeddings fail to encode subtle low-frequency cardiac timings.

2. **Superior Representation for Fine Respiratory Acoustics (ICBHI Task):**
   - On ICBHI 2017 respiratory cycle classification, frozen PANNs embeddings combined with a Lightweight MLP establish the **new overall state-of-the-art across all baselines**:
     - Window Accuracy: **42.91%** (vs 23.82% for CNN, 38.90% for Classical ML)
     - Window Macro F1: **33.91%** (vs 21.62% for CNN, 33.46% for Classical ML)
     - Patient Macro F1: **32.24%** (vs 19.03% for CNN, 21.79% for Classical ML)
     - Normal F1: **56.82%**; Crackle F1: **41.15%**.
   - *Acoustic Mechanism:* AudioSet pretraining exposes CNN14 to transient explosive events and continuous tonal acoustics. These structural priors transfer remarkably well to crackles (<20 ms explosive bursts) and wheezes (>100 ms musical tones), far outperforming a 422k-parameter CNN trained from scratch on small ICBHI data.

3. **Vulnerability of Global AudioSet Embeddings to Environmental Noise (Kaggle Task):**
   - On the Kaggle dataset, PANNs embeddings (Linear SVM: 32.23% Macro F1, 22.06% Patient Macro F1) underperformed both the Custom CNN (35.68% Macro F1) and Classical ML (39.83% Macro F1).
   - On patient level, Classical ML achieved **50.84% Macro F1**, outperforming PANNs by **+28.78%**.
   - *Acoustic Mechanism:* The Kaggle recordings contain significant non-clinical ambient artifacts (background speech, sensor handling friction). PANNs' global average pooling aggregates these prominent environmental acoustic cues into the 2048-d vector, misleading linear classifiers. In contrast, engineered psychoacoustic features (spectral centroid, bandwidth, MFCC acceleration) isolate the spectral decay characteristic of genuine lung sounds.

4. **Classifier Head Dynamics on Frozen Embeddings:**
   - Linear SVM won on Kaggle (Val Macro F1: 0.3566), demonstrating that regularized linear margins prevent overfitting on high-dimensional vectors (2048 dims with $N=1,296$).
   - Lightweight MLP won on ICBHI (Val Macro F1: 0.3504), showing that a nonlinear projection (`2048 -> 128 -> 4`) with balanced sample weighting effectively decouples complex 4-class respiratory acoustics.
   - Scalable RBF SVM (Nystroem kernel approximation + SGD hinge loss) won on Heart (Val Macro F1: 0.5956), training in just 1.18 seconds on 19,117 samples.

---

## 10. Limitations

1. **Frozen Representation Constraint:** PANNs CNN14 weights were strictly frozen. Fine-tuning the upper convolutional layers (`conv_block5`, `conv_block6`) with domain-specific cardiac and respiratory data would likely mitigate the low-frequency domain mismatch observed in the Heart task.
2. **AudioSet Sampling Rate Discrepancy:** PANNs was pretrained on 32 kHz audio, whereas auscultation datasets were standardized at 16 kHz and upsampled. Any pretraining features reliant on >8 kHz frequency bands remained uninformative.
3. **Class Imbalance in Multi-Class Tasks:** Extreme class rarity in ICBHI (`Crackle+Wheeze`: 158 test cycles) and Kaggle (`Asthma`: 7 test patients) produces volatile per-class F1 metrics at the patient level.
4. **Window-Level to Patient-Level Soft-Voting:** Uniform soft-voting across all windows weights noisy or silent segments equally with diagnostically rich segments. Attention-weighted or peak-probability pooling could further enhance patient-level discrimination.

---

## 11. Reproducibility Audit (10-Point Checklist)

| Checkpoint | Requirement | Verification Method | Status |
| :---: | :--- | :--- | :---: |
| **1** | **No Test Data in Preprocessing** | `StandardScaler` fitted strictly on `X_train_raw`. Val and Test transformed. | **PASSED** |
| **2** | **Zero Patient Overlap** | Verified set intersection $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. | **PASSED** |
| **3** | **No Test-Time Model Selection** | Winning classifier selected strictly by highest Validation Macro F1. | **PASSED** |
| **4** | **Single Test Evaluation** | Selected model evaluated exactly once on Held-Out Test split. | **PASSED** |
| **5** | **Frozen Weights Maintained** | Embeddings loaded as fixed `.npz` arrays; zero backpropagation into PANNs backbone. | **PASSED** |
| **6** | **Deterministic Seeds** | All models and estimators initialized with `random_state=42`. | **PASSED** |
| **7** | **Standardized Class Mapping** | Integer label IDs consistently mapped across Train, Val, and Test splits. | **PASSED** |
| **8** | **Model & Scaler Artifacts Stored** | `best_model.joblib`, `scaler.joblib`, and `feature_config.json` saved in experiment directories. | **PASSED** |
| **9** | **Metric Files Stored** | `val_comparison.json` and `test_metrics.json` preserved in `experiments/panns_classifiers/{heart,icbhi,kaggle}/metrics/`. | **PASSED** |
| **10** | **Figures Stored** | `confusion_matrix.png`, `val_confusion_matrix.png`, and `patient_confusion_matrix.png` generated and verified. | **PASSED** |
