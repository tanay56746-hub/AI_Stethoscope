# AI Digital Stethoscope: Final Architecture Design & Integration Plan
**AI Stethoscope Research Project**  
**Date:** September 23, 2026  
**Status:** Architecture Blueprint & Integration Proposal (Strictly Read-Only Analysis)  
**Scope:** Synthesis of Verified Empirical Baselines into a Defensible Multi-Task Digital Stethoscope System

> [!IMPORTANT]
> **Clinical Disclaimer:** This design document proposes a computational architecture for academic and scientific research. The proposed model and algorithms have not undergone clinical trials, regulatory clearance (e.g., FDA 510(k), CE mark), or prospective clinical validation. They must NOT be used for clinical diagnostic decision-making, patient monitoring, or treatment guidance.

---

## 1. Executive Summary

Across the preceding phases of the AI Stethoscope project, three distinct paradigms were empirically benchmarked on identical, patient-isolated held-out test sets across three clinical auscultation tasks:
1. **Custom 2D CNN Baseline (from Scratch):** Standardized Log-Mel spectrogram inputs with a 422k-parameter convolutional backbone.
2. **Classical ML Baseline:** Compact 397-dimensional domain-engineered acoustic features (spectral centroid, bandwidth, rolloff, RMS dynamics, and MFCC trajectories).
3. **Frozen PANNs CNN14 Pretrained Embeddings:** 2048-dimensional transfer features extracted from an 81M-parameter AudioSet foundation model.

The verified experimental results establish three foundational scientific facts:
- **Fact 1:** Pretrained AudioSet representations alone are **not** universally superior across auscultation tasks. While frozen PANNs embeddings achieved the top performance on ICBHI 2017 respiratory cycle classification (**33.91% Macro F1**, **42.91% Accuracy**, **32.24% Patient Macro F1**), they suffered a catastrophic drop on Heart sounds (**59.12% Macro F1** vs **85.90%** for Custom CNN and **86.82%** for Classical ML).
- **Fact 2:** Domain-engineered handcrafted acoustic features demonstrated remarkable resilience, outperforming deep representations on Kaggle Respiratory Disease (**39.83% Window / 50.84% Patient Macro F1**) and matching deep CNNs on Heart sounds (**86.82% Window / 90.59% Patient Macro F1**).
- **Fact 3:** Diagnostic tasks must remain decoupled: acoustic findings (crackles and wheezes) are transient physical phenomena occurring within individual breathing cycles, whereas respiratory diseases (such as Asthma) are patient-level chronic conditions. Conflating wheezes with asthma violates medical reality and degrades diagnostic specificity.

Consequently, a monolithic end-to-end PANNs model is **scientifically indefensible**. Instead, we propose a **Domain-Adapted Hybrid Multi-Task Architecture**:
- A **shared lightweight Auscultation CNN Backbone** (~1.2M parameters) optimized on standardized time-frequency spectrograms.
- An **auxiliary frozen PANNs branch** providing high-level acoustic event embeddings specifically for the **ICBHI Lung Acoustic Head**.
- An **auxiliary 397-dimensional handcrafted physical feature projection** fused with CNN representations to anchor cardiac dynamics and disease classification against ambient environmental noise.
- **Three decoupled task-specific heads** trained via **masked multi-task loss**:
  1. *Heart Pathology Head* (`Normal` vs `Abnormal`).
  2. *Lung Acoustic Cycle Head* (`Normal`, `Crackle`, `Wheeze`, `Crackle+Wheeze`).
  3. *Lung Disease Diagnostic Head* (`Normal`, `Asthma`, `Other`).

---

## 2. Verified Evidence from Existing Experiments

All architectural recommendations in this document are grounded exclusively in the verified test numbers from our three authoritative reports:
- [`analysis/full_cnn_baseline_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/full_cnn_baseline_results.md)
- [`analysis/classical_ml_baseline_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/classical_ml_baseline_results.md)
- [`analysis/panns_classifier_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_results.md)
- [`analysis/panns_classifier_verification.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_verification.md)

### Authoritative Held-Out Test Metric Matrix

| Auscultation Task | Metric Level & Metric | Custom CNN Baseline (from Scratch) | Classical ML Baseline (397-d Handcrafted) | Frozen PANNs CNN14 (2048-d AudioSet) | Empirical Best Paradigm |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Heart Sound**<br>*(PhysioNet 2016)* | **Window Accuracy** | 89.04% | **90.11%** *(RF)* | 67.99% *(RBF SVM)* | **Classical ML (+22.12% over PANNs)** |
| | **Window Macro F1** | 85.90% | **86.82%** *(RF)* | 59.12% *(RBF SVM)* | **Classical ML (+27.70% over PANNs)** |
| | **Window Balanced Acc** | **90.15%** | 89.50% *(RF)* | 60.64% *(RBF SVM)* | **Custom CNN (+29.51% over PANNs)** |
| | **Abnormal Sensitivity** | **92.20%** | 88.38% *(RF)* | 47.19% *(RBF SVM)* | **Custom CNN (+45.01% over PANNs)** |
| | **Normal Specificity** | 88.11% | **90.62%** *(RF)* | 74.09% *(RBF SVM)* | **Classical ML (+16.53% over PANNs)** |
| | **Patient Accuracy** | 91.80% | **93.44%** *(RF)* | 75.18% *(RBF SVM)* | **Classical ML (+18.26% over PANNs)** |
| | **Patient Macro F1** | 88.69% | **90.59%** *(RF)* | 62.99% *(RBF SVM)* | **Classical ML (+27.60% over PANNs)** |
| **ICBHI 2017**<br>*(4-Class Acoustic)*| **Window Accuracy** | 23.82% | 38.90% *(SVM)* | **42.91%** *(MLP)* | **PANNs Embeddings (+4.01% over ClassML)** |
| | **Window Macro F1** | 21.62% | 33.46% *(SVM)* | **33.91%** *(MLP)* | **PANNs Embeddings (+0.45% over ClassML)** |
| | **Window Balanced Acc** | 24.52% | **34.15%** *(SVM)* | 33.93% *(MLP)* | **Classical ML (+0.22% over PANNs)** |
| | **Normal F1** | 17.56% | 52.13% *(SVM)* | **56.82%** *(MLP)* | **PANNs Embeddings (+4.69% over ClassML)** |
| | **Crackle F1** | 39.54% | 34.76% *(SVM)* | **41.15%** *(MLP)* | **PANNs Embeddings (+1.61% over CNN)** |
| | **Wheeze F1** | 14.88% | **23.84%** *(SVM)* | 22.71% *(MLP)* | **Classical ML (+1.13% over PANNs)** |
| | **Crackle+Wheeze F1** | 14.50% | **22.51%** *(SVM)* | 14.97% *(MLP)* | **Classical ML (+7.54% over PANNs)** |
| | **Patient Accuracy** | 16.67% | **70.83%** *(SVM)* | 66.67% *(MLP)* | **Classical ML (+4.16% over PANNs)** |
| | **Patient Macro F1** | 19.03% | 21.79% *(SVM)* | **32.24%** *(MLP)* | **PANNs Embeddings (+10.45% over ClassML)** |
| **Kaggle**<br>*(3-Class Disease)* | **Window Accuracy** | 35.76% | **42.01%** *(SVM)* | 33.33% *(LinSVM)* | **Classical ML (+8.68% over PANNs)** |
| | **Window Macro F1** | 35.68% | **39.83%** *(SVM)* | 32.23% *(LinSVM)* | **Classical ML (+7.60% over PANNs)** |
| | **Window Balanced Acc** | 36.46% | **41.72%** *(SVM)* | 32.51% *(LinSVM)* | **Classical ML (+9.21% over PANNs)** |
| | **Patient Accuracy** | 34.78% | **56.52%** *(SVM)* | 30.43% *(LinSVM)* | **Classical ML (+26.09% over PANNs)** |
| | **Patient Macro F1** | 33.85% | **50.84%** *(SVM)* | 22.06% *(LinSVM)* | **Classical ML (+28.78% over PANNs)** |

---

## 3. Representation Analysis: Selecting the Shared Backbone

The central design dilemma is whether the primary shared backbone should be:
1. An auscultation-specific CNN trained from scratch.
2. A large AudioSet pretrained backbone (PANNs CNN14).
3. The 397-dimensional handcrafted acoustic feature vector.
4. A hybrid multi-modal representation.

### Analysis of Candidates

| Candidate Backbone | Advantages Supported by Evidence | Disadvantages Demonstrated by Evidence | Verdict |
| :--- | :--- | :--- | :---: |
| **Monolithic PANNs CNN14** | Top performance on ICBHI (33.91% window / 32.24% patient macro F1); captures audio transients well. | Catastrophic failure on Heart (59.12% vs 86.82% Macro F1, 47.19% sensitivity); 81M parameters; high CPU latency (~300ms). | **REJECTED AS MONOLITHIC PRIMARY** |
| **Pure Custom CNN (from scratch)** | High performance on Heart (85.90% Macro F1, 92.20% sensitivity); compact (~422k params, <10ms CPU inference). | Fails on small-sample ICBHI (21.62% Macro F1, 16.67% patient accuracy); severe overfitting to dominant class. | **REJECTED AS SOLE BACKBONE** |
| **Pure Handcrafted Features (397-d)** | Highest overall scores on Heart (86.82% F1) and Kaggle (39.83% window, 50.84% patient F1); strong on ICBHI (33.46% F1); lightweight. | Non-differentiable fixed representation; cannot be end-to-end fine-tuned; unable to exploit raw audio spatio-temporal features. | **REJECTED AS SOLE BACKBONE** |
| **Hybrid Auscultation Backbone** | Combines trainable 2D CNN spatio-temporal features with fixed physical acoustic moments and auxiliary PANNs embeddings. | Requires multi-branch fusion logic and slightly more complex forward pass. | **SELECTED AS FINAL ARCHITECTURE** 🏆 |

### Scientific Verdict
- **FACT:** No single representation excels across all three auscultation tasks simultaneously.
- **INFERENCE:** The acoustic physics of heart sounds (low-frequency periodic transients <500 Hz) and adventitious lung sounds (high-frequency explosive bursts and wheezing resonances >200–2000 Hz) require distinct inductive biases.
- **RECOMMENDATION:** The final shared architecture must utilize a **Hybrid Auscultation Representation**: a trainable shared 2D CNN backbone providing core auscultation representations, augmented by task-specific auxiliary projections (PANNs for ICBHI cycles, 397-d handcrafted features for Heart and Kaggle disease).

---

## 4. PANNs Role Analysis

Based on the verified experimental evidence, we address the role of PANNs CNN14:

- **Option A (Primary Shared Backbone):** **REJECTED.** PANNs achieved only 59.12% Macro F1 on Heart sounds (a 27.70% deficit compared to Classical ML). Using PANNs as the sole shared trunk would severely degrade cardiac abnormality detection.
- **Option B (Auxiliary Pretrained Feature Branch across All Tasks):** **REJECTED.** Injecting 2048-dimensional AudioSet embeddings into the Heart and Kaggle heads increased parameter count and noise susceptibility without performance benefit (Kaggle PANNs scored 32.23% vs 39.83% for Classical ML).
- **Option C (Auxiliary Pretrained Branch Exclusively for Lung Acoustic Head):** **ACCEPTED.** PANNs achieved the project's highest scores on ICBHI cycle classification (42.91% window accuracy, 33.91% window macro F1, 41.15% crackle F1, and 32.24% patient macro F1). Its broad environmental pretraining provides rich priors for explosive (<20 ms) and tonal (>100 ms) acoustic bursts.
- **Option D (Exclude from Final Model Entirely):** **REJECTED.** Completely omitting PANNs would forfeit the substantial +12.29% macro F1 improvement demonstrated on ICBHI over the custom CNN.

### Final PANNs Specification
PANNs CNN14 shall serve as an **Auxiliary Pretrained Feature Extractor dedicated to the Lung Acoustic Branch**. Its 2048-dimensional embedding vector is extracted, projected through a linear bottleneck (`Linear(2048, 128)`), and concatenated with the shared CNN acoustic representation before the ICBHI classification head.

---

## 5. Handcrafted Feature Analysis

The 397-dimensional handcrafted feature set includes:
- Temporal mean and standard deviation of 128 Log-Mel frequency bands (256 dims).
- Static MFCCs, $\Delta$ MFCCs, and $\Delta\Delta$ MFCC trajectories (120 dims).
- Frame-level RMS energy dynamics (mean, std, min, max, skewness) (5 dims).
- Physical spectral moments: Spectral Centroid, Bandwidth, Rolloff (85%), and ZCR estimator (16 dims).

### Role in Final Architecture
- **Option A (Completely Excluded):** **REJECTED.** Discarding handcrafted features ignores the top-performing model on Heart sounds (86.82% Macro F1, 93.44% Patient Accuracy) and Kaggle (39.83% Window, 50.84% Patient Macro F1).
- **Option B (Auxiliary Physical Feature Branch):** **ACCEPTED.** 
  - Handcrafted features provide immediate physical grounding (e.g., spectral centroid directly tracks airway narrowing; RMS energy skewness tracks heart valve impact sharpness).
  - Integration: Rather than feeding raw 397-dimensional vectors directly to classifiers, they pass through a lightweight trainable linear projection layer:
    $$\mathbf{z}_{\text{phys}} = \text{LayerNorm}\left(\text{ReLU}\left(\mathbf{W}_{\text{phys}} \mathbf{x}_{\text{397}} + \mathbf{b}_{\text{phys}}\right)\right), \quad \mathbf{W}_{\text{phys}} \in \mathbb{R}^{64 \times 397}$$
  - This 64-dimensional physical vector is concatenated with the pooled CNN representations before entering the Heart Head and Kaggle Disease Head.
- **Option C (Primary Representation):** **REJECTED.** Fixed features prevent task-specific gradient backpropagation from raw spectrograms.
- **Option D (Benchmark Only):** **REJECTED.** The empirical gains on Heart and Kaggle demonstrate clear functional utility.

---

## 6. Heart Head Design

### 6.1 Clinical Objective
Binary classification of cardiac auscultation recordings into `Normal` (Class 0) vs `Abnormal` (Class 1: structural murmurs, valvular regurgitation/stenosis, friction rubs).

### 6.2 Structural Specifications
```
Shared CNN Feature Map (B, 256, T_frames, F_bins)
                    │
                    ▼
         [Frequency Global Pooling] ──> (B, 256, T_frames)
                    │
                    ▼
         [Temporal Rhythm Pooling]  ──> (B, 256)
                    │
                    ├──────────────────────────┐
                    ▼                          ▼
            CNN Vector (B, 256)        Physical Vector (B, 64)
                    │                          │
                    └───────────┬──────────────┘
                                ▼
                      Concatenation (B, 320)
                                │
                                ▼
                       Linear(320 -> 128)
                                │
                                ▼
                      BatchNorm1d + ReLU
                                │
                                ▼
                          Dropout(0.30)
                                │
                                ▼
                       Linear(128 -> 2)
                                │
                                ▼
                   Heart Logits [Normal, Abnormal]
```

### 6.3 Technical Rationale
- **Input Representation:** Fused 320-d vector (256-d pooled CNN features + 64-d projected physical acoustic features).
- **Temporal Rhythm Pooling:** Heart sounds are quasi-periodic (S1-systole-S2-diastole). Averaging across time frames while preserving lower Mel bands (1–40, spanning 20–500 Hz) captures murmur energy envelopes without losing cycle periodicity.
- **Loss Function:** Class-Weighted Cross-Entropy Loss:
  $$w_{\text{Normal}} = 0.655, \quad w_{\text{Abnormal}} = 2.108$$
- **Inference Thresholding:** In clinical cardiac screening, missing an abnormal heart condition carries severe risk. Operating threshold on $P(\text{Abnormal})$ should be tuned on validation data to achieve $\ge 90\%$ Sensitivity while maintaining $\ge 85\%$ Specificity.

---

## 7. Lung Acoustic Head Design (ICBHI 2017)

### 7.1 Clinical Objective
Fine-grained multi-class classification of individual respiratory cycles into four acoustic states:
1. `Normal` (Class 0): Vesicular breath sounds without adventitious acoustics.
2. `Crackle` (Class 1): Explosive, discontinuous crackles (<20 ms).
3. `Wheeze` (Class 2): Musical continuous tonal wheezes (>100 ms).
4. `Crackle+Wheeze` (Class 3): Co-occurring adventitious events within the same cycle.

### 7.2 Structural Specifications
```
Spectrogram (1, 128, 313)                    Raw Audio Cycle (16 kHz)
          │                                              │
          ▼                                              ▼
   Shared Auscultation CNN                      Frozen PANNs CNN14
          │                                              │
          ▼                                              ▼
Feature Map (B, 256, T, F)                     Embedding (B, 2048)
          │                                              │
          ▼                                              ▼
[Masked Temporal Avg Pooling (MTAP)]            Linear Bottleneck(2048 -> 128)
          │                                              │
          ▼                                              ▼
  CNN Vector (B, 256)                           PANNs Vector (B, 128)
          │                                              │
          └──────────────────────┬───────────────────────┘
                                 ▼
                       Concatenation (B, 384)
                                 │
                                 ▼
                        Linear(384 -> 128)
                                 │
                                 ▼
                        LayerNorm + ReLU
                                 │
                                 ▼
                           Dropout(0.25)
                                 │
                                 ▼
                         Linear(128 -> 4)
                                 │
                                 ▼
                   Acoustic Logits [Norm, Crk, Whz, Both]
```

### 7.3 Technical Rationale
- **Masked Temporal Average Pooling (MTAP):** Respiratory cycles vary from 0.8 s to >6.0 s and are zero-padded to 5.0 s (313 frames). Standard global pooling includes silent padding frames, corrupting the feature average. MTAP enforces:
  $$\mathbf{h}_{\text{pooled}} = \frac{\sum_{t=1}^T m(t) \cdot \mathbf{h}(t)}{\sum_{t=1}^T m(t) + \epsilon}$$
  where $m(t) \in \{0, 1\}$ is the boolean validity mask for active audio frames.
- **PANNs Auxiliary Branch:** Verified test results proved that PANNs embeddings improve crackle detection to **41.15% F1** (vs 34.76% in Classical ML) and normal detection to **56.82% F1**. Fusing the 128-d projected PANNs vector directly supplies these rich acoustic burst priors.
- **Loss Function:** Inverse-frequency class-weighted cross-entropy:
  $$w_{\text{Normal}} = 0.449, \quad w_{\text{Crackle}} = 0.911, \quad w_{\text{Wheeze}} = 2.414, \quad w_{\text{Both}} = 3.866$$

---

## 8. Lung Disease Head Design (Kaggle Respiratory Disease)

### 8.1 Clinical Objective
Patient-level classification into three clinical diagnostic categories:
1. `Normal` (Class 0): Clear lungs, healthy subject.
2. `Asthma` (Class 1): Chronic inflammatory airway disease.
3. `Other` (Class 2): Other pulmonary conditions (COPD, Bronchiectasis, Pneumonia, URTI).

### 8.2 Strict Decoupling from Acoustic Head
> [!CAUTION]
> **Clinical & Algorithmic Constraint:**
> Wheeze MUST NOT be equated with Asthma.
> - A patient with COPD or foreign-body aspiration can produce severe wheezing without having asthma.
> - An asthmatic patient between acute exacerbations can present with completely normal vesicular breath sounds.
> - Conflating wheezing and asthma causes false positives in healthy wheezers and false negatives in quiescent asthmatics.
> - The two heads must remain structurally and statistically separate:
>   - *Acoustic Head:* Classifies short-duration acoustic waveforms ($\sim 2\text{--}5$ seconds).
>   - *Disease Head:* Integrates representations across multiple lung zones to infer chronic patient pathology.

### 8.3 Structural Specifications
```
Shared CNN Feature Map (B, 256, T, F)
                   │
                   ▼
       [Global Spatio-Temporal Avg] ──> (B, 256)
                   │
                   ├──────────────────────────┐
                   ▼                          ▼
           CNN Vector (B, 256)        Physical Vector (B, 64)
                   │                          │
                   └───────────┬──────────────┘
                               ▼
                     Concatenation (B, 320)
                               │
                               ▼
                      Linear(320 -> 128)
                               │
                               ▼
                     BatchNorm1d + ReLU
                               │
                               ▼
                         Dropout(0.35)
                               │
                               ▼
                      Linear(128 -> 3)
                               │
                               ▼
                  Disease Logits [Normal, Asthma, Other]
```

### 8.4 Technical Rationale
- **Handcrafted Physical Feature Fusion:** The verified test results demonstrated that classical acoustic features achieved **39.83% Window / 50.84% Patient Macro F1** on Kaggle, outperforming deep models alone. Concatenating the projected 64-d physical feature vector stabilizes disease classification against variable stethoscope placement noise.
- **Patient Aggregation:** Window-level probabilities are aggregated per patient via soft-voting:
  $$\bar{\mathbf{p}}_i = \frac{1}{|W_i|} \sum_{w \in W_i} \text{Softmax}(\mathbf{z}_{\text{disease}}(w)), \quad \hat{y}_i = \arg\max \bar{\mathbf{p}}_i$$

---

## 9. Multi-Task Training Strategy & Masked Task Loss

### 9.1 The Label Disjointness Problem
A major technical challenge in multi-task auscultation learning is that training datasets have mutually exclusive label sets:
- **PhysioNet 2016 (Heart):** Heart labels available ($y_H \in \{0, 1\}$); Lung Acoustic and Disease labels are **missing**.
- **ICBHI 2017 (Acoustic):** Acoustic cycle labels available ($y_A \in \{0, 1, 2, 3\}$); Heart and Disease labels are **missing**.
- **Kaggle (Disease):** Disease labels available ($y_D \in \{0, 1, 2\}$); Heart and Acoustic labels are **missing**.

> [!WARNING]
> **Zero-Imputation Fallacy:**
> Missing labels must **never** be imputed as negative (Class 0). Treating a heart sound recording as "Normal" for lung disease or "Normal" for crackles would inject massive false-negative label noise, corrupting the shared backbone.

### 9.2 Masked Multi-Task Loss Formulation
Each training sample $i$ carries a binary mask vector $\mathbf{m}_i = [m_{H, i}, m_{A, i}, m_{D, i}] \in \{0, 1\}^3$ indicating which ground truth labels are present:

$$\mathcal{L}_{\text{total}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ m_{H, i} \cdot \lambda_H \mathcal{L}_H(\hat{\mathbf{y}}_{H, i}, y_{H, i}) + m_{A, i} \cdot \lambda_A \mathcal{L}_A(\hat{\mathbf{y}}_{A, i}, y_{A, i}) + m_{D, i} \cdot \lambda_D \mathcal{L}_D(\hat{\mathbf{y}}_{D, i}, y_{D, i}) \right]$$

where:
- $\mathcal{L}_H$ is the weighted binary cross-entropy loss for Heart.
- $\mathcal{L}_A$ is the weighted multi-class cross-entropy loss for ICBHI Acoustic cycles.
- $\mathcal{L}_D$ is the weighted multi-class cross-entropy loss for Kaggle Disease diagnosis.
- $\lambda_H, \lambda_A, \lambda_D$ are task balancing hyper-parameters.

```
Sample Input ──> Shared Backbone ──┬──> Heart Head   ──> [Loss_H] × m_H ──┐
                                   ├──> Acoustic Head──> [Loss_A] × m_A ──┼──> Total Loss
                                   └──> Disease Head ──> [Loss_D] × m_D ──┘
```

### 9.3 Batch Sampling Strategy
To prevent dataset size imbalance (Heart: 19,117 samples; ICBHI: 3,804; Kaggle: 1,296) from causing one task to dominate gradient updates, training uses **Stratified Round-Robin Task Batching**:
- Each mini-batch is sampled entirely from one task dataset.
- Batches alternate deterministically: $\text{Batch}_{\text{Heart}} \to \text{Batch}_{\text{ICBHI}} \to \text{Batch}_{\text{Kaggle}}$.
- Gradients from each task update the shared backbone and only that task's active head.
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.01`) with Cosine Annealing.

---

## 10. Loss Design & Task Weighting

### 10.1 Individual Task Loss Formulations
1. **Heart Loss:**
   $$\mathcal{L}_H = - w_{\text{Abnormal}} y_H \log \hat{p}_{\text{Abnormal}} - w_{\text{Normal}} (1 - y_H) \log (1 - \hat{p}_{\text{Abnormal}})$$
2. **Acoustic Cycle Loss:**
   $$\mathcal{L}_A = - \sum_{k=0}^3 w_{A, k} \cdot \mathbb{I}(y_A = k) \cdot \log \hat{p}_{A, k}$$
3. **Disease Diagnosis Loss:**
   $$\mathcal{L}_D = - \sum_{k=0}^2 w_{D, k} \cdot \mathbb{I}(y_D = k) \cdot \log \hat{p}_{D, k}$$

### 10.2 Task Balancing Strategy ($\lambda_H, \lambda_A, \lambda_D$)
- **Baseline Default:** $\lambda_H = 1.0, \lambda_A = 1.0, \lambda_D = 1.0$.
- **Tuning Requirement:** Optimal task weights must **not** be guessed or arbitrarily set. They must be tuned via grid search on the **Validation split** or adaptively estimated using **Uncertainty Weighting** (Kendall et al., 2018):
  $$\mathcal{L}_{\text{multi}} = \frac{1}{2\sigma_H^2}\mathcal{L}_H + \frac{1}{2\sigma_A^2}\mathcal{L}_A + \frac{1}{2\sigma_D^2}\mathcal{L}_D + \log(\sigma_H \sigma_A \sigma_D)$$
  where $\sigma_H, \sigma_A, \sigma_D$ are trainable noise parameters that automatically balance task gradients.

---

## 11. Six Stethoscope Positions: Physical Auscultation Inference Protocol

In physical clinical examinations, lung sounds are recorded across **six standardized anatomical landmarks** (P1 through P6):

```
       ANTERIOR / POSTERIOR THORACIC GRID
       
          Left Side              Right Side
        ┌───────────┐          ┌───────────┐
        │    P1     │          │    P2     │   Upper Lobes / Apical
        ├───────────┤          ├───────────┤
        │    P3     │          │    P4     │   Middle Lobes / Lingular
        ├───────────┤          ├───────────┤
        │    P5     │          │    P6     │   Lower Lobes / Basilar
        └───────────┘          └───────────┘
```

### 11.1 Auscultation Point Topography
- **P1 & P2 (Upper Lobes):** 2nd intercostal space, mid-clavicular line. Primary detection site for upper airway bronchial sounds and apical tuberculosis/pneumonia.
- **P3 & P4 (Middle Zones):** 4th intercostal space, anterior/lateral. Assessment of middle lobe consolidation and generalized asthmatic wheezes.
- **P5 & P6 (Lower Lobes / Bases):** 6th–7th intercostal space, mid-axillary / posterior base. Crucial site for gravity-dependent basilar crackles (pulmonary edema, congestive heart failure, interstitial fibrosis).

### 11.2 Multi-Point Inference Hierarchy

```
[6-Point Digital Auscultation Stream]
   P1 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P1
   P2 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P2
   P3 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P3
   P4 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P4
   P5 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P5
   P6 (Audio) ──> Preprocessing ──> Shared Model ──> Local Prediction P6
                                                           │
                                                           ▼
                             ┌─────────────────────────────┴─────────────────────────────┐
                             ▼                                                           ▼
                [Acoustic Spatial Aggregation]                             [Disease Diagnostic Pooling]
                             │                                                           │
                             ▼                                                           ▼
           - Local Crackle/Wheeze Probability Map                       - Attention-Weighted Patient Pooling
           - Diffuse vs Focal Distinction                               - Final Diagnostic Probability:
             * Diffuse wheeze (P1–P4) -> Asthma/COPD                      Normal / Asthma / Other
             * Unilateral basilar crackles (P5/P6) -> Pneumonia
             * Bilateral basilar crackles (P5+P6) -> Edema
                             │                                                           │
                             └─────────────────────────────┬─────────────────────────────┘
                                                           ▼
                                         [Clinical Visualization Dashboard]
                                         - Hexagonal Thoracic Heatmap
                                         - Auscultation Audio Playback
                                         - Explainability Overlay
```

### 11.3 Crucial Clinical Disclaimers on Spatial Inference
1. **Heart Sounds Are NOT Evaluated at P1–P6:** Heart auscultation requires precordial auscultation sites (Aortic, Pulmonic, Tricuspid, and Mitral/Apex points). The Heart Head must only be activated during designated precordial recording modes.
2. **Local vs Systemic Findings:** Acoustic events can be strictly focal (e.g., crackles localized only to P5 in lobar pneumonia). Averaging away spatial variance destroys diagnostic utility. Local predictions must be presented independently on the thoracic map.

---

## 12. Digital Stethoscope Real-Time Deployment Considerations

Deploying the final model to a physical electronic stethoscope requires solving several hardware-software interfacing challenges:

### 12.1 Preprocessing & Acoustic Coupling Compatibility
- **Sampling Rate Conversion:** Hardware electronic stethoscopes typically record at 44.1 kHz, 48 kHz, or 8 kHz. Incoming audio must pass through an **anti-aliasing FIR low-pass filter (Nyquist cutoff 7.6 kHz)** before resampling to the standardized **16,000 Hz**.
- **Diaphragm Resonance & Contact Rumble:** Stethoscope chest-piece motion generates high-amplitude sub-audible friction artifacts (<20 Hz). A 4th-order **Butterworth High-Pass Filter with cutoff at 30 Hz** must be applied before STFT calculation.
- **Exact Spectrogram Alignment:** The STFT pipeline must strictly reproduce training parameters:
  `sr=16000`, `n_fft=1024`, `hop_length=256`, `n_mels=128`, `f_min=50`, `f_max=8000`, `power=2.0`, `top_db=80.0`.

### 12.2 Real-Time Streaming & Latency Budget
- **Sliding Window Buffer:** Real-time audio streams into a circular ring buffer. A 5.0-second window is extracted every 1.0 second (80% overlap) for real-time inference.
- **Latency Budget:** Total inference latency per 5.0-second buffer must not exceed **150 ms** on edge hardware (Raspberry Pi 4 / mini-PC / laptop):
  - Spectrogram & Feature Extraction: $\sim 25\text{ ms}$
  - CNN Backbone Forward Pass: $\sim 45\text{ ms}$
  - Head Projections & Softmax: $\sim 5\text{ ms}$
  - Total latency $\approx 75\text{ ms}$ (well within the 1000 ms real-time hop).

### 12.3 Robust Gain Normalization
- Different patient body habitus, chest wall thickness, and practitioner pressure cause wide variations in recording volume.
- **Instance-Level Standardization:** Rather than relying on fixed global volume thresholds, each spectrogram slice is standardized independently:
  $$S_{\text{norm}}(f, t) = \frac{S(f, t) - \mu_S}{\sigma_S + \epsilon}$$
  preventing gain-level bias from triggering false abnormality detections.

---

## 13. Proposed Final Architecture Blueprint

```
                      Raw Audio Input (16 kHz, 5.0s Window)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       STFT Preprocessing                             Feature Extraction
   Log-Mel Spectrogram (1, 128, 313)            397-d Physical Acoustic Features
                │                                             │
                ▼                                             ▼
    Shared Auscultation CNN Trunk               Physical Projection Layer
       (4 Conv Blocks, ~1.2M params)                (Linear 397 -> 64 + ReLU)
                │                                             │
                ▼                                             │
      Feature Map (256, T, F)                                 │
                │                                             │
       ┌────────┴────────────────────────┬────────────────────┘
       │                                 │
       ▼                                 ▼
[Heart Branch]                    [Lung Representation]
Global Temporal Pooling (256-d)   Shared Conv Features (256-d)
       │                                 │
       ├──────────────┐                  ├───────────────────────────────┐
       ▼              ▼                  ▼                               ▼
  CNN (256-d)   Physical (64-d)    [Acoustic Branch]              [Disease Branch]
       │              │           Masked Temporal Avg (256-d)    Global Temporal Avg (256-d)
       └──────┬───────┘                  │                               │
              ▼                          ├──────────────┐                ├──────────────┐
       Concat (320-d)                    ▼              ▼                ▼              ▼
              │                     CNN (256-d)   PANNs Aux (128-d) CNN (256-d)   Physical (64-d)
              ▼                          │              │                │              │
       Heart MLP Head                    └──────┬───────┘                └──────┬───────┘
       (Linear 320 -> 128 -> 2)                 ▼                               ▼
              │                          Concat (384-d)                  Concat (320-d)
              ▼                                 │                               │
       HEART PREDICTION                         ▼                               ▼
     (Normal / Abnormal)                Acoustic MLP Head               Disease MLP Head
                                    (Linear 384 -> 128 -> 4)        (Linear 320 -> 128 -> 3)
                                                │                               │
                                                ▼                               ▼
                                       ACOUSTIC PREDICTION              DISEASE PREDICTION
                                   (Norm / Crk / Whz / Both)          (Norm / Asthma / Other)
```

---

## 14. Component Specifications: Trainable vs Frozen

| Architectural Component | Input Dimension | Output Dimension | Trainable / Frozen | Functional Purpose |
| :--- | :--- | :--- | :---: | :--- |
| **STFT Preprocessing** | Audio `(80000,)` | Log-Mel `(1, 128, 313)` | **Fixed (Deterministic)** | Converts raw pressure waveform into normalized time-frequency space. |
| **Physical Feature Extractor** | Spectrogram `(128, 313)` | Feature vector `(397,)` | **Fixed (Deterministic)** | Computes spectral moments, energy dynamics, and MFCC trajectories. |
| **Physical Projection Layer** | `(397,)` | `(64,)` | **Trainable** | Projects handcrafted physical moments into learned embedding space. |
| **Shared Auscultation CNN Trunk**| Log-Mel `(1, 128, 313)` | Feature Map `(256, 19, 8)`| **Trainable** | Learns core hierarchical spatio-temporal auscultation acoustic features. |
| **Heart Classifier Head** | Concat `(320,)` | Logits `(2,)` | **Trainable** | Binary cardiac abnormality detector (`Normal` vs `Abnormal`). |
| **PANNs CNN14 Extractor** | Audio `(80000,)` | Embedding `(2048,)` | **Frozen (Pretrained)** | Generates foundational AudioSet event embeddings for lung cycles. |
| **PANNs Bottleneck Layer** | Embedding `(2048,)` | `(128,)` | **Trainable** | Reduces PANNs dimensionality to prevent head overfitting. |
| **Acoustic Cycle Head** | Concat `(384,)` | Logits `(4,)` | **Trainable** | 4-class respiratory cycle acoustic event classifier. |
| **Disease Classifier Head** | Concat `(320,)` | Logits `(3,)` | **Trainable** | 3-class patient-level respiratory disease diagnostic classifier. |

---

## 15. Required Experiments Before Final Claim

Before this proposed architecture can be declared the definitive final model of the AI Stethoscope project, the following validation experiments must be conducted:

### Category A: Required Experiments (Mandatory for Validation)
1. **Multi-Task vs Independent Head Ablation:**
   - Train the shared backbone jointly on all three tasks using the masked loss formulation.
   - Compare performance against single-task standalone models to verify that multi-task learning does not introduce negative transfer.
2. **PANNs Auxiliary Branch Ablation:**
   - Measure ICBHI performance with and without the auxiliary PANNs branch to verify that the +12.29% F1 gain over pure CNN is preserved within the unified architecture.
3. **Handcrafted Feature Fusion Ablation:**
   - Evaluate the Heart and Kaggle heads with and without the 64-d physical projection to confirm that physical feature fusion maintains the classical ML baseline advantage.
4. **Task Loss Weight ($\lambda_H, \lambda_A, \lambda_D$) Validation Optimization:**
   - Perform grid search or evaluate uncertainty weighting on the validation split to establish optimal multi-task balancing.
5. **Masked Pooling Validation on Variable ICBHI Cycles:**
   - Verify that Masked Temporal Average Pooling (MTAP) outperforms unmasked global pooling when integrated with the shared CNN trunk.

### Category B: Optional / Follow-Up Experiments
1. **PANNs Layer-6 Fine-Tuning Ablation:**
   - Test whether unfreezing the final convolutional block (`conv_block6`) of PANNs specifically during respiratory training improves acoustic event detection without destabilizing representations.
2. **Six-Position Spatial Attention Pooling:**
   - Simulate 6-position inference by grouping multi-site recordings from ICBHI/Kaggle and training a lightweight attention mechanism across points P1–P6.
3. **Live Hardware Latency & Noise Benchmark:**
   - Deploy the exported ONNX model to the physical digital stethoscope prototype and profile real-time inference latency and memory consumption.

---

## 16. Risks, Limitations & Open Questions

1. **Risk of Negative Task Transfer:**
   - *Issue:* Gradient updates from heart sounds (<500 Hz periodic closures) could interfere with lung sound representations (high-frequency adventitious bursts).
   - *Mitigation:* Stratified round-robin batching, task-specific projection heads, and early stopping monitoring joint validation performance.
2. **Missing Precordial Heart Data at Lung Points P1–P6:**
   - *Limitation:* The six physical thoracic points P1–P6 are designed for pulmonary auscultation. Precordial heart sound points (aortic, pulmonic, tricuspid, mitral) are anatomically distinct.
   - *Protocol Requirement:* The user interface must enforce separate examination modes: "Cardiac Exam" (precordial positions) vs "Pulmonary Exam" (positions P1–P6).
3. **Acoustic Device Generalization:**
   - *Limitation:* ICBHI audio was collected across 7 commercial stethoscope devices with distinct frequency responses; the physical prototype will introduce its own acoustic transfer function.
   - *Mitigation:* Pre-deployment equalization and bandpass filtering (30 Hz – 4 kHz) on the hardware stream.

---

## 17. Final Implementation Roadmap

```
[Phase 1: Architecture Construction] (Codebase Setup)
   ├── Implement MultiTaskAudioCNN class (Shared trunk + 3 decoupled heads)
   ├── Implement MaskedMultiTaskLoss module
   └── Implement StratifiedTaskDataLoader for disjoint multi-task batches

[Phase 2: Validation-Governed Training]
   ├── Train MultiTaskAudioCNN on Train splits only
   ├── Monitor joint validation Macro F1 across all three tasks
   └── Select best model checkpoint strictly on Validation performance

[Phase 3: Held-Out Test Evaluation]
   ├── Evaluate final checkpoint ONCE on Held-Out Test splits
   ├── Verify patient-level soft-voting metrics
   └── Validate that multi-task performance matches or exceeds individual baselines

[Phase 4: Hardware & Inference Pipeline Integration]
   ├── Export trained PyTorch model to ONNX runtime format
   ├── Implement real-time circular ring buffer (16 kHz, 5.0s window, 1.0s hop)
   ├── Implement 6-point thoracic heatmap visualization module
   └── Deploy to digital stethoscope embedded platform
```
