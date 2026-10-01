# Verification & Audit Report: PANNs CNN14 Frozen-Embedding Classifier Experiment
**AI Stethoscope Research Project**  
**Date:** September 23, 2026  
**Auditor:** Autonomous Verification Subsystem  
**Scope:** Strict Read-Only Verification of Artifacts, Code, Embedding Caches, Models, and Reports for the PANNs CNN14 Frozen-Embedding Benchmark.

---

## 1. Overall Audit Status

| Audit Item | Scope | Status |
| :---: | :--- | :---: |
| **Audit 1** | Validation-Only Model Selection | **PASS** |
| **Audit 2** | Test Isolation & Leakage Prevention | **PASS** |
| **Audit 3** | Scaler Isolation (Train Fit Only) | **PASS** |
| **Audit 4** | PANNs Embedding Integrity & Immutability | **PASS** |
| **Audit 5** | Split & Patient Isolation (Zero Leakage) | **PASS** |
| **Audit 6** | Patient-Level Aggregation Protocol | **PASS** |
| **Audit 7** | Saved Model Type & Winner Alignment | **PASS** |
| **Audit 8** | Test Metric Consistency Across Artifacts | **PASS** |
| **Audit 9** | Baseline Comparison Number Integrity | **PASS** |
| **Audit 10**| Scientific Claim Rigor & Scope Audit | **PASS WITH QUALIFICATIONS** |
| **OVERALL STATUS** | **STRICT VERIFICATION AUDIT** | **PASSED** |

---

## 2. Audit 1: Validation-Only Model Selection

### 2.1 Verification Checklist
- [x] All five candidate classifiers were trained on the training split only and benchmarked on validation:
  1. Logistic Regression
  2. Linear SVM
  3. RBF SVM
  4. Random Forest
  5. Lightweight MLP
- [x] Selection metric was strictly **Validation Window Macro F1**.
- [x] Zero test data or test metrics were examined or referenced during model selection.

### 2.2 Validation Macro F1 for All Candidates

#### Kaggle Respiratory Disease ($N_{\text{val}}=218$)
| Candidate Classifier | Input Representation | Val Macro F1 | Val Accuracy | Val Balanced Acc | Winner Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear SVM** | Scaled | **0.3566** | **36.24%** | **35.87%** | **SELECTED WINNER** 🏆 |
| Logistic Regression | Scaled | 0.3482 | 34.86% | 34.90% | Candidate |
| Random Forest | Raw | 0.3256 | 35.78% | 33.60% | Candidate |
| RBF SVM | Scaled | 0.3106 | 31.19% | 30.97% | Candidate |
| Lightweight MLP | Scaled | 0.3088 | 33.94% | 31.97% | Candidate |

*Selected Classifier:* **Linear SVM** (exact Validation Macro F1 = **0.3566**).

#### ICBHI 2017 Respiratory Cycles ($N_{\text{val}}=1,549$)
| Candidate Classifier | Input Representation | Val Macro F1 | Val Accuracy | Val Balanced Acc | Winner Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lightweight MLP** | Scaled | **0.3504** | **45.06%** | **35.40%** | **SELECTED WINNER** 🏆 |
| RBF SVM | Scaled | 0.3317 | 42.09% | 33.38% | Candidate |
| Random Forest | Raw | 0.3272 | 46.55% | 32.73% | Candidate |
| Linear SVM | Scaled | 0.3267 | 41.58% | 33.32% | Candidate |
| Logistic Regression | Scaled | 0.3136 | 39.64% | 32.72% | Candidate |

*Selected Classifier:* **Lightweight MLP** (exact Validation Macro F1 = **0.3504**).

#### Heart PhysioNet 2016 ($N_{\text{val}}=2,358$)
| Candidate Classifier | Input Representation | Val Macro F1 | Val Accuracy | Val Balanced Acc | Winner Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RBF SVM (Nystroem + SGD)** | Scaled | **0.5956** | **67.85%** | **61.45%** | **SELECTED WINNER** 🏆 |
| Lightweight MLP | Scaled | 0.5911 | 67.68% | 60.81% | Candidate |
| Logistic Regression | Scaled | 0.5886 | 65.86% | 61.74% | Candidate |
| Random Forest | Raw | 0.5855 | 67.98% | 59.75% | Candidate |
| Linear SVM (SGD) | Scaled | 0.5582 | 68.28% | 55.98% | Candidate |

*Selected Classifier:* **RBF SVM (Nystroem + SGD)** (exact Validation Macro F1 = **0.5956**).

---

## 3. Audit 2: Test Isolation

Inspecting [`scripts/train_panns_classifiers.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/scripts/train_panns_classifiers.py):
1. **Training Partition:** All models execute `clf.fit(X_tr, y_train)` using strictly `X_train_raw` or `X_train_scaled`.
2. **Validation Partition:** Used strictly for computing `val_metrics` (`accuracy`, `balanced_accuracy`, `f1_macro`, etc.) to rank candidate models.
3. **Held-Out Test Partition:** In line 323–333, the test split is loaded into `X_test_use` and evaluated **exactly once** on the single winner:
   ```python
   # Line 323-330
   best_entry = val_benchmark_results[0]
   best_model_name = best_entry["model_name"]
   best_clf, X_test_use, best_rep = trained_clfs[best_model_name]
   ...
   test_preds = best_clf.predict(X_test_use)
   test_probs = best_clf.predict_proba(X_test_use)
   ```
4. **Leakage & Tuning Verification:** No loops over test parameters, no grid search on test, and no post-test selection occurred. Test isolation was 100% maintained.

---

## 4. Audit 3: Scaler Isolation

Inspecting lines 220–225 of [`scripts/train_panns_classifiers.py`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/scripts/train_panns_classifiers.py):
```python
# Step 1: Fitting StandardScaler on Train split only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)
```
- `scaler.fit` / `scaler.fit_transform` was called **exclusively on the training split**.
- Validation and Test splits were processed solely via `scaler.transform`.
- Zero data leakage occurred through the standardizer.

---

## 5. Audit 4: PANNs Embedding Integrity

Read-only inspection of all nine `.npz` embedding caches under `processed/panns_embeddings/`:

| Dataset | Split | Embedding Shape | Dtype | Has NaN? | Has Inf? | Cache File Timestamp |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kaggle** | Train | `(1296, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **Kaggle** | Val | `(218, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **Kaggle** | Test | `(288, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **ICBHI** | Train | `(3804, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **ICBHI** | Val | `(1549, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **ICBHI** | Test | `(1545, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **Heart** | Train | `(19117, 2048)`| `float32` | **False** | **False** | Pre-experiment cache preserved |
| **Heart** | Val | `(2358, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |
| **Heart** | Test | `(2427, 2048)` | `float32` | **False** | **False** | Pre-experiment cache preserved |

- CNN14 weights were completely frozen; no fine-tuning or backpropagation into the extractor occurred.
- Existing embedding cache files were loaded without alteration or regeneration.
- All tensors have embedding dimension exactly 2048, dtype `float32`, and contain zero NaN or infinite values.

---

## 6. Audit 5: Split / Patient Isolation

Using the preserved `patient_ids` arrays stored directly inside each `.npz` file, patient set intersections were computed:

| Dataset | Split Patient Counts | $\text{Train} \cap \text{Val}$ | $\text{Train} \cap \text{Test}$ | $\text{Val} \cap \text{Test}$ | Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kaggle** | Train: 67, Val: 22, Test: 23 | **0 (None)** | **0 (None)** | **0 (None)** | **PASSED (Zero Leakage)** |
| **ICBHI** | Train: 76, Val: 26, Test: 24 | **0 (None)** | **0 (None)** | **0 (None)** | **PASSED (Zero Leakage)** |
| **Heart** | Train: 2,000, Val: 429, Test: 427 | **0 (None)** | **0 (None)** | **0 (None)** | **PASSED (Zero Leakage)** |

Zero subject overlap exists between any combination of splits across all three tasks.

---

## 7. Audit 6: Patient-Level Aggregation Protocol

Inspecting `compute_patient_metrics`:
```python
unique_patients = np.unique(patient_ids)
for pid in unique_patients:
    idx = np.where(patient_ids == pid)[0]
    mean_prob = np.mean(probs[idx], axis=0)
    pred_class = int(np.argmax(mean_prob))
    true_class = int(y_true[idx[0]])
```
1. **Kaggle Respiratory Disease:**
   - Grouped strictly by `patient_ids`.
   - Window predicted probabilities averaged across all windows per subject.
   - Prediction determined via `argmax` over mean probabilities.
   - Held-out test patient count is verified at **exactly 23 patients**.
2. **ICBHI 2017:**
   - Aggregation groups cycles by the preserved patient identifier (24 test patients).
   - In ICBHI, cycles from the same patient can represent different respiratory phases. The aggregation protocol inherits the baseline definition where patient label is taken from `y_true[idx[0]]`, providing direct comparability with the classical ML baseline.
3. **Heart Sound (PhysioNet 2016):**
   - Each recording corresponds to an individual patient/recording entity trial (`a0001`, `b0012`, etc.).
   - Windows segmented from each recording are aggregated via mean soft-voting across the 427 test recording entities.

---

## 8. Audit 7: Saved Model Verification

Direct inspection using `joblib.load()` on the model files saved under `experiments/panns_classifiers/*/models/best_model.joblib`:

| Dataset | Saved Model File | Top-Level Python Class | Inner Wrapped Estimators | Match Validation Winner? |
| :--- | :--- | :--- | :--- | :---: |
| **Kaggle** | `experiments/panns_classifiers/kaggle/models/best_model.joblib` | `DecisionFunctionSoftmaxClassifier` | `LinearSVC(C=0.01, class_weight='balanced', dual=False, random_state=42)` | **YES (Linear SVM)** |
| **ICBHI** | `experiments/panns_classifiers/icbhi/models/best_model.joblib` | `WeightedMLPClassifier` | `MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, early_stopping=True, n_iter_no_change=5, random_state=42)` | **YES (Lightweight MLP)** |
| **Heart** | `experiments/panns_classifiers/heart/models/best_model.joblib` | `NystroemSVMClassifier` | 1. `Nystroem(kernel='rbf', n_components=500, random_state=42)`<br>2. `SGDClassifier(loss='hinge', class_weight='balanced', max_iter=2000, random_state=42)` | **YES (RBF SVM Nystroem+SGD)** |

### Specific Verification on Heart Classifier Implementation:
- **Verified Implementation:** The saved model is **NOT** a standard LibSVM `sklearn.svm.SVC`.
- It is `NystroemSVMClassifier`, an explicit wrapper combining:
  1. `sklearn.kernel_approximation.Nystroem(kernel='rbf', n_components=500)` for nonlinear random feature kernel mapping.
  2. `sklearn.linear_model.SGDClassifier(loss='hinge', class_weight='balanced')` for linear margin optimization.
  3. Sigmoid margin mapping over decision function margins for soft-voting probability output.
- This exactly fulfills the conservative CPU-friendly implementation specified in the implementation plan.

---

## 9. Audit 8: Test Metric Consistency Across Artifacts

Cross-checking reported numbers in [`analysis/panns_classifier_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_results.md) against raw JSON outputs in `experiments/panns_classifiers/*/metrics/test_metrics.json`:

| Dataset | Metric | Value in `test_metrics.json` | Value in `panns_classifier_results.md` | Discrepancy |
| :--- | :--- | :---: | :---: | :---: |
| **Kaggle** | Window Accuracy | 0.3333 (33.33%) | 33.33% | None |
| | Window Balanced Accuracy | 0.3251 (32.51%) | 32.51% | None |
| | Window Macro F1 | 0.3223 | 0.3223 | None |
| | Window Weighted F1 | 0.3263 (32.63%) | 32.63% | None |
| | Patient Accuracy | 0.3043 (30.43%) | 30.43% | None |
| | Patient Balanced Accuracy | 0.2698 (26.98%) | 26.98% | None |
| | Patient Macro F1 | 0.2206 | 0.2206 | None |
| | Patient Weighted F1 | 0.2432 (24.32%) | 24.32% | None |
| **ICBHI** | Window Accuracy | 0.4291 (42.91%) | 42.91% | None |
| | Window Balanced Accuracy | 0.3393 (33.93%) | 33.93% | None |
| | Window Macro F1 | 0.3391 | 0.3391 | None |
| | Window Weighted F1 | 0.4202 (42.02%) | 42.02% | None |
| | Patient Accuracy | 0.6667 (66.67%) | 66.67% | None |
| | Patient Balanced Accuracy | 0.3039 (30.39%) | 30.39% | None |
| | Patient Macro F1 | 0.3224 | 0.3224 | None |
| | Patient Weighted F1 | 0.6217 (62.17%) | 62.17% | None |
| **Heart** | Window Accuracy | 0.6799 (67.99%) | 67.99% | None |
| | Window Balanced Accuracy | 0.6064 (60.64%) | 60.64% | None |
| | Window Macro F1 | 0.5912 | 0.5912 | None |
| | Window Weighted F1 | 0.6951 (69.51%) | 69.51% | None |
| | Window Sensitivity | 0.4719 (47.19%) | 47.19% | None |
| | Window Specificity | 0.7409 (74.09%) | 74.09% | None |
| | Patient Accuracy | 0.7518 (75.18%) | 75.18% | None |
| | Patient Balanced Accuracy | 0.6346 (63.46%) | 63.46% | None |
| | Patient Macro F1 | 0.6299 | 0.6299 | None |
| | Patient Sensitivity | 0.4368 (43.68%) | 43.68% | None |
| | Patient Specificity | 0.8324 (83.24%) | 83.24% | None |

Zero discrepancies exist. All reported numbers match the underlying test metrics JSON artifacts.

---

## 10. Audit 9: Baseline Comparison Number Integrity

Cross-checking baseline numbers in Section 8 of [`analysis/panns_classifier_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_results.md) against historical baseline reports:
- [`analysis/full_cnn_baseline_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/full_cnn_baseline_results.md)
- [`analysis/classical_ml_baseline_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/classical_ml_baseline_results.md)

### 10.1 Heart Sound (PhysioNet 2016)
- **CNN Baseline:** Window Acc = 89.04%, BalAcc = 90.15%, Macro F1 = 85.90%, Sens = 92.20%, Spec = 88.11%, Patient Acc = 91.80%, Patient Macro F1 = 88.69%, Patient Sens = 96.55%, Patient Spec = 90.59%.  
  *Audit:* **EXACT MATCH** with `full_cnn_baseline_results.md` lines 22–36.
- **Classical ML (Random Forest):** Window Acc = 90.11%, BalAcc = 89.50%, Macro F1 = 86.82%, Sens = 88.38%, Spec = 90.62%, Patient Acc = 93.44%, Patient Macro F1 = 90.59%, Patient Sens = 94.25%, Patient Spec = 93.24%.  
  *Audit:* **EXACT MATCH** with `classical_ml_baseline_results.md` lines 22–30.

### 10.2 ICBHI 2017 Respiratory Cycles
- **CNN Baseline:** Window Acc = 23.82%, BalAcc = 24.52%, Macro F1 = 21.62%, Normal F1 = 17.56%, Crackle F1 = 39.54%, Wheeze F1 = 14.88%, Crackle+Wheeze F1 = 14.50%, Patient Acc = 16.67%, Patient Macro F1 = 19.03%.  
  *Audit:* **EXACT MATCH** with `full_cnn_baseline_results.md` and `classical_ml_baseline_results.md` lines 31–39.
- **Classical ML (SVM RBF):** Window Acc = 38.90%, BalAcc = 34.15%, Macro F1 = 33.46%, Normal F1 = 52.13%, Crackle F1 = 34.76%, Wheeze F1 = 23.84%, Crackle+Wheeze F1 = 22.51%, Patient Acc = 70.83%, Patient Macro F1 = 21.79%.  
  *Audit:* **EXACT MATCH** with `classical_ml_baseline_results.md` lines 31–39.

### 10.3 Kaggle Respiratory Disease
- **CNN Baseline:** Window Acc = 35.76%, BalAcc = 36.46%, Macro F1 = 35.68%, Patient Acc = 34.78%, Patient Macro F1 = 33.85%.  
  *Audit:* **EXACT MATCH** with `full_cnn_baseline_results.md` lines 29–34.
- **Classical ML (SVM RBF):** Window Acc = 42.01%, BalAcc = 41.72%, Macro F1 = 39.83%, Patient Acc = 56.52%, Patient Macro F1 = 50.84%.  
  *Audit:* **EXACT MATCH** with `classical_ml_baseline_results.md` lines 40–44.

Zero discrepancies or historical alterations were found.

---

## 11. Audit 10: Scientific Claim Check

Each interpretive claim identified in [`analysis/panns_classifier_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_results.md) was evaluated against empirical evidence:

- **Classification Categories:**
  - **Category A:** Directly supported by this experiment.
  - **Category B:** Reasonable interpretation / plausible acoustic hypothesis, but not directly isolated or proven.
  - **Category C:** Stronger than established by this experiment (unsupported generalization or formal statistical claim without statistical testing).

| # | Specific Report Claim | Claim Text / Context | Audit Classification | Justification & Technical Nuance |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Numerical Superiority on ICBHI | PANNs achieved 42.91% Window Acc and 33.91% Macro F1 vs 23.82% / 21.62% for CNN and 38.90% / 33.46% for Classical ML. | **A (Directly Supported)** | Directly confirmed by test metrics across all three baselines evaluated on identical splits. |
| **2** | "State-of-the-Art" (SOTA) on ICBHI | *"establish the new overall state-of-the-art across all baselines"* | **B (Qualified Interpretation)** | It is the highest score **among the internal AI Stethoscope project baselines**, but the term "state-of-the-art" (SOTA) typically connotes global literature benchmarks, which were not evaluated here. |
| **3** | "Significantly outperforming" | *"...significantly outperforming CNNs trained from scratch..."* | **C (Not Established by Experiment)** | The numerical margin is large (+19.09% acc, +12.29% macro F1), but **no formal hypothesis testing** (e.g. McNemar's test, paired permutation test, bootstrap confidence intervals) was conducted to claim statistical significance at $p < 0.05$. |
| **4** | AudioSet Prior Transfer to Crackles & Wheezes | *"AudioSet pretraining exposes CNN14 to transient explosive events... These structural priors transfer remarkably well to crackles..."* | **B (Reasonable Hypothesis)** | While crackle F1 is higher (41.15%), this experiment did not conduct layer ablation or feature attribution to isolate which specific pretraining classes contributed to this advantage. |
| **5** | Cardiac Acoustic Domain Mismatch (<500 Hz) | *"PANNs embeddings fail to encode subtle low-frequency cardiac timings..."* | **B (Reasonable Hypothesis)** | The drop in cardiac performance is severe (59.12% vs 86.82% Macro F1), and heart sounds are concentrated <500 Hz. However, alternative factors (such as global average pooling destroying temporal valve sequence periodicity) could also explain this drop. |
| **6** | Kaggle Ambient Noise Sensitivity | *"PANNs' global average pooling aggregates these prominent environmental acoustic cues... misleading linear classifiers"* | **B (Reasonable Hypothesis)** | Ambient noise exists in Kaggle, but this benchmark did not conduct an SNR-controlled experiment or noise-filtering ablation to isolate noise from sample-size or class-imbalance effects. |

---

## 12. Issues & Recommended Clarifications

1. **Terminology Clarification:** In [`analysis/panns_classifier_results.md`](file:///c:/Users/Tanay/Downloads/AI_Stethoscope/analysis/panns_classifier_results.md), the phrase *"new overall state-of-the-art across all baselines"* should be read as *"new internal project benchmark high across all evaluated baselines"*.
2. **Statistical Significance Caution:** The word *"significantly"* in Section 9 should be interpreted as *"by a substantial numerical margin"*, rather than as a claim of formal statistical hypothesis significance ($p < 0.05$).
3. **Heart Model Naming:** The saved Heart model is correctly verified as `NystroemSVMClassifier` (Nystroem RBF kernel approximation + SGD linear hinge loss), which is documented accurately in the text.

---

## Summary Conclusion

The verification audit confirms that the PANNs CNN14 classifier experiment was conducted with **complete test isolation, zero data leakage, verified split/patient isolation, and 100% numerical fidelity** between raw code execution, saved `.joblib` models, JSON metrics, and markdown reporting tables.
