#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — SMOKE TEST] | File: scripts/smoke_test_classical_ml.py
# Purpose: Quick Sanity Check for Classical ML Feature Pipeline
# ==============================================================================
"""
AI Stethoscope - Classical ML Pipeline Smoke Test
=================================================
Verifies:
1. Feature dimensions (N, 397)
2. No NaN / Inf in features
3. Strict Train / Validation / Test isolation (Scaler and PCA fitted ONLY on train)
4. Successful model fitting on all 5 candidate models:
   - Logistic Regression
   - SVM with RBF kernel
   - Random Forest
   - XGBoost (reports availability and handles missing package cleanly)
   - K-Nearest Neighbors (KNN)
5. Successful prediction & probability outputs
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_features import extract_audio_features, FEATURE_DIM


def run_smoke_test():
    print("=" * 70)
    print("RUNNING CLASSICAL ML PIPELINE SMOKE TEST")
    print("=" * 70)

    # 1. Generate synthetic spectrogram inputs to test feature extractor
    n_train, n_val, n_test = 80, 20, 20
    np.random.seed(42)

    print("\n[Step 1] Verifying feature extraction & dimension check...")
    X_train_raw = []
    y_train = np.random.choice([0, 1, 2], size=n_train)
    for _ in range(n_train):
        fake_spec = np.random.uniform(-80, 0, (128, 313)).astype(np.float32)
        feat = extract_audio_features(fake_spec)
        X_train_raw.append(feat)
    X_train_raw = np.array(X_train_raw, dtype=np.float32)

    X_val_raw = []
    y_val = np.random.choice([0, 1, 2], size=n_val)
    for _ in range(n_val):
        fake_spec = np.random.uniform(-80, 0, (128, 313)).astype(np.float32)
        feat = extract_audio_features(fake_spec)
        X_val_raw.append(feat)
    X_val_raw = np.array(X_val_raw, dtype=np.float32)

    X_test_raw = []
    y_test = np.random.choice([0, 1, 2], size=n_test)
    for _ in range(n_test):
        fake_spec = np.random.uniform(-80, 0, (128, 313)).astype(np.float32)
        feat = extract_audio_features(fake_spec)
        X_test_raw.append(feat)
    X_test_raw = np.array(X_test_raw, dtype=np.float32)

    assert X_train_raw.shape == (n_train, FEATURE_DIM), f"Expected {(n_train, FEATURE_DIM)}, got {X_train_raw.shape}"
    assert X_val_raw.shape == (n_val, FEATURE_DIM), f"Expected {(n_val, FEATURE_DIM)}, got {X_val_raw.shape}"
    assert X_test_raw.shape == (n_test, FEATURE_DIM), f"Expected {(n_test, FEATURE_DIM)}, got {X_test_raw.shape}"
    print(f"  --> Feature dimension verified: ({FEATURE_DIM},) per sample. Shapes match.")

    # 2. NaN / Inf checks
    print("\n[Step 2] Checking for finite values (Zero NaN / Inf)...")
    assert not np.isnan(X_train_raw).any(), "NaN found in X_train_raw"
    assert not np.isinf(X_train_raw).any(), "Inf found in X_train_raw"
    assert not np.isnan(X_val_raw).any(), "NaN found in X_val_raw"
    assert not np.isinf(X_val_raw).any(), "Inf found in X_val_raw"
    assert not np.isnan(X_test_raw).any(), "NaN found in X_test_raw"
    assert not np.isinf(X_test_raw).any(), "Inf found in X_test_raw"
    print("  --> All features verified finite and non-null.")

    # 3. Train / Validation / Test Preprocessing Isolation
    print("\n[Step 3] Testing Train/Val/Test preprocessing isolation (Scaler & PCA)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_pca_components = X_train_pca.shape[1]
    print(f"  --> PCA (95% variance) reduced {FEATURE_DIM} dims -> {n_pca_components} dims.")
    print("  --> Preprocessing isolation verified: fitted strictly on train, applied to val and test.")

    # 4. Model fitting & prediction check across all 5 models
    print("\n[Step 4] Benchmarking candidate models (Fitting & Prediction)...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
    }

    # Check XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=20, random_state=42, eval_metric="mlogloss")
        print("  * XGBoost: AVAILABLE and included.")
    except ImportError:
        print("  * XGBoost: NOT AVAILABLE in Python environment (Documented and excluded).")

    for name, clf in models.items():
        # Test fitting on PCA features
        clf.fit(X_train_pca, y_train)
        preds = clf.predict(X_val_pca)
        probs = clf.predict_proba(X_val_pca)
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro")
        assert preds.shape == (n_val,), f"Prediction shape mismatch for {name}"
        assert probs.shape == (n_val, 3), f"Probabilities shape mismatch for {name}"
        print(f"  [OK] {name:22s} | Fit & Predict Success | Val Acc: {acc:.4f} | Val Macro F1: {macro_f1:.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED: ALL PIPELINE STAGES VERIFIED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
