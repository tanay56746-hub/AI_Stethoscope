"""
AI Stethoscope - Acoustic Feature Extraction Module
===================================================
Extracts a compact, fixed-length 397-dimensional feature vector from
Log-Mel spectrograms (128, 313) with optional temporal validity masks.

Features included:
1. Log-Mel spectrogram temporal mean (128)
2. Log-Mel spectrogram temporal standard deviation (128)
3. MFCC coefficients temporal mean (20)
4. MFCC coefficients temporal standard deviation (20)
5. Delta MFCC temporal mean (20)
6. Delta MFCC temporal standard deviation (20)
7. Delta-Delta MFCC temporal mean (20)
8. Delta-Delta MFCC temporal standard deviation (20)
9. RMS energy statistics: mean, std, min, max, skew (5)
10. Spectral Centroid statistics: mean, std, min, max (4)
11. Spectral Bandwidth statistics: mean, std, min, max (4)
12. Spectral Rolloff (85%) statistics: mean, std, min, max (4)
13. Zero-Crossing Rate (ZCR) statistics: mean, std, min, max (4)

Total Feature Dimension: 397
"""

from typing import Optional, List, Dict, Tuple
import numpy as np
import scipy.fftpack
from scipy import stats
import librosa

N_MELS = 128
TIME_FRAMES = 313
N_MFCC = 20
F_MIN = 20.0
F_MAX = 2000.0

# Pre-calculate Mel frequency bin centers (128,)
MEL_FREQUENCIES = librosa.mel_frequencies(n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX).astype(np.float32)

FEATURE_DIM = 397


def _build_feature_names() -> List[str]:
    names = []
    # 1. Log-Mel mean (128)
    for i in range(N_MELS):
        names.append(f"logmel_mean_bin{i}")
    # 2. Log-Mel std (128)
    for i in range(N_MELS):
        names.append(f"logmel_std_bin{i}")
    # 3. MFCC mean & std (20 + 20)
    for i in range(N_MFCC):
        names.append(f"mfcc{i}_mean")
    for i in range(N_MFCC):
        names.append(f"mfcc{i}_std")
    # 4. Delta MFCC mean & std (20 + 20)
    for i in range(N_MFCC):
        names.append(f"delta_mfcc{i}_mean")
    for i in range(N_MFCC):
        names.append(f"delta_mfcc{i}_std")
    # 5. Delta-Delta MFCC mean & std (20 + 20)
    for i in range(N_MFCC):
        names.append(f"delta2_mfcc{i}_mean")
    for i in range(N_MFCC):
        names.append(f"delta2_mfcc{i}_std")
    # 6. RMS energy stats (5)
    for s in ["mean", "std", "min", "max", "skew"]:
        names.append(f"rms_{s}")
    # 7. Spectral Centroid stats (4)
    for s in ["mean", "std", "min", "max"]:
        names.append(f"spec_centroid_{s}")
    # 8. Spectral Bandwidth stats (4)
    for s in ["mean", "std", "min", "max"]:
        names.append(f"spec_bandwidth_{s}")
    # 9. Spectral Rolloff stats (4)
    for s in ["mean", "std", "min", "max"]:
        names.append(f"spec_rolloff_{s}")
    # 10. ZCR stats (4)
    for s in ["mean", "std", "min", "max"]:
        names.append(f"zcr_{s}")
    return names


FEATURE_NAMES = _build_feature_names()
assert len(FEATURE_NAMES) == FEATURE_DIM, f"Expected {FEATURE_DIM} names, got {len(FEATURE_NAMES)}"


def compute_deltas(feats: np.ndarray, width: int = 9) -> np.ndarray:
    """
    Vectorized computation of delta coefficients along time axis (-1).
    Equivalent to librosa.feature.delta with edge padding.
    """
    if feats.shape[-1] < 3:
        return np.zeros_like(feats)

    half = width // 2
    pad = np.pad(feats, ((0, 0), (half, half)), mode="edge")
    kernel = np.arange(-half, half + 1, dtype=np.float32)
    denom = 2.0 * float(np.sum(np.arange(1, half + 1) ** 2))

    # Apply 1D conv along last dimension
    # Pad shape: (C, T + 2*half)
    deltas = np.zeros_like(feats)
    for i in range(feats.shape[0]):
        deltas[i] = np.convolve(pad[i], kernel[::-1], mode="valid") / denom
    return deltas.astype(np.float32)


def extract_audio_features(
    spectrogram: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract fixed-length 397-dimensional feature vector from a Log-Mel spectrogram.

    Parameters:
        spectrogram: np.ndarray of shape (128, 313) or (1, 128, 313).
        mask: Optional binary temporal mask of shape (313,) where 1.0 indicates
              active acoustic signal and 0.0 indicates zero-padding.

    Returns:
        np.ndarray of shape (397,) float32 with finite values.
    """
    # Normalize input shape
    if spectrogram.ndim == 3 and spectrogram.shape[0] == 1:
        S_db = spectrogram[0].astype(np.float32)
    elif spectrogram.ndim == 2:
        S_db = spectrogram.astype(np.float32)
    else:
        raise ValueError(f"Unexpected spectrogram shape {spectrogram.shape}; expected (128, 313) or (1, 128, 313)")

    # Temporal masking
    if mask is not None:
        mask = np.asarray(mask).squeeze()
        valid_indices = np.where(mask > 0.5)[0]
        if len(valid_indices) > 5:
            S_act = S_db[:, valid_indices]
        else:
            # Fallback if mask is degenerate
            S_act = S_db
    else:
        S_act = S_db

    # Guarantee non-empty
    if S_act.shape[1] == 0:
        S_act = S_db

    # Replace any non-finite values if present
    S_act = np.nan_to_num(S_act, nan=-80.0, posinf=0.0, neginf=-80.0)

    # 1. Log-Mel spectrogram mean & std across time (128 + 128 = 256 dims)
    mel_mean = np.mean(S_act, axis=1)
    mel_std = np.std(S_act, axis=1)

    # 2. MFCC coefficients (DCT-II along frequency axis)
    mfcc = scipy.fftpack.dct(S_act, type=2, axis=0, norm="ortho")[:N_MFCC]  # (20, T)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 3. Delta MFCC (20 + 20 dims)
    d_mfcc = compute_deltas(mfcc, width=9)
    d_mean = np.mean(d_mfcc, axis=1)
    d_std = np.std(d_mfcc, axis=1)

    # 4. Delta-Delta MFCC (20 + 20 dims)
    d2_mfcc = compute_deltas(d_mfcc, width=9)
    d2_mean = np.mean(d2_mfcc, axis=1)
    d2_std = np.std(d2_mfcc, axis=1)

    # 5. Linear power spectrum for physical acoustic quantities
    # Clamp dB to reasonable range [-100, +20] before exponentiating
    S_clamped = np.clip(S_act, -100.0, 20.0)
    P = np.power(10.0, S_clamped / 10.0)  # (128, T)
    P_sum = np.sum(P, axis=0) + 1e-12     # (T,)

    # RMS Energy Statistics (5 dims)
    rms = np.sqrt(np.mean(P, axis=0))      # (T,)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_min = float(np.min(rms))
    rms_max = float(np.max(rms))
    rms_skew = float(stats.skew(rms)) if len(rms) > 2 and rms_std > 1e-7 else 0.0
    rms_stats = [rms_mean, rms_std, rms_min, rms_max, rms_skew]

    # Spectral Centroid (4 dims)
    centroid = np.sum(MEL_FREQUENCIES[:, None] * P, axis=0) / P_sum  # (T,)
    sc_stats = [float(np.mean(centroid)), float(np.std(centroid)), float(np.min(centroid)), float(np.max(centroid))]

    # Spectral Bandwidth (4 dims)
    dev_sq = (MEL_FREQUENCIES[:, None] - centroid[None, :]) ** 2
    bandwidth = np.sqrt(np.sum(dev_sq * P, axis=0) / P_sum)  # (T,)
    sb_stats = [float(np.mean(bandwidth)), float(np.std(bandwidth)), float(np.min(bandwidth)), float(np.max(bandwidth))]

    # Spectral Rolloff 85% (4 dims)
    cum_energy = np.cumsum(P, axis=0)
    thresh = 0.85 * P_sum[None, :]
    rolloff_idx = np.argmax(cum_energy >= thresh, axis=0)
    rolloff = MEL_FREQUENCIES[rolloff_idx]  # (T,)
    ro_stats = [float(np.mean(rolloff)), float(np.std(rolloff)), float(np.min(rolloff)), float(np.max(rolloff))]

    # Spectral Zero-Crossing Rate Estimator (Rice's second moment formula) (4 dims)
    m2 = np.sum((MEL_FREQUENCIES[:, None] ** 2) * P, axis=0) / P_sum
    zcr_est = np.sqrt(np.maximum(0.0, m2)) / (np.pi * F_MAX)
    zcr_stats = [float(np.mean(zcr_est)), float(np.std(zcr_est)), float(np.min(zcr_est)), float(np.max(zcr_est))]

    # Concatenate all features into single 397-d vector
    features = np.concatenate([
        mel_mean,       # 128
        mel_std,        # 128
        mfcc_mean,      # 20
        mfcc_std,       # 20
        d_mean,         # 20
        d_std,          # 20
        d2_mean,        # 20
        d2_std,         # 20
        np.array(rms_stats, dtype=np.float32),   # 5
        np.array(sc_stats, dtype=np.float32),    # 4
        np.array(sb_stats, dtype=np.float32),    # 4
        np.array(ro_stats, dtype=np.float32),    # 4
        np.array(zcr_stats, dtype=np.float32),   # 4
    ]).astype(np.float32)

    # Final finite check
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features
