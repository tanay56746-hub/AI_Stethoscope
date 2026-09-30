#!/usr/bin/env python3
# ==============================================================================
# [LAYER: BACKEND INFERENCE ENGINE] | File: backend/inference_engine.py
# Purpose: Singleton Cached Inference Engine & Waveform/Spectrogram Generator
# ==============================================================================
"""
AI Stethoscope - Inference Engine for Demo API
==============================================
Wraps the offline inference pipeline from `backend.predict_audio` with model
caching and rich visualization data (downsampled waveform and log-mel matrix)
for real-time web UI demonstration.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessor import AudioPreprocessor
from src.features.audio_features import extract_audio_features
from backend.predict_audio import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PANNS_WEIGHTS,
    HEART_CLASSES,
    ACOUSTIC_CLASSES,
    DISEASE_CLASSES,
    load_panns_backbone,
    load_multitask_model,
    preprocess_for_panns,
)


class StethoscopeInferenceEngine:
    """Cached inference engine for the AI Stethoscope demonstration."""

    _instance = None

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        panns_weights_path: Path = DEFAULT_PANNS_WEIGHTS,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.panns_weights_path = Path(panns_weights_path).resolve()
        self.preprocessor = AudioPreprocessor()

        print("[Engine] Loading PANNs CNN14 backbone...")
        self.panns_model = load_panns_backbone(self.panns_weights_path)
        print("[Engine] PANNs CNN14 backbone loaded.")

        print("[Engine] Loading Multi-Task Stethoscope model...")
        self.multitask_model = load_multitask_model(self.checkpoint_path)
        print("[Engine] Multi-Task Stethoscope model loaded successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = StethoscopeInferenceEngine()
        return cls._instance

    def analyze_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Executes end-to-end inference on the audio file, returning model predictions,
        probabilities, and visualization representations (waveform + log-mel).
        """
        start_time = time.time()
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. Load and Preprocess for CNN & Physical Features
        raw_audio, orig_sr, orig_shape, orig_dur = self.preprocessor.load_raw_wav(audio_path)
        if len(raw_audio) == 0:
            raise ValueError(f"Audio file has zero samples: {audio_path.name}")

        audio_dc = self.preprocessor.remove_dc_offset(raw_audio)
        audio_hp = self.preprocessor.apply_highpass_filter(audio_dc, orig_sr)
        audio_16k = self.preprocessor.resample_to_target(audio_hp, orig_sr)
        audio_norm = self.preprocessor.normalize_peak(audio_16k)
        audio_fixed = self.preprocessor.pad_or_window_fixed_length(audio_norm)

        # Log-Mel (128, 313)
        log_mel = self.preprocessor.compute_log_mel(audio_fixed)
        mel_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()

        # Physical acoustic features (397,)
        phys_features = extract_audio_features(log_mel)
        phys_tensor = torch.from_numpy(phys_features).unsqueeze(0).float()

        # 2. Extract PANNs Embedding (2048,)
        panns_embedding = preprocess_for_panns(audio_path, self.panns_model)

        # 3. Model Forward Pass
        with torch.no_grad():
            outputs = self.multitask_model(
                mel=mel_tensor,
                acoustic_features=phys_tensor,
                panns_embedding=panns_embedding,
            )
            heart_probs = torch.softmax(outputs["heart_logits"], dim=-1)[0].cpu().numpy()
            acoustic_probs = torch.softmax(outputs["acoustic_logits"], dim=-1)[0].cpu().numpy()
            disease_probs = torch.softmax(outputs["disease_logits"], dim=-1)[0].cpu().numpy()

        inference_duration = time.time() - start_time

        # 4. Generate Downsampled Waveform (~300 points)
        # Using peak-envelope downsampling so spikes are preserved
        n_points = 300
        step = max(1, len(audio_fixed) // n_points)
        waveform_downsampled = []
        for i in range(0, len(audio_fixed), step):
            chunk = audio_fixed[i : i + step]
            if len(chunk) > 0:
                # pick value with maximum absolute amplitude in window
                idx = np.argmax(np.abs(chunk))
                waveform_downsampled.append(round(float(chunk[idx]), 4))
            if len(waveform_downsampled) >= n_points:
                break

        # 5. Generate Downsampled Spectrogram (~64 frequency bins x 100 time frames)
        # log_mel shape is (128, 313).
        # We downsample by taking every 2nd bin along freq, and every 3rd frame along time
        spec_reduced = log_mel[::2, ::3]  # shape: (64, 105)
        # Normalize to 0.0 - 1.0 range for simple color mapping on canvas
        spec_min = float(np.min(spec_reduced))
        spec_max = float(np.max(spec_reduced))
        if spec_max - spec_min > 1e-6:
            spec_normalized = (spec_reduced - spec_min) / (spec_max - spec_min)
        else:
            spec_normalized = np.zeros_like(spec_reduced)
        # Round values to 3 decimals to keep JSON payload lightweight
        spec_matrix = np.round(spec_normalized, 3).tolist()

        # 5. Results dictionary
        heart_idx = int(np.argmax(heart_probs))
        acoustic_idx = int(np.argmax(acoustic_probs))
        disease_idx = int(np.argmax(disease_probs))

        return {
            "success": True,
            "audio": {
                "filename": audio_path.name,
                "duration": round(orig_dur, 2),
                "analyzed_duration": 5.0,
                "sample_rate": 16000,
                "original_sample_rate": orig_sr,
                "channels": 1 if raw_audio.ndim == 1 else raw_audio.shape[0],
            },
            "heart": {
                "prediction": HEART_CLASSES[heart_idx],
                "probabilities": {
                    c: round(float(heart_probs[i]), 4)
                    for i, c in enumerate(HEART_CLASSES)
                },
            },
            "lung_acoustic": {
                "prediction": ACOUSTIC_CLASSES[acoustic_idx],
                "probabilities": {
                    c: round(float(acoustic_probs[i]), 4)
                    for i, c in enumerate(ACOUSTIC_CLASSES)
                },
            },
            "lung_disease": {
                "prediction": DISEASE_CLASSES[disease_idx],
                "probabilities": {
                    c: round(float(disease_probs[i]), 4)
                    for i, c in enumerate(DISEASE_CLASSES)
                },
            },
            "inference_time": round(inference_duration, 3),
            "waveform": waveform_downsampled,
            "spectrogram": {
                "matrix": spec_matrix,
                "freq_bins": len(spec_matrix),
                "time_frames": len(spec_matrix[0]) if len(spec_matrix) > 0 else 0,
                "f_min": 20,
                "f_max": 2000,
            },
        }


if __name__ == "__main__":
    test_audio = PROJECT_ROOT / "datasets" / "Heart" / "Raw" / "training" / "training-b" / "b0032.wav"
    print(f"Testing engine on: {test_audio}")
    engine = StethoscopeInferenceEngine.get_instance()
    res = engine.analyze_audio(test_audio)
    print("Inference completed in:", res["inference_time"], "seconds")
    print("Heart prediction:", res["heart"])
    print("Lung acoustic prediction:", res["lung_acoustic"])
    print("Lung disease prediction:", res["lung_disease"])
    print("Waveform points:", len(res["waveform"]))
    print("Spectrogram dims:", res["spectrogram"]["freq_bins"], "x", res["spectrogram"]["time_frames"])
