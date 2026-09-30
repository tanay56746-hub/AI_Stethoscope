#!/usr/bin/env python3
# ==============================================================================
# [LAYER: BACKEND INFERENCE CLI] | File: backend/predict_audio.py
# Purpose: End-to-End Offline WAV Audio Inference Pipeline (CNN + Phys + PANNs)
# ==============================================================================
"""
AI Stethoscope - Offline WAV Audio Inference Pipeline
======================================================
Executes end-to-end inference for a single input WAV file using the verified
final multi-task model (CNN + Physical Features + PANNs Embeddings).

Usage:
    python backend/predict_audio.py --audio <path_to_wav>
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import load_env_file
from src.features.preprocessor import AudioPreprocessor
from src.features.audio_features import extract_audio_features
from src.models.panns_cnn14 import Cnn14
from src.models.final_multitask_model import FinalMultiTaskStethoscopeModel

load_env_file()

DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "MODEL_CHECKPOINT_PATH",
        str(
            PROJECT_ROOT
            / "experiments"
            / "multitask_ablations"
            / "cnn_physical_panns"
            / "checkpoints"
            / "best_checkpoint.pt"
        ),
    )
)
if not DEFAULT_CHECKPOINT.is_absolute():
    DEFAULT_CHECKPOINT = PROJECT_ROOT / DEFAULT_CHECKPOINT

DEFAULT_PANNS_WEIGHTS = Path(
    os.environ.get(
        "PANNS_WEIGHTS_PATH",
        str(PROJECT_ROOT / "pretrained_weights" / "Cnn14_mAP=0.431.pth"),
    )
)
if not DEFAULT_PANNS_WEIGHTS.is_absolute():
    DEFAULT_PANNS_WEIGHTS = PROJECT_ROOT / DEFAULT_PANNS_WEIGHTS

# Task class labels
HEART_CLASSES = ["Normal", "Abnormal"]
ACOUSTIC_CLASSES = ["Normal", "Crackle", "Wheeze", "Crackle+Wheeze"]
DISEASE_CLASSES = ["Normal", "Asthma", "Other"]


def load_panns_backbone(weights_path: Path) -> Cnn14:
    """Instantiate and load the frozen PANNs CNN14 feature extractor."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"PANNs pretrained weights not found at: {weights_path}\n"
            f"Please ensure official Cnn14 weights exist."
        )
    panns_model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50.0,
        fmax=14000.0,
        classes_num=527,
    )
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    panns_model.load_state_dict(state_dict, strict=True)
    panns_model.eval()
    for param in panns_model.parameters():
        param.requires_grad = False
    return panns_model


def load_multitask_model(checkpoint_path: Path) -> FinalMultiTaskStethoscopeModel:
    """Instantiate and load the verified final multi-task model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure the final multi-task checkpoint exists."
        )
    model = FinalMultiTaskStethoscopeModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state"]
        if "model_state" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def preprocess_for_cnn_and_physical(audio_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize audio at 16 kHz, compute Log-Mel spectrogram (1, 1, 128, 313)
    and extract 397-dimensional physical acoustic features.
    """
    preprocessor = AudioPreprocessor()
    raw_audio, orig_sr, orig_shape, orig_dur = preprocessor.load_raw_wav(audio_path)

    if len(raw_audio) == 0:
        raise ValueError(f"WAV file contains no audio samples: {audio_path}")

    # Standard preprocessing pipeline
    audio = preprocessor.remove_dc_offset(raw_audio)
    audio = preprocessor.apply_highpass_filter(audio, orig_sr)
    audio = preprocessor.resample_to_target(audio, orig_sr)
    audio = preprocessor.normalize_peak(audio)
    audio_fixed = preprocessor.pad_or_window_fixed_length(audio)

    # Log-Mel (128, 313)
    log_mel = preprocessor.compute_log_mel(audio_fixed)
    mel_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 313)

    # Physical features (397,)
    phys_features = extract_audio_features(log_mel)
    phys_tensor = torch.from_numpy(phys_features).unsqueeze(0).float()  # (1, 397)

    return mel_tensor, phys_tensor


def preprocess_for_panns(audio_path: Path, panns_model: Cnn14) -> torch.Tensor:
    """
    Standardize audio at 32 kHz, fixed 5.0 seconds (160,000 samples),
    peak normalize, and extract 2048-dimensional frozen embedding.
    """
    wav, sr = sf.read(str(audio_path), dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav = wav - np.mean(wav)

    target_sr = 32000
    if sr != target_sr:
        wav_32k = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_mq")
    else:
        wav_32k = wav

    target_len = int(5.0 * target_sr)  # 160,000 samples
    if len(wav_32k) > target_len:
        wav_32k = wav_32k[:target_len]
    elif len(wav_32k) < target_len:
        wav_32k = np.pad(wav_32k, (0, target_len - len(wav_32k)), mode="constant")

    max_val = np.max(np.abs(wav_32k))
    if max_val > 1e-7:
        wav_32k = (wav_32k / max_val) * 0.95

    panns_input = torch.from_numpy(wav_32k).unsqueeze(0).float()  # (1, 160000)

    with torch.no_grad():
        panns_output = panns_model(panns_input)
        embedding = panns_output["embedding"]  # (1, 2048)

    return embedding


def run_inference(
    audio_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    panns_weights_path: Path = DEFAULT_PANNS_WEIGHTS,
) -> dict:
    """Execute full offline pipeline on a single WAV file."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Specified audio file not found: {audio_path}")

    # 1. Feature Extraction
    mel_tensor, phys_tensor = preprocess_for_cnn_and_physical(audio_path)
    panns_model = load_panns_backbone(panns_weights_path)
    panns_embedding = preprocess_for_panns(audio_path, panns_model)

    # 2. Multi-Task Forward Pass
    model = load_multitask_model(checkpoint_path)

    with torch.no_grad():
        outputs = model(
            mel=mel_tensor,
            acoustic_features=phys_tensor,
            panns_embedding=panns_embedding,
        )
        heart_probs = torch.softmax(outputs["heart_logits"], dim=-1)[0].cpu().numpy()
        acoustic_probs = torch.softmax(outputs["acoustic_logits"], dim=-1)[0].cpu().numpy()
        disease_probs = torch.softmax(outputs["disease_logits"], dim=-1)[0].cpu().numpy()

    # 3. Interpret Predictions
    heart_pred_idx = int(np.argmax(heart_probs))
    acoustic_pred_idx = int(np.argmax(acoustic_probs))
    disease_pred_idx = int(np.argmax(disease_probs))

    return {
        "audio_file": str(audio_path),
        "heart": {
            "prediction": HEART_CLASSES[heart_pred_idx],
            "probabilities": {
                c: float(heart_probs[i]) for i, c in enumerate(HEART_CLASSES)
            },
        },
        "acoustic": {
            "prediction": ACOUSTIC_CLASSES[acoustic_pred_idx],
            "probabilities": {
                c: float(acoustic_probs[i]) for i, c in enumerate(ACOUSTIC_CLASSES)
            },
        },
        "disease": {
            "prediction": DISEASE_CLASSES[disease_pred_idx],
            "probabilities": {
                c: float(disease_probs[i]) for i, c in enumerate(DISEASE_CLASSES)
            },
        },
    }


def print_formatted_results(results: dict):
    """Print results in standard human-readable terminal format."""
    print("========================================")
    print("AI STETHOSCOPE OFFLINE INFERENCE")
    print("========================================")
    print("")
    print("Audio:")
    print(f"{results['audio_file']}")
    print("")
    print("HEART")
    print(f"Prediction: {results['heart']['prediction']}")
    print("Probabilities:")
    for c, p in results['heart']['probabilities'].items():
        print(f"  {c}: {p * 100:.2f}%")
    print("")
    print("LUNG ACOUSTIC")
    print(f"Prediction: {results['acoustic']['prediction']}")
    print("Probabilities:")
    for c, p in results['acoustic']['probabilities'].items():
        print(f"  {c}: {p * 100:.2f}%")
    print("")
    print("LUNG DISEASE")
    print(f"Prediction: {results['disease']['prediction']}")
    print("Probabilities:")
    for c, p in results['disease']['probabilities'].items():
        print(f"  {c}: {p * 100:.2f}%")
    print("")
    print("NOTE:")
    print("These are machine-learning predictions from the trained research model")
    print("and are not a medical diagnosis.")
    print("========================================")


def main():
    parser = argparse.ArgumentParser(
        description="Run offline WAV audio inference with the AI Stethoscope final model."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to the WAV audio file for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained model checkpoint (default: cnn_physical_panns best checkpoint).",
    )
    parser.add_argument(
        "--panns-weights",
        type=str,
        default=str(DEFAULT_PANNS_WEIGHTS),
        help="Path to official PANNs Cnn14 pretrained weights.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    panns_weights_path = Path(args.panns_weights).resolve()

    try:
        results = run_inference(
            audio_path=audio_path,
            checkpoint_path=checkpoint_path,
            panns_weights_path=panns_weights_path,
        )
        print_formatted_results(results)
    except Exception as e:
        print(f"ERROR: Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
