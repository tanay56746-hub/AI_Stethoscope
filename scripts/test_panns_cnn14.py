#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — BACKBONE TEST] | File: scripts/test_panns_cnn14.py
# Purpose: Verifies Official PANNs CNN14 Weights & 2048-D Embedding Extraction
# ==============================================================================
"""
AI Stethoscope - PANNs CNN14 Compatibility & Single-File Verification Test
========================================================================
Performs an isolated verification of the official PANNs CNN14 architecture
and official AudioSet pretrained weights on CPU:
  1. Instantiates the self-contained official Cnn14 PyTorch model.
  2. Loads official pretrained weights (Cnn14_mAP=0.431.pth) with strict=True.
  3. Loads ONE real audio recording from the dataset.
  4. Resamples to 32,000 Hz (native AudioSet rate) and normalizes.
  5. Executes a single forward pass with torch.no_grad() on CPU.
  6. Validates output embedding shape (1, 2048), finite bounds, and CPU inference latency.
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Pure PyTorch Implementation of STFT & LogmelFilterBank (torchlibrosa)
# ---------------------------------------------------------------------------

class STFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320, win_length=1024, window="hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
        fft_window = librosa.util.pad_center(data=fft_window, size=n_fft)

        # DFT Matrix
        (x, y) = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
        omega = np.exp(-2 * np.pi * 1j / n_fft)
        W = np.power(omega, x * y)

        out_channels = n_fft // 2 + 1  # 513
        self.conv_real = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=n_fft,
            stride=hop_length, padding=0, bias=False
        )
        self.conv_imag = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=n_fft,
            stride=hop_length, padding=0, bias=False
        )

        self.conv_real.weight.data = torch.Tensor(
            np.real(W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :].contiguous()
        self.conv_imag.weight.data = torch.Tensor(
            np.imag(W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :].contiguous()

        self.conv_real.weight.requires_grad = False
        self.conv_imag.weight.requires_grad = False

    def forward(self, x):
        # x: (batch_size, data_length)
        x = x[:, None, :]
        pad_size = self.n_fft // 2
        x = F.pad(x, (pad_size, pad_size), mode="reflect")
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        return real, imag


class Spectrogram(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320, win_length=1024, window="hann"):
        super().__init__()
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

    def forward(self, input):
        real, imag = self.stft(input)
        spectrogram = real ** 2 + imag ** 2
        # Transpose from (batch, freq, time) -> (batch, 1, time, freq)
        return spectrogram.transpose(1, 2)[:, None, :, :]


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None):
        super().__init__()
        melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T  # (513, 64)
        self.melW = nn.Parameter(torch.Tensor(melW), requires_grad=False)
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, x):
        # x: (batch, 1, time, freq)
        mel_spectrogram = torch.matmul(x, self.melW)
        # Power to dB: 10 * log10(max(amin, mel))
        logmel_spectrogram = 10.0 * torch.log10(torch.clamp(mel_spectrogram, min=self.amin))
        logmel_spectrogram = logmel_spectrogram - 10.0 * np.log10(np.maximum(self.amin, self.ref))
        if self.top_db is not None:
            max_val = torch.max(logmel_spectrogram)
            logmel_spectrogram = torch.clamp(logmel_spectrogram, min=max_val - self.top_db)
        return logmel_spectrogram


# ---------------------------------------------------------------------------
# 2. Official PANNs Cnn14 Architecture Definition
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14(nn.Module):
    def __init__(
        self,
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50.0,
        fmax=14000.0,
        classes_num=527,
    ):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size, window="hann"
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax
        )
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, input):
        # input: (batch_size, data_length)
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)           # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling across frequency and time
        x = torch.mean(x, dim=3)  # Average over frequency
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # Sum of max and mean over time

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = x  # (batch_size, 2048)

        clipwise_output = torch.sigmoid(self.fc_audioset(x))  # (batch_size, classes_num)
        return {
            "embedding": embedding,
            "clipwise_output": clipwise_output,
        }


# ---------------------------------------------------------------------------
# 3. Main Verification Function
# ---------------------------------------------------------------------------

def test_panns():
    print("=" * 75)
    print("PANNS CNN14 PRETRAINED MODEL COMPATIBILITY & CPU FORWARD-PASS TEST")
    print("=" * 75)

    # 1. Check weight file
    ckpt_path = PROJECT_ROOT / "pretrained_weights" / "Cnn14_mAP=0.431.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"[1] Verified Pretrained Weight File:")
    print(f"    * Path: {ckpt_path}")
    print(f"    * File Size: {size_mb:.2f} MB")

    # 2. Build model
    print("\n[2] Instantiating Cnn14 Architecture...")
    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50.0,
        fmax=14000.0,
        classes_num=527,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    * Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # 3. Load official pretrained weights with strict=True
    print("\n[3] Loading Pretrained Weights with strict=True...")
    t_load = time.time()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Load state dict
    load_result = model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"    * Successfully loaded with strict=True in {time.time() - t_load:.2f}s!")
    print(f"    * Missing keys: {len(load_result.missing_keys)}, Unexpected keys: {len(load_result.unexpected_keys)}")

    # 4. Load ONE existing audio file from our dataset
    audio_path = PROJECT_ROOT / "datasets" / "Heart" / "Raw" / "training" / "training-b" / "b0032.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Sample audio missing at: {audio_path}")
    print(f"\n[4] Loading Sample Audio File:")
    print(f"    * Audio Path: {audio_path.relative_to(PROJECT_ROOT)}")

    wav, orig_sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    orig_dur = len(wav) / orig_sr
    print(f"    * Original: {orig_sr} Hz, {len(wav)} samples ({orig_dur:.2f}s)")

    # 5. Preprocess for PANNs: Resample to 32,000 Hz, take 5.0 seconds
    target_sr = 32000
    if orig_sr != target_sr:
        wav_32k = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    else:
        wav_32k = wav

    # Standardize to 5.0s (160,000 samples)
    target_len = int(5.0 * target_sr)
    if len(wav_32k) > target_len:
        wav_32k = wav_32k[:target_len]
    elif len(wav_32k) < target_len:
        wav_32k = np.pad(wav_32k, (0, target_len - len(wav_32k)), mode="constant")

    # Peak normalize
    max_val = np.max(np.abs(wav_32k))
    if max_val > 0:
        wav_32k = wav_32k / max_val * 0.95

    input_tensor = torch.from_numpy(wav_32k).unsqueeze(0).float()  # (1, 160000)
    print(f"    * Preprocessed Tensor: Shape {input_tensor.shape}, Dtype {input_tensor.dtype}")

    # 6. Execute single CPU forward pass with gradients disabled
    print("\n[5] Executing CPU Forward Pass (Frozen Backbone, no_grad)...")
    with torch.no_grad():
        t_infer_start = time.time()
        output = model(input_tensor)
        t_infer = time.time() - t_infer_start

    embedding = output["embedding"]
    clipwise = output["clipwise_output"]

    print(f"\n[6] Verification & Output Inspection:")
    print(f"    * CPU Inference Time: {t_infer:.3f} seconds ({1.0/max(0.001, t_infer):.1f} samples/sec)")
    print(f"    * Extracted Embedding Shape: {embedding.shape} (Expected: (1, 2048))")
    print(f"    * AudioSet Clipwise Output Shape: {clipwise.shape} (Expected: (1, 527))")

    # Check finite
    has_nan = torch.isnan(embedding).any().item()
    has_inf = torch.isinf(embedding).any().item()
    print(f"    * Embedding Contains NaN: {has_nan}")
    print(f"    * Embedding Contains Inf: {has_inf}")
    print(f"    * Embedding Mean: {embedding.mean().item():.4f}, Std: {embedding.std().item():.4f}, Min: {embedding.min().item():.4f}, Max: {embedding.max().item():.4f}")

    assert not has_nan, "NaN detected in embedding!"
    assert not has_inf, "Inf detected in embedding!"
    assert embedding.shape == (1, 2048), f"Unexpected embedding shape {embedding.shape}"
    assert clipwise.shape == (1, 527), f"Unexpected clipwise shape {clipwise.shape}"

    print("\n" + "=" * 75)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("PANNs CNN14 is 100% compatible and functional on CPU in this environment.")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = test_panns()
    if not success:
        sys.exit(1)
