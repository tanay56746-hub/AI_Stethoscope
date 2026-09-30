# ==============================================================================
# [LAYER: ML FEATURE PREPROCESSING] | File: src/features/preprocessor.py
# Purpose: Standardized 16 kHz Audio Filtering, Windowing & Log-Mel Extraction
# ==============================================================================
"""
AI Stethoscope - Standardized Audio Preprocessor
================================================
Modular, configurable audio preprocessor for Heart and Lung auscultation signals:
  RAW WAV -> Mono -> DC Offset Removal -> 20 Hz High-pass -> Resample to 16 kHz
  -> Peak Normalization -> Fixed 5-second Framing (80,000 samples) -> Log-Mel Spectrogram.
"""

from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa

DEFAULT_CONFIG = {
    "target_sample_rate": 16000,
    "highpass_cutoff": 20.0,
    "highpass_order": 4,
    "target_peak": 0.95,
    "target_duration": 5.0,  # seconds
    "target_samples": 80000,  # 5.0 * 16000
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 20.0,
    "f_max": 2000.0,
    "power": 2.0,
    "top_db": 80.0,
}


class AudioPreprocessor:
    """
    Modular, configurable audio preprocessor for Heart and Lung auscultation signals.
    """

    def __init__(self, config: dict = None):
        self.cfg = DEFAULT_CONFIG.copy()
        if config:
            self.cfg.update(config)

        self.target_sr = self.cfg["target_sample_rate"]
        self.target_duration = self.cfg["target_duration"]
        self.target_samples = int(self.target_sr * self.target_duration)
        self.highpass_cutoff = self.cfg["highpass_cutoff"]
        self.highpass_order = self.cfg["highpass_order"]
        self.target_peak = self.cfg["target_peak"]

        # Mel Spectrogram parameters
        self.n_fft = self.cfg["n_fft"]
        self.hop_length = self.cfg["hop_length"]
        self.n_mels = self.cfg["n_mels"]
        self.f_min = self.cfg["f_min"]
        self.f_max = self.cfg["f_max"]
        self.power = self.cfg["power"]
        self.top_db = self.cfg["top_db"]

    def load_raw_wav(self, file_path: Path):
        """Read WAV file safely and convert to float32."""
        sr, data = wavfile.read(str(file_path))
        orig_shape = data.shape
        orig_dur = len(data) / float(sr) if sr > 0 else 0.0

        # Mono conversion
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Normalize integer PCM to [-1.0, 1.0]
        if np.issubdtype(data.dtype, np.integer):
            max_v = float(np.iinfo(data.dtype).max)
            audio = data.astype(np.float32) / max_v
        elif np.issubdtype(data.dtype, np.floating):
            audio = data.astype(np.float32)
        else:
            audio = data.astype(np.float32)

        return audio, sr, orig_shape, orig_dur

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC electrical offset / mean centering."""
        return audio - np.mean(audio)

    def apply_highpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a Butterworth high-pass filter around 20 Hz to remove
        sub-audible chest contact friction and baseline drift.
        """
        nyquist = sr / 2.0
        cutoff = self.highpass_cutoff
        if cutoff >= nyquist * 0.95:
            return audio

        sos = signal.butter(self.highpass_order, cutoff, btype="highpass", fs=sr, output="sos")
        filtered = signal.sosfiltfilt(sos, audio)
        return filtered.astype(np.float32)

    def resample_to_target(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Band-limited sinc resampling to target_sr (16,000 Hz)."""
        if orig_sr == self.target_sr:
            return audio
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr, res_type="soxr_vhq")
        return resampled.astype(np.float32)

    def normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """Peak amplitude normalization to target_peak (0.95)."""
        max_abs = np.max(np.abs(audio))
        if max_abs > 1e-7:
            normalized = (audio / max_abs) * self.target_peak
        else:
            normalized = audio
        return normalized.astype(np.float32)

    def pad_or_window_fixed_length(self, audio: np.ndarray, pad_mode: str = "end") -> np.ndarray:
        """
        Ensure fixed-length output of exactly self.target_samples (80,000 samples = 5.0s @ 16kHz).
        Truncates if longer; zero-pads if shorter.
        """
        n = len(audio)
        if n == self.target_samples:
            return audio
        elif n > self.target_samples:
            return audio[: self.target_samples]
        else:
            pad_len = self.target_samples - n
            if pad_mode == "end":
                padded = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)
            elif pad_mode == "center":
                pad_l = pad_len // 2
                pad_r = pad_len - pad_l
                padded = np.pad(audio, (pad_l, pad_r), mode="constant", constant_values=0.0)
            else:
                padded = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)
            return padded.astype(np.float32)

    def compute_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Log-Mel spectrogram.
        Parameters: sr=16000, n_fft=1024, hop_length=256, n_mels=128, f_min=20, f_max=2000.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
        return log_mel.astype(np.float32)

    def process_continuous_audio(self, file_path: Path):
        """Full pipeline for continuous audio (Heart, Kaggle)."""
        raw_audio, orig_sr, orig_shape, orig_dur = self.load_raw_wav(file_path)

        audio = self.remove_dc_offset(raw_audio)
        audio = self.apply_highpass_filter(audio, orig_sr)
        audio = self.resample_to_target(audio, orig_sr)
        audio = self.normalize_peak(audio)
        audio_fixed = self.pad_or_window_fixed_length(audio)
        log_mel = self.compute_log_mel(audio_fixed)

        processed_dur = len(audio_fixed) / float(self.target_sr)

        meta = {
            "orig_sr": orig_sr,
            "orig_dur": orig_dur,
            "target_sr": self.target_sr,
            "processed_dur": processed_dur,
            "waveform_shape": audio_fixed.shape,
            "mel_shape": log_mel.shape,
            "wav_min": float(np.min(audio_fixed)),
            "wav_max": float(np.max(audio_fixed)),
            "mel_min": float(np.min(log_mel)),
            "mel_max": float(np.max(log_mel)),
            "has_nan": bool(np.isnan(audio_fixed).any() or np.isnan(log_mel).any()),
            "has_inf": bool(np.isinf(audio_fixed).any() or np.isinf(log_mel).any()),
        }
        return raw_audio, audio_fixed, log_mel, meta

    def process_icbhi_cycle(self, raw_audio: np.ndarray, orig_sr: int, start_sec: float, end_sec: float):
        """
        Extract a single annotated respiratory cycle from an ICBHI recording
        and run it through the standardized preprocessing pipeline.
        """
        audio_dc = self.remove_dc_offset(raw_audio)
        audio_hp = self.apply_highpass_filter(audio_dc, orig_sr)
        audio_16k = self.resample_to_target(audio_hp, orig_sr)

        start_idx = max(0, int(round(start_sec * self.target_sr)))
        end_idx = min(len(audio_16k), int(round(end_sec * self.target_sr)))
        cycle_audio = audio_16k[start_idx:end_idx]

        orig_cycle_dur = (end_idx - start_idx) / float(self.target_sr)

        cycle_norm = self.normalize_peak(cycle_audio)
        cycle_fixed = self.pad_or_window_fixed_length(cycle_norm)
        log_mel = self.compute_log_mel(cycle_fixed)

        meta = {
            "orig_sr": orig_sr,
            "orig_dur": orig_cycle_dur,
            "target_sr": self.target_sr,
            "processed_dur": len(cycle_fixed) / float(self.target_sr),
            "waveform_shape": cycle_fixed.shape,
            "mel_shape": log_mel.shape,
            "wav_min": float(np.min(cycle_fixed)),
            "wav_max": float(np.max(cycle_fixed)),
            "mel_min": float(np.min(log_mel)),
            "mel_max": float(np.max(log_mel)),
            "has_nan": bool(np.isnan(cycle_fixed).any() or np.isnan(log_mel).any()),
            "has_inf": bool(np.isinf(cycle_fixed).any() or np.isinf(log_mel).any()),
        }
        return cycle_audio, cycle_fixed, log_mel, meta
