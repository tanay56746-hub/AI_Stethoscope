# ==============================================================================
# [LAYER: ML MODEL ARCHITECTURE] | File: src/models/panns_cnn14.py
# Purpose: Frozen PANNs CNN14 AudioSet Pretrained Backbone (2048-D Embedding)
# ==============================================================================
"""
AI Stethoscope - PANNs CNN14 Foundation Backbone Architecture
=============================================================
Self-contained PyTorch implementation of STFT, LogmelFilterBank, and the
official AudioSet Cnn14 architecture for 2048-dimensional feature extraction.
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


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
