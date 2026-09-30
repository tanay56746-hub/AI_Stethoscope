# ==============================================================================
# [LAYER: ML FEATURE EXTRACTION] | File: src/features/__init__.py
# Purpose: Exports AudioPreprocessor and 397-D Physical Acoustic Feature Extractor
# ==============================================================================
"""
AI Stethoscope - Feature Extraction & Preprocessing Package
===========================================================
Provides standardized audio preprocessing and 397-D physical acoustic feature extraction.
"""
from src.features.audio_features import extract_audio_features, FEATURE_DIM, FEATURE_NAMES
from src.features.preprocessor import AudioPreprocessor, DEFAULT_CONFIG

__all__ = [
    "extract_audio_features",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "AudioPreprocessor",
    "DEFAULT_CONFIG",
]
