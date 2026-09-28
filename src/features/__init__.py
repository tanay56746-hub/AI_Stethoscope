"""
AI Stethoscope - Feature Extraction Package
==========================================
Provides reusable acoustic feature engineering tools for classical ML baselines.
"""
from src.features.audio_features import extract_audio_features, FEATURE_DIM, FEATURE_NAMES

__all__ = ["extract_audio_features", "FEATURE_DIM", "FEATURE_NAMES"]
