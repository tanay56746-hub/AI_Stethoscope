# ==============================================================================
# [LAYER: ML MODEL ARCHITECTURE] | File: src/models/__init__.py
# Purpose: Exports All Stethoscope Neural Network Architectures (CNN, MultiTask, PANNs)
# ==============================================================================
"""
AI Stethoscope - Model Architectures Package
"""
from src.models.cnn_baseline import (
    AudioCNNBaseline,
    build_heart_cnn,
    build_icbhi_cnn,
    build_kaggle_cnn,
)
from src.models.final_multitask_model import FinalMultiTaskStethoscopeModel
from src.models.panns_cnn14 import Cnn14

__all__ = [
    "AudioCNNBaseline",
    "build_heart_cnn",
    "build_icbhi_cnn",
    "build_kaggle_cnn",
    "FinalMultiTaskStethoscopeModel",
    "Cnn14",
]
