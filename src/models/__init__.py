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

__all__ = [
    "AudioCNNBaseline",
    "build_heart_cnn",
    "build_icbhi_cnn",
    "build_kaggle_cnn",
    "FinalMultiTaskStethoscopeModel",
]
