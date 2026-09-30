# ==============================================================================
# [LAYER: ML LOSS FUNCTIONS] | File: src/losses/__init__.py
# Purpose: Exports Multi-Task & Class-Balanced Loss Formulations
# ==============================================================================
"""
AI Stethoscope - Losses Package
===============================
Provides masked multi-task loss functions for disjoint auscultation datasets.
"""

from src.losses.multitask_loss import MaskedMultiTaskLoss

__all__ = ["MaskedMultiTaskLoss"]
