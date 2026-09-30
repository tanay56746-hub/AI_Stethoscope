#!/usr/bin/env python3
# ==============================================================================
# [LAYER: RESEARCH SCRIPT — MODEL TEST] | File: scripts/test_final_multitask_model.py
# Purpose: Verifies FinalMultiTaskStethoscopeModel Forward Pass & Parameter Count
# ==============================================================================
"""
AI Stethoscope - Final Multi-Task Architecture Smoke Tests
==========================================================
Executes rigorous, lightweight smoke tests for FinalMultiTaskStethoscopeModel,
MaskedMultiTaskLoss, and AuscultationMultiTaskAdapter.

Tests:
  A. Heart-only sample forward + backward pass
  B. ICBHI-only sample forward + backward pass
  C. Kaggle-only sample forward + backward pass
  D. Mixed batch containing non-overlapping task availability masks
  E. Gradient isolation: verify inactive heads receive zero gradient when mask=0
  F. PANNs dimension test: verify (B, 2048) -> (B, 128) and loud failure on mismatch
  G. Architectural parameter count audit by major branch
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.final_multitask_model import FinalMultiTaskStethoscopeModel
from src.losses.multitask_loss import MaskedMultiTaskLoss
from src.models.dataset_adapter import AuscultationMultiTaskAdapter


def test_heart_only_batch():
    print("\n[TEST A] Heart-Only Batch Smoke Test...")
    model = FinalMultiTaskStethoscopeModel()
    criterion = MaskedMultiTaskLoss()
    B = 4

    mel = torch.randn(B, 1, 128, 313)
    acoustic_features = torch.randn(B, 397)
    heart_label = torch.randint(0, 2, (B,))

    # Format batch
    samples = [
        AuscultationMultiTaskAdapter.format_heart_sample(
            mel=mel[i],
            heart_label=int(heart_label[i].item()),
            acoustic_features=acoustic_features[i],
        )
        for i in range(B)
    ]
    batch = AuscultationMultiTaskAdapter.collate_multitask_batch(samples)

    # Forward
    preds = model(
        mel=batch["inputs"]["mel"],
        acoustic_features=batch["inputs"]["acoustic_features"],
    )

    # Check output shapes
    assert preds["heart_logits"].shape == (B, 2), f"Wrong heart shape: {preds['heart_logits'].shape}"
    assert preds["acoustic_logits"].shape == (B, 4), f"Wrong acoustic shape: {preds['acoustic_logits'].shape}"
    assert preds["disease_logits"].shape == (B, 3), f"Wrong disease shape: {preds['disease_logits'].shape}"
    assert not torch.isnan(preds["heart_logits"]).any(), "NaN in heart logits!"

    # Loss
    loss_dict = criterion(preds, batch["targets"], batch["masks"])
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss), "Heart total loss is not finite!"
    assert loss_dict["acoustic_loss"].item() == 0.0, "Acoustic loss must be 0 for heart batch!"
    assert loss_dict["disease_loss"].item() == 0.0, "Disease loss must be 0 for heart batch!"

    # Backward
    model.zero_grad()
    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"

    print(f"  * PASS: Heart loss = {total_loss.item():.4f}, shapes verified, gradients finite.")


def test_icbhi_only_batch():
    print("\n[TEST B] ICBHI-Only Batch Smoke Test...")
    model = FinalMultiTaskStethoscopeModel()
    criterion = MaskedMultiTaskLoss()
    B = 4

    mel = torch.randn(B, 1, 128, 313)
    temporal_mask = torch.ones(B, 313)
    temporal_mask[:, 150:] = 0.0  # Simulate 50% padded silence
    panns_emb = torch.randn(B, 2048)
    acoustic_label = torch.randint(0, 4, (B,))

    samples = [
        AuscultationMultiTaskAdapter.format_icbhi_sample(
            mel=mel[i],
            acoustic_label=int(acoustic_label[i].item()),
            temporal_mask=temporal_mask[i],
            panns_embedding=panns_emb[i],
        )
        for i in range(B)
    ]
    batch = AuscultationMultiTaskAdapter.collate_multitask_batch(samples)

    preds = model(
        mel=batch["inputs"]["mel"],
        panns_embedding=batch["inputs"]["panns_embedding"],
        temporal_mask=batch["inputs"]["temporal_mask"],
    )

    assert preds["acoustic_logits"].shape == (B, 4)
    assert not torch.isnan(preds["acoustic_logits"]).any()

    loss_dict = criterion(preds, batch["targets"], batch["masks"])
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss)
    assert loss_dict["heart_loss"].item() == 0.0, "Heart loss must be 0 for ICBHI batch!"
    assert loss_dict["disease_loss"].item() == 0.0, "Disease loss must be 0 for ICBHI batch!"

    model.zero_grad()
    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()

    print(f"  * PASS: ICBHI loss = {total_loss.item():.4f}, MTAP active, gradients finite.")


def test_kaggle_only_batch():
    print("\n[TEST C] Kaggle-Only Batch Smoke Test...")
    model = FinalMultiTaskStethoscopeModel()
    criterion = MaskedMultiTaskLoss()
    B = 4

    mel = torch.randn(B, 1, 128, 313)
    acoustic_features = torch.randn(B, 397)
    disease_label = torch.randint(0, 3, (B,))

    samples = [
        AuscultationMultiTaskAdapter.format_kaggle_sample(
            mel=mel[i],
            disease_label=int(disease_label[i].item()),
            acoustic_features=acoustic_features[i],
        )
        for i in range(B)
    ]
    batch = AuscultationMultiTaskAdapter.collate_multitask_batch(samples)

    preds = model(
        mel=batch["inputs"]["mel"],
        acoustic_features=batch["inputs"]["acoustic_features"],
    )

    assert preds["disease_logits"].shape == (B, 3)
    assert not torch.isnan(preds["disease_logits"]).any()

    loss_dict = criterion(preds, batch["targets"], batch["masks"])
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss)
    assert loss_dict["heart_loss"].item() == 0.0, "Heart loss must be 0 for Kaggle batch!"
    assert loss_dict["acoustic_loss"].item() == 0.0, "Acoustic loss must be 0 for Kaggle batch!"

    model.zero_grad()
    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()

    print(f"  * PASS: Kaggle loss = {total_loss.item():.4f}, shapes verified, gradients finite.")


def test_mixed_task_batch():
    print("\n[TEST D] Mixed-Task Batch Smoke Test...")
    model = FinalMultiTaskStethoscopeModel()
    criterion = MaskedMultiTaskLoss()

    # 2 Heart, 2 ICBHI, 2 Kaggle samples in one batch
    samples = []
    # Heart
    for _ in range(2):
        samples.append(
            AuscultationMultiTaskAdapter.format_heart_sample(
                mel=torch.randn(1, 128, 313),
                heart_label=1,
                acoustic_features=torch.randn(397),
            )
        )
    # ICBHI
    for _ in range(2):
        t_mask = torch.ones(313)
        t_mask[180:] = 0.0
        samples.append(
            AuscultationMultiTaskAdapter.format_icbhi_sample(
                mel=torch.randn(1, 128, 313),
                acoustic_label=2,
                temporal_mask=t_mask,
                panns_embedding=torch.randn(2048),
            )
        )
    # Kaggle
    for _ in range(2):
        samples.append(
            AuscultationMultiTaskAdapter.format_kaggle_sample(
                mel=torch.randn(1, 128, 313),
                disease_label=1,
                acoustic_features=torch.randn(397),
            )
        )

    batch = AuscultationMultiTaskAdapter.collate_multitask_batch(samples)
    B = len(samples)  # 6

    preds = model(
        mel=batch["inputs"]["mel"],
        acoustic_features=batch["inputs"]["acoustic_features"],
        panns_embedding=batch["inputs"]["panns_embedding"],
        temporal_mask=batch["inputs"]["temporal_mask"],
    )

    assert preds["heart_logits"].shape == (6, 2)
    assert preds["acoustic_logits"].shape == (6, 4)
    assert preds["disease_logits"].shape == (6, 3)

    loss_dict = criterion(preds, batch["targets"], batch["masks"])
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss)
    assert loss_dict["num_active_heart"].item() == 2
    assert loss_dict["num_active_acoustic"].item() == 2
    assert loss_dict["num_active_disease"].item() == 2

    model.zero_grad()
    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()

    print(f"  * PASS: Mixed batch loss = {total_loss.item():.4f}, all 3 active losses contributed cleanly.")


def test_gradient_isolation():
    print("\n[TEST E] Gradient Isolation Smoke Test...")
    model = FinalMultiTaskStethoscopeModel()
    criterion = MaskedMultiTaskLoss()
    B = 3

    mel = torch.randn(B, 1, 128, 313)
    acoustic_features = torch.randn(B, 397)

    # Pure Heart batch: heart_mask=1, acoustic_mask=0, disease_mask=0
    preds = model(mel=mel, acoustic_features=acoustic_features)
    targets = {
        "heart_targets": torch.tensor([0, 1, 0]),
        "acoustic_targets": torch.tensor([0, 0, 0]),
        "disease_targets": torch.tensor([0, 0, 0]),
    }
    masks = {
        "heart_mask": torch.tensor([1.0, 1.0, 1.0]),
        "acoustic_mask": torch.tensor([0.0, 0.0, 0.0]),
        "disease_mask": torch.tensor([0.0, 0.0, 0.0]),
    }

    loss_dict = criterion(preds, targets, masks)
    model.zero_grad()
    loss_dict["total_loss"].backward()

    # Verify heart head has gradients
    heart_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heart_head.parameters())
    assert heart_has_grad, "Heart head should have gradients!"

    # Verify acoustic head has ZERO gradient
    acoustic_grad_sum = sum(p.grad.abs().sum().item() for p in model.acoustic_head.parameters() if p.grad is not None)
    assert acoustic_grad_sum == 0.0, f"Acoustic head received gradient when mask=0! Sum={acoustic_grad_sum}"

    # Verify disease head has ZERO gradient
    disease_grad_sum = sum(p.grad.abs().sum().item() for p in model.disease_head.parameters() if p.grad is not None)
    assert disease_grad_sum == 0.0, f"Disease head received gradient when mask=0! Sum={disease_grad_sum}"

    print("  * PASS: Inactive heads (acoustic, disease) received strictly zero gradients when mask=0.")


def test_panns_dimension_enforcement():
    print("\n[TEST F] PANNs Dimension Enforcement Test...")
    model = FinalMultiTaskStethoscopeModel()
    B = 2

    # Correct 2048-d input
    correct_panns = torch.randn(B, 2048)
    proj = model.panns_branch(correct_panns)
    assert proj.shape == (B, 128), f"Expected (B, 128), got {proj.shape}"
    print(f"  * Correct dimension passed: {correct_panns.shape} -> {proj.shape}")

    # Incorrect dimension: test that it fails loudly
    wrong_dims = [128, 512, 1024, 4096]
    for wdim in wrong_dims:
        wrong_panns = torch.randn(B, wdim)
        try:
            model.panns_branch(wrong_panns)
            assert False, f"Expected ValueError for wrong dimension {wdim}, but call succeeded!"
        except ValueError as e:
            print(f"  * Caught expected ValueError loudly for dimension {wdim}: {e}")

    print("  * PASS: PANNs dimension mismatch fails loudly as required.")


def report_parameter_counts():
    print("\n[TEST G] Architectural Parameter Breakdown...")
    model = FinalMultiTaskStethoscopeModel()
    param_counts = model.count_parameters()

    print("=" * 65)
    print(f"{'Branch / Component':35s} | {'Parameters':15s} | {'% of Total':10s}")
    print("=" * 65)
    total = param_counts["total"]
    for k, v in param_counts.items():
        if k in ("total", "total_trainable"):
            continue
        pct = (v / total) * 100.0
        print(f"{k:35s} | {v:15,d} | {pct:8.2f}%")
    print("-" * 65)
    print(f"{'TOTAL TRAINABLE PARAMETERS':35s} | {param_counts['total_trainable']:15,d} | 100.00%")
    print(f"{'TOTAL MODEL PARAMETERS':35s} | {param_counts['total']:15,d} | 100.00%")
    print("=" * 65)
    print("NOTE: Does NOT include the 81.8M PANNs CNN14 backbone, which is frozen")
    print("      and provides precomputed 2048-d embeddings to the final model.")
    return param_counts


def main():
    print("=" * 70)
    print("  STARTING FINAL MULTI-TASK MODEL SMOKE TESTS")
    print("=" * 70)

    test_heart_only_batch()
    test_icbhi_only_batch()
    test_kaggle_only_batch()
    test_mixed_task_batch()
    test_gradient_isolation()
    test_panns_dimension_enforcement()
    param_counts = report_parameter_counts()

    print("\n" + "=" * 70)
    print("ALL FINAL MULTI-TASK MODEL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
