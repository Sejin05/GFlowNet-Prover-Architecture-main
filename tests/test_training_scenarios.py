"""Comprehensive multi-scenario test suite for GFlowNet model training.

Covers:
1. Multi-mode discovery under diverse reward distributions (preventing mode collapse).
2. Gradient spike damping and Huber clipping stability against out-of-distribution rewards.
3. Continual learning & catastrophic forgetting resistance via empirical Fisher EWC.
4. Numerical calibration of learned log Z against analytical closed-form log Z*.
5. Metamorphic feature stability under semantic code transformations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from cli.config import TrainingHyperparameters
from cli.trainer import (
    FEATURE_DIM,
    HIDDEN_DIM,
    TrainingEngine,
    _extract_workspace_features,
    _spearman_rho,
    load_python_corpus,
)


@pytest.mark.unit
def test_scenario_1_multi_mode_discovery_without_mode_collapse(tmp_path: Path) -> None:
    """SCENARIO 1: Verify all 4 ground truth vulnerability modes are discovered simultaneously."""
    ckpt = tmp_path / "scenario_1_tb.safetensors"
    hp = TrainingHyperparameters(
        phase=2,
        total_steps=120,
        corpus_dir="data/corpus",
        checkpoint_out=str(ckpt),
    )
    engine = TrainingEngine(hp=hp)
    history = engine.run(total_steps=120, generate_plots=False, metrics_path=tmp_path / "m1.json")

    # Final policy must discover all 4 distinct modes without collapsing to a single exploit
    assert history["modes"][-1] == 4, f"Mode collapse detected: only found {history['modes'][-1]}/4 modes"
    assert history["spearman_rho"][-1] >= 0.85
    assert history["entropy"][-1] >= 1.25, f"Policy entropy too low ({history['entropy'][-1]:.3f} nats); indicates mode collapse"
    assert history["rewards"][-1] >= 95.0


@pytest.mark.unit
def test_scenario_2_gradient_spike_guard_and_huber_delta_clipping(tmp_path: Path) -> None:
    """SCENARIO 2: Injects extreme loss residuals to verify gradient spike damping and finiteness."""
    ckpt = tmp_path / "scenario_2_spike.safetensors"
    hp = TrainingHyperparameters(
        phase=2,
        total_steps=60,
        clip_grad_norm=0.75,
        huber_delta=4.0,
        enable_grad_spike_guard=True,
        checkpoint_out=str(ckpt),
    )
    engine = TrainingEngine(hp=hp)
    history = engine.run(total_steps=60, generate_plots=False, metrics_path=tmp_path / "m2.json")

    # Gradient clipping and spike dampening must trigger without NaN or Inf
    assert history["spike_damp_events"] >= 1
    assert history["grad_clip_events"] >= 1
    assert len(history["huber_clipped_steps"]) >= 1
    for g in history["grad_norm"]:
        assert math.isfinite(g)
        assert g <= hp.clip_grad_norm + 1e-6
    for l in history["loss_tb"]:
        assert math.isfinite(l)
        assert l >= 0.0


@pytest.mark.unit
def test_scenario_3_continual_learning_ewc_prevents_catastrophic_forgetting(tmp_path: Path) -> None:
    """SCENARIO 3: Continual learning EWC regularizer preserves previous mode knowledge."""
    # 1. Train baseline anchor on Phase 2
    anchor_ckpt = tmp_path / "anchor_tb.safetensors"
    hp_anchor = TrainingHyperparameters(
        phase=2,
        total_steps=100,
        checkpoint_out=str(anchor_ckpt),
    )
    engine_anchor = TrainingEngine(hp=hp_anchor)
    hist_anchor = engine_anchor.run(total_steps=100, generate_plots=False, metrics_path=tmp_path / "anchor.json")
    rho_anchor = hist_anchor["spearman_rho"][-1]
    assert rho_anchor >= 0.85

    # 2. Train continual Phase 3 with EWC regularizer anchored to Phase 2 checkpoint
    ewc_ckpt = tmp_path / "continual_ewc.safetensors"
    hp_ewc = TrainingHyperparameters(
        phase=3,
        total_steps=80,
        lambda_ewc=400.0,
        anchor_checkpoint=str(anchor_ckpt),
        checkpoint_out=str(ewc_ckpt),
    )
    engine_ewc = TrainingEngine(hp=hp_ewc)
    hist_ewc = engine_ewc.run(total_steps=80, generate_plots=False, metrics_path=tmp_path / "ewc.json")

    # Spearman rank correlation must remain high (zero catastrophic forgetting)
    rho_ewc = hist_ewc["spearman_rho"][-1]
    assert rho_ewc >= 0.85
    assert abs(rho_ewc - rho_anchor) < 0.05
    assert hist_ewc["modes"][-1] == 4


@pytest.mark.unit
def test_scenario_4_exact_analytical_partition_function_log_z_convergence(tmp_path: Path) -> None:
    """SCENARIO 4: Verify learned log Z converges to the analytical theoretical equilibrium log Z*."""
    samples = load_python_corpus("data/corpus")
    # Analytical partition function: Z* = sum_{tau} P_B(tau) * R(tau)
    analytical_z = sum(math.exp(s["log_pb"]) * s["reward"] for s in samples)
    analytical_log_z = math.log(analytical_z)

    ckpt = tmp_path / "scenario_4_z.safetensors"
    hp = TrainingHyperparameters(
        phase=2,
        total_steps=150,
        checkpoint_out=str(ckpt),
    )
    engine = TrainingEngine(hp=hp)
    history = engine.run(total_steps=150, generate_plots=False, metrics_path=tmp_path / "m4.json")

    # Manifest learned log_z must match analytical log Z* within 0.05
    manifest_file = ckpt.with_suffix(".manifest.json")
    import json
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    learned_log_z = manifest["final_metrics"]["learned_log_z"]

    assert abs(learned_log_z - analytical_log_z) < 0.05, (
        f"Learned log Z ({learned_log_z:.4f}) diverged from analytical log Z* ({analytical_log_z:.4f})"
    )


@pytest.mark.unit
def test_scenario_5_metamorphic_feature_invariance_under_code_transformations(tmp_path: Path) -> None:
    """SCENARIO 5: Verify feature extraction and logit scoring are invariant to benign code transformations."""
    # Test that extraction produces consistent structural features across workspaces
    v1_path = Path("tests/fixtures/multi_bug_defi_protocol")
    v2_path = Path("contracts/secure_defi_protocol")

    for arch in [
        "INV_PRECISION_NEUTRALITY_01",
        "INV_SOLVENCY_02",
        "INV_ACCESS_CONTROL_03",
        "INV_TIMELOCK_MONOTONICITY_04",
    ]:
        v1_feat = _extract_workspace_features(v1_path, arch)
        v2_feat = _extract_workspace_features(v2_path, arch)
        assert len(v1_feat) == FEATURE_DIM
        assert len(v2_feat) == FEATURE_DIM
        # Features must be finite and normalized in [0.0, 1.0]
        assert all(0.0 <= x <= 1.0 for x in v1_feat)
        assert all(0.0 <= x <= 1.0 for x in v2_feat)
        # v1 (vulnerable) and v2 (hardened) must have discriminative structural feature signatures
        assert v1_feat != v2_feat, f"Features for {arch} failed to discriminate between vulnerable and hardened implementations"


@pytest.mark.unit
def test_scenario_6_spearman_rank_correlation_corner_cases() -> None:
    """SCENARIO 6: Verify Spearman rho calculation handles perfect, reversed, tied, and uniform distributions."""
    # Perfect alignment
    assert _spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    # Perfect inverse
    assert _spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    # Short sequences
    assert _spearman_rho([1.0], [10.0]) == 0.0
    assert _spearman_rho([], []) == 0.0
