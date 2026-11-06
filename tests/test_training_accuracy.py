"""Mathematical accuracy, checkpoint binary integrity, and config schema tests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from cli.config import ConfigManager, TrainingHyperparameters
from cli.trainer import TrainingEngine, inspect_safetensors_header, serialize_safetensors


@pytest.mark.unit
def test_safetensors_binary_roundtrip_and_corruption_rejection(tmp_path: Path) -> None:
    raw = serialize_safetensors(
        tensors={
            "log_z": ([1], [4.60517]),
            "weights": ([2, 3], [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]),
        },
        metadata={"phase": "2", "format": "pt"},
    )
    sf_file = tmp_path / "valid.safetensors"
    sf_file.write_bytes(raw)

    header = inspect_safetensors_header(sf_file)
    assert header["__metadata__"]["phase"] == "2"
    assert header["log_z"]["dtype"] == "F32"
    assert header["weights"]["shape"] == [2, 3]

    # Unpack raw float32 payload and verify exact IEEE-754 values
    (header_len,) = struct.unpack("<Q", raw[:8])
    assert header_len % 8 == 0
    payload = raw[8 + header_len :]
    w_start, w_end = header["weights"]["data_offsets"]
    unpacked = struct.unpack("<6f", payload[w_start:w_end])
    assert list(unpacked) == pytest.approx([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], rel=1e-6)

    # Raw ASCII fake checkpoint must be rejected!
    ascii_fake = tmp_path / "fake.safetensors"
    ascii_fake.write_bytes(b"GFN_CHECKPOINT_PHASE_2_STEPS_400")
    with pytest.raises(ValueError):
        inspect_safetensors_header(ascii_fake)

    # Shape mismatch must raise ValueError on serialization
    with pytest.raises(ValueError):
        serialize_safetensors({"bad": ([2, 2], [1.0, 2.0])})


@pytest.mark.unit
@pytest.mark.parametrize("phase", [1, 2, 3])
def test_all_three_training_phases_ema_and_clipping_invariants(tmp_path: Path, phase: int) -> None:
    ckpt = tmp_path / f"phase_{phase}.safetensors"
    metrics = tmp_path / f"metrics_{phase}.json"
    hp = TrainingHyperparameters(
        phase=phase,
        total_steps=90,
        lr_warmup_steps=15,
        clip_grad_norm=0.85,
        huber_delta=6.0,
        ema_decay=0.90,
        checkpoint_out=str(ckpt),
    )
    engine = TrainingEngine(hp=hp)
    hist = engine.run(total_steps=90, generate_plots=False, metrics_path=metrics)

    # Verify exact EMA recurrence: ema[t] = beta * ema[t-1] + (1 - beta) * loss[t]
    losses = hist["loss_tb"]
    emas = hist["loss_ema"]
    assert emas[0] == pytest.approx(losses[0], rel=1e-9)
    for idx in range(1, len(losses)):
        expected_ema = 0.90 * emas[idx - 1] + 0.10 * losses[idx]
        assert emas[idx] == pytest.approx(expected_ema, rel=1e-9)

    # Verify all metrics are strictly finite, non-negative, and Phase 2 grad clipping obeys ceiling
    assert all(math.isfinite(v) and v >= 0.0 for v in losses)
    if phase in (1, 2):
        assert losses[-1] < losses[0]
    assert all(math.isfinite(r) for r in hist["rewards"])
    assert all(math.isfinite(s) and -1.0 <= s <= 1.0 for s in hist["spearman_rho"])
    if phase == 2:
        assert all(g <= hp.clip_grad_norm + 1e-9 for g in hist["grad_norm"])
        assert hist["spike_damp_events"] >= 1
        assert len(hist["huber_clipped_steps"]) >= 1

    # Verify manifest SHA-256 matches binary SafeTensors file bytes
    manifest = json.loads(ckpt.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert manifest["sha256_hash"] == actual_sha


@pytest.mark.unit
def test_real_repository_toml_configs_parse_accurately(tmp_path: Path) -> None:
    # Phase 1 pretrain_mstm.toml uses [training], [model] hidden_dim, [optim] lr, [loss] huber_delta
    hp_mstm = ConfigManager.load_training_config("configs/pretrain_mstm.toml")
    assert hp_mstm.phase == 1
    assert hp_mstm.epochs == 20
    assert hp_mstm.mask_ratio == pytest.approx(0.30)
    assert hp_mstm.lr_policy == pytest.approx(3e-4)
    assert hp_mstm.rgat_hidden_dim == 512
    assert hp_mstm.huber_delta == pytest.approx(1.0)

    # Phase 2 curriculum_tb.toml
    hp_tb = ConfigManager.load_training_config("configs/curriculum_tb.toml")
    assert hp_tb.phase == 2
    assert hp_tb.lr_policy == pytest.approx(3e-4)
    assert hp_tb.lr_log_z == pytest.approx(5e-3)
    assert hp_tb.replay_capacity == 100_000
    assert hp_tb.boundary_trigger == pytest.approx(0.85)
    assert hp_tb.worker_threads == 8

    # Audit config default_audit.toml
    audit_cfg = ConfigManager.load_audit_config("configs/default_audit.toml")
    assert audit_cfg.foundry_profile.evm_version == "cancun"
    assert audit_cfg.search_budget.max_depth >= 1

    # Corrupt TOML must raise ValueError instead of failing silently
    corrupt = tmp_path / "corrupt.toml"
    corrupt.write_text("[optim\nlr_policy = ???\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ConfigManager.load_training_config(corrupt)
