"""Comprehensive fast CLI, context bundler, audit, and visualization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.audit import run_audit
from cli.config import ConfigManager, TRAINING_PRESETS, TrainingHyperparameters
from cli.plotter import generate_training_graphs, render_ascii_chart, render_terminal_summary
from cli.runner import build_parser, main
from cli.trainer import TrainingEngine, inspect_safetensors_header


@pytest.mark.unit
def test_config_defaults_and_presets(tmp_path: Path) -> None:
    hp = TrainingHyperparameters()
    assert hp.phase == 2
    assert hp.lr_policy == 3e-4
    assert hp.huber_delta == 10.0

    for preset in ("smoke_test", "production_defi", "deep_exploit", "quick_dev"):
        assert preset in TRAINING_PRESETS

    save_file = tmp_path / "test_cfg.toml"
    ConfigManager.save_training_config(hp, save_file)
    assert save_file.exists()

    loaded = ConfigManager.load_training_config(save_file)
    assert loaded.lr_policy == hp.lr_policy
    assert loaded.huber_delta == hp.huber_delta
    assert loaded.rgat_hidden_dim == hp.rgat_hidden_dim
    assert loaded.replay_capacity == hp.replay_capacity


@pytest.mark.unit
def test_lr_schedule() -> None:
    hp = TrainingHyperparameters(lr_policy=1e-3, lr_log_z=1e-2, lr_warmup_steps=10)
    engine = TrainingEngine(hp=hp)

    lr_w1, lz_w1 = engine.compute_lr_schedule(0, 100)
    lr_w2, lz_w2 = engine.compute_lr_schedule(5, 100)
    assert lr_w1 < lr_w2
    assert lz_w1 < lz_w2

    lr_peak, _ = engine.compute_lr_schedule(10, 100)
    lr_end, _ = engine.compute_lr_schedule(99, 100)
    assert lr_peak == pytest.approx(1e-3)
    assert lr_peak > lr_end
    assert lr_end >= 1e-4


@pytest.mark.unit
def test_training_engine_execution_fast_no_plot(tmp_path: Path) -> None:
    ckpt_file = tmp_path / "test_model.safetensors"
    metrics_file = tmp_path / "test_metrics.json"
    hp = TrainingHyperparameters(
        total_steps=25,
        checkpoint_out=str(ckpt_file),
        lr_warmup_steps=5,
    )
    callbacks = []
    engine = TrainingEngine(hp=hp)
    history = engine.run(
        total_steps=25,
        on_step_callback=lambda s, m: callbacks.append((s, m["loss"])),
        generate_plots=False,
        metrics_path=metrics_file,
    )

    assert len(history["steps"]) == 25
    assert len(callbacks) == 25
    assert ckpt_file.exists()
    assert ckpt_file.with_suffix(".manifest.json").exists()
    assert metrics_file.exists()

    header = inspect_safetensors_header(ckpt_file)
    assert "log_z" in header
    assert "loss_ema_trace" in header


@pytest.mark.integration
def test_plotter_ascii_and_figure(tmp_path: Path) -> None:
    dummy_history = {
        "steps": list(range(1, 31)),
        "loss_tb": [10.0 / (i + 1) for i in range(30)],
        "loss_ema": [9.0 / (i + 1) for i in range(30)],
        "rewards": [1.0 + i * 2.0 for i in range(30)],
        "modes": [min(16, i // 2) for i in range(30)],
        "spearman_rho": [0.3 + 0.02 * i for i in range(30)],
        "entropy": [2.0 - 0.03 * i for i in range(30)],
        "grad_norm": [0.8 - 0.01 * i for i in range(30)],
        "lr_policy": [3e-4] * 30,
        "huber_clipped_steps": [1, 2],
        "huber_clipped_vals": [5.0, 3.3],
        "phase": 2,
    }

    chart = render_ascii_chart(dummy_history["loss_ema"], title="Test Loss")
    assert "Test Loss" in chart
    assert "*" in chart or "o" in chart
    assert render_ascii_chart([], title="Empty") == "Empty: (no data)"

    terminal_summary = render_terminal_summary(dummy_history)
    assert "GFlowNet-Prover Training Progress" in terminal_summary

    fig_path = tmp_path / "dashboard.png"
    out = generate_training_graphs(dummy_history, output_path=fig_path, dpi=40)
    assert out.exists()
    raw = out.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(raw) > 2_000


@pytest.mark.unit
def test_cli_commands_context_plot_errors_and_precedence(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--steps", "10", "--no-plot"])
    assert args.command == "train"
    assert args.steps == 10
    assert args.no_plot is True

    assert main(["benchmark"]) == 0
    assert main(["status"]) == 0

    # Context export in JSON and Markdown against real multi-bug Solidity fixture
    ctx_json = tmp_path / "context.json"
    assert main(["context", "--project", "tests/fixtures/multi_bug_defi_protocol", "--output", str(ctx_json), "--format", "json"]) == 0
    ctx_data = json.loads(ctx_json.read_text(encoding="utf-8"))
    assert ctx_data["format"] == "json"
    assert ctx_data["contract_count"] >= 4
    all_contracts = {c for item in ctx_data["files"] for c in item["contracts"]}
    assert {"VaultErc4626", "AmmPool", "LendingMarket", "GovTimelock"}.issubset(all_contracts)

    ctx_md = tmp_path / "context.md"
    assert main(["context", "--project", "tests/fixtures/multi_bug_defi_protocol", "--output", str(ctx_md), "--format", "markdown"]) == 0
    md_text = ctx_md.read_text(encoding="utf-8")
    assert "VaultErc4626" in md_text
    assert "GovTimelock" in md_text

    # Plot error handling on missing & malformed JSON
    assert main(["plot", "--metrics-file", str(tmp_path / "missing.json")]) == 1
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    assert main(["plot", "--metrics-file", str(bad_json)]) == 1

    # Config + explicit preset precedence test
    cfg_path = tmp_path / "custom.toml"
    custom_hp = TrainingHyperparameters(lr_policy=2e-4, rgat_hidden_dim=256)
    ConfigManager.save_training_config(custom_hp, cfg_path)
    out_ckpt = tmp_path / "out.safetensors"
    out_metrics = tmp_path / "metrics.json"
    assert main([
        "train",
        "--config", str(cfg_path),
        "--preset", "smoke_test",
        "--steps", "8",
        "--checkpoint-out", str(out_ckpt),
        "--metrics-file", str(out_metrics),
        "--no-plot",
    ]) == 0
    manifest = json.loads(out_ckpt.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    # Explicit preset `smoke_test` overrides lr_policy to 5e-4 while preserving config's rgat_hidden_dim=256
    assert manifest["hyperparameters"]["optim"]["lr_policy"] == pytest.approx(5e-4)
    assert manifest["hyperparameters"]["model"]["rgat_hidden_dim"] == 256


@pytest.mark.unit
def test_audit_isolated_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit_out"
    report = run_audit(
        project_path="tests/fixtures/multi_bug_defi_protocol",
        output_dir=out_dir,
        max_trajectories=50,
        fast_mode=True,
    )
    assert (out_dir / "audit_report.json").exists()
    assert isinstance(report, dict)
    findings = report.get("findings", [])
    assert len(findings) >= 3
    for finding in findings:
        target = finding.get("contract_target") or (
            finding.get("trajectory_steps", [{}])[-1].get("target_contract")
        )
        assert target and target != "N/A"


def test_contrastive_corpus_and_phase_convergence(tmp_path: Path) -> None:
    """Verify corpus extraction (v1 vulnerable + v2 hardened) and multi-mode TB convergence."""
    from cli.trainer import load_python_corpus

    samples = load_python_corpus("data/corpus")
    assert len(samples) == 8
    assert sum(1 for s in samples if s["is_vulnerable"]) == 4
    assert sum(1 for s in samples if not s["is_vulnerable"]) == 4

    hp = TrainingHyperparameters(
        phase=2,
        total_steps=80,
        corpus_dir="data/corpus",
        checkpoint_out=str(tmp_path / "contrastive_tb.safetensors"),
    )
    trainer = TrainingEngine(hp)
    history = trainer.run(
        total_steps=80,
        generate_plots=False,
        metrics_path=tmp_path / "contrastive_metrics.json",
    )
    assert history["loss_tb"][-1] < history["loss_tb"][0]
    assert history["modes"][-1] == 4
    assert history["spearman_rho"][-1] >= 0.85


@pytest.mark.slow
def test_audit_hardened_secure_contracts_zero_false_positives(tmp_path: Path) -> None:
    """Ensure auditing `contracts/secure_defi_protocol` yields 0 findings and 0% FPR."""
    out_dir = tmp_path / "audit_secure_v2"
    report = run_audit(
        project_path="contracts/secure_defi_protocol",
        output_dir=out_dir,
        max_trajectories=32,
    )
    assert (out_dir / "audit_report.json").exists()
    assert report.get("verified_findings_count") == 0
    assert report.get("findings") == []
    assert not (out_dir / "ExploitPoC.t.sol").exists()


@pytest.mark.unit
def test_reporter_markdown_and_sarif_generation(tmp_path: Path) -> None:
    from cli.reporter import generate_markdown_advisory, generate_sarif_report, run_report

    sample_report = {
        "report_metadata": {
            "title": "GFlowNet-Prover Audit",
            "version": "1.0.0",
            "timestamp": "2026-11-05T00:00:00Z",
            "evaluation_status": "CERTIFIED_ZERO_ERRORS",
        },
        "target_project": "./tests/fixtures/multi_bug_defi_protocol",
        "verified_findings": [
            {
                "finding_id": "FINDING-TEST-001",
                "severity": "CRITICAL",
                "target_contract": "VaultErc4626",
                "violated_invariant_id": "INV_PRECISION_NEUTRALITY_01",
                "description": "Share inflation attack test",
                "trajectory_steps": [
                    {
                        "step_idx": 0,
                        "caller_role": "Attacker0",
                        "function_signature": "deposit(uint256)",
                        "target_contract": "VaultErc4626",
                        "decoded_args": ["1"],
                    }
                ],
                "foundry_poc_solidity_code": "// PoC Code",
                "remediation_patch_diff": "// Patch Diff",
                "minimal_negated_precondition": "require(assets > 0);",
            }
        ],
    }

    md_out = generate_markdown_advisory(sample_report)
    assert "GFlowNet-Prover Autonomous Security Advisory" in md_out
    assert "VaultErc4626" in md_out
    assert "INV_PRECISION_NEUTRALITY_01" in md_out
    assert "CWE-682" in md_out
    assert "SWC-101" in md_out

    sarif_out = generate_sarif_report(sample_report)
    assert sarif_out["version"] == "2.1.0"
    assert len(sarif_out["runs"][0]["results"]) == 1
    assert sarif_out["runs"][0]["results"][0]["ruleId"] == "INV_PRECISION_NEUTRALITY_01"
    driver_rules = sarif_out["runs"][0]["tool"]["driver"]["rules"]
    assert len(driver_rules) == 1
    assert driver_rules[0]["properties"]["cwe"] == "CWE-682"
    assert driver_rules[0]["properties"]["swc"] == "SWC-101"
    assert "https://swcregistry.io" in driver_rules[0]["helpUri"]
    assert "https://cwe.mitre.org" in driver_rules[0]["help"]["markdown"]

    # Test file export via run_report
    rep_file = tmp_path / "sample_audit.json"
    rep_file.write_text(json.dumps(sample_report), encoding="utf-8")

    md_dest = tmp_path / "SECURITY_ADVISORY.md"
    sarif_dest = tmp_path / "report.sarif"
    assert run_report(rep_file, md_dest, "markdown") == 0
    assert md_dest.exists()
    assert run_report(rep_file, sarif_dest, "sarif") == 0
    assert sarif_dest.exists()


@pytest.mark.unit
def test_cli_report_command_invocation(tmp_path: Path) -> None:
    rep_file = tmp_path / "audit_report.json"
    rep_file.write_text(json.dumps({"verified_findings": []}), encoding="utf-8")
    out_file = tmp_path / "advisory.md"

    exit_code = main(["report", "--input", str(rep_file), "--output", str(out_file), "--format", "markdown"])
    assert exit_code == 0
    assert out_file.exists()


@pytest.mark.unit
def test_cli_auxiliary_commands_invocation(tmp_path: Path) -> None:
    """Verify status, benchmark/eval, review, and context CLI entrypoints execute cleanly."""
    assert main(["status"]) == 0
    assert main(["info"]) == 0
    assert main(["benchmark"]) == 0
    assert main(["benchmark", "--profile"]) == 0
    assert main(["benchmark", "--detailed", "--profile"]) == 0
    assert main(["eval"]) == 0
    assert main(["review"]) == 0

    ctx_out = tmp_path / "exported_context.md"
    assert main([
        "context",
        "--project", "tests/fixtures/multi_bug_defi_protocol",
        "--output", str(ctx_out),
        "--format", "markdown",
    ]) == 0
    assert ctx_out.exists()
    assert len(ctx_out.read_text(encoding="utf-8")) > 0


@pytest.mark.unit
def test_cli_audit_patch_formats_and_config(tmp_path: Path) -> None:
    """Verify audit patch format options ('require' and 'custom-error') and config loading."""
    out_dir_req = tmp_path / "audit_req"
    assert main([
        "audit",
        "--project", "tests/fixtures/multi_bug_defi_protocol",
        "--output-dir", str(out_dir_req),
        "--patch-format", "require",
        "--max-trajectories", "16",
    ]) == 0
    patch_req = (out_dir_req / "remediation.patch").read_text(encoding="utf-8")
    assert "require(" in patch_req

    out_dir_err = tmp_path / "audit_err"
    assert main([
        "audit",
        "--project", "tests/fixtures/multi_bug_defi_protocol",
        "--output-dir", str(out_dir_err),
        "--patch-format", "custom-error",
        "--max-trajectories", "16",
    ]) == 0
    patch_err = (out_dir_err / "remediation.patch").read_text(encoding="utf-8")
    assert "revert " in patch_err
    assert "error " in patch_err

    # Test loading patch_format from TOML configuration
    cfg_file = tmp_path / "custom_audit.toml"
    cfg_file.write_text(
        "[verifier]\n"
        "patch_format = \"custom-error\"\n",
        encoding="utf-8",
    )
    loaded_cfg = ConfigManager.load_audit_config(cfg_file)
    assert loaded_cfg.patch_format == "custom-error"



