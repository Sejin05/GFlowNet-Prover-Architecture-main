"""Unit and regression tests for the Automated Code-Review Framework (`scripts/code_review.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.runner import main as cli_main
from scripts.code_review import CodeReviewEngine, run_review_cli


@pytest.mark.unit
def test_live_workspace_passes_code_review_zero_errors(tmp_path: Path) -> None:
    engine = CodeReviewEngine()
    report = engine.run_all()

    assert report["status"] == "PASS", f"Unexpected ERROR findings: {report['findings']}"
    assert report["summary"]["ERROR"] == 0

    inv = report["inventory"]
    assert inv["workspace_crates_count"] >= 9
    assert inv["stub_crates_count"] >= 8
    assert inv["rust_unit_tests"] >= 85
    assert inv["rust_integration_tests"] >= 5
    assert inv["solidity_fixture_contracts"] == 4
    assert inv["solidity_benign_tests"] == 12

    # CLI subcommand `python main.py review --output ...`
    out_json = tmp_path / "review.json"
    assert cli_main(["review", "--output", str(out_json)]) == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"


@pytest.mark.unit
def test_code_review_detects_planted_defects_in_synthetic_workspace(tmp_path: Path) -> None:
    # Construct a minimal synthetic workspace containing intentional safety/soundness/security bugs
    crate_src = tmp_path / "crates" / "gfn-demo" / "src"
    crate_src.mkdir(parents=True)
    (tmp_path / "crates" / "gfn-demo" / "Cargo.toml").write_text(
        '[package]\nname = "gfn-demo"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    # Missing #![deny(...)] AND uses .unwrap() in production code
    (crate_src / "lib.rs").write_text(
        "pub fn bad() -> u32 {\n    let x: Option<u32> = Some(1);\n    x.unwrap()\n}\n",
        encoding="utf-8",
    )

    # Planted tautological PoC assertion
    ver_src = tmp_path / "crates" / "gfn-verifier" / "src"
    ver_src.mkdir(parents=True)
    (tmp_path / "crates" / "gfn-verifier" / "Cargo.toml").write_text(
        '[package]\nname = "gfn-verifier"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (ver_src / "lib.rs").write_text(
        "#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]\n",
        encoding="utf-8",
    )
    (ver_src / "poc_synthesizer.rs").write_text(
        'pub fn bad_poc() -> &\'static str { "assertGt(ATTACKER0.balance, 0, \\"bad\\");" }\n',
        encoding="utf-8",
    )
    (ver_src / "forge_gate.rs").write_text(
        "pub struct ConcreteForgeVerifier;\n",
        encoding="utf-8",
    )

    # Planted path-traversal-vulnerable static handler
    cli_src = tmp_path / "crates" / "gfn-cli" / "src"
    cli_src.mkdir(parents=True)
    (tmp_path / "crates" / "gfn-cli" / "Cargo.toml").write_text(
        '[package]\nname = "gfn-cli"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (cli_src / "lib.rs").write_text(
        "#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]\n",
        encoding="utf-8",
    )
    (cli_src / "server.rs").write_text(
        "pub async fn serve_static_asset(file: String) {}\n",
        encoding="utf-8",
    )

    # Planted NaN infinite-loop KS function
    mon_src = tmp_path / "crates" / "gfn-monitor" / "src"
    mon_src.mkdir(parents=True)
    (tmp_path / "crates" / "gfn-monitor" / "Cargo.toml").write_text(
        '[package]\nname = "gfn-monitor"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (mon_src / "lib.rs").write_text(
        "#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]\n",
        encoding="utf-8",
    )
    (mon_src / "drift.rs").write_text(
        "pub fn compute_two_sample_ks(p: &[f64], q: &[f64]) -> f64 { 0.0 }\n",
        encoding="utf-8",
    )

    # Planted ASCII fake checkpoint & timeout-free subprocess call
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "broken.safetensors").write_bytes(b"ASCII_FAKE")

    py_cli = tmp_path / "cli"
    py_cli.mkdir()
    (py_cli / "audit.py").write_text(
        "import subprocess\nsubprocess.run(['echo', 'hi'])\n",
        encoding="utf-8",
    )

    report = CodeReviewEngine(tmp_path).run_all()
    assert report["status"] == "FAIL"
    codes = {f["code"] for f in report["findings"] if f["severity"] == "ERROR"}
    assert {"RS001", "RS002", "ST004", "SEC001", "SEC002", "SEC003", "SEC004", "SND001", "SND003"}.issubset(codes)

    assert run_review_cli(["--root", str(tmp_path), "--json"]) == 1
