"""Automated Static & Semantic Code-Review Framework for GFlowNet-Prover.

Executes 7 deterministic codebase analyzers in <150ms with zero external dependencies:
  1. Rust Safety & Lint Policy Auditor (`#![deny(...)]`, non-test `.unwrap()`/`.expect()`, `unsafe` audit)
  2. Stub, Patch & SafeTensors Checkpoint Integrity Detector
  3. Portability & Build Hygiene Scanner (`Dockerfile`, `Makefile`, Cargo toolchain configs)
  4. Security & Subprocess Surface Scanner (HTTP path traversal, sandbox isolation, timeouts)
  5. Verifier, EVM & Numerical Soundness Checker (tautological PoC oracle & NaN domain guards)
  6. Configuration Schema Cross-Validator (`configs/*.toml` roundtrip & parser coverage)
  7. Live Architecture & Test Inventory Mapper (prevents documentation/metric drift)

Usage:
  python scripts/code_review.py
  python scripts/code_review.py --json --output review_report.json
  python main.py review
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# Ensure project root is importable when invoked directly as `python scripts/code_review.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config import ConfigManager, TrainingHyperparameters
from cli.trainer import inspect_safetensors_header


REQUIRED_RUST_LINTS = (
    "clippy::unwrap_used",
    "clippy::expect_used",
    "unsafe_op_in_unsafe_fn",
)


@dataclass
class ReviewFinding:
    analyzer: str
    severity: str  # "ERROR" | "WARN" | "INFO"
    file: str
    line: int
    code: str
    message: str


def _strip_rust_comments_and_strings(line: str) -> str:
    """Remove `//` line comments and double-quoted string literals for lightweight token scanning."""
    out: List[str] = []
    in_str = False
    escape = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        if ch == '"':
            in_str = True
            out.append('""')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class CodeReviewEngine:
    """Deterministic multi-analyzer static & semantic code reviewer."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or PROJECT_ROOT).resolve()
        self.findings: List[ReviewFinding] = []
        self.inventory: Dict[str, Any] = {}

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def _add(
        self,
        analyzer: str,
        severity: str,
        path: Path,
        line: int,
        code: str,
        message: str,
    ) -> None:
        self.findings.append(
            ReviewFinding(
                analyzer=analyzer,
                severity=severity,
                file=self._rel(path),
                line=line,
                code=code,
                message=message,
            )
        )

    # ------------------------------------------------------------------
    # 1. Rust Safety & Lint Policy Auditor
    # ------------------------------------------------------------------
    def analyze_rust_safety_and_lints(self) -> None:
        analyzer = "rust_safety"
        crates_dir = self.root / "crates"
        tests_dir = self.root / "tests"

        entrypoints: List[Path] = []
        if crates_dir.exists():
            for crate in sorted(crates_dir.glob("gfn-*")):
                for entry_name in ("lib.rs", "main.rs"):
                    candidate = crate / "src" / entry_name
                    if candidate.exists():
                        entrypoints.append(candidate)
        if tests_dir.exists():
            entrypoints.extend(sorted(tests_dir.glob("*.rs")))

        for ep in entrypoints:
            head = "\n".join(ep.read_text(encoding="utf-8", errors="replace").splitlines()[:8])
            missing = [lint for lint in REQUIRED_RUST_LINTS if lint not in head]
            if missing:
                self._add(
                    analyzer,
                    "ERROR",
                    ep,
                    1,
                    "RS001",
                    f"Missing required `#![deny(...)]` lints: {', '.join(missing)}",
                )

        # Scan all non-stub workspace Rust files for non-test `.unwrap(` / `.expect(` and `unsafe` blocks
        unsafe_workspace_count = 0
        if crates_dir.exists():
            for rs_file in sorted(crates_dir.glob("gfn-*/src/**/*.rs")):
                lines = rs_file.read_text(encoding="utf-8", errors="replace").splitlines()
                in_test_mod = False
                for idx, raw_line in enumerate(lines, start=1):
                    stripped = raw_line.strip()
                    if stripped.startswith("#[cfg(test)]"):
                        in_test_mod = True
                    cleaned = _strip_rust_comments_and_strings(raw_line)
                    if re.search(r"\bunsafe\s*\{|\bunsafe\s+fn\b|\bunsafe\s+impl\b", cleaned):
                        unsafe_workspace_count += 1
                    if not in_test_mod:
                        if ".unwrap(" in cleaned:
                            self._add(
                                analyzer,
                                "ERROR",
                                rs_file,
                                idx,
                                "RS002",
                                "Forbidden `.unwrap()` in production Rust module",
                            )
                        if ".expect(" in cleaned:
                            self._add(
                                analyzer,
                                "ERROR",
                                rs_file,
                                idx,
                                "RS003",
                                "Forbidden `.expect()` in production Rust module",
                            )

        if unsafe_workspace_count > 0:
            self._add(
                analyzer,
                "INFO",
                self.root / "Cargo.toml",
                1,
                "RS004",
                f"Audited {unsafe_workspace_count} `unsafe` site(s) in workspace crates under `unsafe_op_in_unsafe_fn`.",
            )

    # ------------------------------------------------------------------
    # 2. Stub, Patch & Checkpoint Integrity Detector
    # ------------------------------------------------------------------
    def analyze_stubs_deps_and_checkpoints(self) -> None:
        analyzer = "stubs_and_artifacts"
        cargo_toml = self.root / "Cargo.toml"
        if cargo_toml.exists():
            data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
            patches = data.get("patch", {}).get("crates-io", {})
            if patches:
                patched_names = ", ".join(sorted(patches.keys()))
                self._add(
                    analyzer,
                    "WARN",
                    cargo_toml,
                    97,
                    "ST001",
                    f"Workspace uses {len(patches)} `[patch.crates-io]` local stubs ({patched_names}); "
                    "gate behind Cargo features (`stub-torch` vs `cuda-torch`) for real GPU training.",
                )

        # Detect declared dependencies never referenced in crate source
        crates_dir = self.root / "crates"
        if crates_dir.exists():
            for manifest_path in sorted(crates_dir.glob("gfn-*/Cargo.toml")):
                manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
                deps = list(manifest.get("dependencies", {}).keys())
                src_files = list((manifest_path.parent / "src").rglob("*.rs"))
                combined_src = "\n".join(
                    p.read_text(encoding="utf-8", errors="replace") for p in src_files
                )
                unused: List[str] = []
                for dep in deps:
                    rust_ident = dep.replace("-", "_")
                    if rust_ident in ("thiserror", "serde"):
                        continue
                    if rust_ident not in combined_src:
                        unused.append(dep)
                if unused:
                    self._add(
                        analyzer,
                        "INFO",
                        manifest_path,
                        9,
                        "ST002",
                        f"Declared dependencies not yet referenced in `src/`: {', '.join(sorted(unused))}",
                    )

        # Validate every `.safetensors` checkpoint on disk
        ckpt_dir = self.root / "checkpoints"
        if ckpt_dir.exists():
            for sf_path in sorted(ckpt_dir.glob("*.safetensors")):
                try:
                    header = inspect_safetensors_header(sf_path)
                    tensor_keys = [k for k in header.keys() if k != "__metadata__"]
                    if not tensor_keys:
                        self._add(
                            analyzer,
                            "ERROR",
                            sf_path,
                            1,
                            "ST003",
                            "SafeTensors checkpoint contains zero tensor entries",
                        )
                except Exception as exc:
                    self._add(
                        analyzer,
                        "ERROR",
                        sf_path,
                        1,
                        "ST004",
                        f"Invalid `.safetensors` binary format: {exc}",
                    )

    # ------------------------------------------------------------------
    # 3. Portability & Build Hygiene Scanner
    # ------------------------------------------------------------------
    def analyze_portability_and_build(self) -> None:
        analyzer = "portability"
        user_path_re = re.compile(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s\"']+", re.IGNORECASE)

        for rel_candidate in (
            ".cargo/config.toml",
            "rust-toolchain.toml",
            "crates/stubs/generate_torch_sys.ps1",
        ):
            p = self.root / rel_candidate
            if not p.exists():
                continue
            for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if user_path_re.search(line):
                    self._add(
                        analyzer,
                        "WARN",
                        p,
                        idx,
                        "PT001",
                        "Contains machine-specific absolute user path; parameterize via env var for CI/Linux portability.",
                    )
                    break

        dockerfile = self.root / "Dockerfile"
        if dockerfile.exists():
            df_text = dockerfile.read_text(encoding="utf-8", errors="replace")
            if "COPY tests/" not in df_text:
                self._add(
                    analyzer,
                    "ERROR",
                    dockerfile,
                    28,
                    "PT002",
                    "Dockerfile omits `COPY tests/ ./tests/` required by `crates/gfn-cli/Cargo.toml` `[[test]]` targets.",
                )

        makefile = self.root / "Makefile"
        if makefile.exists():
            mk_text = makefile.read_text(encoding="utf-8", errors="replace")
            if "--lib --bins --tests" not in mk_text:
                self._add(
                    analyzer,
                    "ERROR",
                    makefile,
                    11,
                    "PT003",
                    "Makefile `cargo test` omits `--lib --bins --tests`, triggering slow empty `Doc-tests` builds.",
                )

    # ------------------------------------------------------------------
    # 4. Security & Subprocess Surface Scanner
    # ------------------------------------------------------------------
    def analyze_security_and_subprocess_hygiene(self) -> None:
        analyzer = "security"
        server_rs = self.root / "crates" / "gfn-cli" / "src" / "server.rs"
        if server_rs.exists():
            srv_text = server_rs.read_text(encoding="utf-8", errors="replace")
            if 'file.contains("..")' not in srv_text or "file.contains('\\\\')" not in srv_text:
                self._add(
                    analyzer,
                    "ERROR",
                    server_rs,
                    710,
                    "SEC001",
                    "`serve_static_asset` lacks explicit `..` and Windows backslash `\\` path-traversal guards.",
                )

        forge_gate_rs = self.root / "crates" / "gfn-verifier" / "src" / "forge_gate.rs"
        if forge_gate_rs.exists():
            fg_text = forge_gate_rs.read_text(encoding="utf-8", errors="replace")
            if "SandboxWorkspace" not in fg_text:
                self._add(
                    analyzer,
                    "ERROR",
                    forge_gate_rs,
                    30,
                    "SEC002",
                    "`ConcreteForgeVerifier` mutates live `project_root` instead of using an isolated sandbox directory.",
                )
            if "tokio::time::timeout" not in fg_text:
                self._add(
                    analyzer,
                    "ERROR",
                    forge_gate_rs,
                    73,
                    "SEC003",
                    "`ConcreteForgeVerifier` spawns `forge` subprocesses without `tokio::time::timeout`.",
                )

        # Check Python subprocess.run calls for explicit timeout
        cli_dir = self.root / "cli"
        if cli_dir.exists():
            for py_file in sorted(cli_dir.glob("*.py")):
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if (
                            isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "subprocess"
                            and node.func.attr == "run"
                        ):
                            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                            if "timeout" not in kw_names:
                                self._add(
                                    analyzer,
                                    "ERROR",
                                    py_file,
                                    node.lineno,
                                    "SEC004",
                                    "`subprocess.run(...)` invoked without explicit `timeout=` bound.",
                                )

    # ------------------------------------------------------------------
    # 5. Verifier, EVM & Numerical Soundness Checker
    # ------------------------------------------------------------------
    def analyze_verifier_and_numerical_soundness(self) -> None:
        analyzer = "soundness"
        poc_rs = self.root / "crates" / "gfn-verifier" / "src" / "poc_synthesizer.rs"
        if poc_rs.exists():
            poc_src = poc_rs.read_text(encoding="utf-8", errors="replace")
            non_test_poc = poc_src.split("#[cfg(test)]")[0]
            if "assertGt(ATTACKER0.balance, 0" in non_test_poc:
                self._add(
                    analyzer,
                    "ERROR",
                    poc_rs,
                    227,
                    "SND001",
                    "Tautological PoC oracle `assertGt(ATTACKER0.balance, 0)` after `vm.deal(ATTACKER0, 100 ether)`!",
                )

        audit_py = self.root / "cli" / "audit.py"
        if audit_py.exists():
            audit_src = audit_py.read_text(encoding="utf-8", errors="replace")
            if "assertTrue(true)" in audit_src:
                self._add(
                    analyzer,
                    "ERROR",
                    audit_py,
                    117,
                    "SND002",
                    "Tautological `assertTrue(true)` in generated fallback PoC.",
                )

        drift_rs = self.root / "crates" / "gfn-monitor" / "src" / "drift.rs"
        if drift_rs.exists():
            drift_src = drift_rs.read_text(encoding="utf-8", errors="replace")
            ks_idx = drift_src.find("pub fn compute_two_sample_ks")
            if ks_idx != -1:
                ks_body = drift_src[ks_idx : ks_idx + 700]
                if "is_finite()" not in ks_body:
                    self._add(
                        analyzer,
                        "ERROR",
                        drift_rs,
                        221,
                        "SND003",
                        "`compute_two_sample_ks` lacks `is_finite()` guard and hangs infinitely on `NaN` inputs.",
                    )

        fixture_toml = self.root / "tests" / "fixtures" / "multi_bug_defi_protocol" / "foundry.toml"
        if fixture_toml.exists():
            ft_text = fixture_toml.read_text(encoding="utf-8", errors="replace")
            if "storageLayout" not in ft_text:
                self._add(
                    analyzer,
                    "ERROR",
                    fixture_toml,
                    1,
                    "SND004",
                    "Fixture `foundry.toml` omits `extra_output = [\"storageLayout\"]`, causing dummy 1-slot extraction.",
                )

    # ------------------------------------------------------------------
    # 6. Configuration Schema Cross-Validator
    # ------------------------------------------------------------------
    def analyze_config_schemas(self) -> None:
        analyzer = "config_schema"
        configs_dir = self.root / "configs"
        if not configs_dir.exists():
            return

        for toml_path in sorted(configs_dir.glob("*.toml")):
            try:
                raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
                if "project_source" in raw or "search_budget" in raw:
                    ConfigManager.load_audit_config(toml_path)
                else:
                    hp = ConfigManager.load_training_config(toml_path)
                    dumped = hp.to_toml_dict()
                    assert isinstance(dumped, dict) and "optim" in dumped
            except Exception as exc:
                self._add(
                    analyzer,
                    "ERROR",
                    toml_path,
                    1,
                    "CFG001",
                    f"Configuration failed schema validation: {exc}",
                )

    # ------------------------------------------------------------------
    # 7. Live Architecture & Test Inventory Mapper
    # ------------------------------------------------------------------
    def build_architecture_and_test_inventory(self) -> None:
        rust_crates: List[Dict[str, Any]] = []
        crates_dir = self.root / "crates"
        total_rust_loc = 0
        total_rust_unit_tests = 0
        total_rust_pub_items = 0

        pub_re = re.compile(r"^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait|type)\b")
        test_attr_re = re.compile(r"^\s*#\[(?:tokio::)?test\b")

        if crates_dir.exists():
            for crate_dir in sorted(crates_dir.glob("gfn-*")):
                rs_files = sorted((crate_dir / "src").rglob("*.rs"))
                loc = 0
                tests_cnt = 0
                pub_cnt = 0
                for f in rs_files:
                    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                    loc += len(lines)
                    tests_cnt += sum(1 for ln in lines if test_attr_re.search(ln))
                    pub_cnt += sum(1 for ln in lines if pub_re.search(ln))
                total_rust_loc += loc
                total_rust_unit_tests += tests_cnt
                total_rust_pub_items += pub_cnt
                rust_crates.append(
                    {
                        "crate": crate_dir.name,
                        "files": len(rs_files),
                        "loc": loc,
                        "pub_symbols": pub_cnt,
                        "unit_tests": tests_cnt,
                    }
                )

        rust_integration_tests = 0
        tests_dir = self.root / "tests"
        if tests_dir.exists():
            for rs_test in sorted(tests_dir.glob("*.rs")):
                lines = rs_test.read_text(encoding="utf-8", errors="replace").splitlines()
                rust_integration_tests += sum(1 for ln in lines if test_attr_re.search(ln))

        py_files = sorted(list((self.root / "cli").glob("*.py")) + list(self.root.glob("*.py")))
        py_loc = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in py_files if p.is_file())

        py_test_count = 0
        if tests_dir.exists():
            for pt in sorted(tests_dir.glob("test_*.py")):
                tree = ast.parse(pt.read_text(encoding="utf-8"), filename=str(pt))
                py_test_count += sum(
                    1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                )

        sol_test_count = 0
        sol_contracts = 0
        fixture_root = self.root / "tests" / "fixtures" / "multi_bug_defi_protocol"
        if fixture_root.exists():
            for sol in sorted((fixture_root / "src").glob("*.sol")):
                sol_contracts += 1
            for sol_t in sorted((fixture_root / "test").glob("*.sol")):
                txt = sol_t.read_text(encoding="utf-8", errors="replace")
                sol_test_count += len(re.findall(r"\bfunction\s+test_[A-Za-z0-9_]*\s*\(", txt))

        stub_crates = sorted(p.parent.name for p in (crates_dir / "stubs").glob("*/Cargo.toml")) if (crates_dir / "stubs").exists() else []

        self.inventory = {
            "workspace_crates_count": len(rust_crates),
            "stub_crates_count": len(stub_crates),
            "rust_Crates": rust_crates,
            "rust_total_loc": total_rust_loc,
            "rust_pub_symbols": total_rust_pub_items,
            "rust_unit_tests": total_rust_unit_tests,
            "rust_integration_tests": rust_integration_tests,
            "python_cli_loc": py_loc,
            "python_tests": py_test_count,
            "solidity_fixture_contracts": sol_contracts,
            "solidity_benign_tests": sol_test_count,
        }

    def run_all(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.findings.clear()
        self.analyze_rust_safety_and_lints()
        self.analyze_stubs_deps_and_checkpoints()
        self.analyze_portability_and_build()
        self.analyze_security_and_subprocess_hygiene()
        self.analyze_verifier_and_numerical_soundness()
        self.analyze_config_schemas()
        self.build_architecture_and_test_inventory()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        return {
            "status": "PASS" if counts["ERROR"] == 0 else "FAIL",
            "elapsed_ms": round(elapsed_ms, 2),
            "summary": counts,
            "inventory": self.inventory,
            "findings": [asdict(f) for f in self.findings],
        }


def format_human_report(report: Dict[str, Any]) -> str:
    inv = report["inventory"]
    s = report["summary"]
    lines = [
        "",
        "=" * 82,
        f"  GFlowNet-Prover Automated Code-Review Report  [{report['status']}] ({report['elapsed_ms']:.1f} ms)",
        "=" * 82,
        f"  Workspace Crates : {inv['workspace_crates_count']} crates ({inv['rust_total_loc']:,} LOC, {inv['rust_pub_symbols']} pub symbols) + {inv['stub_crates_count']} stubs",
        f"  Test Inventory   : {inv['rust_unit_tests']} Rust unit + {inv['rust_integration_tests']} Rust integration + {inv['python_tests']} Python + {inv['solidity_benign_tests']} Solidity",
        f"  Findings Summary : {s['ERROR']} ERROR | {s['WARN']} WARN | {s['INFO']} INFO",
        "-" * 82,
    ]
    for f in report["findings"]:
        sev = f["severity"]
        loc = f"{f['file']}:{f['line']}"
        lines.append(f"  [{sev:<5}] {f['code']} ({loc}) -> {f['message']}")
    lines.append("=" * 82)
    return "\n".join(lines)


def run_review_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated static & semantic code-review tool")
    parser.add_argument("--root", type=str, default=None, help="Workspace root directory")
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on WARN as well as ERROR")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args(argv)

    engine = CodeReviewEngine(Path(args.root) if args.root else None)
    report = engine.run_all()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_human_report(report))

    if report["summary"]["ERROR"] > 0:
        return 1
    if args.strict and report["summary"]["WARN"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_review_cli())
