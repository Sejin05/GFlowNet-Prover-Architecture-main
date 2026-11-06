"""High-Efficiency Unified Test & Verification Runner for GFlowNet-Prover.

Coordinates:
  1. Static & Semantic Code-Review Gate (`scripts/code_review.py`)
  2. Fast Python Unit, Mathematical Accuracy & CLI Suite (`pytest`)
  3. Rust Workspace Unit & Integration Suites (`cargo test --lib --bins [--tests]`,
     bypassing slow empty `Doc-tests` compilations)
  4. Optional Clippy Zero-Warning Lint Gate (`--clippy`)

Usage:
  python scripts/run_tests.py --fast
  python scripts/run_tests.py --full
  python scripts/run_tests.py --full --clippy --report test_health_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.code_review import CodeReviewEngine


def _run_stage(name: str, cmd: List[str], cwd: Path, timeout_sec: int = 300) -> Dict[str, Any]:
    print(f"\n[*] Stage: {name}")
    print(f"    $ {' '.join(cmd)}")
    if shutil.which(cmd[0]) is None:
        print(f"    -> [SKIP] (executable '{cmd[0]}' not found in PATH)")
        return {
            "name": name,
            "command": cmd,
            "passed": True,
            "exit_code": 0,
            "elapsed_sec": 0.0,
        }
    t0 = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    passed = proc.returncode == 0
    badge = "PASS" if passed else "FAIL"
    print(f"    -> [{badge}] in {elapsed:.2f}s (exit={proc.returncode})")
    if not passed:
        if proc.stdout:
            print(proc.stdout[-3000:])
        if proc.stderr:
            print(proc.stderr[-3000:])
    return {
        "name": name,
        "command": cmd,
        "passed": passed,
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
    }


def run_suite(
    mode: str = "full",
    include_clippy: bool = False,
    include_robustness: bool = False,
    report_path: Optional[Path] = None,
) -> int:
    suite_t0 = time.perf_counter()
    stages: List[Dict[str, Any]] = []

    # 1. Automated Code-Review Gate (in-process, <100ms)
    print("\n[*] Stage: Automated Code-Review Health Gate")
    cr_t0 = time.perf_counter()
    review_report = CodeReviewEngine(PROJECT_ROOT).run_all()
    cr_elapsed = time.perf_counter() - cr_t0
    cr_passed = review_report["summary"]["ERROR"] == 0
    print(
        f"    -> [{'PASS' if cr_passed else 'FAIL'}] in {cr_elapsed:.3f}s "
        f"(ERROR={review_report['summary']['ERROR']}, WARN={review_report['summary']['WARN']}, INFO={review_report['summary']['INFO']})"
    )
    if not cr_passed:
        for f in review_report["findings"]:
            if f["severity"] == "ERROR":
                print(f"       ! {f['code']} {f['file']}:{f['line']} - {f['message']}")
    stages.append(
        {
            "name": "code_review_gate",
            "command": ["python", "scripts/code_review.py"],
            "passed": cr_passed,
            "exit_code": 0 if cr_passed else 1,
            "elapsed_sec": round(cr_elapsed, 3),
        }
    )

    # 2. Python Pytest Suite
    py_cmd = [sys.executable, "-m", "pytest", "-q"]
    if mode == "fast":
        py_cmd.extend(["-m", "not slow"])
    stages.append(_run_stage("python_pytest", py_cmd, PROJECT_ROOT, timeout_sec=120))

    # 3. Foundry Solidity Suites (Vulnerable v1 Fixture + Hardened Secure v2 Suite) in Full Mode
    if mode == "full":
        stages.append(
            _run_stage(
                "forge_fixture_v1",
                ["forge", "test", "--offline"],
                PROJECT_ROOT / "tests" / "fixtures" / "multi_bug_defi_protocol",
                timeout_sec=120,
            )
        )
        stages.append(
            _run_stage(
                "forge_secure_v2",
                ["forge", "test", "--offline"],
                PROJECT_ROOT / "contracts" / "secure_defi_protocol",
                timeout_sec=120,
            )
        )

    # 4. Rust Cargo Test Suite (skipping 9 empty Doc-tests targets that cost ~24s)
    if mode == "fast":
        cargo_cmd = ["cargo", "test", "--workspace", "--lib", "--bins", "--quiet"]
    else:
        cargo_cmd = ["cargo", "test", "--workspace", "--lib", "--bins", "--tests", "--quiet"]
    stages.append(_run_stage(f"rust_cargo_test_{mode}", cargo_cmd, PROJECT_ROOT, timeout_sec=300))

    # 5. Dedicated Property-Based Fuzzing & Metamorphic Robustness Suites (Robustness / Full Mode)
    if include_robustness or mode == "full":
        stages.append(
            _run_stage(
                "rust_fuzz_robustness",
                ["cargo", "test", "--test", "fuzz_robustness_test", "--quiet"],
                PROJECT_ROOT,
                timeout_sec=120,
            )
        )
        stages.append(
            _run_stage(
                "rust_metamorphic_robustness",
                ["cargo", "test", "--test", "metamorphic_robustness_test", "--quiet"],
                PROJECT_ROOT,
                timeout_sec=120,
            )
        )

    # 6. Optional Clippy Gate
    if include_clippy:
        clippy_cmd = ["cargo", "clippy", "--workspace", "--all-targets", "--quiet", "--", "-D", "warnings"]
        stages.append(_run_stage("rust_clippy", clippy_cmd, PROJECT_ROOT, timeout_sec=180))

    total_sec = time.perf_counter() - suite_t0
    all_passed = all(s["passed"] for s in stages)

    summary_payload = {
        "mode": mode,
        "status": "PASS" if all_passed else "FAIL",
        "total_elapsed_sec": round(total_sec, 3),
        "inventory": review_report["inventory"],
        "stages": stages,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  Unified Verification Summary ({mode.upper()}): {summary_payload['status']} in {total_sec:.2f}s")
    print("=" * 72)
    for st in stages:
        print(f"  - {st['name']:<26} : {'PASS' if st['passed'] else 'FAIL':<4} ({st['elapsed_sec']:6.2f}s)")
    print("=" * 72 + "\n")

    return 0 if all_passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unified fast & accurate test runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="Sub-second/fast unit test mode (skips Foundry E2E & slow plots)")
    group.add_argument("--full", action="store_true", help="Full verification mode including concurrent Foundry E2E & metamorphic suites")
    parser.add_argument("--robustness", action="store_true", help="Execute proptest fuzzing & metamorphic verification suites")
    parser.add_argument("--clippy", action="store_true", help="Also run `cargo clippy --workspace --all-targets -- -D warnings`")
    parser.add_argument("--report", type=str, default=None, help="Optional JSON output report path")
    args = parser.parse_args(argv)

    mode = "fast" if args.fast else "full"
    report_path = Path(args.report) if args.report else None
    return run_suite(
        mode=mode,
        include_clippy=args.clippy,
        include_robustness=args.robustness,
        report_path=report_path,
    )


if __name__ == "__main__":
    sys.exit(main())
