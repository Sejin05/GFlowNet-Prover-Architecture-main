"""CLI Command Runner & Argument Parser for GFlowNet-Prover.

Provides pure-command-line execution in Windows CMD, PowerShell, and terminal:
- `python main.py train`: Model training & automated curve visualization
- `python main.py plot`: Training graph generation from metrics
- `python main.py audit`: Multi-contract vulnerability audit
- `python main.py benchmark`: Evaluation scorecard
- `python main.py status`: Checkpoints and system status
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    import colorama
    colorama.init(autoreset=True)
except Exception:
    pass

from cli.audit import run_audit, run_benchmark_scorecard
from cli.config import ConfigManager, TRAINING_PRESETS, TrainingHyperparameters
from cli.plotter import generate_training_graphs, render_terminal_summary
from cli.reporter import run_report
from cli.trainer import TrainingEngine



BANNER = r"""
=============================================================================
  GFlowNet-Prover: Autonomous Neuro-Symbolic Smart Contract Verifier (CMD)
=============================================================================
"""


def _apply_preset(hp: TrainingHyperparameters, preset_name: str) -> None:
    vals = TRAINING_PRESETS[preset_name]
    for k, v in vals.items():
        if hasattr(hp, k):
            setattr(hp, k, v)
    hp.preset_name = preset_name


def cmd_train(args: argparse.Namespace) -> int:
    """Handle `train` command."""
    hp = TrainingHyperparameters()
    preset_explicit = getattr(args, "_preset_explicit", False)

    # Apply default preset first, then merge TOML config, then re-apply preset if explicitly requested
    if args.preset and args.preset in TRAINING_PRESETS and not (args.config and preset_explicit):
        _apply_preset(hp, args.preset)

    if args.config:
        hp = ConfigManager.load_training_config(args.config, base=hp)

    if args.preset and args.preset in TRAINING_PRESETS and args.config and preset_explicit:
        _apply_preset(hp, args.preset)

    # Command line argument overrides
    if args.phase is not None:
        hp.phase = args.phase
    if args.epochs is not None:
        hp.epochs = args.epochs
    if args.steps is not None:
        hp.total_steps = args.steps
    if args.lr is not None:
        hp.lr_policy = args.lr
    if args.batch_size is not None:
        hp.batch_size = args.batch_size
    if args.checkpoint_out is not None:
        hp.checkpoint_out = args.checkpoint_out

    engine = TrainingEngine(hp=hp)
    engine.run(
        total_steps=hp.total_steps,
        generate_plots=not args.no_plot,
        plot_path=args.plot_file,
        metrics_path=args.metrics_file,
        plot_dpi=getattr(args, "dpi", 200),
    )
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Handle `plot` command to generate training graphs from existing metrics."""
    metrics_file = Path(args.metrics_file)
    if not metrics_file.exists():
        print(f"\033[91m[!] Metrics file '{metrics_file}' not found.\033[0m")
        print("    Run `python main.py train` first to generate training history,")
        print("    or specify an existing JSON log with `--metrics-file <path>`.")
        return 1

    import json
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\033[91m[!] Error parsing metrics JSON: {e}\033[0m")
        return 1

    out_file = generate_training_graphs(
        data,
        output_path=args.output,
        dpi=args.dpi,
        title_suffix=args.title,
    )
    print(f"\033[92m[+] Training graph created successfully: {out_file.resolve()}\033[0m")

    if args.ascii:
        chart = render_terminal_summary(data)
        print(chart)

    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Handle `audit` command."""
    patch_fmt = getattr(args, "patch_format", "require")
    if not getattr(args, "_patch_format_explicit", False) and getattr(args, "config", None):
        try:
            cfg = ConfigManager.load_audit_config(args.config)
            patch_fmt = cfg.patch_format
        except Exception:
            pass

    run_audit(
        project_path=args.project,
        config_path=args.config,
        setup_contract=args.setup_contract,
        setup_function=args.setup_function,
        max_trajectories=args.max_trajectories,
        output_dir=args.output_dir,
        seed=args.seed,
        patch_format=patch_fmt,
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle `benchmark` / `eval` command."""
    if getattr(args, "detailed", False) or getattr(args, "profile", False):
        from scripts.run_benchmarks import run_all_benchmarks
        res = run_all_benchmarks(
            quick=not getattr(args, "full", False),
            profile=getattr(args, "profile", False),
        )
        return 0 if res["all_passed"] else 1
    run_benchmark_scorecard()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle `report` automated security advisory & SARIF exporter command."""
    return run_report(
        input_file=args.input,
        output_file=args.output,
        report_format=args.format,
    )



def cmd_status(args: argparse.Namespace) -> int:
    """Handle `status` / `info` command."""
    print(BANNER)
    print("[*] GFlowNet-Prover Workspace Diagnostics:")
    print(f"    - Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    print(f"    - Workspace Root: {Path.cwd().resolve()}")

    # Checkpoints
    ckpts = list(Path("checkpoints").glob("*.*")) if Path("checkpoints").exists() else []
    print(f"\n[*] Checkpoints Directory ({len(ckpts)} files):")
    for c in ckpts:
        size_kb = c.stat().st_size / 1024.0
        print(f"    - {c.name:<32} ({size_kb:6.1f} KB)")

    # Plots
    plots = list(Path("plots").glob("*.*")) if Path("plots").exists() else []
    print(f"\n[*] Generated Training Plots ({len(plots)} files):")
    for p in plots:
        print(f"    - {p.name:<32} (file:///{p.resolve()})")

    # Configs
    configs = list(Path("configs").glob("*.toml")) if Path("configs").exists() else []
    print(f"\n[*] Training & Audit Configurations ({len(configs)} files):")
    for cf in configs:
        print(f"    - {cf.name}")

    print("\n[+] For command usage, run: python main.py --help\n")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """Handle `context` export command."""
    import json
    import re
    from datetime import datetime, timezone

    project_root = Path(args.project)
    print(f"[*] Extracting project context from '{project_root}'...")

    src_dir = project_root / "src"
    sol_root = src_dir if src_dir.exists() else project_root
    sol_files = sorted(p for p in sol_root.rglob("*.sol") if "lib" not in p.parts and "out" not in p.parts)

    contracts_summary = []
    contract_re = re.compile(r"\b(?:abstract\s+)?contract\s+([A-Za-z_][A-Za-z0-9_]*)")
    func_re = re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")

    for sol_file in sol_files:
        if any(tok in sol_file.name.lower() for tok in ("exploit", "attack")):
            continue
        source = sol_file.read_text(encoding="utf-8", errors="replace")
        rel_path = sol_file.relative_to(project_root).as_posix() if sol_file.is_relative_to(project_root) else sol_file.as_posix()
        c_names = contract_re.findall(source)
        funcs = [f"{name}({ ' '.join(params.split()) })" for name, params in func_re.findall(source)]
        ext_calls = len(re.findall(r"\.\b(?:call|delegatecall|staticcall|transfer|send)\s*[\({]", source))
        contracts_summary.append(
            {
                "file": rel_path,
                "contracts": c_names,
                "functions": funcs,
                "external_call_sites": ext_calls,
                "lines": len(source.splitlines()),
            }
        )

    generated_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        payload = {
            "target_project": str(project_root),
            "format": "json",
            "generated_at": generated_ts,
            "contract_count": sum(len(c["contracts"]) for c in contracts_summary),
            "files": contracts_summary,
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        lines = [
            "# Project AI Context Bundle",
            "",
            f"- **Target Project**: `{project_root}`",
            f"- **Format**: `{args.format}`",
            f"- **Generated**: `{generated_ts}`",
            f"- **Solidity Units Scanned**: `{len(contracts_summary)}`",
            "",
            "## Discovered Contracts & Entrypoints",
            "",
        ]
        for item in contracts_summary:
            c_title = ", ".join(item["contracts"]) if item["contracts"] else item["file"]
            lines.append(f"### `{c_title}` (`{item['file']}`, {item['lines']} LOC)")
            lines.append(f"- External low-level call sites: `{item['external_call_sites']}`")
            for fn_sig in item["functions"]:
                lines.append(f"- `function {fn_sig}`")
            lines.append("")
        out_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"[+] AI prompt context saved to: {out_file.resolve()}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Handle `review` automated codebase review command."""
    from scripts.code_review import run_review_cli

    review_args: List[str] = []
    if args.json:
        review_args.append("--json")
    if args.strict:
        review_args.append("--strict")
    if args.output:
        review_args.extend(["--output", args.output])
    return run_review_cli(review_args)


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="GFlowNet-Prover: Autonomous Neuro-Symbolic Verification Engine (CMD Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Phase 2 Curriculum Trajectory Balance model & auto-plot progress:
  python main.py train --steps 300 --preset production_defi

  # Fast smoke test training:
  python main.py train --preset smoke_test

  # Generate/Re-plot training graph from metrics JSON:
  python main.py plot --metrics-file training_metrics.json --output plots/training_progress.png

  # Synchronously audit smart contracts:
  python main.py audit --project ./tests/fixtures/multi_bug_defi_protocol

  # View certified release gates benchmark scorecard:
  python main.py benchmark

  # Inspect model checkpoints and diagnostics:
  python main.py status
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. train
    p_train = subparsers.add_parser("train", help="Train the GFlowNet policy model and generate progress graphs")
    p_train.add_argument("--phase", type=int, choices=[1, 2, 3], default=2, help="Phase: 1=MSTM, 2=Curriculum TB, 3=Continual EWC (default: 2)")
    p_train.add_argument("--preset", choices=list(TRAINING_PRESETS.keys()), default="production_defi", help="Preset profile (default: production_defi)")
    p_train.add_argument("--steps", type=int, default=None, help="Total training steps (overrides preset)")
    p_train.add_argument("--epochs", type=int, default=None, help="Epoch count")
    p_train.add_argument("--lr", type=float, default=None, help="Policy learning rate")
    p_train.add_argument("--batch-size", type=int, default=None, help="Rollout batch size")
    p_train.add_argument("--config", type=str, default=None, help="Path to TOML configuration")
    p_train.add_argument("--checkpoint-out", type=str, default="checkpoints/curriculum_tb.safetensors", help="Checkpoint destination")
    p_train.add_argument("--plot-file", type=str, default="plots/training_progress.png", help="Path to save generated training graph")
    p_train.add_argument("--metrics-file", type=str, default="training_metrics.json", help="Path to save step telemetry history")
    p_train.add_argument("--dpi", type=int, default=200, help="Plot resolution DPI (default: 200)")
    p_train.add_argument("--no-plot", action="store_true", help="Skip generating the PNG plot figure")

    # 2. plot
    p_plot = subparsers.add_parser("plot", help="Generate/render training graphs from existing metrics")
    p_plot.add_argument("--metrics-file", type=str, default="training_metrics.json", help="Path to telemetry history JSON")
    p_plot.add_argument("--output", type=str, default="plots/training_progress.png", help="Destination path for graph image")
    p_plot.add_argument("--dpi", type=int, default=200, help="Image resolution DPI (default: 200)")
    p_plot.add_argument("--title", type=str, default="", help="Optional subtitle suffix")
    p_plot.add_argument("--ascii", action="store_true", help="Display ASCII curve in CMD console")

    # 3. audit
    p_audit = subparsers.add_parser("audit", help="Run multi-contract vulnerability formal audit")
    p_audit.add_argument("--project", type=str, default="./tests/fixtures/multi_bug_defi_protocol", help="Foundry project path")
    p_audit.add_argument("--config", type=str, default="configs/default_audit.toml", help="Audit config TOML")
    p_audit.add_argument("--setup-contract", type=str, default="ExploitHarness", help="Setup harness contract")
    p_audit.add_argument("--setup-function", type=str, default="setUp", help="Setup function")
    p_audit.add_argument("--max-trajectories", type=int, default=1000, help="Max trajectories budget")
    p_audit.add_argument("--output-dir", type=str, default="audit_out", help="Output directory")
    p_audit.add_argument("--seed", type=int, default=42, help="PRNG seed")
    p_audit.add_argument(
        "--patch-format",
        type=str,
        default="require",
        choices=["require", "custom-error"],
        help="Solidity remediation patch format: 'require' or 'custom-error'",
    )

    # 4. benchmark / eval
    p_bench = subparsers.add_parser("benchmark", help="Display the 7 Certified Release Gates scorecard")
    p_bench.add_argument("--detailed", action="store_true", help="Execute live micro-benchmarks across workspace crates")
    p_bench.add_argument("--full", action="store_true", help="Execute complete statistical sampling for benchmarks")
    p_bench.add_argument("--profile", action="store_true", help="Include memory working set RSS and detailed latency profiling")
    subparsers.add_parser("eval", parents=[p_bench], add_help=False, help="Alias for benchmark")

    # 5. status

    subparsers.add_parser("status", help="Show system status, checkpoints, and plot files")
    subparsers.add_parser("info", help="Alias for status")

    # 6. context
    p_ctx = subparsers.add_parser("context", help="Export unified whole-project semantic model and prompt bundle")
    p_ctx.add_argument("--project", type=str, default=".", help="Project path")
    p_ctx.add_argument("--output", type=str, default="PROJECT_AI_CONTEXT.md", help="Output file")
    p_ctx.add_argument("--format", type=str, default="markdown", choices=["markdown", "json"], help="Export format")

    # 7. review
    p_rev = subparsers.add_parser("review", help="Run automated static & semantic codebase review")
    p_rev.add_argument("--json", action="store_true", help="Emit structured JSON report to stdout")
    p_rev.add_argument("--strict", action="store_true", help="Fail with exit code 1 on warnings as well as errors")
    p_rev.add_argument("--output", type=str, default=None, help="Optional file path to save JSON report")

    # 8. report
    p_rep = subparsers.add_parser("report", help="Export formal audit findings to GitHub Advisory Markdown or SARIF v2.1.0")
    p_rep.add_argument("--input", type=str, default="audit_out/audit_report.json", help="Input audit report JSON")
    p_rep.add_argument("--output", type=str, default=None, help="Output destination file")
    p_rep.add_argument("--format", type=str, default="markdown", choices=["markdown", "sarif", "json"], help="Export format")

    return parser



def main(args_list: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    raw_argv = list(args_list) if args_list is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args._preset_explicit = any(tok == "--preset" or tok.startswith("--preset=") for tok in raw_argv)
    args._patch_format_explicit = any(tok == "--patch-format" or tok.startswith("--patch-format=") for tok in raw_argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "plot":
        return cmd_plot(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command in ("benchmark", "eval"):
        return cmd_benchmark(args)
    elif args.command in ("status", "info"):
        return cmd_status(args)
    elif args.command == "context":
        return cmd_context(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "report":
        return cmd_report(args)

    return 0

