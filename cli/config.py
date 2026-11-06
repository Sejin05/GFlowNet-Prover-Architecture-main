"""Configuration Loader, Validator, and Preset Manager for GFlowNet-Prover CLI."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    try:
        import tomli as toml_reader  # type: ignore
    except ImportError:
        toml_reader = None  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore


@dataclass
class ProjectSourceConfig:
    workspace_path: str = "."
    setup_contract: str = "ExploitHarnessTest"
    setup_function: str = "setUp"
    target_contracts: List[str] = field(default_factory=list)


@dataclass
class FoundryProfileConfig:
    solc_version: str = "0.8.24"
    evm_version: str = "cancun"
    fork_block_number: Optional[int] = 18_000_000


@dataclass
class SearchBudgetConfig:
    max_trajectories: int = 100_000
    max_depth: int = 15
    batch_size: int = 64
    wall_clock_timeout_sec: int = 900
    smt_timeout_ms: int = 5000
    enable_online_adaptation: bool = True


@dataclass
class RewardConfig:
    alpha_boundary: float = 1.0
    beta_novelty: float = 0.5
    gamma_exploit: float = 100.0
    m_unsat_penalty: float = 1e-3
    r_min: float = 1e-6
    lambda_boundary: float = 5.0


@dataclass
class SymbolicConfig:
    boundary_trigger: float = 0.85
    max_smt_workers: int = 6
    enable_unsat_cache: bool = True
    lmdb_cache_path: str = "cache/lmdb"
    smt_timeout_ms: int = 5000


@dataclass
class AuditConfig:
    project_source: ProjectSourceConfig = field(default_factory=ProjectSourceConfig)
    foundry_profile: FoundryProfileConfig = field(default_factory=FoundryProfileConfig)
    search_budget: SearchBudgetConfig = field(default_factory=SearchBudgetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    patch_format: str = "require"
    seed: int = 42


@dataclass
class TrainingHyperparameters:
    phase: int = 2  # 1: MSTM Pre-train, 2: Curriculum TB, 3: Continual EWC
    preset_name: str = "Production DeFi Tuning"
    corpus_dir: str = "data/corpus"
    checkpoint_out: str = "checkpoints/curriculum_tb.safetensors"
    anchor_checkpoint: str = "checkpoints/pretrained_mstm.safetensors"
    compute_device: str = "CUDA GPU 0 (Kind::BFloat16)"
    worker_threads: int = 16
    epochs: int = 10
    total_steps: int = 500
    mask_ratio: float = 0.30
    lr_policy: float = 3e-4
    lr_log_z: float = 5e-3
    lr_warmup_steps: int = 50
    huber_delta: float = 10.0
    alpha_boundary: float = 1.0
    beta_novelty: float = 0.5
    gamma_exploit: float = 100.0
    m_unsat_penalty: float = 1e-3
    boundary_trigger: float = 0.85
    lambda_ewc: float = 400.0
    replay_capacity: int = 100_000
    batch_size: int = 64
    max_steps: int = 15
    rgat_hidden_dim: int = 512
    rgat_num_layers: int = 6
    rgat_heads: int = 8
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.95
    logit_clip_bound: float = 30.0
    entropy_reg_coeff: float = 0.01
    target_entropy_nats: float = 0.35
    enable_grad_spike_guard: bool = True
    max_residual_clip: float = 100.0
    seed: int = 42

    def to_toml_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "rgat_hidden_dim": self.rgat_hidden_dim,
                "rgat_num_layers": self.rgat_num_layers,
                "rgat_heads": self.rgat_heads,
                "hgt_hidden_dim": 512,
                "hgt_num_layers": 4,
                "hgt_heads": 8,
            },
            "optim": {
                "lr_policy": self.lr_policy,
                "lr_log_z": self.lr_log_z,
                "lr_warmup_steps": self.lr_warmup_steps,
                "lr_min": 1e-5,
                "weight_decay": 1e-4,
                "clip_grad_norm": self.clip_grad_norm,
            },
            "reward": {
                "alpha_boundary": self.alpha_boundary,
                "beta_novelty": self.beta_novelty,
                "gamma_exploit": self.gamma_exploit,
                "m_unsat_penalty": self.m_unsat_penalty,
                "r_min": 1e-6,
                "lambda_boundary": 5.0,
            },
            "rollout": {
                "max_steps": self.max_steps,
                "batch_size": self.batch_size,
                "temperature": 1.0,
                "epsilon_uniform": 0.05,
                "num_workers": self.worker_threads,
            },
            "replay_buffer": {
                "capacity": self.replay_capacity,
                "alpha_per": 0.6,
                "beta_per": 0.4,
                "eps_per": 1e-5,
            },
            "tb_loss": {
                "huber_delta": self.huber_delta,
                "reward_min": 1e-6,
                "max_residual_clip": self.max_residual_clip,
            },
            "stability": {
                "ema_decay": self.ema_decay,
                "logit_clip_bound": self.logit_clip_bound,
                "entropy_reg_coeff": self.entropy_reg_coeff,
                "target_entropy_nats": self.target_entropy_nats,
                "enable_grad_spike_guard": self.enable_grad_spike_guard,
            },
            "symbolic": {
                "boundary_trigger": self.boundary_trigger,
                "smt_timeout_ms": 5000,
            },
            "continual": {
                "lambda_ewc": self.lambda_ewc,
                "anchor_checkpoint": self.anchor_checkpoint,
            },
        }


TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke_test": {
        "lr_policy": 5e-4,
        "lr_log_z": 1e-2,
        "lr_warmup_steps": 20,
        "total_steps": 100,
        "clip_grad_norm": 1.0,
        "ema_decay": 0.90,
        "enable_grad_spike_guard": True,
        "huber_delta": 10.0,
        "alpha_boundary": 1.0,
        "beta_novelty": 0.25,
        "gamma_exploit": 50.0,
        "boundary_trigger": 0.85,
        "lambda_ewc": 400.0,
        "replay_capacity": 10_000,
        "worker_threads": 4,
        "epochs": 2,
        "batch_size": 16,
    },
    "production_defi": {
        "lr_policy": 3e-4,
        "lr_log_z": 5e-3,
        "lr_warmup_steps": 50,
        "total_steps": 500,
        "clip_grad_norm": 1.0,
        "ema_decay": 0.95,
        "enable_grad_spike_guard": True,
        "huber_delta": 10.0,
        "alpha_boundary": 1.0,
        "beta_novelty": 0.50,
        "gamma_exploit": 100.0,
        "boundary_trigger": 0.85,
        "lambda_ewc": 400.0,
        "replay_capacity": 100_000,
        "worker_threads": 16,
        "epochs": 10,
        "batch_size": 64,
    },
    "deep_exploit": {
        "lr_policy": 1.5e-4,
        "lr_log_z": 5e-3,
        "lr_warmup_steps": 100,
        "total_steps": 1000,
        "clip_grad_norm": 0.75,
        "ema_decay": 0.98,
        "enable_grad_spike_guard": True,
        "huber_delta": 10.0,
        "alpha_boundary": 1.25,
        "beta_novelty": 1.50,
        "gamma_exploit": 250.0,
        "boundary_trigger": 0.78,
        "lambda_ewc": 400.0,
        "replay_capacity": 250_000,
        "worker_threads": 32,
        "epochs": 25,
        "batch_size": 64,
    },
    "quick_dev": {
        "lr_policy": 4e-4,
        "lr_log_z": 8e-3,
        "lr_warmup_steps": 30,
        "total_steps": 200,
        "clip_grad_norm": 1.0,
        "ema_decay": 0.92,
        "enable_grad_spike_guard": True,
        "huber_delta": 10.0,
        "alpha_boundary": 1.0,
        "beta_novelty": 0.40,
        "gamma_exploit": 80.0,
        "boundary_trigger": 0.85,
        "lambda_ewc": 400.0,
        "replay_capacity": 20_000,
        "worker_threads": 8,
        "epochs": 5,
        "batch_size": 32,
    },
}


def _format_toml_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        inner = ", ".join(_format_toml_val(x) for x in v)
        return f"[{inner}]"
    return str(v)


def dump_toml(data: Dict[str, Any]) -> str:
    """Deterministic TOML serializer."""
    if tomli_w is not None:
        return tomli_w.dumps(data)
    lines: List[str] = []
    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_format_toml_val(v)}")
    for section, values in data.items():
        if isinstance(values, dict):
            lines.append(f"\n[{section}]")
            for k, v in values.items():
                if v is not None:
                    lines.append(f"{k} = {_format_toml_val(v)}")
    return "\n".join(lines).strip() + "\n"


class ConfigManager:
    """Manages reading and writing TOML configurations."""

    @staticmethod
    def load_training_config(
        path: str | Path,
        base: Optional[TrainingHyperparameters] = None,
    ) -> TrainingHyperparameters:
        p = Path(path)
        hp = TrainingHyperparameters(**asdict(base)) if base is not None else TrainingHyperparameters()
        if not p.exists() or toml_reader is None:
            return hp

        try:
            with open(p, "rb") as f:
                data = toml_reader.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse training TOML configuration '{p}': {exc}") from exc

        if "training" in data:
            tr = data["training"]
            hp.corpus_dir = str(tr.get("dataset_path", tr.get("corpus_dir", hp.corpus_dir)))
            hp.batch_size = int(tr.get("batch_size", hp.batch_size))
            hp.epochs = int(tr.get("num_epochs", tr.get("epochs", hp.epochs)))
            hp.mask_ratio = float(tr.get("mask_ratio", hp.mask_ratio))
            if "total_steps" in tr:
                hp.total_steps = int(tr["total_steps"])
            if "phase" in tr:
                hp.phase = int(tr["phase"])
            elif "tb_loss" not in data:
                hp.phase = 1

        if "model" in data:
            md = data["model"]
            hp.rgat_hidden_dim = int(md.get("rgat_hidden_dim", md.get("hidden_dim", hp.rgat_hidden_dim)))
            hp.rgat_num_layers = int(md.get("rgat_num_layers", md.get("num_layers", hp.rgat_num_layers)))
            hp.rgat_heads = int(md.get("rgat_heads", md.get("num_heads", hp.rgat_heads)))

        if "optim" in data:
            opt = data["optim"]
            hp.lr_policy = float(opt.get("lr_policy", opt.get("lr", hp.lr_policy)))
            hp.lr_log_z = float(opt.get("lr_log_z", hp.lr_log_z))
            hp.lr_warmup_steps = int(opt.get("lr_warmup_steps", hp.lr_warmup_steps))
            hp.clip_grad_norm = float(opt.get("clip_grad_norm", hp.clip_grad_norm))

        if "reward" in data:
            r = data["reward"]
            hp.alpha_boundary = float(r.get("alpha_boundary", hp.alpha_boundary))
            hp.beta_novelty = float(r.get("beta_novelty", hp.beta_novelty))
            hp.gamma_exploit = float(r.get("gamma_exploit", hp.gamma_exploit))
            hp.m_unsat_penalty = float(r.get("m_unsat_penalty", hp.m_unsat_penalty))

        if "rollout" in data:
            ro = data["rollout"]
            hp.batch_size = int(ro.get("batch_size", hp.batch_size))
            hp.max_steps = int(ro.get("max_steps", hp.max_steps))
            hp.worker_threads = int(ro.get("num_workers", hp.worker_threads))

        if "replay_buffer" in data:
            rb = data["replay_buffer"]
            hp.replay_capacity = int(rb.get("capacity", hp.replay_capacity))

        if "tb_loss" in data:
            tb = data["tb_loss"]
            hp.huber_delta = float(tb.get("huber_delta", hp.huber_delta))
            hp.max_residual_clip = float(tb.get("max_residual_clip", hp.max_residual_clip))
        elif "loss" in data:
            ls = data["loss"]
            hp.huber_delta = float(ls.get("huber_delta", hp.huber_delta))

        if "stability" in data:
            st = data["stability"]
            hp.ema_decay = float(st.get("ema_decay", hp.ema_decay))
            hp.logit_clip_bound = float(st.get("logit_clip_bound", hp.logit_clip_bound))
            hp.entropy_reg_coeff = float(st.get("entropy_reg_coeff", hp.entropy_reg_coeff))
            hp.target_entropy_nats = float(st.get("target_entropy_nats", hp.target_entropy_nats))
            hp.enable_grad_spike_guard = bool(st.get("enable_grad_spike_guard", hp.enable_grad_spike_guard))

        if "symbolic" in data:
            sym = data["symbolic"]
            hp.boundary_trigger = float(sym.get("boundary_trigger", hp.boundary_trigger))

        if "continual" in data:
            ct = data["continual"]
            hp.lambda_ewc = float(ct.get("lambda_ewc", hp.lambda_ewc))
            hp.anchor_checkpoint = str(ct.get("anchor_checkpoint", hp.anchor_checkpoint))

        return hp

    @staticmethod
    def load_audit_config(path: str | Path) -> AuditConfig:
        p = Path(path)
        cfg = AuditConfig()
        if not p.exists() or toml_reader is None:
            return cfg

        try:
            with open(p, "rb") as f:
                data = toml_reader.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse audit TOML configuration '{p}': {exc}") from exc

        if "seed" in data:
            cfg.seed = int(data["seed"])
        if "project_source" in data:
            ps = data["project_source"]
            cfg.project_source.workspace_path = str(ps.get("workspace_path", cfg.project_source.workspace_path))
            cfg.project_source.setup_contract = str(ps.get("setup_contract", cfg.project_source.setup_contract))
            cfg.project_source.setup_function = str(ps.get("setup_function", cfg.project_source.setup_function))
            if "target_contracts" in ps:
                cfg.project_source.target_contracts = [str(x) for x in ps["target_contracts"]]
        if "foundry_profile" in data:
            fp = data["foundry_profile"]
            cfg.foundry_profile.solc_version = str(fp.get("solc_version", cfg.foundry_profile.solc_version))
            cfg.foundry_profile.evm_version = str(fp.get("evm_version", cfg.foundry_profile.evm_version))
            if "fork_block_number" in fp:
                cfg.foundry_profile.fork_block_number = int(fp["fork_block_number"])
        if "search_budget" in data:
            sb = data["search_budget"]
            cfg.search_budget.max_trajectories = int(sb.get("max_trajectories", cfg.search_budget.max_trajectories))
            cfg.search_budget.max_depth = int(sb.get("max_depth", cfg.search_budget.max_depth))
            cfg.search_budget.batch_size = int(sb.get("batch_size", cfg.search_budget.batch_size))
            cfg.search_budget.wall_clock_timeout_sec = int(sb.get("wall_clock_timeout_sec", cfg.search_budget.wall_clock_timeout_sec))
            cfg.search_budget.smt_timeout_ms = int(sb.get("smt_timeout_ms", cfg.search_budget.smt_timeout_ms))
            cfg.search_budget.enable_online_adaptation = bool(sb.get("enable_online_adaptation", cfg.search_budget.enable_online_adaptation))
        if "reward" in data:
            rw = data["reward"]
            cfg.reward.alpha_boundary = float(rw.get("alpha_boundary", cfg.reward.alpha_boundary))
            cfg.reward.beta_novelty = float(rw.get("beta_novelty", cfg.reward.beta_novelty))
            cfg.reward.gamma_exploit = float(rw.get("gamma_exploit", cfg.reward.gamma_exploit))
            cfg.reward.m_unsat_penalty = float(rw.get("m_unsat_penalty", cfg.reward.m_unsat_penalty))
            cfg.reward.r_min = float(rw.get("r_min", cfg.reward.r_min))
            cfg.reward.lambda_boundary = float(rw.get("lambda_boundary", cfg.reward.lambda_boundary))
        if "symbolic" in data:
            sm = data["symbolic"]
            cfg.symbolic.boundary_trigger = float(sm.get("boundary_trigger", cfg.symbolic.boundary_trigger))
            cfg.symbolic.max_smt_workers = int(sm.get("max_smt_workers", cfg.symbolic.max_smt_workers))
            cfg.symbolic.enable_unsat_cache = bool(sm.get("enable_unsat_cache", cfg.symbolic.enable_unsat_cache))
            cfg.symbolic.lmdb_cache_path = str(sm.get("lmdb_cache_path", cfg.symbolic.lmdb_cache_path))
            cfg.symbolic.smt_timeout_ms = int(sm.get("smt_timeout_ms", cfg.symbolic.smt_timeout_ms))
        if "patch_format" in data:
            cfg.patch_format = str(data["patch_format"])
        elif "verifier" in data and "patch_format" in data["verifier"]:
            cfg.patch_format = str(data["verifier"]["patch_format"])

        return cfg

    @staticmethod
    def save_training_config(hp: TrainingHyperparameters, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        content = dump_toml(hp.to_toml_dict())
        out.write_text(content, encoding="utf-8")
        return out
