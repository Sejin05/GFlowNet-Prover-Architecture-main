# Modular Programming Plan & Agent Implementation Roadmap: GFlowNet-Prover (Pure Rust / Foundry Edition)

---

## Section A: Complete Pure-Rust Repository File Tree

```text
gflownet-prover/
├── Cargo.toml                                    # Workspace Cargo manifest (resolver = "2", edition = "2021")
├── rust-toolchain.toml                           # Pinned Rust 1.82.0 toolchain with clippy & rustfmt
├── Dockerfile                                    # Multi-stage CUDA 12.4 + LibTorch 2.5.1 + Rust + Foundry container
├── Makefile                                      # Build, clippy, test, criterion bench, and audit targets
├── configs/
│   ├── default_audit.toml                        # Default Phase 3 online adaptation & audit configuration
│   ├── pretrain_mstm.toml                        # Phase 1 Masked State-Transition Modeling config
│   └── curriculum_tb.toml                        # Phase 2 Trajectory Balance bug-discovery config
├── crates/
│   ├── gfn-common/                               # Core types, error hierarchy, math utilities & REST schemas
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs                          # `GfnError` (`thiserror`) & `GfnResult<T>`
│   │       ├── schemas.rs                        # `FoundryContractArtifact`, `AuditSubmissionRequest`, `VerifiedFinding`
│   │       └── math.rs                           # `u256_to_log2_norm`, sinusoidal chunk encodings & Spearman rank correlation
│   ├── gfn-extractor/                            # Module 1A: Foundry compilation, Solar AST/CFG & HSDG builder
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── compiler.rs                       # Programmatic `foundry-compilers` & `svm-rs` wrapper
│   │       ├── cfg_lifter.rs                     # `solar-parse` + EVM bytecode SSA CFG & dominator lifter (`petgraph`)
│   │       ├── proxy_resolver.rs                 # EIP-1967 / EIP-2535 Diamond dynamic storage slot merger
│   │       └── hsdg_builder.rs                   # Builds and validates `HeteroGraphTensors` (`tch::Tensor` + `rkyv`)
│   ├── gfn-invariants/                           # Module 1B: Automated Safety Invariant Synthesis (I)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── templates.rs                      # Solvency, Monotonicity, AccessControl, PrecisionNeutrality templates
│   │       ├── hgt.rs                            # 4-layer Heterogeneous Graph Transformer (`tch::nn`)
│   │       └── compiler.rs                       # Compiles invariants into `revm` closures & SMT-LIBv2 `QF_BV` strings
│   ├── gfn-evm/                                  # Module 3A: High-speed in-memory `revm` CoW Arena (65,000+ TPS)
│   │   ├── Cargo.toml
│   │   ├── benches/
│   │   │   └── revm_tps_bench.rs                 # Criterion benchmark asserting >= 65,000 TPS/thread
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── arena.rs                          # `RevmArena`, `PinnedStateBuffer`, and `rayon` parallel batch stepping
│   │       ├── inspector.rs                      # Custom `revm::Inspector` tracking `SLOAD`/`SSTORE` & `JUMPI` branch distance
│   │       └── cheatcodes.rs                     # Native handlers for `vm.prank`, `vm.warp`, `vm.roll`, and `deal`
│   ├── gfn-symbolic/                             # Module 3B: Native Concolic EVM & Bitwuzla/Z3 Portfolio Solver
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── concolic_evm.rs                   # Symbolic 256-bit BitVector path constraint translator (`QF_ABV`)
│   │       ├── solver.rs                         # `PortfolioSmtSolver` (`bitwuzla-sys` + `z3`) with hard timeouts
│   │       ├── concretizer.rs                    # Partial storage concretization for non-linear/Keccak256 guards
│   │       └── unsat_cache.rs                    # L1 `DashMap` + L2 `heed` (LMDB) cache keyed by `xxh3_128`
│   ├── gfn-engine/                               # Module 2: Trajectory GFlowNet Engine (`tch-rs` CUDA)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── rgat.rs                           # 6-layer `StateGraphRgatEncoder` (Hidden Dim = 512, `Kind::BFloat16`)
│   │       ├── policy.rs                         # `AutoregressiveForwardPolicyHead`, `BackwardPolicyHead`, `LogPartitionNet`
│   │       ├── reward_rnd.rs                     # Random Network Distillation (RND) & composite reward `R(s_n)`
│   │       ├── tb_loss.rs                        # Huber-clipped `TrajectoryBalanceLoss` in `Kind::Float` (`f32`) + EWC
│   │       ├── replay_buffer.rs                  # Lock-free Fenwick sum-tree `PrioritizedReplayBuffer`
│   │       └── rollout_worker.rs                 # Orchestrates GPU-to-`RevmArena` rollouts & online adaptation
│   ├── gfn-verifier/                             # Module 4: Concrete `forge test` Gate, PoC & Patch Synthesizer
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── forge_gate.rs                     # Executes `forge test --offline` and enforces 0.0% False Positive rate
│   │       ├── poc_synthesizer.rs                # Renders turnkey `ExploitPoC.t.sol` Foundry test scripts
│   │       └── patch_synthesizer.rs              # Minimal Negated Precondition extractor & `solar-ast` unified diff generator
│   ├── gfn-monitor/                              # Drift detection (MMD, PSI, KS) & Prometheus telemetry
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── drift.rs                          # Spectral MMD, Opcode PSI, Storage JS-Divergence, and RND KS-test
│   │       └── telemetry.rs                      # `metrics-exporter-prometheus` & structured JSON `tracing` setup
│   ├── gfn-trading/                              # Concentrated AMM liquidity math & cyclic arbitrage
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                            # Uniswap v3 pool simulation, tick spacing, swap routing
│   ├── gfn-cli/                                  # Binary entrypoint (`gfn-forge`) & `axum` REST/SSE server
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs                           # `clap` CLI (`audit`, `pretrain`, `train`, `serve`)
│   │       ├── server.rs                         # `axum` `/v1/audits` & `/v1/policy/sample-step` microservice
│   │       ├── auditor.rs                        # Full Rust-native formal verification auditor
│   │       ├── trainer.rs                        # Rust-native SafeTensors trainer & MSTM/TB optimization
│   │       └── context_bundler.rs                # AI prompt & semantic model context extractor
│   └── stubs/                                    # 9 portable offline stubs for self-contained CI builds
├── scripts/                                      # Unified Automation, Verification & Benchmark Scripts
│   ├── setup_env.ps1                             # Dynamic MinGW toolchain configuration for Windows GNU
│   ├── run_tests.py                              # Unified fast/full test runner (code review, pytest, cargo)
│   ├── run_benchmarks.py                         # High-precision Criterion performance & memory SLA runner
│   └── code_review.py                            # 7-analyzer static & semantic code reviewer (<150ms)
├── cli/                                          # Pure Python Headless CLI, Training & Reporting Engine
│   ├── runner.py                                 # CMD argument parser & subcommand dispatcher
│   ├── config.py                                 # Hyperparameter schemas, presets & TOML serializer
│   ├── trainer.py                                # Gradient training engine & SafeTensors IEEE-754 serializer
│   ├── audit.py                                  # Formal audit runner & 7 Release Gates scorecard
│   ├── plotter.py                                # Matplotlib 6-panel dashboard & ASCII curve visualizer
│   └── reporter.py                               # Security advisory Markdown & SARIF v2.1.0 generator
├── main.py                                       # Primary CLI entrypoint (`python main.py <CMD>`)
├── app.py                                        # CLI alias (`python app.py <CMD>`)
└── tests/                                        # Workspace-level integration, metamorphic, fuzzing & unit tests
    ├── fixtures/
    │   └── multi_bug_defi_protocol/              # 4-contract Foundry suite (ERC-4626 + FlashLoan + Governance bugs)
    ├── end_to_end_audit_test.rs                  # Full pipeline integration test asserting 3/3 bugs found & 0% FPR
    ├── metamorphic_robustness_test.rs            # `proptest` AST obfuscation & storage permutation invariance tests
    ├── fuzz_robustness_test.rs                   # Property-based fuzzing robustness test
    ├── test_cli.py                               # Python CLI, preset, config & audit tests
    ├── test_code_review.py                       # Code review framework unit & mutation tests
    ├── test_training_accuracy.py                 # SafeTensors roundtrip, EMA & math accuracy tests
    └── test_training_scenarios.py                # Multi-mode, spike guard, EWC, & log Z convergence tests
```

---

## Section B: Chronological Step-by-Step Implementation Plan

Downstream coding agents must implement the Rust workspace strictly in the following 7 sequential milestones. Every crate must compile with `#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]` and pass `cargo test` before proceeding to the next milestone.

### Milestone 1: Workspace Foundation, `gfn-common`, and `gfn-evm` (`RevmArena`)
1. Initialize the root `Cargo.toml` workspace, `rust-toolchain.toml` (`1.82.0`), and `crates/gfn-common/src/{lib.rs, error.rs, schemas.rs, math.rs}`.
2. Implement `crates/gfn-evm/src/{lib.rs, arena.rs, inspector.rs, cheatcodes.rs}` using `revm = "19.0.0"`, `foundry-evm = "1.0.0"`, and `rayon = "1.10.0"`.
3. Run `cargo test -p gfn-common -p gfn-evm` and `cargo bench -p gfn-evm` to verify bit-exact CoW state snapshotting and $\ge 65,000\text{ TPS/thread}$.

### Milestone 2: Module 1A — `gfn-extractor` (Foundry Compiler, Solar CFG & HSDG Builder)
1. Implement `crates/gfn-extractor/src/compiler.rs` using `foundry-compilers = "0.11.0"` and `svm-rs = "0.5.12"` to extract ASTs, ABIs, deployed bytecode, and storage layouts.
2. Implement `crates/gfn-extractor/src/{cfg_lifter.rs, proxy_resolver.rs}` using `solar-parse = "0.1.2"` and `petgraph = "0.6.5"` to extract SSA CFG nodes, dominator frontiers, and EIP-1967 proxy target slots.
3. Implement `crates/gfn-extractor/src/hsdg_builder.rs` to build and validate `HeteroGraphTensors` with exact node feature dimensions (`StorageNode: 64`, `FunctionNode: 64`, `ConditionNode: 48`, `ExternalCallNode: 32`) and `u256_to_log2_norm`.
4. Verify with `cargo test -p gfn-extractor`.

### Milestone 3: Module 1B — `gfn-invariants` (Automated Safety Invariant Synthesis $\mathcal{I}$)
1. Implement `crates/gfn-invariants/src/templates.rs` covering Solvency, Monotonicity, AccessControl, and PrecisionNeutrality.
2. Implement `crates/gfn-invariants/src/hgt.rs` (4-layer `HgtConvLayer` in `tch::nn`) and `compiler.rs` (compiling candidate invariants into vectorized `RevmArena` state evaluators and SMT-LIBv2 `QF_BV` assertions).
3. Verify with `cargo test -p gfn-invariants`.

### Milestone 4: Module 2 — `gfn-engine` (Trajectory GFlowNet Engine & RND Curiosity)
1. Implement `crates/gfn-engine/src/rgat.rs` (6-layer `StateGraphRgatEncoder`, hidden dim 512, 8 heads, sparse scatter-softmax attention in `tch::Kind::BFloat16`).
2. Implement `crates/gfn-engine/src/policy.rs` (`AutoregressiveForwardPolicyHead`, `BackwardPolicyHead`, and `LogPartitionNet`).
3. Implement `crates/gfn-engine/src/reward_rnd.rs` (orthogonal Target/Predictor RND networks and composite reward $R(s_n)$), `tb_loss.rs` ($\mathcal{L}_{TB}$ in `tch::Kind::Float`), and `replay_buffer.rs` (`PrioritizedReplayBuffer`).
4. Verify with `cargo test -p gfn-engine`.

### Milestone 5: Module 3B — `gfn-symbolic` (Concolic EVM + Bitwuzla/Z3 Solver Bridge)
1. Implement `crates/gfn-symbolic/src/unsat_cache.rs` (L1 `DashMap` + L2 `heed` LMDB cache keyed by `xxh3_128`).
2. Implement `crates/gfn-symbolic/src/{concretizer.rs, concolic_evm.rs, solver.rs}` interfacing `bitwuzla-sys = "0.6.0"` and `z3 = "0.12.1"` with hard `5,000 ms` `Terminator` cutoffs and Unsat Core extraction.
3. Wire `crates/gfn-engine/src/rollout_worker.rs` so narrow arithmetic branches ($R_{\text{boundary}} > 0.85$ or `ArgDomain::SymbolicBv256Solve`) automatically invoke `PortfolioSmtSolver` and update $M_{\text{unsat}}(\tau)$.
4. Verify with `cargo test -p gfn-symbolic`.

### Milestone 6: Module 4 — `gfn-verifier` (Concrete Forge Gate, PoC & Patch Synthesizer)
1. Implement `crates/gfn-verifier/src/poc_synthesizer.rs` to generate turnkey `ExploitPoC.t.sol` files with `vm.prank`, `vm.warp`, `vm.roll`, `deal`, and concrete witness calldata.
2. Implement `crates/gfn-verifier/src/forge_gate.rs` to execute `forge test --match-test test_ExploitPoC -vvvv --offline` via `tokio::process::Command` and enforce the 0.0% False Positive gate.
3. Implement `crates/gfn-verifier/src/patch_synthesizer.rs` to extract the Minimal Negated Precondition from the SMT branch trace, inject the `require(...)` or CEI reorder via `solar-ast` source spans, and verify non-regression across existing project tests.
4. Verify with `cargo test -p gfn-verifier`.

### Milestone 7: `gfn-monitor`, `gfn-cli` (`axum` Server + CLI) & Integration Tests
1. Implement `crates/gfn-monitor/src/{drift.rs, telemetry.rs}` and `crates/gfn-cli/src/{main.rs, server.rs}`.
2. Run full workspace integration and metamorphic test suites (`cargo test --workspace --release`).

---

## Section C: Rust Interface Contracts, Trait Signatures & Error Protocols

### C.1 Unified Error Hierarchy (`crates/gfn-common/src/error.rs`)
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GfnError {
    #[error("[GFN_ERR_101] Foundry compilation or artifact extraction failed: {reason}")]
    CompilationArtifact { reason: String },

    #[error("[GFN_ERR_102] Invalid HSDG topology or non-finite feature tensor: {details}")]
    InvalidGraphTopology { details: String },

    #[error("[GFN_ERR_201] RevmArena execution error on env_idx={env_idx}: {message}")]
    RevmExecution { env_idx: usize, message: String },

    #[error("[GFN_ERR_301] SMT solver timed out after {timeout_ms} ms on branch_pc={branch_pc}")]
    SmtSolverTimeout { branch_pc: usize, timeout_ms: u64 },

    #[error("[GFN_ERR_401] Candidate exploit rejected by concrete `forge test` gate: {stdout}")]
    FalsePositiveRejected { stdout: String },

    #[error("[GFN_ERR_501] LibTorch tensor error: {0}")]
    Tensor(#[from] tch::TchError),

    #[error("[GFN_ERR_502] LMDB persistence error: {0}")]
    Storage(#[from] heed::Error),
}

pub type GfnResult<T> = Result<T, GfnError>;
```

### C.2 Module 3A Interface: `gfn-evm` (`crates/gfn-evm/src/arena.rs`)
```rust
use alloy_primitives::{Address, Bytes, B256, U256};
use gfn_common::error::GfnResult;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct StepBatchInput {
    pub caller_indices: Vec<u8>,          // Length B, actor persona index in 0..=4
    pub target_contracts: Vec<Address>,   // Length B, 20-byte target contract addresses
    pub calldata_batch: Vec<Bytes>,       // Length B, ABI-encoded transaction payloads
    pub warp_deltas: Vec<u64>,            // Length B, vm.warp seconds delta
    pub roll_deltas: Vec<u64>,            // Length B, vm.roll blocks delta
}

#[derive(Debug)]
pub struct StepBatchOutput {
    pub storage_words: Vec<U256>,         // Flattened (B * N_slots) active 256-bit storage values
    pub storage_dyn_features: Vec<f32>,   // Flattened (B * N_slots * 32) SIMD-computed features
    pub branch_telemetry: Vec<f32>,       // Flattened (B * N_cond * 8) branch distances & hit counts
    pub env_context: Vec<f32>,            // Flattened (B * 32) normalized actor balances & block context
    pub reverted_mask: Vec<bool>,         // Length B
    pub state_hashes: Vec<B256>,          // Length B, Blake3 state trie digests
}

pub struct RevmArena {
    batch_size: usize,
    num_slots: usize,
    num_conds: usize,
    // Internal rayon ThreadPool & Arc<CacheDB<EmptyDB>> base snapshot s_0
}

impl RevmArena {
    pub fn new(
        project_root: &Path,
        setup_contract: &str,
        setup_function: &str,
        batch_size: usize,
        num_threads: usize,
        seed: u64,
    ) -> GfnResult<Self>;

    pub fn reset_all_to_s0(&mut self) -> GfnResult<StepBatchOutput>;

    pub fn reset_env_indices(&mut self, env_indices: &[usize]) -> GfnResult<StepBatchOutput>;

    pub fn step_batch(&mut self, input: &StepBatchInput) -> GfnResult<StepBatchOutput>;

    pub fn snapshot_env(&mut self, env_idx: usize) -> GfnResult<u64>;

    pub fn restore_snapshot(&mut self, env_idx: usize, snapshot_id: u64) -> GfnResult<()>;
}
```

### C.3 Module 1 Interfaces: `gfn-extractor` & `gfn-invariants`
```rust
use alloy_primitives::U256;
use gfn_common::error::GfnResult;
use gfn_common::schemas::{FoundryContractArtifact, SynthesizedInvariant};
use std::path::Path;
use tch::{Device, Tensor};

pub struct HeteroGraphTensors {
    pub storage_x: Tensor,                // Shape: [N_storage, 64], Kind::Float
    pub function_x: Tensor,               // Shape: [N_func, 64], Kind::Float
    pub condition_x: Tensor,              // Shape: [N_cond, 48], Kind::Float
    pub ext_call_x: Tensor,               // Shape: [N_ext, 32], Kind::Float
    pub edge_indices: [Tensor; 8],        // Each [2, E_r], Kind::Int64
    pub edge_attrs: [Tensor; 8],          // Each [E_r, 16], Kind::Float
}

pub struct HsdgBuilder {
    project_root: std::path::PathBuf,
    solc_version: String,
}

impl HsdgBuilder {
    pub fn new(project_root: &Path, solc_version: &str) -> Self;

    pub fn extract_artifacts(&self) -> GfnResult<Vec<FoundryContractArtifact>>;

    pub fn build_static_graph(
        &self,
        artifacts: &[FoundryContractArtifact],
        device: Device,
    ) -> GfnResult<HeteroGraphTensors>;

    pub fn update_dynamic_batch(
        base_graph: &HeteroGraphTensors,
        batch_size: usize,
        storage_dyn_features: &[f32],
        branch_telemetry: &[f32],
        device: Device,
    ) -> GfnResult<HeteroGraphTensors>;
}

pub struct HgtInvariantSynthesizer {
    vs: tch::nn::VarStore,
    hidden_dim: i64,
}

impl HgtInvariantSynthesizer {
    pub fn new(device: Device, hidden_dim: i64) -> GfnResult<Self>;

    pub fn synthesize_and_verify(
        &self,
        static_graph: &HeteroGraphTensors,
        warmup_storage_traces: &[Vec<U256>],
    ) -> GfnResult<Vec<SynthesizedInvariant>>;
}
```

### C.4 Module 2 Interfaces: `gfn-engine` (`rgat.rs`, `policy.rs`, `tb_loss.rs`)
```rust
use gfn_common::error::GfnResult;
use gfn_extractor::HeteroGraphTensors;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct ActionBatchTensor {
    pub caller_id: Tensor,                // Shape: [B], Kind::Int64 in 0..=5 (5 = Stop)
    pub func_node_id: Tensor,             // Shape: [B], Kind::Int64 in 0..N_func
    pub arg_domains: Tensor,              // Shape: [B, 8], Kind::Int64 in 0..=7
    pub env_op: Tensor,                   // Shape: [B], Kind::Int64 in 0..=5
    pub log_pf: Tensor,                   // Shape: [B], Kind::Float (exact f32 sum of conditional log-probs)
}

pub struct StateGraphRgatEncoder {
    layers: Vec<RgatConvLayer>,
    ctx_mlp: nn::Sequential,
}

impl StateGraphRgatEncoder {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_layers: usize, num_heads: i64) -> Self;

    /// Returns (global_ctx [B, 512], func_embeddings [B, N_func, 512], storage_embeddings [B, N_storage, 512])
    pub fn forward_t(
        &self,
        batched_graph: &HeteroGraphTensors,
        env_context: &Tensor,
        train: bool,
    ) -> GfnResult<(Tensor, Tensor, Tensor)>;
}

pub struct AutoregressiveForwardPolicyHead {
    caller_mlp: nn::Sequential,
    func_query_proj: nn::Linear,
    domain_mlp: nn::Sequential,
    env_mlp: nn::Sequential,
}

impl AutoregressiveForwardPolicyHead {
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self;

    pub fn sample_actions(
        &self,
        global_ctx: &Tensor,
        func_embeddings: &Tensor,
        callable_mask: &Tensor,
        temperature: f64,
        epsilon_uniform: f64,
    ) -> GfnResult<ActionBatchTensor>;

    pub fn evaluate_log_pf(
        &self,
        global_ctx: &Tensor,
        func_embeddings: &Tensor,
        callable_mask: &Tensor,
        actions: &ActionBatchTensor,
    ) -> GfnResult<Tensor>;
}

pub struct TrajectoryBalanceLoss {
    pub huber_delta: f64,
    pub reward_min: f64,
}

impl TrajectoryBalanceLoss {
    /// Computes Huber-clipped Trajectory Balance loss L_TB in strict Kind::Float (f32) precision.
    /// Returns (mean_tb_loss, abs_tb_residuals_per_trajectory).
    pub fn forward(
        &self,
        log_z: &Tensor,
        sum_log_pf: &Tensor,
        sum_log_pb: &Tensor,
        terminal_rewards: &Tensor,
    ) -> GfnResult<(Tensor, Tensor)>;
}
```

### C.5 Module 3B & Module 4 Interfaces: `gfn-symbolic` & `gfn-verifier`
```rust
use alloy_primitives::{Bytes, U256};
use gfn_common::error::GfnResult;
use gfn_common::schemas::{TrajectoryRecord, VerifiedFinding};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtStatus {
    Sat,
    Unsat,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SmtResolution {
    pub status: SmtStatus,
    pub concrete_witness_calldata: Option<Bytes>,
    pub unsat_core_hash: Option<[u8; 16]>, // xxh3_128 digest
    pub m_unsat_multiplier: f32,           // 1.0 for Sat, 1e-3 for Unsat, 0.5 for Unknown
    pub solve_time_ms: f64,
}

pub struct PortfolioSmtSolver {
    timeout_ms: u64,
    // Internal L1 DashMap + L2 heed::Env cache & rayon solver pool
}

impl PortfolioSmtSolver {
    pub fn new(lmdb_cache_path: &Path, timeout_ms: u64, num_workers: usize) -> GfnResult<Self>;

    pub fn solve_branch_violation(
        &self,
        state_storage_snapshot: &BTreeMap<u64, U256>,
        target_function_sig: &str,
        path_constraints_smt2: &str,
        negated_invariant_smt2: &str,
    ) -> GfnResult<SmtResolution>;
}

pub struct ConcreteForgeVerifier {
    project_root: std::path::PathBuf,
}

impl ConcreteForgeVerifier {
    pub fn new(project_root: &Path) -> Self;

    /// Writes `ExploitPoC.t.sol`, executes `forge test --offline`, synthesizes AST patch,
    /// and validates non-regression. Returns `Err(GfnError::FalsePositiveRejected)` if `forge test` fails.
    pub async fn verify_and_synthesize(
        &self,
        candidate_trajectory: &TrajectoryRecord,
        violated_invariant_id: &str,
    ) -> GfnResult<VerifiedFinding>;
}
```

---

## Section D: Unit & Integration Testing Suites with Deterministic Exit Criteria

Every milestone is gated by `cargo test --workspace --release` and `cargo clippy --workspace --all-targets -- -D warnings`. All tests run deterministically with fixed `ChaCha8Rng` seed `42` and zero external network calls.

### D.1 Unit Test Suite Specifications (Per-Crate `#[cfg(test)]` & `benches/`)

| Crate Target | Test / Benchmark Module | Concrete Test Assertions | Deterministic Exit Criteria |
| :--- | :--- | :--- | :--- |
| `gfn-evm` | `benches/revm_tps_bench.rs` & `arena::tests` | 1. Assert `RevmArena::reset_all_to_s0()` produces identical `state_hashes` across all $B=64$ slots.<br>2. Assert `snapshot_env` and `restore_snapshot` restore modified storage slots bit-for-bit.<br>3. Benchmark 100,000 steps across 8 `rayon` threads and assert throughput $\ge 65,000\text{ TPS/thread}$. | `cargo test -p gfn-evm` exits `0`; memory RSS delta over 10,000 CoW fork cycles is `< 5 MB`. |
| `gfn-extractor` | `hsdg_builder::tests` | 1. Assert `storage_x.size() == [N_s, 64]`, `function_x.size() == [N_f, 64]`, `condition_x.size() == [N_c, 48]`, `ext_call_x.size() == [N_e, 32]`.<br>2. Pass `U256::MAX` to `u256_to_log2_norm` and assert `(val - 1.0).abs() < 1e-6` and `storage_x.isfinite().all()` is `true`.<br>3. Assert EIP-1967 proxy storage merges implementation `FunctionNode` edges onto proxy `StorageNode`s. | `cargo test -p gfn-extractor` exits `0`; $100\%$ finite tensors; line coverage $\ge 92\%$. |
| `gfn-invariants` | `hgt::tests` | 1. Run `HgtInvariantSynthesizer` on `VaultErc4626Fixture` and assert Solvency ($\sum \text{shares} \times \text{price} \le \text{assets}$) and PrecisionNeutrality invariants are emitted with confidence $\ge 0.90$.<br>2. Assert candidate invariants violated during benign warmup are filtered out ($0\%$ false invariant lock-in). | `cargo test -p gfn-invariants` exits `0`; all 4 canonical invariant templates verified on synthetic state tensors. |
| `gfn-engine` | `tb_loss::tests` | 1. Construct a synthetic 4-depth EVM DAG with 3 distinct high-reward terminal exploit states ($R = 100.0$) and 61 dead-end states ($R = 0.01$).<br>2. Train `StateGraphRgatEncoder` + `AutoregressiveForwardPolicyHead` with `TrajectoryBalanceLoss` for 500 steps.<br>3. Assert $\mathcal{L}_{TB}$ decreases by $\ge 95\%$, Spearman $\rho(\log P_F(\tau), \log R(\tau)) \ge 0.90$, and all 3 modes are sampled within 100 rollouts (whereas `PpoRgatAgent` collapses to 1 mode). | `cargo test -p gfn-engine` exits `0`; `sum_log_pf.kind() == Kind::Float`; zero `NaN` gradients under `Kind::BFloat16` forward pass. |
| `gfn-symbolic` | `solver::tests` | 1. Submit a narrow 256-bit arithmetic guard `(bvudiv (bvmul x #x...2710) #x...26f2) == #x...cc07c9` and `(bvurem x #x...11) == 3` to `PortfolioSmtSolver`.<br>2. Assert `status == SmtStatus::Sat`, `m_unsat_multiplier == 1.0`, and concrete witness satisfies the EVM branch.<br>3. Submit contradictory guard `x > 100 && x < 50` and assert `status == SmtStatus::Unsat`, `m_unsat_multiplier == 1e-3`, and subsequent query hits `unsat_cache` in `< 0.05 ms`. | `cargo test -p gfn-symbolic` exits `0`; Bitwuzla/Z3 portfolio returns exact 32-byte big-endian `U256` witness; L1/L2 cache hit verified. |
| `gfn-verifier` | `forge_gate::tests` | 1. Pass a valid flash-loan + reentrancy trajectory to `ConcreteForgeVerifier` and assert generated `ExploitPoC.t.sol` passes `forge test`.<br>2. Pass a fabricated hallucinated trajectory (where invariant holds) and assert `Err(GfnError::FalsePositiveRejected)` is returned.<br>3. Apply synthesized patch diff and assert `ExploitPoC.t.sol` reverts while `BenignUserTest.t.sol` passes. | `cargo test -p gfn-verifier` exits `0`; 0.0% False Positive enforcement verified; patch non-regression check passes. |

### D.2 Workspace Integration & Metamorphic Test Suite (`tests/`)
1. **`tests/end_to_end_audit_test.rs`**:
   - Invokes the full audit pipeline on `./tests/fixtures/multi_bug_defi_protocol` with `max_trajectories = 2000` (a 4-contract lending + AMM vault protocol containing 3 distinct vulnerabilities: First-Depositor ERC-4626 rounding inflation, Read-Only Reentrancy via pool callback, and Unprotected Governance Timelock bypass).
   - **Deterministic Exit Criteria**:
     - All 3 distinct vulnerabilities are discovered (`result.verified_findings.len() == 3`).
     - Every generated `ExploitPoC_<id>.t.sol` passes concrete `forge test` execution (`FPR == 0.0%`).
     - All 3 generated unified diff patches compile cleanly and block their respective PoCs without breaking the 12 benign protocol unit tests.
2. **`tests/metamorphic_robustness_test.rs`**:
   - Uses `proptest` to apply `ObfuscateNames` and `PermuteStorage` transformations to `multi_bug_defi_protocol` and re-runs the audit with fixed seed `42`.
   - **Deterministic Exit Criteria**: All 3 vulnerabilities are still discovered within 2,000 trajectories with identical SMT witness satisfaction.
