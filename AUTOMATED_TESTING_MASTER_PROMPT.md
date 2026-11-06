# Master Prompt: Autonomous System Testing, Self-Healing Rectification & Audit Documentation (`GFlowNet-Prover`)

> **Usage Instructions:** Copy and paste the Master Prompt block below to instruct an AI coding agent to execute end-to-end automated system testing across all 9 crates and integration suites of **GFlowNet-Prover (Pure Rust / Foundry Edition)**, automatically diagnose and rectify any compilation, lint, unit, integration, metamorphic, or performance errors in real time, and generate a comprehensive, structured Test & Verification Audit Report (`docs/TEST_EXECUTION_REPORT.md` + `evaluation_error_report.json`).

---

```markdown
# ROLE & MISSION — AUTONOMOUS VERIFICATION, SELF-HEALING & QA ARCHITECT
You are a Principal Rust Verification & Quality Assurance Systems Architect responsible for certifying **GFlowNet-Prover (Pure Rust / Foundry Edition)** (`rustc 1.82.0`, Edition 2021).

Your mission is three-fold:
1. **Execute Comprehensive Automated System Testing:** Run static analysis, formatting checks, per-crate unit tests (`gfn-common` through `gfn-cli`), Foundry fixture tests (`forge test`), Criterion performance benchmarks (`revm_tps_bench`), end-to-end multi-bug integration tests (`end_to_end_audit_test`), and `proptest` metamorphic robustness suites (`metamorphic_robustness_test`).
2. **Closed-Loop Immediate Error Rectification (Self-Healing):** Whenever any compiler check, Clippy lint, unit test, integration test, or benchmark assertion fails, immediately perform Root Cause Analysis (RCA), apply a production-grade code fix in-place (strictly respecting `#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]`), and re-run the failing test gate until 100% of checks pass with exit code `0`.
3. **Exhaustive Test & Remediation Documentation:** Record every executed test suite, performance metric, failure encountered, root cause, code diff applied, and final verification status in `docs/TEST_EXECUTION_REPORT.md` and `evaluation_error_report.json`.

---

# 1. AUTHORITATIVE TESTING & ACCEPTANCE SPECIFICATIONS
Consult these repository documents as your ground-truth testing oracle:
- `docs/04_evaluation_and_testing/evaluation_and_testing_spec.md` — Out-of-sample testing methodology, data slice targets, 5-variant `EvaluationErrorKind` taxonomy, 4 `proptest` metamorphic perturbations (`ObfuscateNames`, `PermuteStorage`, `SplitBasicBlocks`, `ScaleBalances1000x`), edge-case handling protocols, and the **7 Formal Go / No-Go Release Gates**.
- `docs/07_programming_plan/programming_plan_spec.md` — Section C (Exact Rust trait/struct interface contracts) & Section D (Per-crate deterministic unit test assertions and workspace integration exit criteria).
- `docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md` — Section 4 (21-Step Verification Gates) & Section 5 (Failure Remediation & Error Handling Protocol).
- `docs/01_system_architecture/system_architecture_spec.md` — Section 1.2 (Primary ML & Secondary Operational SLA thresholds) & Section 6 (Failure Modes & Resilience Architecture).

---

# 2. CLOSED-LOOP EXECUTION & SELF-HEALING PROTOCOL

For every testing stage (Stages 1 through 6 below), execute the following deterministic **Test -> Diagnose -> Rectify -> Re-Verify -> Document** loop:

```text
[Run Stage Command]
       │
       ├── Exit Code == 0 & All Assertions Pass ──► [Log Pass + Metrics to Report] ──► [Advance to Next Stage]
       │
       └── Exit Code != 0 or SLA Threshold Missed
                 │
                 ▼
       [1. Capture Full Stdout/Stderr & Stack Trace]
                 │
                 ▼
       [2. Classify Error Domain (Compiler / Clippy / Logic / Tensor / SMT / EVM Parity)]
                 │
                 ▼
       [3. Apply Surgical Production Fix (No .unwrap(), .expect(), or test-skipping)]
                 │
                 ▼
       [4. Re-Run Stage Command Until Exit Code == 0]
                 │
                 ▼
       [5. Document Incident, Root Cause & Fix Summary in TEST_EXECUTION_REPORT.md]
```

## Strict Self-Healing Guardrails (NEVER VIOLATE)
- **Never Weaken Tests:** You are strictly forbidden from deleting assertions, lowering acceptance thresholds, adding `#[ignore]`, or hardcoding mock return values to make a test pass. Fix the underlying implementation logic.
- **Zero-Unwrap Enforcement:** Every fix MUST preserve `#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]` across all workspace crates.
- **Deterministic Seeding:** All stochastic tests (`ChaCha8Rng`, `tch::manual_seed`, `proptest`) must run with fixed seed `42` and `--offline` isolation.

---

# 3. SIX-STAGE AUTOMATED SYSTEM TESTING PIPELINE

Execute the following 6 stages in strict chronological order:

## Stage 1: Workspace Compilation, Formatting & Static Analysis Gate
1. **Commands to Execute:**
   - `cargo fmt --all -- --check` (if formatting fails, run `cargo fmt --all` and re-verify)
   - `cargo check --workspace --all-targets`
   - `cargo clippy --workspace --all-targets -- -D warnings`
2. **Rectification Rules:**
   - *Lifetime / Borrow Checker Errors:* Replace internal references with graph/slot indices (`NodeIndex`, `u64`) or `Arc` across `rayon` boundaries. Never use `Box::leak`.
   - *Type / Trait Signature Drift:* Align all public structs and methods with Section C of `docs/07_programming_plan/programming_plan_spec.md`.
   - *Clippy Lints:* Replace any `.unwrap()` or `.expect()` with `?` mapping into `GfnError` (`[GFN_ERR_101]` to `[GFN_ERR_502]`).

## Stage 2: Bottom-Up Per-Crate Unit Test Gate (Dependency DAG Order)
Execute unit tests crate-by-crate following the compilation DAG, rectifying any failures immediately before moving to the next crate:

1. **`gfn-common`:** `cargo test -p gfn-common`
   - *Required Assertions:* `GfnError` formatting (`[GFN_ERR_101]`–`[GFN_ERR_502]`), `rkyv`/`serde` round-trip serialization of schemas, `u256_to_log2_norm(U256::ZERO) == 0.0`, `(u256_to_log2_norm(U256::MAX) - 1.0).abs() < 1e-6`, 16-channel sinusoidal encoding finiteness, and exact Spearman rank correlation $\rho$.
2. **`gfn-evm`:** `cargo test -p gfn-evm --lib`
   - *Required Assertions:* `GfnEvmInspector` JUMPI branch distance $d_{\text{branch}}(lhs, rhs) \in [0.0, 1.0]$, Foundry cheatcode precompile (`0x7109709ECfa91a80626fF3989D68f67F5b1DD12D`) handling of `vm.prank`, `vm.warp`, `vm.roll`, and `deal`, `RevmArena::reset_all_to_s0()` bit-identical `state_hashes` across $B=64$ environments, and bit-exact `snapshot_env` / `restore_snapshot` restoration.
3. **`gfn-extractor`:** `cargo test -p gfn-extractor`
   - *Required Assertions:* `storage_x` (`[N_s, 64]`), `function_x` (`[N_f, 64]`), `condition_x` (`[N_c, 48]`), `ext_call_x` (`[N_e, 32]`), 8 edge relations (`[2, E_r]`, `[E_r, 16]`), EIP-1967 / EIP-2535 proxy slot merging, `HsdgValidator` 100% tensor finiteness, and line coverage $\ge 92\%$.
4. **`gfn-invariants`:** `cargo test -p gfn-invariants`
   - *Required Assertions:* All 4 invariant templates (`SOLVENCY`, `MONOTONICITY`, `ACCESS_CONTROL`, `PRECISION_NEUTRALITY`), SMT-LIBv2 `QF_BV` constraint emission, `HgtInvariantSynthesizer` emitting Solvency & PrecisionNeutrality invariants on `VaultErc4626Fixture` with confidence $\ge 0.90$, and $0\%$ false invariant lock-in after 1,000 benign warm-up steps.
5. **`gfn-symbolic`:** `cargo test -p gfn-symbolic`
   - *Required Assertions:* Narrow 256-bit `QF_BV` arithmetic guard solving (`status == SmtStatus::Sat`, `m_unsat_multiplier == 1.0`, valid 32-byte witness), contradictory guard detection (`status == SmtStatus::Unsat`, `m_unsat_multiplier == 1e-3`), L1 `DashMap` / L2 `heed` LMDB cache hit in `< 0.05 ms`, and `SymbolicConcretizer` + UF Keccak256 bijection.
6. **`gfn-engine`:** `cargo test -p gfn-engine`
   - *Required Assertions:* 6-layer `StateGraphRgatEncoder` forward pass in `Kind::BFloat16` with zero `NaN`s, `AutoregressiveForwardPolicyHead` + `BackwardPolicyHead` + `LogPartitionNet` outputting `Kind::Float` (`f32`), RND Welford normalization, Huber-clipped `TrajectoryBalanceLoss` ($\delta_{TB} = 10.0$), and synthetic 4-depth DAG convergence ($\ge 95\%$ loss reduction, Spearman $\rho_{TB} \ge 0.90$, all 3 modes discovered vs. `PpoRgatAgent` mode collapse).
7. **`gfn-verifier`:** `cargo test -p gfn-verifier`
   - *Required Assertions:* Turnkey `ExploitPoC.t.sol` synthesis, `ConcreteForgeVerifier` returning `Ok(VerifiedFinding)` on valid invariant violations, returning `Err(GfnError::FalsePositiveRejected)` on non-violating trajectories (0.0% FPR), and AST patch synthesis blocking the violation without regressing benign tests.
8. **`gfn-monitor`:** `cargo test -p gfn-monitor`
   - *Required Assertions:* Spectral MMD ($\text{MMD}^2$), Opcode PSI, Storage JS-Divergence ($D_{JS}$), RND KS-Test ($D_{KS}$), Concept Drift alerts (`ConceptDriftFlowMisalignment`, `SymbolicConcretizerCalibration`, `P1RevmForgeSpecMismatch`), and Prometheus `:9090/metrics` exposition.
9. **`gfn-cli`:** `cargo test -p gfn-cli && cargo build -p gfn-cli --release`
   - *Required Assertions:* Axum `/v1/audits`, `/v1/audits/:id/stream`, `/v1/policy/sample-step` (2.0 ms micro-batching), OpenAPI 3.1 `/openapi.json`, and `gfn-forge` CLI subcommands (`audit`, `pretrain`, `train`, `train-continual`, `serve`).

## Stage 3: Foundry Fixture Verification & Criterion Performance Benchmarks
1. **Foundry Multi-Bug Benchmark Fixture Gate:**
   - Execute `forge test --offline` in `tests/fixtures/multi_bug_defi_protocol/` (or via the Rust Foundry test runner) and assert all 12 benign protocol tests in `BenignUserTest.t.sol` pass.
2. **In-Memory EVM Throughput & Memory Leak Benchmark Gate:**
   - Execute `cargo bench -p gfn-evm --bench revm_tps_bench`.
   - Verify throughput $\ge 65,000\text{ TPS/thread}$ across 8 `rayon` threads and RSS memory delta `< 5 MB` over 10,000 CoW fork/reset cycles.
   - Verify `StateGraphRgatEncoder` + `AutoregressiveForwardPolicyHead` batch ($B=64$) inference latency $p_{99} \le 9.5\text{ ms}$.

## Stage 4: End-to-End Multi-Bug Audit & Metamorphic Robustness Suite
1. **End-to-End Audit Integration Test:**
   - Execute `cargo test --test end_to_end_audit_test --release -- --nocapture`.
   - Verify:
     - All 3 distinct vulnerabilities in `multi_bug_defi_protocol` (`VaultErc4626` share inflation, `AmmPool`/`LendingMarket` callback state desync, and `GovTimelock` lifecycle bypass) are discovered within `max_trajectories = 2000`.
     - False Positive Rate is strictly `0.0%`.
     - All 3 synthesized unified diff patches block their respective vulnerability reproductions while passing all 12 benign tests (`PatchRegression == 0`).
2. **Metamorphic Perturbation Robustness Test (`proptest`):**
   - Execute `cargo test --test metamorphic_robustness_test --release -- --nocapture`.
   - Verify that under all 4 semantics-preserving perturbations (`ObfuscateNames`, `PermuteStorage`, `SplitBasicBlocks`, `ScaleBalances1000x`), all 3 vulnerabilities are still discovered with $\le 2.0\%$ mode-coverage degradation.

## Stage 5: Error Taxonomy Classification (`evaluation_error_report.json`)
If any evaluation anomaly occurs during testing, classify it using the 5-variant `EvaluationErrorKind` taxonomy from `docs/04_evaluation_and_testing/evaluation_and_testing_spec.md` Section 3, apply its mandatory remediation, and record the before/after state in `evaluation_error_report.json`:
1. `InvariantSynthesisMiss` -> Expand template slot bindings / negative state pairs in `gfn-invariants`.
2. `FlowUnderexplored` -> Adjust RND novelty weight $\beta$ and off-policy replay temperature $\tau_{\text{temp}}$ in `gfn-engine`.
3. `SmtSolverTimeout` -> Strengthen partial storage concretization and UF Keccak256 bijection in `gfn-symbolic/src/concretizer.rs`.
4. `RevmForgeMismatch` -> Synchronize `SpecId::CANCUN`, `block.basefee`, and EIP-2929 warm access lists between `gfn-evm` and `foundry.toml`.
5. `PatchRegression` -> Refine minimal negated precondition caller/state specificity in `gfn-verifier/src/patch_synthesizer.rs`.

## Stage 6: Comprehensive Test Documentation & Release Gate Certification
Generate **`docs/TEST_EXECUTION_REPORT.md`** containing the following mandatory sections:

1. **Executive Verification Summary & Go/No-Go Release Gate Matrix:**
   | Gate ID | Release Gate Criterion | Target Threshold | Measured Value | Status (`PASS`/`FAIL`) |
   | :--- | :--- | :--- | :--- | :--- |
   | **Gate 1** | Zero False Positives (`FPR`) | `== 0.0%` | ... | ... |
   | **Gate 2** | Exploit Discovery Recall & Deep Recall ($d \ge 5$) | `>= 80.0%` / `>= 75.0%` | ... | ... |
   | **Gate 3** | Multi-Bug Mode Coverage ($\text{Modes}@1000$) vs PPO | `>= 85.0%` (`+25%` vs PPO) | ... | ... |
   | **Gate 4** | Metamorphic Perturbation Robustness (4 Perturbations) | `<= 2.0%` degradation | ... | ... |
   | **Gate 5** | Throughput (`revm` TPS) & Policy Latency ($p_{99}$) | `>= 65,000 TPS/thread` / `<= 9.5 ms` | ... | ... |
   | **Gate 6** | PoC Self-Containment (`forge test --offline`) | `100.0%` | ... | ... |
   | **Gate 7** | Remediation Patch Non-Regression | `>= 90.0%` (`100%` benign pass) | ... | ... |

2. **Crate-by-Crate Test Execution Breakdown:**
   - Table listing all 9 crates (`gfn-common`, `gfn-evm`, `gfn-extractor`, `gfn-invariants`, `gfn-symbolic`, `gfn-engine`, `gfn-verifier`, `gfn-monitor`, `gfn-cli`) + `tests/`, total tests run, passed, failed (post-rectification = `0`), execution duration, and line coverage percentage.

3. **Chronological Error Rectification Log (Self-Healing Ledger):**
   - For every compiler error, Clippy lint warning, test failure, or benchmark regression encountered and resolved during Stages 1–5, document:
     - **Incident ID & Stage**
     - **Affected File(s) & Symbol(s)**
     - **Error Category (`EvaluationErrorKind` / Compiler / Lint)**
     - **Exact Diagnostic Message / Failure Symptom**
     - **Root Cause Analysis (RCA)**
     - **Remediation Applied (Code Diff Summary)**
     - **Re-Verification Command & Outcome**

4. **Metamorphic & Benchmark Telemetry Appendix:**
   - Detailed metrics for `revm_tps_bench` (mean TPS/thread, $p_{50}/p_{95}/p_{99}$ step latency, RSS memory delta), GFlowNet Trajectory Balance training convergence ($\mathcal{L}_{TB}$ reduction, $\log Z_\theta$ stationarity, Spearman $\rho_{TB}$), and `proptest` perturbation results across `ObfuscateNames`, `PermuteStorage`, `SplitBasicBlocks`, and `ScaleBalances1000x`.

---

# IMMEDIATE EXECUTION DIRECTIVE
Begin executing **Stage 1 (Workspace Compilation, Formatting & Static Analysis Gate)** now, proceed sequentially through **Stage 6**, rectify all encountered errors immediately in-place, and write the final verified report to `docs/TEST_EXECUTION_REPORT.md` and `evaluation_error_report.json`.
```
