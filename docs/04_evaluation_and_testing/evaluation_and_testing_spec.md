# Evaluation & Testing Specification: GFlowNet-Prover (Pure Rust / Foundry Edition)

## 1. Out-of-Sample Testing Methodology

To rigorously evaluate zero-day vulnerability discovery without memorization bias, evaluation is conducted via the Rust benchmark runner (`cargo test --workspace --release` and `cargo bench`) across three strictly disjoint out-of-sample benchmark suites:
1. **Suite A: Historical Post-Mortem Holdout (`DeFiHackLabs-2026-Holdout`, $N=120$ protocols)**:
   - Real-world multi-contract exploits occurring after the temporal training cutoff (`2026-01-01`), stripped of all exploit scripts and comments.
2. **Suite B: Multi-Modal Exploit Benchmark (`MultiBug-Foundry-Bench`, $N=50$ protocols)**:
   - Complex DeFi protocols (Lending Markets, Concentrated Liquidity AMMs, Liquid Staking Vaults, Cross-Chain Bridges) where each protocol contains between **3 and 5 distinct, orthogonal vulnerability root causes** (e.g., an ERC-4626 share inflation bug, a price oracle manipulation bug, and a timelock bypass bug in the same codebase). Used to measure **GFlowNet Mode Coverage ($\text{Modes}@K$)** against RL/PPO mode collapse.
3. **Suite C: Certified Safe Protocol Suite (`OpenZeppelin-AaveV3-Clean`, $N=40$ protocols)**:
   - Battle-tested, formally verified production contracts with no known vulnerabilities, used to verify the **0.0% False Positive Guarantee** and measure wasted SMT solver overhead on safe branches.

---

## 2. Data Slice Performance Metrics & Acceptance Targets

All models are evaluated across two primary slicing dimensions: **Vulnerability Taxonomy Slice** and **Exploit Sequence Depth Slice ($d = \text{min transactions required}$)**.

### 2.1 Performance by Vulnerability Taxonomy Slice
| Data Slice (Vulnerability Class) | Min Recall@$K=1000$ | Min Mode Coverage ($\text{Modes}@1000$) | Max Allowed False Positive Rate | Median Time-to-First-Exploit |
| :--- | :--- | :--- | :--- | :--- |
| **Slice 1: Conservation of Value / Flash-Loan Solvency** | $\ge 82.0\%$ | $\ge 85.0\%$ | $0.0\%$ | $\le 140\text{ s}$ |
| **Slice 2: Rounding & Precision Neutrality (ERC-4626 / AMM)** | $\ge 88.0\%$ | $\ge 90.0\%$ | $0.0\%$ | $\le 75\text{ s}$ |
| **Slice 3: Cross-Contract Reentrancy & Read-Only Reentrancy** | $\ge 85.0\%$ | $\ge 85.0\%$ | $0.0\%$ | $\le 110\text{ s}$ |
| **Slice 4: State Machine Monotonicity & Governance Lifecycle** | $\ge 78.0\%$ | $\ge 80.0\%$ | $0.0\%$ | $\le 190\text{ s}$ |
| **Slice 5: Privilege Escalation & Access Control Isolation** | $\ge 94.0\%$ | $\ge 92.0\%$ | $0.0\%$ | $\le 35\text{ s}$ |

### 2.2 Performance by Transaction Sequence Depth Slice ($d$)
| Trajectory Depth Slice ($d$ steps) | Standalone Foundry Fuzzer Baseline | Standalone Concolic/SMT Baseline | PPO + RGAT RL Baseline (`PpoRgatAgent`) | **GFlowNet-Prover Target** |
| :--- | :--- | :--- | :--- | :--- |
| **Shallow ($1 \le d \le 2$)** | $85.0\%$ | $96.0\%$ | $92.0\%$ | **$\ge 98.0\%$** |
| **Medium ($3 \le d \le 5$)** | $34.0\%$ | $28.0\%$ (Path Explosion) | $68.0\%$ (Mode Collapse) | **$\ge 86.0\%$** |
| **Deep ($6 \le d \le 10$)** | $4.0\%$ | $0.0\%$ (Timeout) | $31.0\%$ | **$\ge 76.0\%$** |
| **Ultra-Deep ($11 \le d \le 15$)** | $0.0\%$ | $0.0\%$ (Timeout) | $8.0\%$ | **$\ge 55.0\%$** |

---

## 3. Error Taxonomy Analysis

Every failure mode during evaluation is automatically categorized into a 5-variant Rust enum `EvaluationErrorKind` serialized to `evaluation_error_report.json`:

1. **`EvaluationErrorKind::InvariantSynthesisMiss`**:
   - *Symptom*: The GFlowNet explores the vulnerable state transition, but no synthesized invariant $\mathcal{I}_m(s_n)$ in `crates/gfn-invariants` triggers on the corrupted state.
   - *Diagnostic Signal*: Ground-truth PoC state $s^*$ satisfies $\mathcal{I}(s^*) = \text{true}$.
   - *Remediation*: Retrain `HgtInvariantSynthesizer` with augmented negative state pairs and expand domain invariant templates.
2. **`EvaluationErrorKind::FlowUnderexplored`**:
   - *Symptom*: The required sequence of `FunctionNode` calls is never sampled within $K=5000$ trajectories.
   - *Diagnostic Signal*: Zero trajectories in `PrioritizedReplayBuffer` reach the target `ConditionNode` dominator frontier.
   - *Remediation*: Increase RND novelty coefficient $\beta$ and off-policy replay temperature $\tau_{\text{temp}}$.
3. **`EvaluationErrorKind::SmtSolverTimeout`**:
   - *Symptom*: GFlowNet reaches the exact guarded branch, hands over to `gfn-symbolic` (`Bitwuzla` / `Z3`), and the solver times out at $5,000\text{ ms}$.
   - *Diagnostic Signal*: `smt_status_flag == SmtStatus::Unknown` on the critical `ConditionNode` due to nested `keccak256` or 256-bit polynomial division.
   - *Remediation*: Apply concretization of non-critical storage variables in `crates/gfn-symbolic/src/concretizer.rs` before invoking Bitwuzla.
4. **`EvaluationErrorKind::RevmForgeMismatch`**:
   - *Symptom*: Candidate exploit succeeds in `RevmArena` ($R_{\text{exploit}} = 1.0$) but fails concrete verification in `gfn-verifier` `forge test`.
   - *Diagnostic Signal*: `gfn_forge_verification_rejected_total > 0` (caused by gas metering discrepancy or precompile state difference).
   - *Remediation*: Enforce exact `revm::primitives::SpecId::CANCUN` parity and synchronize `block.basefee` / warm storage access lists (`EIP-2929`) between `crates/gfn-evm` and `foundry.toml`.
5. **`EvaluationErrorKind::PatchRegression`**:
   - *Symptom*: Counterexample-guided patch blocks the exploit (`ExploitPoC.t.sol` reverts as expected) but causes an existing unit test in the target repository's `test/` directory to fail.
   - *Diagnostic Signal*: Non-zero exit code on `forge test --exclude-match-contract ExploitPoC`.
   - *Remediation*: Refine minimal negated precondition strengthening in `crates/gfn-verifier/src/patch_synthesizer.rs` to include caller/state specificity.

---

## 4. Perturbation Robustness & Metamorphic Testing (`proptest`)

To verify that the 6-layer RGAT and HGT Invariant Synthesizer rely on deep semantic state dependencies rather than superficial syntactic artifacts, every model checkpoint must pass **4 Semantics-Preserving Metamorphic Perturbation Tests** (generated via `proptest = "1.5.0"` over `solar-ast` trees) on `MultiBug-Foundry-Bench` with **$\le 2.0\%$ degradation in $\text{Modes}@1000$**:

1. **Perturbation 1: Identifier & ABI Obfuscation (`ObfuscateNames`)**:
   - All contract names, function names, and state variable names are hashed to random hexadecimal identifiers (`func_a91f3c(...)`, `slot_77b201`), altering all 4-byte function selectors $f_{\text{sel}}$ while keeping the `petgraph` HSDG topology and storage layout identical.
2. **Perturbation 2: Storage Slot Reordering & Dead-Slot Injection (`PermuteStorage`)**:
   - Inject 15 dummy `uint256` storage variables interleaved across the contract inheritance hierarchy, shifting physical `slot_id` indices while preserving `ReadsFrom` and `WritesTo` graph edges.
3. **Perturbation 3: Control-Flow Graph Equivalence (`SplitBasicBlocks`)**:
   - Rewrite `require(A && B)` statements in `solar-ast` into nested `if (!A) revert(); if (!B) revert();` structures and insert no-op arithmetic instructions, testing invariance of the `ConditionNode` and `Guards` edge representations.
4. **Perturbation 4: Initial Liquidity Scale Shift (`ScaleBalances1000x`)**:
   - Multiply all initial token endowments and pool reserves in `setUp()` by $10^3$ and $10^{-3}$, verifying that `u256_to_log2_norm(v)` and symbolic domain partitioning generalize across asset decimal scales (`USDC` 6 decimals vs `WETH` 18 decimals).

---

## 5. Edge-Case Handling Protocols

| Edge-Case Scenario | Structural Challenge | Mandatory Handling Protocol |
| :--- | :--- | :--- |
| **EIP-1967 / UUPS / Diamond Proxies (EIP-2535)** | `DELEGATECALL` executes implementation bytecode against the proxy's storage trie; static AST inspection sees empty proxy storage. | `crates/gfn-extractor/src/proxy_resolver.rs` detects EIP-1967 slots (`0x360894...`) and Diamond `facetAddress` mappings during `setUp()` execution in `revm`, merging the implementation contract's `FunctionNode`s and `ConditionNode`s directly onto the proxy's `StorageNode`s. |
| **Flash-Loan & ERC-777 / ERC-1363 Callbacks** | Exploit requires deploying an attacker contract that implements `uniswapV3FlashCallback`, `executeOperation`, or `onERC721Received` to re-enter the target protocol mid-transaction. | The `Attacker0` actor persona in `crates/gfn-evm` is pre-deployed as a universal **Dynamic Callback Router Contract** capable of executing nested GFlowNet sub-actions $(a_{t,1}, a_{t,2})$ when invoked mid-frame by a target `ExternalCallNode`. |
| **Inline Assembly (`yul`) & Custom Errors** | Direct `sload`/`sstore` and `mstore` inside `assembly { ... }` blocks bypass high-level Solidity AST variable references. | Bytecode CFG lifter in `crates/gfn-extractor/src/cfg_lifter.rs` lifts Yul `SLOAD` (`0x54`)/`SSTORE` (`0x55`) opcodes into explicit `ReadsFrom`/`WritesTo` edges linked via runtime `revm::Inspector` opcode tracing during initial warm-up. |
| **Precompiles & `keccak256` in SMT** | Symbolic execution over `KECCAK256` (`0x20`) or `ecrecover` (`0x01`) causes Bitwuzla/Z3 to hang on cryptographic inversion. | `crates/gfn-symbolic/src/concolic_evm.rs` applies **Uninterpreted Function (`UF`) + Concrete Hash Table Bijection**: known storage key hashes observed in `revm` are injected as concrete equality axioms rather than bit-blasting the Keccak-f[1600] permutation. |

---

## 6. Formal Deployment Readiness Criteria (Go / No-Go Release Gate)

A compiled GFlowNet-Prover release candidate binary is blocked from production promotion unless **all 7 automated release gates** pass in CI:

1. **Zero False Positives**: $\text{FPR} == 0.0\%$ across all $N=210$ combined test protocols (`Suite A + Suite B + Suite C`).
2. **Exploit Discovery Recall**: Recall@$K=1000 \ge 80.0\%$ on `Suite A` and Deep Sequence Recall ($d \ge 5$) $\ge 75.0\%$.
3. **Multi-Bug Mode Coverage**: $\text{Modes}@1000 \ge 85.0\%$ on `Suite B`, outperforming the `PpoRgatAgent` baseline by $\ge +25.0$ percentage points.
4. **Metamorphic Robustness**: Recall drop under `ObfuscateNames` and `PermuteStorage` perturbations is $\le 2.0\%$.
5. **Throughput & Latency Compliance**: `gfn-evm` achieves $\ge 65,000\text{ TPS/thread}$ (`criterion` benchmark) and `gfn-engine` RGAT policy inference $p_{99} \le 9.5\text{ ms}$ ($B=64$).
6. **PoC Self-Containment**: $100\%$ of emitted `ExploitPoC.t.sol` files compile and pass `forge test --match-test test_ExploitPoC` in a clean container with `--offline` enabled.
7. **Patch Non-Regression**: $\ge 90.0\%$ of synthesized counterexample patches block the exploit while passing $100\%$ of the target project's existing benign unit test suite.
