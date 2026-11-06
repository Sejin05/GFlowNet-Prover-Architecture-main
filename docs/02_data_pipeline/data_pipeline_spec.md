# Data Engineering & Pipeline Specification: GFlowNet-Prover (Pure Rust / Foundry Edition)

## 1. Data Ingestion Pipeline & Artifact Extraction

The data pipeline is implemented entirely in Rust (`crates/gfn-extractor`), ingesting raw multi-contract Solidity repositories and transforming them into two primary mathematical structures:
1. **Static Heterogeneous State-Dependency Graph (HSDG)** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ constructed in-memory via `petgraph::stable_graph::StableDiGraph`, converted to contiguous Compressed Sparse Row (CSR) / Coordinate (COO) `tch::Tensor` structures, and persisted via zero-copy `rkyv` archives inside `heed` (LMDB) shards.
2. **Dynamic EVM State Tensors** $s_t$ and **Formal Safety Invariant Specifications** $\mathcal{I}(s)$ evaluated natively at every transaction step.

### 1.1 Programmatic Foundry & Solar AST/CFG Extraction
For each target repository root `project_root: &Path`, the Rust `ProjectExtractor` executes the following deterministic sequence without spawning external Python processes:
1. **Compiler Resolution (`svm-rs` + `foundry-config`)**: Parse `foundry.toml` and `.sol` pragma headers via `semver`; bind the exact `solc` binary (`0.8.0` to `0.8.28`) via `foundry_compilers::Project`.
2. **AST, Bytecode & Storage Layout Generation**:
   Configure `foundry_compilers::ProjectPathsConfig` and `SolcConfig` with `OutputSelection` requesting `ast`, `abi`, `evm.deployedBytecode.object`, and `storageLayout`.
3. **Native Rust AST & Bytecode CFG Lifting (`solar-parse` + `revm-interpreter`)**:
   - Parse the Solidity source units using `solar_parse::Parser` to resolve inheritance hierarchies, modifier execution chains, and unchecked arithmetic blocks.
   - Lift deployed EVM bytecode and AST statements into a Static Single Assignment (SSA) Control Flow Graph (`petgraph::stable_graph::StableDiGraph<BasicBlockNode, CfgEdge>`) with dominance frontier computation (`petgraph::algo::dominators::simple_fast`) to extract `ReadsFrom`, `WritesTo`, `Guards`, and `Invokes` semantic edges.

---

## 2. Rust Struct Schemas & Field-Level Data Types

All pipeline schemas are strongly typed at compile time using `serde`, `rkyv` (for zero-copy deserialization), and `alloy-primitives`.

### 2.1 Raw Foundry Artifact Schema (`FoundryContractArtifact`)
Defined in `crates/gfn-common/src/schemas.rs`:

```rust
use alloy_json_abi::JsonAbi;
use alloy_primitives::{Bytes, B256};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct FoundryContractArtifact {
    pub project_id: Uuid,                        // Deterministic UUIDv5 from repo tree SHA-256
    pub contract_name: String,                   // Validated regex: ^[A-Za-z_][A-Za-z0-9_]*$
    pub source_path: String,                     // Relative POSIX path (e.g., "src/vaults/LendingPool.sol")
    pub solc_version: String,                    // Exact compiler string (e.g., "0.8.24+commit.e11b9ed9")
    pub deployed_bytecode_hash: B256,            // Keccak256 digest of deployed bytecode
    pub deployed_bytecode: Vec<u8>,              // Raw EVM deployed bytecode (len >= 2)
    pub abi_json_str: String,                    // Canonical serialized Ethereum Contract ABI
    pub storage_slots: Vec<StorageSlotRecord>,   // Ordered storage layout records
}
```

### 2.2 Storage Slot Schema (`StorageSlotRecord`)
| Struct Field | Rust Type | Width / Range | Description & Packing Rules |
| :--- | :--- | :--- | :--- |
| `slot_id` | `u64` (`U256` base) | $[0, 2^{64}-1]$ | Base storage slot index assigned by `solc` (or synthetic hash index for dynamic slots). |
| `byte_offset` | `u8` | $[0, 31]$ | Starting byte offset within the 32-byte EVM word for packed variables. |
| `byte_size` | `u8` | $[1, 32]$ | Number of bytes occupied in the slot (e.g., `uint128` = 16, `address` = 20, `bool` = 1). |
| `solidity_type` | `SoliditySlotType` (`#[repr(u8)]`) | Enum ($0..18$) | Normalized type: `Uint256`, `Uint128`, `Int256`, `Address`, `Bool`, `Bytes32`, `MappingAddressUint`, `Struct`, `DynamicArray`, etc. |
| `is_dynamic_mapping` | `bool` | `{true, false}` | `true` if slot is computed via `keccak256(abi.encode(key, base_slot))`. |
| `semantic_role` | `StorageSemanticRole` (`#[repr(u8)]`) | Enum ($0..8$) | Classified role: `0:AssetBalance`, `1:TotalSupply`, `2:PoolReserve`, `3:DebtShare`, `4:OraclePrice`, `5:AdminRole`, `6:ReentrancyLock`, `7:GenericState`. |

---

## 3. Heterogeneous State-Dependency Graph (HSDG) Tensor Specifications

The HSDG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is encapsulated in a Rust `HeteroGraphTensors` struct holding type-specific node feature matrices (`tch::Tensor`) and relation-specific COO/CSR sparse index tensors (`edge_index` of shape $[2, E_{\text{rel}}]$, `Kind::Int64`).

### 3.1 Node Feature Schemas ($\mathcal{V}$)
All node feature matrices are projected into a common hidden dimension $d = 512$ (`Kind::BFloat16` during GPU forward passes, `Kind::Float` on host/disk) via type-specific `tch::nn::Linear` stems, initialized from the following raw feature layouts:

#### 1. `StorageNode` (`storage_x`, shape: $[N_{\text{storage}}, 64]$, `Kind::Float`)
- **Cols `[0..18]`**: One-hot encoding of `SoliditySlotType` (18 categories).
- **Cols `[18..26]`**: One-hot encoding of `StorageSemanticRole` (8 categories).
- **Cols `[26..28]`**: Normalized packing coordinates: `f32::from(byte_offset) / 32.0` and `f32::from(byte_size) / 32.0`.
- **Cols `[28..32]`**: Flags: `[is_dynamic_mapping, is_immutable, is_constant, is_proxy_implementation_slot]`.
- **Cols `[32..64]`**: **Dynamic Runtime State Feature Vector** (computed in Rust SIMD worker threads at every step $t$ from `alloy_primitives::U256` $v_t$):
  - `[32..40]`: Exact log-scaled magnitude features of the 256-bit value $v \in [0, 2^{256}-1]$ computed via `U256::bit_len()` and top-64-bit mantissa extraction:
    $$\phi_{\text{mag}}(v) = \frac{\log_2(v + 1)}{256.0}, \quad \phi_{\text{delta}}(v_t, v_{t-1}) = \text{sign}(v_t - v_{t-1}) \cdot \frac{\log_2(|v_t - v_{t-1}| + 1)}{256.0}$$
  - `[40..48]`: Boundary indicator flags (exact match to `U256::ZERO`, `U256::from(1)`, `U128::MAX`, `U256::MAX`, initial $s_0$ value, or known actor addresses).
  - `[48..64]`: 16-dimensional sinusoidal positional encoding of the 32-byte word split into sixteen `u16` chunks (`v.to_be_bytes::<32>()`).

#### 2. `FunctionNode` (`function_x`, shape: $[N_{\text{func}}, 64]$, `Kind::Float`)
- **Cols `[0..4]`**: Visibility one-hot (`External`, `Public`, `Internal`, `Private`).
- **Cols `[4..8]`**: State mutability one-hot (`Payable`, `NonPayable`, `View`, `Pure`).
- **Cols `[8..16]`**: Access modifier & structural flags (`has_only_owner_or_role`, `has_non_reentrant`, `is_fallback_or_receive`, `is_flashloan_callback`, `contains_inline_assembly`, `contains_unchecked_math`, `emits_events`, `is_constructor_or_init`).
- **Cols `[16..24]`**: Structural complexity metrics (log-normalized CFG cyclomatic complexity, basic block count, external call count, storage write count, argument count, return count, loop depth, opcode count).
- **Cols `[24..64]`**: 40-dimensional deterministic projection of the 4-byte function selector $f_{\text{sel}}$ (`[u8; 4]`) combined with mean-pooled EVM/SSA opcode histogram across the function's CFG basic blocks.

#### 3. `ConditionNode` (`condition_x`, shape: $[N_{\text{cond}}, 48]$, `Kind::Float`)
- **Cols `[0..6]`**: Guard origin one-hot (`Require`, `RevertCustomError`, `Assert`, `IfBranch`, `ModifierCheck`, `LoopBound`).
- **Cols `[6..16]`**: Comparison & arithmetic operator profile (`EQ`, `NEQ`, `LT`, `LE`, `GT`, `GE`, `MUL_DIV`, `MOD`, `BITWISE_AND_OR`, `CALLER_CHECK`).
- **Cols `[16..24]`**: Constant threshold scale features ($\log_2$ magnitude of constant `U256` literals in the branch predicate).
- **Cols `[24..32]`**: **Dynamic Runtime Branch Telemetry** (updated at step $t$ by `revm::Inspector`):
  - `[24]`: `branch_hit_frequency` in current trajectory $\tau$.
  - `[25]`: `branch_distance_metric` $\in [0.0, 1.0]$ computed on `JUMPI` (`0x57`) operands `(lhs: U256, rhs: U256)`:
    $$d_{\text{branch}}(lhs, rhs) = 1.0 - \frac{\log_2(|lhs - rhs| + 1)}{256.0}$$
  - `[26]`: `smt_status_flag` (`0.0`: unqueried, `0.5`: `SAT` witness known, `-1.0`: `UNSAT` proven for current prefix).
  - `[27..32]`: Dynamic constraint complexity indicators (symbolic variable count, AST depth, non-linear `MUL`/`DIV`/`KECCAK256` count).
- **Cols `[32..48]`**: 16-dimensional structural embedding of the AST/SSA predicate tree.

#### 4. `ExternalCallNode` (`ext_call_x`, shape: $[N_{\text{ext}}, 32]$, `Kind::Float`)
- **Cols `[0..6]`**: Call opcode type one-hot (`CALL`, `DELEGATECALL`, `STATICCALL`, `CALLCODE`, `CREATE`, `CREATE2`).
- **Cols `[6..14]`**: Target classification (`Erc20Transfer`, `Erc20TransferFrom`, `UniswapV2V3Pool`, `ChainlinkOracle`, `FlashloanProvider`, `UserCallbackHook`, `KnownProtocolContract`, `ArbitraryExternalAddress`).
- **Cols `[14..18]`**: Reentrancy risk indicators (`writes_storage_after_call`, `forwards_all_gas`, `sends_native_eth`, `target_is_user_controlled`).
- **Cols `[18..32]`**: Target selector and gas/value flow descriptors.

### 3.2 Edge Index & Edge Attribute Schemas ($\mathcal{E}$)
Every relation $r \in \{0..7\}$ (`ReadsFrom`, `WritesTo`, `Guards`, `Invokes`, plus their 4 reverse relations `Rev*`) stores `edge_index[r]` of shape $[2, E_r]$ (`Kind::Int64`) and `edge_attr[r]` of shape $[E_r, 16]$ (`Kind::Float`):

1. `EdgeRelation::ReadsFrom` (`Function -> Storage` and `Condition -> Storage`):
   - `edge_attr`: Encodes CFG basic block depth of the `SLOAD`, whether the read occurs inside a natural loop, and whether the read feeds a division denominator or multiplication.
2. `EdgeRelation::WritesTo` (`Function -> Storage`):
   - `edge_attr`: Encodes mutation operator (`Assign`, `AddAssign`, `SubAssign`, `MulAssign`, `Delete`), basic block index, and whether the write occurs before or after an `ExternalCallNode` (Checks-Effects-Interactions ordering).
3. `EdgeRelation::Guards` (`Condition -> Function` and `Condition -> Storage`):
   - `edge_attr`: Encodes control-dependence polarity (`True` branch vs `False` branch dominance) and dominator tree distance in `petgraph`.
4. `EdgeRelation::Invokes` (`Function -> Function` and `Function -> ExternalCall`):
   - `edge_attr`: Encodes call site order within the caller function, value transfer flag (`msg.value > 0`), and calldata taint propagation mask.

---

## 4. Automated Data Validation Rules (`HsdgValidator`)

Every extracted `HeteroGraphTensors` and `RevmArena` snapshot must pass `HsdgValidator::validate(&graph)` before entering training or auditing:

1. **Graph Connectivity & Non-Emptiness Rule**:
   - `n_storage >= 1` and `n_function >= 1`. Pure libraries/interfaces with zero state slots are inlined into calling contracts during CFG construction.
   - Every edge $(u, v)$ in `edge_index[r]` must satisfy $0 \le u < N_{\text{src}(r)}$ and $0 \le v < N_{\text{dst}(r)}$. Out-of-bounds indices return `Err(GfnError::InvalidGraphTopology)`.
2. **Strict `U256`-to-`f32` Log-Scaling Rule (Zero `NaN`/`Inf` Guarantee)**:
   - Because EVM `U256::MAX` ($\approx 1.1579 \times 10^{77}$) overflows IEEE-754 `f32::MAX` ($\approx 3.4028 \times 10^{38}$), **direct casting of `U256` to `f32` is impossible at compile time** (`U256` does not implement `Into<f32>`).
   - Instead, `u256_to_log2_norm(val: U256) -> f32` computes:
     ```rust
     #[inline]
     pub fn u256_to_log2_norm(val: U256) -> f32 {
         if val.is_zero() {
             return 0.0_f32;
         }
         let bit_len = val.bit_len() as u32; // 1..=256
         let shift = bit_len.saturating_sub(53);
         let top_bits: u64 = (val >> shift).to::<u64>();
         let log2_exact = (top_bits as f64).log2() + (shift as f64);
         (log2_exact / 256.0_f64).clamp(0.0, 1.0) as f32
     }
     ```
   - All tensors are verified via `tensor.isfinite().all().int64_value(&[]) == 1`.
3. **Storage Layout Alignment Verification**:
   - For every packed storage slot `slot_id`, the sum of `byte_size` across variables sharing `slot_id` must satisfy $\sum \text{byte\_size} \le 32$, and byte intervals $[\text{byte\_offset}, \text{byte\_offset} + \text{byte\_size})$ must be mutually disjoint.
4. **Foundry Harness Determinism Check**:
   - The initial state $s_0$ produced by executing `setUp()` in `RevmArena` is executed twice with identical block parameters (`block.number = 18_000_000`, `block.timestamp = 1_700_000_000`); the resulting `blake3` storage trie hashes must match bit-for-bit.

---

## 5. Data Leakage Mitigation & Cross-Validation Splitting Strategy

Smart contract datasets exhibit severe near-duplicate code cloning (e.g., thousands of Uniswap V2 and ERC-4626 forks) and temporal vulnerability disclosure patterns. To prevent data leakage across Training (Phases 1 & 2), Validation, and Held-Out Zero-Day Evaluation:

### 5.1 Three-Tier Leakage Prevention Protocol
1. **AST & Bytecode MinHash Deduplication (Fork Clustering in Rust)**:
   - Compute 128-permutation MinHash signatures (`xxh3_64` with 128 seeds) over 5-shingles of normalized EVM/SSA opcode sequences for every project repository.
   - Cluster all repositories using a lock-free Union-Find (`DisjointSet`) at a Jaccard similarity threshold of $\tau_{\text{sim}} = 0.70$.
   - **Hard Partition Rule**: All repositories belonging to the same `fork_cluster_id` are assigned strictly to the same split (`Train`, `Val`, or `Test`).
2. **Temporal Cutoff Boundary**:
   - Phase 1 (Pre-training) and Phase 2 (Bug-Discovery Training) datasets are strictly restricted to contracts deployed or exploits disclosed prior to `2025-06-01T00:00:00Z`.
   - The Validation split covers `2025-06-01` to `2025-12-31`.
   - The Held-Out Test split (Zero-Day Evaluation Benchmark) comprises exploits and protocol audits from `2026-01-01` onward, plus completely unseen protocol archetypes held out by `fork_cluster_id`.
3. **Zero Ground-Truth Leakage in Invariant Synthesis**:
   - During evaluation, `ProjectExtractor` strips all files matching `*Exploit*.t.sol`, `*Attack*.sol`, audit comments, and historical transaction traces before running `solar-parse` and `HgtInvariantSynthesizer`.

### 5.2 Stratified Group K-Fold Cross-Validation ($K=5$)
For hyperparameter tuning during Phase 2:
- Grouping variable: `fork_cluster_id` (ensuring zero cross-fold structural leakage).
- Stratification label: Primary vulnerability taxonomy enum (`0: ReentrancyStateDesync`, `1: FlashLoanPriceManipulation`, `2: RoundingShareInflation`, `3: AccessControlTakeover`, `4: MonotonicityViolation`).
- Each of the 5 folds preserves an exact $20\%$ allocation of each vulnerability class while keeping entire protocol families isolated within a single fold.
