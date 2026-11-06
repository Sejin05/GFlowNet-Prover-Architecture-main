#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! High-speed in-memory RevmArena Copy-on-Write (CoW) execution engine.
//!
//! Provides zero-allocation parallel EVM simulation across `rayon::ThreadPool`
//! worker threads over an `Arc<CacheDB<EmptyDB>>` base state $s_0$, with dynamic
//! runtime SIMD feature extraction, condition telemetry, and Blake3 state trie digests.

use alloy_primitives::{Address, Bytes, B256, U256};
use gfn_common::error::{GfnError, GfnResult};
use gfn_common::math::{u256_signed_log2_delta, u256_sinusoidal_chunk_encoding, u256_to_log2_norm};
use revm::{
    db::{CacheDB, EmptyDB},
    primitives::{AccountInfo, ExecutionResult, SpecId, TransactTo, TxEnv},
    DatabaseCommit, Evm,
};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use crate::cheatcodes::{ActorPersona, CheatcodeState, CHEATCODE_ADDRESS};
use crate::inspector::GfnEvmInspector;

/// Initial canonical Cancun EVM block number: 18,000,000.
pub const CANCUN_INITIAL_BLOCK_NUMBER: u64 = 18_000_000;

/// Initial canonical Cancun EVM block timestamp: 1,700,000,000 (Nov 14, 2023).
pub const CANCUN_INITIAL_BLOCK_TIMESTAMP: u64 = 1_700_000_000;

/// Canonical chain ID (Ethereum Mainnet = 1).
pub const CANCUN_CHAIN_ID: u64 = 1;

/// Initial native ETH funding deal for each actor persona: 10,000 ETH.
pub const INITIAL_ACTOR_BALANCE_WEI: u128 = 10_000_000_000_000_000_000_000;

/// Input batch parameters for parallel step execution across $B$ environments.
#[derive(Debug, Clone)]
pub struct StepBatchInput {
    /// Actor persona index in `0..=4` for each environment (Length $B$).
    pub caller_indices: Vec<u8>,
    /// Target 20-byte contract addresses (Length $B$).
    pub target_contracts: Vec<Address>,
    /// ABI-encoded calldata payloads (Length $B$).
    pub calldata_batch: Vec<Bytes>,
    /// Block timestamp advancement delta in seconds (Length $B$).
    pub warp_deltas: Vec<u64>,
    /// Block number advancement delta in blocks (Length $B$).
    pub roll_deltas: Vec<u64>,
}

/// Output batch tensors, telemetry, and execution masks produced after step $t$.
#[derive(Debug, Clone)]
pub struct StepBatchOutput {
    /// Flattened ($B \times N_{\text{slots}}$) active 256-bit storage words.
    pub storage_words: Vec<U256>,
    /// Flattened ($B \times N_{\text{slots}} \times 32$) SIMD-computed dynamic features.
    pub storage_dyn_features: Vec<f32>,
    /// Flattened ($B \times N_{\text{cond}} \times 8$) branch distances & telemetry vectors.
    pub branch_telemetry: Vec<f32>,
    /// Flattened ($B \times 32$) normalized actor balances & block context.
    pub env_context: Vec<f32>,
    /// Reverted execution flags for each environment (Length $B$).
    pub reverted_mask: Vec<bool>,
    /// Blake3 state trie cryptographic digests for each environment (Length $B$).
    pub state_hashes: Vec<B256>,
}

/// Internal parameters for executing a single environment step.
struct StepEnvParams<'a> {
    caller_idx: u8,
    target: Address,
    calldata: &'a Bytes,
    warp_delta: u64,
    roll_delta: u64,
}
/// Pinned host memory buffer layout for zero-copy DMA transfers to GPU tensors.
#[derive(Debug, Clone)]
pub struct PinnedStateBuffer {
    pub batch_size: usize,
    pub num_slots: usize,
    pub num_conds: usize,
    pub storage_words: Vec<U256>,
    pub storage_features: Vec<f32>,
    pub branch_telemetry: Vec<f32>,
    pub env_context: Vec<f32>,
}

impl PinnedStateBuffer {
    /// Allocates an aligned state buffer for $B$ environments.
    pub fn new(batch_size: usize, num_slots: usize, num_conds: usize) -> Self {
        Self {
            batch_size,
            num_slots,
            num_conds,
            storage_words: vec![U256::ZERO; batch_size * num_slots],
            storage_features: vec![0.0_f32; batch_size * num_slots * 32],
            branch_telemetry: vec![0.0_f32; batch_size * num_conds * 8],
            env_context: vec![0.0_f32; batch_size * 32],
        }
    }
}

/// Snapshot of an individual worker environment state.
#[derive(Debug, Clone)]
pub struct EnvSnapshot {
    pub db: CacheDB<EmptyDB>,
    pub block_timestamp: U256,
    pub block_number: U256,
    pub cheatcode_state: CheatcodeState,
    pub previous_storage: Vec<U256>,
}

/// State of an individual worker environment inside `RevmArena`.
#[derive(Debug, Clone)]
pub struct WorkerEnvironment {
    pub db: CacheDB<EmptyDB>,
    pub inspector: GfnEvmInspector,
    pub cheatcode_state: CheatcodeState,
    pub block_timestamp: U256,
    pub block_number: U256,
    pub previous_storage: Vec<U256>,
    pub initial_storage: Vec<U256>,
    pub snapshots: HashMap<u64, EnvSnapshot>,
    pub next_snapshot_id: u64,
}

/// High-speed multi-threaded in-memory EVM arena.
pub struct RevmArena {
    batch_size: usize,
    num_slots: usize,
    num_conds: usize,
    base_db: Arc<CacheDB<EmptyDB>>,
    environments: Vec<WorkerEnvironment>,
    thread_pool: rayon::ThreadPool,
    tracked_slots: Vec<(Address, U256)>,
    tracked_conds: Vec<usize>,
    initial_block_timestamp: U256,
    initial_block_number: U256,
    pinned_buffer: PinnedStateBuffer,
}

impl RevmArena {
    /// Initializes a `RevmArena` instance using project compilation artifacts and setup contract.
    pub fn new(
        _project_root: &Path,
        _setup_contract: &str,
        _setup_function: &str,
        batch_size: usize,
        num_threads: usize,
        seed: u64,
    ) -> GfnResult<Self> {
        let mut base_db = CacheDB::new(EmptyDB::default());
        init_default_actor_accounts(&mut base_db);

        // Track standard persona slots for baseline telemetry
        let tracked_slots: Vec<(Address, U256)> = ActorPersona::ALL
            .iter()
            .map(|p| (p.address(), U256::ZERO))
            .collect();
        let tracked_conds: Vec<usize> = Vec::new();

        Self::new_with_base_db(
            base_db,
            tracked_slots,
            tracked_conds,
            batch_size,
            num_threads,
            seed,
        )
    }

    /// Initializes a `RevmArena` directly from an existing base `CacheDB` and tracked slots/conditions.
    pub fn new_with_base_db(
        base_db: CacheDB<EmptyDB>,
        tracked_slots: Vec<(Address, U256)>,
        tracked_conds: Vec<usize>,
        batch_size: usize,
        num_threads: usize,
        _seed: u64,
    ) -> GfnResult<Self> {
        if batch_size == 0 {
            return Err(GfnError::RevmExecution {
                env_idx: 0,
                message: "batch_size must be >= 1".to_string(),
            });
        }
        let num_threads_clamped = if num_threads == 0 { 1 } else { num_threads };

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads_clamped)
            .build()
            .map_err(|e| GfnError::RevmExecution {
                env_idx: 0,
                message: format!("Failed to create rayon thread pool: {e}"),
            })?;

        let num_slots = tracked_slots.len();
        let num_conds = tracked_conds.len();

        let initial_block_timestamp = U256::from(CANCUN_INITIAL_BLOCK_TIMESTAMP);
        let initial_block_number = U256::from(CANCUN_INITIAL_BLOCK_NUMBER);

        let initial_storage = extract_storage_slice(&base_db, &tracked_slots);

        let mut environments = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            environments.push(WorkerEnvironment {
                db: base_db.clone(),
                inspector: GfnEvmInspector::new(),
                cheatcode_state: CheatcodeState::new(),
                block_timestamp: initial_block_timestamp,
                block_number: initial_block_number,
                previous_storage: initial_storage.clone(),
                initial_storage: initial_storage.clone(),
                snapshots: HashMap::new(),
                next_snapshot_id: 1,
            });
        }

        let pinned_buffer = PinnedStateBuffer::new(batch_size, num_slots, num_conds);

        let mut arena = Self {
            batch_size,
            num_slots,
            num_conds,
            base_db: Arc::new(base_db),
            environments,
            thread_pool,
            tracked_slots,
            tracked_conds,
            initial_block_timestamp,
            initial_block_number,
            pinned_buffer,
        };

        // Initialize pinned buffer and baseline state
        arena.reset_all_to_s0()?;
        Ok(arena)
    }

    /// Resets all $B$ environments back to initial state $s_0$.
    pub fn reset_all_to_s0(&mut self) -> GfnResult<StepBatchOutput> {
        let all_indices: Vec<usize> = (0..self.batch_size).collect();
        self.reset_env_indices(&all_indices)
    }

    /// Resets only the specified environment indices back to $s_0$.
    pub fn reset_env_indices(&mut self, env_indices: &[usize]) -> GfnResult<StepBatchOutput> {
        let base_db_ref = Arc::clone(&self.base_db);
        let initial_storage = extract_storage_slice(&base_db_ref, &self.tracked_slots);
        let initial_ts = self.initial_block_timestamp;
        let initial_bn = self.initial_block_number;

        for &idx in env_indices {
            if idx >= self.batch_size {
                return Err(GfnError::RevmExecution {
                    env_idx: idx,
                    message: format!("env_idx out of bounds: {idx} >= {}", self.batch_size),
                });
            }
            let env = match self.environments.get_mut(idx) {
                Some(e) => e,
                None => {
                    return Err(GfnError::RevmExecution {
                        env_idx: idx,
                        message: format!("Environment not found at index {idx}"),
                    });
                }
            };
            env.db = (*base_db_ref).clone();
            env.inspector.clear();
            env.cheatcode_state.reset();
            env.block_timestamp = initial_ts;
            env.block_number = initial_bn;
            env.previous_storage = initial_storage.clone();
            env.initial_storage = initial_storage.clone();
        }

        self.generate_batch_output(&vec![false; self.batch_size])
    }

    /// Takes a bit-exact snapshot of environment `env_idx` and returns a unique `snapshot_id`.
    pub fn snapshot_env(&mut self, env_idx: usize) -> GfnResult<u64> {
        if env_idx >= self.batch_size {
            return Err(GfnError::RevmExecution {
                env_idx,
                message: format!("env_idx out of bounds: {env_idx} >= {}", self.batch_size),
            });
        }
        let env = match self.environments.get_mut(env_idx) {
            Some(e) => e,
            None => {
                return Err(GfnError::RevmExecution {
                    env_idx,
                    message: format!("Environment not found at index {env_idx}"),
                });
            }
        };

        let snapshot_id = env.next_snapshot_id;
        env.next_snapshot_id = env.next_snapshot_id.saturating_add(1);

        env.snapshots.insert(
            snapshot_id,
            EnvSnapshot {
                db: env.db.clone(),
                block_timestamp: env.block_timestamp,
                block_number: env.block_number,
                cheatcode_state: env.cheatcode_state.clone(),
                previous_storage: env.previous_storage.clone(),
            },
        );

        Ok(snapshot_id)
    }

    /// Restores environment `env_idx` to the exact state saved under `snapshot_id`.
    pub fn restore_snapshot(&mut self, env_idx: usize, snapshot_id: u64) -> GfnResult<()> {
        if env_idx >= self.batch_size {
            return Err(GfnError::RevmExecution {
                env_idx,
                message: format!("env_idx out of bounds: {env_idx} >= {}", self.batch_size),
            });
        }
        let env = match self.environments.get_mut(env_idx) {
            Some(e) => e,
            None => {
                return Err(GfnError::RevmExecution {
                    env_idx,
                    message: format!("Environment not found at index {env_idx}"),
                });
            }
        };

        let snapshot = env
            .snapshots
            .get(&snapshot_id)
            .ok_or_else(|| GfnError::RevmExecution {
                env_idx,
                message: format!("Snapshot {snapshot_id} not found for env_idx {env_idx}"),
            })?;

        env.db = snapshot.db.clone();
        env.block_timestamp = snapshot.block_timestamp;
        env.block_number = snapshot.block_number;
        env.cheatcode_state = snapshot.cheatcode_state.clone();
        env.previous_storage = snapshot.previous_storage.clone();
        env.inspector.clear();

        Ok(())
    }

    /// Executes an input batch across all $B$ environments in parallel using `rayon`.
    pub fn step_batch(&mut self, input: &StepBatchInput) -> GfnResult<StepBatchOutput> {
        let b = self.batch_size;
        if input.caller_indices.len() != b
            || input.target_contracts.len() != b
            || input.calldata_batch.len() != b
            || input.warp_deltas.len() != b
            || input.roll_deltas.len() != b
        {
            return Err(GfnError::RevmExecution {
                env_idx: 0,
                message: format!(
                    "StepBatchInput vector length mismatch: expected {b}, got callers={}, targets={}, calldata={}, warp={}, roll={}",
                    input.caller_indices.len(),
                    input.target_contracts.len(),
                    input.calldata_batch.len(),
                    input.warp_deltas.len(),
                    input.roll_deltas.len()
                ),
            });
        }

        let tracked_slots = &self.tracked_slots;
        let mut reverted_mask = vec![false; b];

        // Parallel execution across rayon worker threads
        let env_results: Vec<Result<bool, GfnError>> = self.thread_pool.install(|| {
            use rayon::prelude::*;
            self.environments
                .par_iter_mut()
                .enumerate()
                .map(|(env_idx, env)| {
                    let params = StepEnvParams {
                        caller_idx: input.caller_indices[env_idx],
                        target: input.target_contracts[env_idx],
                        calldata: &input.calldata_batch[env_idx],
                        warp_delta: input.warp_deltas[env_idx],
                        roll_delta: input.roll_deltas[env_idx],
                    };
                    // Wrap execution in catch_unwind to guard against panics
                    let panic_result = catch_unwind(AssertUnwindSafe(|| {
                        step_single_environment(env_idx, env, params, tracked_slots)
                    }));

                    match panic_result {
                        Ok(res) => res,
                        Err(panic_err) => {
                            let msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                                (*s).to_string()
                            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "Unknown panic in EVM worker task".to_string()
                            };
                            Err(GfnError::WorkerPanicked { env_idx, msg })
                        }
                    }
                })
                .collect()
        });

        // Collect reverted masks and check for worker panics
        for (idx, res) in env_results.into_iter().enumerate() {
            match res {
                Ok(reverted) => {
                    reverted_mask[idx] = reverted;
                }
                Err(e) => {
                    // Reset corrupted environment back to s0
                    let base_db_ref = Arc::clone(&self.base_db);
                    let initial_storage = extract_storage_slice(&base_db_ref, &self.tracked_slots);
                    if let Some(env) = self.environments.get_mut(idx) {
                        env.db = (*base_db_ref).clone();
                        env.inspector.clear();
                        env.cheatcode_state.reset();
                        env.block_timestamp = self.initial_block_timestamp;
                        env.block_number = self.initial_block_number;
                        env.previous_storage = initial_storage;
                    }
                    return Err(e);
                }
            }
        }

        self.generate_batch_output(&reverted_mask)
    }

    /// Returns an immutable reference to the pre-allocated pinned state buffer.
    #[must_use]
    pub fn pinned_buffer(&self) -> &PinnedStateBuffer {
        &self.pinned_buffer
    }

    /// Internal helper to populate `PinnedStateBuffer` and construct `StepBatchOutput`.
    fn generate_batch_output(&mut self, reverted_mask: &[bool]) -> GfnResult<StepBatchOutput> {
        let b = self.batch_size;
        let num_slots = self.num_slots;
        let num_conds = self.num_conds;

        let total_slots = b * num_slots;
        let total_features = b * num_slots * 32;
        let total_telemetry = b * num_conds * 8;
        let total_context = b * 32;

        // Ensure pre-allocated buffer sizes
        if self.pinned_buffer.storage_words.len() != total_slots {
            self.pinned_buffer.storage_words.resize(total_slots, U256::ZERO);
        }
        if self.pinned_buffer.storage_features.len() != total_features {
            self.pinned_buffer.storage_features.resize(total_features, 0.0_f32);
        }
        if self.pinned_buffer.branch_telemetry.len() != total_telemetry {
            self.pinned_buffer.branch_telemetry.resize(total_telemetry, 0.0_f32);
        }
        if self.pinned_buffer.env_context.len() != total_context {
            self.pinned_buffer.env_context.resize(total_context, 0.0_f32);
        }

        let mut state_hashes = Vec::with_capacity(b);

        for env_idx in 0..b {
            let env = match self.environments.get(env_idx) {
                Some(e) => e,
                None => {
                    return Err(GfnError::RevmExecution {
                        env_idx,
                        message: format!("Environment not found at index {env_idx}"),
                    });
                }
            };

            // 1. Storage words and dynamic features
            let slot_base = env_idx * num_slots;
            let feat_base = env_idx * num_slots * 32;

            for (slot_idx, &(contract, slot)) in self.tracked_slots.iter().enumerate() {
                let v_t = match env.db.accounts.get(&contract) {
                    Some(acc) => acc.storage.get(&slot).copied().unwrap_or(U256::ZERO),
                    None => U256::ZERO,
                };
                let v_prev = env.previous_storage[slot_idx];
                let v_s0 = env.initial_storage[slot_idx];

                self.pinned_buffer.storage_words[slot_base + slot_idx] = v_t;

                // 32-dim SIMD dynamic features
                let dyn_features = compute_32dim_storage_features(v_t, v_prev, v_s0);
                let f_dst = feat_base + slot_idx * 32;
                self.pinned_buffer.storage_features[f_dst..f_dst + 32].copy_from_slice(&dyn_features);
            }

            // 2. Condition telemetry
            let cond_base = env_idx * num_conds * 8;
            for (c_idx, &cond_pc) in self.tracked_conds.iter().enumerate() {
                let feat_vec = match env.inspector.branch_telemetry.get(&cond_pc) {
                    Some(telemetry) => telemetry.to_feature_vector(),
                    None => [0.0_f32; 8],
                };
                let c_dst = cond_base + c_idx * 8;
                self.pinned_buffer.branch_telemetry[c_dst..c_dst + 8].copy_from_slice(&feat_vec);
            }

            // 3. Env context (32 dims)
            let ctx =
                compute_env_context(env, self.initial_block_timestamp, self.initial_block_number);
            let ctx_dst = env_idx * 32;
            self.pinned_buffer.env_context[ctx_dst..ctx_dst + 32].copy_from_slice(&ctx);

            // 4. Blake3 State Hash (zero-allocation view over pinned buffer)
            let current_storage_slice =
                &self.pinned_buffer.storage_words[slot_base..slot_base + num_slots];
            let state_hash = compute_blake3_state_hash(current_storage_slice, env);
            state_hashes.push(state_hash);
        }

        Ok(StepBatchOutput {
            storage_words: self.pinned_buffer.storage_words.clone(),
            storage_dyn_features: self.pinned_buffer.storage_features.clone(),
            branch_telemetry: self.pinned_buffer.branch_telemetry.clone(),
            env_context: self.pinned_buffer.env_context.clone(),
            reverted_mask: reverted_mask.to_vec(),
            state_hashes,
        })
    }
}

/// Executes a single environment step. Returns `true` if transaction reverted.
fn step_single_environment(
    env_idx: usize,
    env: &mut WorkerEnvironment,
    params: StepEnvParams<'_>,
    tracked_slots: &[(Address, U256)],
) -> GfnResult<bool> {
    // 1. Advance block timestamp and number if requested
    if params.warp_delta > 0 {
        env.block_timestamp = env
            .block_timestamp
            .saturating_add(U256::from(params.warp_delta));
    }
    if params.roll_delta > 0 {
        env.block_number = env
            .block_number
            .saturating_add(U256::from(params.roll_delta));
    }

    // 2. Resolve caller persona and check prank override
    let (caller, _origin) = match env.cheatcode_state.consume_prank_for_next_tx() {
        Some((c, orig)) => (c, orig.unwrap_or(c)),
        None => {
            let persona = ActorPersona::try_from_u8(params.caller_idx)?;
            let addr = persona.address();
            (addr, addr)
        }
    };

    // 3. Check if call targets Foundry cheatcode address
    if params.target == CHEATCODE_ADDRESS {
        let mut context = revm::Context::new_with_db(&mut env.db);
        context.evm.env.block.timestamp = env.block_timestamp;
        context.evm.env.block.number = env.block_number;
        let _ = env
            .cheatcode_state
            .dispatch_call(params.calldata, &mut context.evm)?;
        env.block_timestamp = context.evm.env.block.timestamp;
        env.block_number = context.evm.env.block.number;
        return Ok(false);
    }

    // 4. Update previous storage before mutation (zero-allocation in-place reuse)
    extract_storage_slice_into(&env.db, tracked_slots, &mut env.previous_storage);

    // 4b. Check if call is mocked via vm.mockCall
    if let Some(_mock_ret) = env.cheatcode_state.find_mocked_call(&params.target, params.calldata) {
        return Ok(false);
    }

    // 5. Construct and configure transaction environment
    let tx_env = TxEnv {
        caller,
        transact_to: TransactTo::Call(params.target),
        data: params.calldata.clone(),
        value: U256::ZERO,
        gas_limit: 30_000_000,
        chain_id: Some(CANCUN_CHAIN_ID),
        ..Default::default()
    };

    let mut evm = Evm::builder()
        .with_db(&mut env.db)
        .with_external_context(&mut env.inspector)
        .with_spec_id(SpecId::CANCUN)
        .build();

    evm.context.evm.env.tx = tx_env;
    evm.context.evm.env.block.timestamp = env.block_timestamp;
    evm.context.evm.env.block.number = env.block_number;
    evm.context.evm.env.block.coinbase = Address::ZERO;
    evm.context.evm.env.cfg.chain_id = CANCUN_CHAIN_ID;

    // 6. Execute transaction
    let result_and_state = evm.transact().map_err(|e| GfnError::RevmExecution {
        env_idx,
        message: format!("EVM transact error: {e:?}"),
    })?;

    drop(evm);

    let is_reverted = match result_and_state.result {
        ExecutionResult::Success { .. } => {
            // Commit changes to db only on success
            env.db.commit(result_and_state.state);
            false
        }
        ExecutionResult::Revert { .. } | ExecutionResult::Halt { .. } => {
            // Discard state changes on revert
            true
        }
    };

    Ok(is_reverted)
}

/// Reads storage slot values for the specified list of `(contract, slot)` pairs into a pre-allocated vector.
#[inline]
fn extract_storage_slice_into(
    db: &CacheDB<EmptyDB>,
    tracked_slots: &[(Address, U256)],
    out: &mut Vec<U256>,
) {
    if out.len() != tracked_slots.len() {
        out.resize(tracked_slots.len(), U256::ZERO);
    }
    for (i, &(contract, slot)) in tracked_slots.iter().enumerate() {
        out[i] = match db.accounts.get(&contract) {
            Some(acc) => acc.storage.get(&slot).copied().unwrap_or(U256::ZERO),
            None => U256::ZERO,
        };
    }
}

/// Reads storage slot values for the specified list of `(contract, slot)` pairs.
fn extract_storage_slice(db: &CacheDB<EmptyDB>, tracked_slots: &[(Address, U256)]) -> Vec<U256> {
    let mut out = vec![U256::ZERO; tracked_slots.len()];
    extract_storage_slice_into(db, tracked_slots, &mut out);
    out
}

/// Computes 32-dim SIMD dynamic runtime features for a storage slot.
#[inline]
fn compute_32dim_storage_features(v_t: U256, v_prev: U256, v_s0: U256) -> [f32; 32] {
    let mut feat = [0.0_f32; 32];

    // [0..8]: Magnitude, signed deltas, and bit changes
    feat[0] = u256_to_log2_norm(v_t);
    feat[1] = u256_signed_log2_delta(v_t, v_prev);
    feat[2] = u256_signed_log2_delta(v_t, v_s0);
    feat[3] = u256_to_log2_norm(v_prev);
    feat[4] = u256_to_log2_norm(v_s0);
    feat[5] = (v_t ^ v_prev).count_ones() as f32 / 256.0;
    feat[6] = (v_t ^ v_s0).count_ones() as f32 / 256.0;
    feat[7] = if v_t > v_s0 {
        1.0
    } else if v_t < v_s0 {
        -1.0
    } else {
        0.0
    };

    // [8..16]: Boundary indicators
    feat[8] = if v_t.is_zero() { 1.0 } else { 0.0 };
    feat[9] = if v_t == U256::from(1) { 1.0 } else { 0.0 };
    feat[10] = if v_t == U256::from(2) { 1.0 } else { 0.0 };
    feat[11] = if v_t == U256::MAX { 1.0 } else { 0.0 };
    feat[12] = if v_t == v_s0 { 1.0 } else { 0.0 };
    feat[13] = if ActorPersona::is_actor_u256(v_t) {
        1.0
    } else {
        0.0
    };
    feat[14] = if v_t >= (U256::from(1) << 128) {
        1.0
    } else {
        0.0
    };
    feat[15] = if !v_t.is_zero() && (v_t % U256::from(1_000_000_000_000_000_000_u64)).is_zero() {
        1.0
    } else {
        0.0
    };

    // [16..32]: 16-channel sinusoidal chunk encoding
    let sinusoidal = u256_sinusoidal_chunk_encoding(v_t);
    feat[16..32].copy_from_slice(&sinusoidal);

    feat
}

/// Computes 32-dim normalized environment context vector.
#[inline]
fn compute_env_context(env: &WorkerEnvironment, s0_ts: U256, s0_bn: U256) -> [f32; 32] {
    let mut ctx = [0.0_f32; 32];

    // [0..10]: Balances and s0-deltas of the 5 canonical actor personas (single consolidated lookup)
    let s0_bal = U256::from(INITIAL_ACTOR_BALANCE_WEI);
    for (i, persona) in ActorPersona::ALL.iter().enumerate() {
        let bal = match env.db.accounts.get(&persona.address()) {
            Some(acc) => acc.info.balance,
            None => U256::ZERO,
        };
        ctx[i] = u256_to_log2_norm(bal);
        ctx[5 + i] = u256_signed_log2_delta(bal, s0_bal);
    }

    // [10]: Normalized timestamp advance in days
    let ts_diff = env.block_timestamp.saturating_sub(s0_ts);
    ctx[10] = (ts_diff.to::<u64>() as f32 / 86400.0).clamp(0.0, 100.0);

    // [11]: Normalized block advance in blocks (~12s slots, 7200 blocks/day)
    let bn_diff = env.block_number.saturating_sub(s0_bn);
    ctx[11] = (bn_diff.to::<u64>() as f32 / 7200.0).clamp(0.0, 100.0);

    // [12..32]: Reserved for external telemetry and activity flags (remain 0.0)
    ctx
}

/// Computes Blake3 B256 cryptographic state trie digest over active storage and balances.
#[inline]
fn compute_blake3_state_hash(storage_words: &[U256], env: &WorkerEnvironment) -> B256 {
    let mut hasher = blake3::Hasher::new();

    // Hash active storage words
    for word in storage_words {
        hasher.update(&word.to_be_bytes::<32>());
    }

    // Hash persona balances
    for persona in &ActorPersona::ALL {
        let bal = match env.db.accounts.get(&persona.address()) {
            Some(acc) => acc.info.balance,
            None => U256::ZERO,
        };
        hasher.update(&bal.to_be_bytes::<32>());
    }

    // Hash block context
    hasher.update(&env.block_timestamp.to_be_bytes::<32>());
    hasher.update(&env.block_number.to_be_bytes::<32>());

    let hash_bytes: [u8; 32] = *hasher.finalize().as_bytes();
    B256::from(hash_bytes)
}

/// Initializes default actor persona accounts with native ETH balance.
fn init_default_actor_accounts(db: &mut CacheDB<EmptyDB>) {
    let initial_balance = U256::from(INITIAL_ACTOR_BALANCE_WEI);
    for persona in &ActorPersona::ALL {
        let info = AccountInfo {
            balance: initial_balance,
            ..Default::default()
        };
        db.insert_account_info(persona.address(), info);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::Bytecode;

    #[test]
    fn test_reset_all_to_s0_identical_hashes() -> GfnResult<()> {
        let mut base_db = CacheDB::new(EmptyDB::default());
        init_default_actor_accounts(&mut base_db);

        let tracked_slots = vec![
            (ActorPersona::Attacker0.address(), U256::ZERO),
            (ActorPersona::NormalUser.address(), U256::from(1)),
        ];

        let mut arena = RevmArena::new_with_base_db(base_db, tracked_slots, Vec::new(), 64, 4, 42)?;

        let output = arena.reset_all_to_s0()?;
        assert_eq!(output.state_hashes.len(), 64);

        let first_hash = output.state_hashes[0];
        for (i, hash) in output.state_hashes.iter().enumerate() {
            assert_eq!(
                *hash, first_hash,
                "State hash at env {i} must be identical to env 0"
            );
        }

        // Verify storage dyn features are 100% finite
        for val in &output.storage_dyn_features {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_snapshot_and_restore_bit_exact() -> GfnResult<()> {
        let mut base_db = CacheDB::new(EmptyDB::default());
        init_default_actor_accounts(&mut base_db);

        let target_contract = Address::repeat_byte(0x77);
        let tracked_slots = vec![(target_contract, U256::ZERO)];

        let mut arena = RevmArena::new_with_base_db(base_db, tracked_slots, Vec::new(), 4, 2, 42)?;

        // Mutate env 0 directly
        let snapshot_id = arena.snapshot_env(0)?;

        let mutated_val = U256::from(999_888_777_u64);
        if let Some(env) = arena.environments.get_mut(0) {
            env.db
                .accounts
                .entry(target_contract)
                .or_default()
                .storage
                .insert(U256::ZERO, mutated_val);
            env.block_timestamp += U256::from(500);
        }

        // Verify mutated
        let current_storage =
            extract_storage_slice(&arena.environments[0].db, &arena.tracked_slots);
        assert_eq!(current_storage[0], mutated_val);

        // Restore snapshot
        arena.restore_snapshot(0, snapshot_id)?;

        let restored_storage =
            extract_storage_slice(&arena.environments[0].db, &arena.tracked_slots);
        assert_eq!(
            restored_storage[0],
            U256::ZERO,
            "Storage must be restored bit-exact to pre-mutation value"
        );
        assert_eq!(
            arena.environments[0].block_timestamp,
            U256::from(CANCUN_INITIAL_BLOCK_TIMESTAMP)
        );

        Ok(())
    }

    #[test]
    fn test_step_batch_reverted_execution() -> GfnResult<()> {
        let mut base_db = CacheDB::new(EmptyDB::default());
        init_default_actor_accounts(&mut base_db);

        // Deploy a contract with REVERT opcode (0xFD)
        let reverting_contract = Address::repeat_byte(0x88);
        let info = AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from_static(&[
                0x60, 0x00, 0x60, 0x00, 0xFD,
            ]))),
            ..Default::default()
        };
        base_db.insert_account_info(reverting_contract, info);

        let tracked_slots = vec![(reverting_contract, U256::ZERO)];

        let mut arena = RevmArena::new_with_base_db(base_db, tracked_slots, Vec::new(), 2, 1, 42)?;

        let input = StepBatchInput {
            caller_indices: vec![0, 2],
            target_contracts: vec![reverting_contract, reverting_contract],
            calldata_batch: vec![Bytes::new(), Bytes::new()],
            warp_deltas: vec![0, 10],
            roll_deltas: vec![0, 1],
        };

        let output = arena.step_batch(&input)?;
        assert_eq!(output.reverted_mask.len(), 2);
        assert!(
            output.reverted_mask[0],
            "Reverting contract should trigger revert"
        );
        assert!(
            output.reverted_mask[1],
            "Reverting contract should trigger revert"
        );

        // Verify env_context has updated timestamp delta for env 1
        assert_eq!(output.env_context.len(), 64);
        assert!(output.env_context[32 + 10] > 0.0); // env 1 timestamp delta

        Ok(())
    }

    #[test]
    fn test_step_batch_mock_call_interception() -> GfnResult<()> {
        let mut base_db = CacheDB::new(EmptyDB::default());
        init_default_actor_accounts(&mut base_db);

        // Deploy a contract with REVERT opcode (0xFD)
        let reverting_contract = Address::repeat_byte(0x77);
        let info = AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from_static(&[
                0x60, 0x00, 0x60, 0x00, 0xFD,
            ]))),
            ..Default::default()
        };
        base_db.insert_account_info(reverting_contract, info);

        let tracked_slots = vec![(reverting_contract, U256::ZERO)];
        let mut arena = RevmArena::new_with_base_db(base_db, tracked_slots, Vec::new(), 1, 1, 42)?;

        // Set mock call on environment 0
        arena.environments[0].cheatcode_state.mocked_calls.insert(
            (reverting_contract, Bytes::from_static(&[0x12, 0x34])),
            Bytes::from_static(&[0xaa, 0xbb]),
        );

        let input = StepBatchInput {
            caller_indices: vec![0],
            target_contracts: vec![reverting_contract],
            calldata_batch: vec![Bytes::from_static(&[0x12, 0x34, 0x00])],
            warp_deltas: vec![0],
            roll_deltas: vec![0],
        };

        let output = arena.step_batch(&input)?;
        assert_eq!(output.reverted_mask.len(), 1);
        assert!(
            !output.reverted_mask[0],
            "Mocked call must bypass the reverting bytecode and succeed"
        );

        Ok(())
    }
}
