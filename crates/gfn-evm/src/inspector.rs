#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Zero-allocation EVM Inspector tracking storage accesses, memory/keccak pairs,
//! and narrow arithmetic branch distance telemetry for GFlowNet-Prover.

use alloy_primitives::{Address, Bytes, B256, U256};
use gfn_common::math::u256_to_log2_norm;
use revm::{
    interpreter::{opcode, CallInputs, CallOutcome, Interpreter},
    primitives::db::Database,
    EvmContext, Inspector,
};
use std::collections::{BTreeMap, HashMap};

/// Record of a concrete storage read or write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageAccessRecord {
    pub contract: Address,
    pub slot: U256,
    pub is_write: bool,
    pub value: U256,
    pub step_pc: usize,
}

/// Record of a Keccak256 pre-image and resulting hash for SMT bijection tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakPreimageRecord {
    pub preimage: Bytes,
    pub hash: B256,
}

/// Telemetry metrics for a conditional branch (JUMPI).
#[derive(Debug, Clone, PartialEq)]
pub struct BranchTelemetry {
    pub pc: usize,
    pub hit_count: u32,
    pub branch_distance: f32,
    pub condition_taken: bool,
    pub smt_status: f32,
    pub symbolic_vars: f32,
    pub ast_depth: f32,
    pub non_linear_ops: f32,
}

impl BranchTelemetry {
    /// Serializes the branch telemetry into an 8-channel vector for the `ConditionNode`.
    #[inline]
    pub fn to_feature_vector(&self) -> [f32; 8] {
        let hit_freq = (self.hit_count as f32).min(100.0) / 100.0;
        let narrow_trigger = if self.branch_distance > 0.85 {
            1.0_f32
        } else {
            0.0_f32
        };
        let taken_flag = if self.condition_taken {
            1.0_f32
        } else {
            0.0_f32
        };

        [
            hit_freq,
            self.branch_distance,
            self.smt_status,
            self.symbolic_vars,
            self.ast_depth,
            self.non_linear_ops,
            narrow_trigger,
            taken_flag,
        ]
    }
}

/// Custom Revm Inspector capturing execution traces, storage accesses, and branch telemetry.
#[derive(Debug, Clone, Default)]
pub struct GfnEvmInspector {
    /// Storage accesses recorded during transaction execution.
    pub storage_accesses: Vec<StorageAccessRecord>,
    /// Keccak256 pre-image to hash mapping.
    pub keccak_table: BTreeMap<B256, Bytes>,
    /// Active branch telemetry indexed by Program Counter (PC).
    pub branch_telemetry: HashMap<usize, BranchTelemetry>,
    /// Last comparison distance observed immediately preceding a conditional jump.
    last_comparison_distance: Option<f32>,
    /// Pending keccak preimage slice tracking [offset, size].
    pending_keccak: Option<(usize, usize)>,
    /// Pending SLOAD access index waiting for step_end post-value on stack top.
    pending_sload: Option<usize>,
    /// Pending SSTORE slot waiting for step_end post-value.
    pending_sstore: Option<(Address, U256, usize)>,
}

impl GfnEvmInspector {
    /// Instantiates a new empty inspector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all recorded accesses, tables, and branch telemetry.
    pub fn clear(&mut self) {
        self.storage_accesses.clear();
        self.keccak_table.clear();
        self.branch_telemetry.clear();
        self.last_comparison_distance = None;
        self.pending_keccak = None;
        self.pending_sload = None;
        self.pending_sstore = None;
    }

    /// Resets runtime execution traces between transaction steps while preserving cumulative telemetry.
    pub fn reset_step_traces(&mut self) {
        self.storage_accesses.clear();
        self.last_comparison_distance = None;
        self.pending_keccak = None;
        self.pending_sload = None;
        self.pending_sstore = None;
    }

    /// Computes the exact normalized arithmetic branch distance:
    /// d_branch(lhs, rhs) = 1.0 - (log2(|lhs - rhs| + 1) / 256.0)
    #[inline]
    pub fn calculate_branch_distance(lhs: U256, rhs: U256) -> f32 {
        let abs_diff = if lhs >= rhs { lhs - rhs } else { rhs - lhs };
        let norm_diff = u256_to_log2_norm(abs_diff);
        (1.0_f32 - norm_diff).clamp(0.0_f32, 1.0_f32)
    }

    /// Collects flattened branch telemetry features for up to `num_conds` condition nodes.
    pub fn collect_telemetry_matrix(&self, num_conds: usize) -> Vec<f32> {
        let mut matrix = vec![0.0_f32; num_conds * 8];
        let mut sorted_pcs: Vec<&usize> = self.branch_telemetry.keys().collect();
        sorted_pcs.sort();

        for (idx, &pc) in sorted_pcs.iter().take(num_conds).enumerate() {
            if let Some(telemetry) = self.branch_telemetry.get(pc) {
                let vec = telemetry.to_feature_vector();
                let base = idx * 8;
                matrix[base..base + 8].copy_from_slice(&vec);
            }
        }
        matrix
    }
}

impl<DB: Database> Inspector<DB> for GfnEvmInspector {
    #[inline]
    fn step(&mut self, interp: &mut Interpreter, _context: &mut EvmContext<DB>) {
        let op = interp.current_opcode();
        let pc = interp.program_counter();
        let contract = interp.contract.target_address;

        match op {
            // SLOAD: [slot] -> [value]
            opcode::SLOAD => {
                if let Ok(slot) = interp.stack().peek(0) {
                    let access_idx = self.storage_accesses.len();
                    self.storage_accesses.push(StorageAccessRecord {
                        contract,
                        slot,
                        is_write: false,
                        value: U256::ZERO,
                        step_pc: pc,
                    });
                    self.pending_sload = Some(access_idx);
                }
            }
            // SSTORE: [slot, value]
            opcode::SSTORE => {
                if let (Ok(slot), Ok(val)) = (interp.stack().peek(0), interp.stack().peek(1)) {
                    self.storage_accesses.push(StorageAccessRecord {
                        contract,
                        slot,
                        is_write: true,
                        value: val,
                        step_pc: pc,
                    });
                    self.pending_sstore = Some((contract, slot, pc));
                }
            }
            // Arithmetic and bitwise comparison opcodes: EQ, LT, GT, SLT, SGT
            opcode::EQ | opcode::LT | opcode::GT | opcode::SLT | opcode::SGT => {
                if let (Ok(lhs), Ok(rhs)) = (interp.stack().peek(0), interp.stack().peek(1)) {
                    let d = Self::calculate_branch_distance(lhs, rhs);
                    self.last_comparison_distance = Some(d);
                }
            }
            // ISZERO: [val] -> [val == 0]
            opcode::ISZERO => {
                if let Ok(val) = interp.stack().peek(0) {
                    if self.last_comparison_distance.is_none() {
                        let d = Self::calculate_branch_distance(val, U256::ZERO);
                        self.last_comparison_distance = Some(d);
                    }
                }
            }
            // JUMPI: [dest, condition]
            opcode::JUMPI => {
                if let Ok(cond) = interp.stack().peek(1) {
                    let condition_taken = !cond.is_zero();
                    let d_branch = self.last_comparison_distance.unwrap_or(if condition_taken {
                        1.0
                    } else {
                        0.0
                    });

                    let entry = self.branch_telemetry.entry(pc).or_insert(BranchTelemetry {
                        pc,
                        hit_count: 0,
                        branch_distance: d_branch,
                        condition_taken,
                        smt_status: 0.0,
                        symbolic_vars: 0.0,
                        ast_depth: 1.0,
                        non_linear_ops: 0.0,
                    });

                    entry.hit_count = entry.hit_count.saturating_add(1);
                    entry.branch_distance = d_branch;
                    entry.condition_taken = condition_taken;
                }
            }
            // KECCAK256: [offset, size]
            opcode::KECCAK256 => {
                if let (Ok(offset), Ok(size)) = (interp.stack().peek(0), interp.stack().peek(1)) {
                    let off_usize = offset.saturating_to::<usize>();
                    let size_usize = size.saturating_to::<usize>();
                    if size_usize <= 4096 {
                        self.pending_keccak = Some((off_usize, size_usize));
                    }
                }
            }
            _ => {}
        }
    }

    #[inline]
    fn step_end(&mut self, interp: &mut Interpreter, _context: &mut EvmContext<DB>) {
        // Populate loaded storage value from stack top after SLOAD completes
        if let Some(access_idx) = self.pending_sload.take() {
            if let Ok(loaded_val) = interp.stack().peek(0) {
                if let Some(rec) = self.storage_accesses.get_mut(access_idx) {
                    rec.value = loaded_val;
                }
            }
        }

        // Capture KECCAK256 pre-image and digest
        if let Some((offset, size)) = self.pending_keccak.take() {
            if let Ok(hash_val) = interp.stack().peek(0) {
                let mem_len = interp.shared_memory.len();
                if offset.saturating_add(size) <= mem_len {
                    let slice = interp.shared_memory.slice(offset, size);
                    let preimage = Bytes::copy_from_slice(slice);
                    let hash: B256 = hash_val.into();
                    self.keccak_table.insert(hash, preimage);
                }
            }
        }
    }

    #[inline]
    fn call(
        &mut self,
        _context: &mut EvmContext<DB>,
        _inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_branch_distance_exact_match() {
        let val = U256::from(1000);
        let dist = GfnEvmInspector::calculate_branch_distance(val, val);
        assert_eq!(dist, 1.0_f32, "Exact match should have distance 1.0");
    }

    #[test]
    fn test_calculate_branch_distance_max_diff() {
        let dist = GfnEvmInspector::calculate_branch_distance(U256::ZERO, U256::MAX);
        assert!(
            dist < 0.001_f32,
            "Maximum difference should yield distance ~0.0"
        );
    }

    #[test]
    fn test_calculate_branch_distance_monotonicity() {
        let base = U256::from(100);
        let near = U256::from(101);
        let mid = U256::from(1_000_000);
        let far = U256::MAX;

        let d_near = GfnEvmInspector::calculate_branch_distance(base, near);
        let d_mid = GfnEvmInspector::calculate_branch_distance(base, mid);
        let d_far = GfnEvmInspector::calculate_branch_distance(base, far);

        assert!(d_near > d_mid);
        assert!(d_mid > d_far);
        assert!(d_near <= 1.0_f32 && d_far >= 0.0_f32);
    }

    #[test]
    fn test_branch_telemetry_feature_vector() {
        let telemetry = BranchTelemetry {
            pc: 0x42,
            hit_count: 10,
            branch_distance: 0.90,
            condition_taken: true,
            smt_status: 0.5,
            symbolic_vars: 2.0,
            ast_depth: 3.0,
            non_linear_ops: 1.0,
        };

        let vec = telemetry.to_feature_vector();
        assert_eq!(vec.len(), 8);
        assert_eq!(vec[0], 0.10_f32); // 10 / 100
        assert_eq!(vec[1], 0.90_f32); // branch_distance
        assert_eq!(vec[2], 0.5_f32); // smt_status
        assert_eq!(vec[3], 2.0_f32); // symbolic_vars
        assert_eq!(vec[4], 3.0_f32); // ast_depth
        assert_eq!(vec[5], 1.0_f32); // non_linear_ops
        assert_eq!(vec[6], 1.0_f32); // narrow_trigger because 0.90 > 0.85
        assert_eq!(vec[7], 1.0_f32); // taken_flag
    }

    #[test]
    fn test_telemetry_matrix_collection() {
        let mut inspector = GfnEvmInspector::new();
        inspector.branch_telemetry.insert(
            0x10,
            BranchTelemetry {
                pc: 0x10,
                hit_count: 5,
                branch_distance: 0.88,
                condition_taken: false,
                smt_status: 0.0,
                symbolic_vars: 1.0,
                ast_depth: 2.0,
                non_linear_ops: 0.0,
            },
        );

        let matrix = inspector.collect_telemetry_matrix(4);
        assert_eq!(matrix.len(), 32); // 4 conds * 8 channels
        assert_eq!(matrix[0], 0.05_f32);
        assert_eq!(matrix[1], 0.88_f32);
        assert_eq!(matrix[6], 1.0_f32); // 0.88 > 0.85
        assert_eq!(matrix[7], 0.0_f32); // condition_taken: false
    }
}
