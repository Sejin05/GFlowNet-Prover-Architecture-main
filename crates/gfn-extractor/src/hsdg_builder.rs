#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Heterogeneous State-Dependency Graph (HSDG) Builder and Tensor Validator.
//!
//! Transforms compiled Foundry contract artifacts and lifted CFG structures into strictly typed,
//! zero-copy `HeteroGraphTensors` suitable for forward passes in `HgtInvariantSynthesizer`
//! and `StateGraphRgatEncoder`.

use crate::cfg_lifter::{HsdgNode, ProjectCfgGraph, SsaCfgLifter};
use crate::compiler::ProjectExtractor;
use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::FoundryContractArtifact;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

/// Strongly typed heterogeneous state-dependency graph tensors.
#[derive(Debug)]
pub struct HeteroGraphTensors {
    /// Storage node features: Shape `[N_storage, 64]`, `Kind::Float`.
    pub storage_x: Tensor,
    /// Function node features: Shape `[N_func, 64]`, `Kind::Float`.
    pub function_x: Tensor,
    /// Condition node features: Shape `[N_cond, 48]`, `Kind::Float`.
    pub condition_x: Tensor,
    /// External call node features: Shape `[N_ext, 32]`, `Kind::Float`.
    pub ext_call_x: Tensor,
    /// Sparse edge indices for each relation: 8 tensors of shape `[2, E_r]`, `Kind::Int64`.
    pub edge_indices: [Tensor; 8],
    /// Sparse edge attributes for each relation: 8 tensors of shape `[E_r, 16]`, `Kind::Float`.
    pub edge_attrs: [Tensor; 8],
}

impl Clone for HeteroGraphTensors {
    fn clone(&self) -> Self {
        Self {
            storage_x: self.storage_x.copy(),
            function_x: self.function_x.copy(),
            condition_x: self.condition_x.copy(),
            ext_call_x: self.ext_call_x.copy(),
            edge_indices: [
                self.edge_indices[0].copy(),
                self.edge_indices[1].copy(),
                self.edge_indices[2].copy(),
                self.edge_indices[3].copy(),
                self.edge_indices[4].copy(),
                self.edge_indices[5].copy(),
                self.edge_indices[6].copy(),
                self.edge_indices[7].copy(),
            ],
            edge_attrs: [
                self.edge_attrs[0].copy(),
                self.edge_attrs[1].copy(),
                self.edge_attrs[2].copy(),
                self.edge_attrs[3].copy(),
                self.edge_attrs[4].copy(),
                self.edge_attrs[5].copy(),
                self.edge_attrs[6].copy(),
                self.edge_attrs[7].copy(),
            ],
        }
    }
}

/// Automated data validator verifying graph connectivity, topological bounds, and numerical finiteness.
pub struct HsdgValidator;

impl HsdgValidator {
    /// Validates an extracted `HeteroGraphTensors` against all canonical HSDG invariants.
    pub fn validate(graph: &HeteroGraphTensors) -> GfnResult<()> {
        let n_storage = graph.storage_x.size()[0];
        let n_func = graph.function_x.size()[0];
        let n_cond = graph.condition_x.size()[0];
        let n_ext = graph.ext_call_x.size()[0];

        // 1. Graph Connectivity & Non-Emptiness Rule: N_storage >= 1 and N_function >= 1
        if n_storage < 1 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!("HSDG requires at least 1 storage node, found {n_storage}"),
            });
        }
        if n_func < 1 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!("HSDG requires at least 1 function node, found {n_func}"),
            });
        }

        // 2. Validate exact feature widths
        if graph.storage_x.size()[1] != 64 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!(
                    "storage_x width must be 64, found {}",
                    graph.storage_x.size()[1]
                ),
            });
        }
        if graph.function_x.size()[1] != 64 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!(
                    "function_x width must be 64, found {}",
                    graph.function_x.size()[1]
                ),
            });
        }
        if graph.condition_x.size()[1] != 48 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!(
                    "condition_x width must be 48, found {}",
                    graph.condition_x.size()[1]
                ),
            });
        }
        if graph.ext_call_x.size()[1] != 32 {
            return Err(GfnError::InvalidGraphTopology {
                details: format!(
                    "ext_call_x width must be 32, found {}",
                    graph.ext_call_x.size()[1]
                ),
            });
        }

        // 3. Strict 100% Finite Values Guarantee (Zero NaN / Inf)
        let tensors = [
            &graph.storage_x,
            &graph.function_x,
            &graph.condition_x,
            &graph.ext_call_x,
        ];
        for (i, t) in tensors.iter().enumerate() {
            let is_finite = t.isfinite().all().int64_value(&[]);
            if is_finite != 1 {
                return Err(GfnError::InvalidGraphTopology {
                    details: format!(
                        "Node feature tensor index {i} contains non-finite values (NaN/Inf)"
                    ),
                });
            }
        }

        // 4. Edge relations index bounds and finite attributes
        for (rel_idx, (edge_index, edge_attr)) in graph
            .edge_indices
            .iter()
            .zip(graph.edge_attrs.iter())
            .enumerate()
        {
            if edge_index.size()[0] != 2 {
                return Err(GfnError::InvalidGraphTopology {
                    details: format!("Edge index relation {rel_idx} must have first dimension 2"),
                });
            }
            if edge_attr.size()[1] != 16 {
                return Err(GfnError::InvalidGraphTopology {
                    details: format!("Edge attr relation {rel_idx} must have feature width 16"),
                });
            }

            let num_edges = edge_index.size()[1];
            if num_edges > 0 {
                let is_finite = edge_attr.isfinite().all().int64_value(&[]);
                if is_finite != 1 {
                    return Err(GfnError::InvalidGraphTopology {
                        details: format!(
                            "Edge attribute relation {rel_idx} contains non-finite values"
                        ),
                    });
                }

                // Check index bounds: Determine max allowed source and destination
                let (max_src, max_dst) = match rel_idx {
                    0 => (n_func.max(n_cond), n_storage), // ReadsFrom: Function/Condition -> Storage
                    1 => (n_func, n_storage),             // WritesTo: Function -> Storage
                    2 => (n_cond, n_func.max(n_storage)), // Guards: Condition -> Function/Storage
                    3 => (n_func, n_func.max(n_ext)), // Invokes: Function -> Function/ExternalCall
                    4 => (n_storage, n_func.max(n_cond)), // RevReadsFrom: Storage -> Function/Condition
                    5 => (n_storage, n_func),             // RevWritesTo: Storage -> Function
                    6 => (n_func.max(n_storage), n_cond), // RevGuards: Function/Storage -> Condition
                    7 => (n_func.max(n_ext), n_func), // RevInvokes: Function/ExternalCall -> Function
                    _ => (n_storage.max(n_func), n_storage.max(n_func)),
                };

                let src_max = edge_index.select(0, 0).max().int64_value(&[]);
                let dst_max = edge_index.select(0, 1).max().int64_value(&[]);

                if src_max >= max_src || dst_max >= max_dst {
                    return Err(GfnError::InvalidGraphTopology {
                        details: format!(
                            "Edge index relation {rel_idx} out of bounds: src_max={src_max} (limit {max_src}), dst_max={dst_max} (limit {max_dst})"
                        ),
                    });
                }
            }
        }

        Ok(())
    }
}

/// HSDG Builder constructing `HeteroGraphTensors` from repository sources.
pub struct HsdgBuilder {
    pub project_root: PathBuf,
    pub solc_version: String,
}

impl HsdgBuilder {
    /// Creates a new `HsdgBuilder`.
    pub fn new(project_root: &Path, solc_version: &str) -> Self {
        Self {
            project_root: project_root.to_path_buf(),
            solc_version: solc_version.to_string(),
        }
    }

    /// Programmatically compiles the target repository and extracts raw `FoundryContractArtifact`s.
    pub fn extract_artifacts(&self) -> GfnResult<Vec<FoundryContractArtifact>> {
        let extractor = ProjectExtractor::new(&self.project_root, &self.solc_version);
        extractor.extract()
    }

    /// Constructs the static `HeteroGraphTensors` from extracted compilation artifacts.
    pub fn build_static_graph(
        &self,
        artifacts: &[FoundryContractArtifact],
        device: Device,
    ) -> GfnResult<HeteroGraphTensors> {
        let cfg_graph = SsaCfgLifter::lift_artifacts(artifacts)?;
        Self::convert_cfg_to_tensors(&cfg_graph, device)
    }

    /// Converts an in-memory `ProjectCfgGraph` into strictly typed `HeteroGraphTensors`.
    pub fn convert_cfg_to_tensors(
        cfg_graph: &ProjectCfgGraph,
        device: Device,
    ) -> GfnResult<HeteroGraphTensors> {
        let n_storage = cfg_graph.storage_indices.len();
        let n_func = cfg_graph.function_indices.len();
        let n_cond = cfg_graph.condition_indices.len();
        let n_ext = cfg_graph.ext_call_indices.len();

        // 1. Build Storage Node Feature Matrix [N_storage, 64]
        let mut storage_data = Vec::with_capacity(n_storage.saturating_mul(64));
        for &node_idx in &cfg_graph.storage_indices {
            let mut row = [0.0f32; 64];
            if let Some(HsdgNode::Storage(s)) = cfg_graph.graph.node_weight(node_idx) {
                // One-hot slot type: 0..18
                let type_idx = (s.solidity_type as usize).min(17);
                row[type_idx] = 1.0;

                // One-hot semantic role: 18..26
                let role_idx = 18 + (s.semantic_role as usize).min(7);
                row[role_idx] = 1.0;

                // Packing coordinates: 26..28
                row[26] = (s.byte_offset as f32 / 32.0).clamp(0.0, 1.0);
                row[27] = (s.byte_size as f32 / 32.0).clamp(0.0, 1.0);

                // Flags: 28..32
                row[28] = if s.is_dynamic_mapping { 1.0 } else { 0.0 };
                row[29] = if s.is_immutable { 1.0 } else { 0.0 };
                row[30] = if s.is_constant { 1.0 } else { 0.0 };
                row[31] = if s.is_proxy_implementation { 1.0 } else { 0.0 };

                // 32..64 initialized to 0.0 in static graph
            }
            storage_data.extend_from_slice(&row);
        }

        // 2. Build Function Node Feature Matrix [N_func, 64]
        let mut func_data = Vec::with_capacity(n_func.saturating_mul(64));
        for &node_idx in &cfg_graph.function_indices {
            let mut row = [0.0f32; 64];
            if let Some(HsdgNode::Function(f)) = cfg_graph.graph.node_weight(node_idx) {
                // Visibility: 0..4
                let vis_idx = (f.visibility as usize).min(3);
                row[vis_idx] = 1.0;

                // Mutability: 4..8
                let mut_idx = 4 + (f.mutability as usize).min(3);
                row[mut_idx] = 1.0;

                // Flags: 8..16
                row[8] = if f.has_only_owner_or_role { 1.0 } else { 0.0 };
                row[9] = if f.has_non_reentrant { 1.0 } else { 0.0 };
                row[10] = if f.is_fallback_or_receive { 1.0 } else { 0.0 };
                row[11] = if f.is_flashloan_callback { 1.0 } else { 0.0 };
                row[12] = if f.contains_inline_assembly { 1.0 } else { 0.0 };
                row[13] = if f.contains_unchecked_math { 1.0 } else { 0.0 };
                row[14] = if f.emits_events { 1.0 } else { 0.0 };
                row[15] = if f.is_constructor_or_init { 1.0 } else { 0.0 };

                // Complexity metrics: 16..24
                row[16] = (f.cyclomatic_complexity / 50.0).clamp(0.0, 1.0);
                row[17] = (f.basic_block_count as f32 / 100.0).clamp(0.0, 1.0);
                row[18] = (f.external_call_count as f32 / 10.0).clamp(0.0, 1.0);
                row[19] = (f.storage_write_count as f32 / 20.0).clamp(0.0, 1.0);

                // Selector projection and opcode histogram: 24..64
                for (k, &b) in f.selector.iter().enumerate() {
                    row[24 + k] = (b as f32) / 255.0;
                }
                for (k, &hist) in f.opcode_histogram.iter().enumerate() {
                    row[28 + k] = hist;
                }
            }
            func_data.extend_from_slice(&row);
        }

        // 3. Build Condition Node Feature Matrix [N_cond, 48]
        let effective_cond = n_cond.max(1);
        let mut cond_data = Vec::with_capacity(effective_cond.saturating_mul(48));
        if n_cond == 0 {
            cond_data.extend_from_slice(&[0.0f32; 48]);
        } else {
            for &node_idx in &cfg_graph.condition_indices {
                let mut row = [0.0f32; 48];
                if let Some(HsdgNode::Condition(c)) = cfg_graph.graph.node_weight(node_idx) {
                    let origin_idx = (c.guard_origin as usize).min(5);
                    row[origin_idx] = 1.0;

                    for (k, &prof) in c.operator_profile.iter().enumerate() {
                        row[6 + k] = if prof { 1.0 } else { 0.0 };
                    }
                    row[16] = c.constant_threshold_scale.clamp(0.0, 1.0);
                    row[17] = (c.ast_depth as f32 / 20.0).clamp(0.0, 1.0);
                }
                cond_data.extend_from_slice(&row);
            }
        }

        // 4. Build External Call Node Feature Matrix [N_ext, 32]
        let effective_ext = n_ext.max(1);
        let mut ext_data = Vec::with_capacity(effective_ext.saturating_mul(32));
        if n_ext == 0 {
            ext_data.extend_from_slice(&[0.0f32; 32]);
        } else {
            for &node_idx in &cfg_graph.ext_call_indices {
                let mut row = [0.0f32; 32];
                if let Some(HsdgNode::ExternalCall(e)) = cfg_graph.graph.node_weight(node_idx) {
                    let opcode_idx = (e.call_opcode as usize).min(5);
                    row[opcode_idx] = 1.0;

                    let target_idx = 6 + (e.target_classification as usize).min(7);
                    row[target_idx] = 1.0;

                    row[14] = if e.writes_storage_after_call {
                        1.0
                    } else {
                        0.0
                    };
                    row[15] = if e.forwards_all_gas { 1.0 } else { 0.0 };
                    row[16] = if e.sends_native_eth { 1.0 } else { 0.0 };
                    row[17] = if e.target_is_user_controlled {
                        1.0
                    } else {
                        0.0
                    };
                }
                ext_data.extend_from_slice(&row);
            }
        }

        // 5. Construct node tensors
        let storage_x = Tensor::from_slice(&storage_data)
            .reshape([n_storage as i64, 64])
            .to_device(device);
        let function_x = Tensor::from_slice(&func_data)
            .reshape([n_func as i64, 64])
            .to_device(device);
        let condition_x = Tensor::from_slice(&cond_data)
            .reshape([effective_cond as i64, 48])
            .to_device(device);
        let ext_call_x = Tensor::from_slice(&ext_data)
            .reshape([effective_ext as i64, 32])
            .to_device(device);

        // 6. Partition edges by 8 relations
        let mut rel_sources: [Vec<i64>; 8] = Default::default();
        let mut rel_targets: [Vec<i64>; 8] = Default::default();
        let mut rel_attrs: [Vec<f32>; 8] = Default::default();

        let node_id_map: std::collections::HashMap<_, _> = cfg_graph
            .storage_indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, (0u8, i as i64)))
            .chain(
                cfg_graph
                    .function_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| (idx, (1u8, i as i64))),
            )
            .chain(
                cfg_graph
                    .condition_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| (idx, (2u8, i as i64))),
            )
            .chain(
                cfg_graph
                    .ext_call_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| (idx, (3u8, i as i64))),
            )
            .collect();

        for edge_ref in cfg_graph.graph.edge_references() {
            let u_raw = edge_ref.source();
            let v_raw = edge_ref.target();
            let weight = edge_ref.weight();

            if let (Some(&(_type_u, u_id)), Some(&(_type_v, v_id))) =
                (node_id_map.get(&u_raw), node_id_map.get(&v_raw))
            {
                let r_idx = weight.relation as usize;
                rel_sources[r_idx].push(u_id);
                rel_targets[r_idx].push(v_id);
                rel_attrs[r_idx].extend_from_slice(&weight.attributes);
            }
        }

        let mut edge_indices_tensors: [Tensor; 8] = empty_tensor_array();
        let mut edge_attrs_tensors: [Tensor; 8] = empty_tensor_array();

        for r in 0..8 {
            let num_edges = rel_attrs[r].len() / 16;
            if num_edges == 0 {
                edge_indices_tensors[r] = Tensor::zeros([2, 0], (Kind::Int64, device));
                edge_attrs_tensors[r] = Tensor::zeros([0, 16], (Kind::Float, device));
            } else {
                let mut flat = Vec::with_capacity(num_edges.saturating_mul(2));
                flat.extend_from_slice(&rel_sources[r]);
                flat.extend_from_slice(&rel_targets[r]);
                edge_indices_tensors[r] = Tensor::from_slice(&flat)
                    .reshape([2, num_edges as i64])
                    .to_kind(Kind::Int64)
                    .to_device(device);
                edge_attrs_tensors[r] = Tensor::from_slice(&rel_attrs[r])
                    .reshape([num_edges as i64, 16])
                    .to_device(device);
            }
        }

        let hetero_graph = HeteroGraphTensors {
            storage_x,
            function_x,
            condition_x,
            ext_call_x,
            edge_indices: edge_indices_tensors,
            edge_attrs: edge_attrs_tensors,
        };

        HsdgValidator::validate(&hetero_graph)?;
        Ok(hetero_graph)
    }

    /// Updates dynamic runtime features (SIMD storage features and branch telemetry) at step $t$.
    pub fn update_dynamic_batch(
        base_graph: &HeteroGraphTensors,
        batch_size: usize,
        storage_dyn_features: &[f32],
        branch_telemetry: &[f32],
        device: Device,
    ) -> GfnResult<HeteroGraphTensors> {
        let n_storage = base_graph.storage_x.size()[0] as usize;
        let n_cond = base_graph.condition_x.size()[0] as usize;

        let expected_storage_len = batch_size.saturating_mul(n_storage).saturating_mul(32);
        let expected_branch_len = batch_size.saturating_mul(n_cond).saturating_mul(8);

        if storage_dyn_features.len() < expected_storage_len {
            return Err(GfnError::MathDomainError {
                details: format!(
                    "storage_dyn_features length {} less than expected {}",
                    storage_dyn_features.len(),
                    expected_storage_len
                ),
            });
        }

        if branch_telemetry.len() < expected_branch_len {
            return Err(GfnError::MathDomainError {
                details: format!(
                    "branch_telemetry length {} less than expected {}",
                    branch_telemetry.len(),
                    expected_branch_len
                ),
            });
        }

        let mut updated = base_graph.clone();
        let effective_b = batch_size.max(1);

        // 1. Update storage dynamic columns [32..64] (averaged across active batch with finite clamping)
        let mut storage_data = vec![0.0f32; n_storage * 64];
        updated
            .storage_x
            .copy_data(&mut storage_data, n_storage * 64);
        for i in 0..n_storage {
            for col in 0..32 {
                let mut acc = 0.0f32;
                let mut count = 0usize;
                for b_idx in 0..effective_b {
                    let flat_idx = (b_idx * n_storage + i) * 32 + col;
                    if let Some(&v) = storage_dyn_features.get(flat_idx) {
                        if v.is_finite() {
                            acc += v.clamp(-100.0, 100.0);
                            count += 1;
                        }
                    }
                }
                let val = if count > 0 { acc / (count as f32) } else { 0.0 };
                storage_data[i * 64 + 32 + col] = val;
            }
        }
        updated.storage_x = Tensor::from_slice(&storage_data)
            .reshape([n_storage as i64, 64])
            .to_device(device);

        // 2. Update condition telemetry columns [24..32] (averaged across active batch with finite clamping)
        let mut cond_data = vec![0.0f32; n_cond * 48];
        updated.condition_x.copy_data(&mut cond_data, n_cond * 48);
        for i in 0..n_cond {
            for col in 0..8 {
                let mut acc = 0.0f32;
                let mut count = 0usize;
                for b_idx in 0..effective_b {
                    let flat_idx = (b_idx * n_cond + i) * 8 + col;
                    if let Some(&v) = branch_telemetry.get(flat_idx) {
                        if v.is_finite() {
                            acc += v.clamp(-100.0, 100.0);
                            count += 1;
                        }
                    }
                }
                let val = if count > 0 { acc / (count as f32) } else { 0.0 };
                cond_data[i * 48 + 24 + col] = val;
            }
        }
        updated.condition_x = Tensor::from_slice(&cond_data)
            .reshape([n_cond as i64, 48])
            .to_device(device);

        HsdgValidator::validate(&updated)?;
        Ok(updated)
    }
}

fn empty_tensor_array() -> [Tensor; 8] {
    [
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use gfn_common::math::u256_to_log2_norm;
    use gfn_common::schemas::{SoliditySlotType, StorageSemanticRole, StorageSlotRecord};

    #[test]
    fn test_tensor_shapes_and_finite_guarantee() -> GfnResult<()> {
        let artifact = FoundryContractArtifact {
            project_id: uuid::Uuid::nil(),
            contract_name: "MockVault".to_string(),
            source_path: "src/MockVault.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0u8; 32],
            deployed_bytecode: vec![0x60, 0x00, 0x54, 0x60, 0x20, 0x57, 0x00],
            abi_json_str:
                r#"[{"type":"function","name":"withdraw","inputs":[{"type":"uint256"}]}]"#
                    .to_string(),
            storage_slots: vec![
                StorageSlotRecord {
                    slot_id: 0,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::AssetBalance,
                },
                StorageSlotRecord {
                    slot_id: 1,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::TotalSupply,
                },
            ],
        };

        let builder = HsdgBuilder::new(Path::new("."), "0.8.24");
        let graph = builder.build_static_graph(&[artifact], Device::Cpu)?;

        let n_s = graph.storage_x.size()[0];
        let n_f = graph.function_x.size()[0];
        let n_c = graph.condition_x.size()[0];
        let n_e = graph.ext_call_x.size()[0];

        assert_eq!(graph.storage_x.size(), vec![n_s, 64]);
        assert_eq!(graph.function_x.size(), vec![n_f, 64]);
        assert_eq!(graph.condition_x.size(), vec![n_c, 48]);
        assert_eq!(graph.ext_call_x.size(), vec![n_e, 32]);

        assert_eq!(graph.storage_x.isfinite().all().int64_value(&[]), 1);
        assert_eq!(graph.function_x.isfinite().all().int64_value(&[]), 1);
        assert_eq!(graph.condition_x.isfinite().all().int64_value(&[]), 1);
        assert_eq!(graph.ext_call_x.isfinite().all().int64_value(&[]), 1);

        Ok(())
    }

    #[test]
    fn test_u256_max_log2_norm_boundary() {
        let max_val = U256::MAX;
        let norm = u256_to_log2_norm(max_val);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_validator_rejects_empty_storage() {
        let empty_storage = HeteroGraphTensors {
            storage_x: Tensor::zeros([0, 64], (Kind::Float, Device::Cpu)),
            function_x: Tensor::zeros([1, 64], (Kind::Float, Device::Cpu)),
            condition_x: Tensor::zeros([1, 48], (Kind::Float, Device::Cpu)),
            ext_call_x: Tensor::zeros([1, 32], (Kind::Float, Device::Cpu)),
            edge_indices: [
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
                Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            ],
            edge_attrs: [
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
                Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            ],
        };

        let result = HsdgValidator::validate(&empty_storage);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_dynamic_batch() -> GfnResult<()> {
        let artifact = FoundryContractArtifact {
            project_id: uuid::Uuid::nil(),
            contract_name: "MockVault".to_string(),
            source_path: "src/MockVault.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0u8; 32],
            deployed_bytecode: vec![0x60, 0x00, 0x54, 0x60, 0x20, 0x57, 0x00],
            abi_json_str: r#"[{"type":"function","name":"deposit","inputs":[]}]"#.to_string(),
            storage_slots: vec![StorageSlotRecord {
                slot_id: 0,
                byte_offset: 0,
                byte_size: 32,
                solidity_type: SoliditySlotType::Uint256,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AssetBalance,
            }],
        };

        let builder = HsdgBuilder::new(Path::new("."), "0.8.24");
        let base_graph = builder.build_static_graph(&[artifact], Device::Cpu)?;

        let n_s = base_graph.storage_x.size()[0] as usize;
        let n_c = base_graph.condition_x.size()[0] as usize;

        let storage_dyn = vec![0.5f32; n_s * 32];
        let branch_dyn = vec![0.8f32; n_c * 8];

        let updated = HsdgBuilder::update_dynamic_batch(
            &base_graph,
            1,
            &storage_dyn,
            &branch_dyn,
            Device::Cpu,
        )?;

        // Assert updated dynamic features
        let s_val = updated.storage_x.get(0).get(32).double_value(&[]);
        assert!((s_val - 0.5).abs() < 1e-5);

        let b_val = updated.condition_x.get(0).get(24).double_value(&[]);
        assert!((b_val - 0.8).abs() < 1e-5);

        Ok(())
    }
}
