#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Control Flow Graph (CFG) lifter and semantic relation extractor.
//!
//! Lifts EVM bytecode and Solidity source units into a Heterogeneous State-Dependency Graph
//! (HSDG) using `petgraph::stable_graph::StableDiGraph`, extracting 8 semantic relation edge types.

use gfn_common::error::GfnResult;
use gfn_common::schemas::{FoundryContractArtifact, SoliditySlotType, StorageSemanticRole};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;

/// 8 Canonical edge relation types in HSDG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EdgeRelation {
    ReadsFrom = 0,
    WritesTo = 1,
    Guards = 2,
    Invokes = 3,
    RevReadsFrom = 4,
    RevWritesTo = 5,
    RevGuards = 6,
    RevInvokes = 7,
}

impl EdgeRelation {
    /// Returns the reverse relation.
    pub fn reverse(self) -> Self {
        match self {
            Self::ReadsFrom => Self::RevReadsFrom,
            Self::WritesTo => Self::RevWritesTo,
            Self::Guards => Self::RevGuards,
            Self::Invokes => Self::RevInvokes,
            Self::RevReadsFrom => Self::ReadsFrom,
            Self::RevWritesTo => Self::WritesTo,
            Self::RevGuards => Self::Guards,
            Self::RevInvokes => Self::Invokes,
        }
    }
}

/// Node data for `StorageNode` (Shape: 64 features).
#[derive(Debug, Clone, PartialEq)]
pub struct StorageNodeData {
    pub slot_id: u64,
    pub solidity_type: SoliditySlotType,
    pub semantic_role: StorageSemanticRole,
    pub byte_offset: u8,
    pub byte_size: u8,
    pub is_dynamic_mapping: bool,
    pub is_immutable: bool,
    pub is_constant: bool,
    pub is_proxy_implementation: bool,
}

/// Node data for `FunctionNode` (Shape: 64 features).
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionNodeData {
    pub selector: [u8; 4],
    pub visibility: u8, // 0: External, 1: Public, 2: Internal, 3: Private
    pub mutability: u8, // 0: Payable, 1: NonPayable, 2: View, 3: Pure
    pub has_only_owner_or_role: bool,
    pub has_non_reentrant: bool,
    pub is_fallback_or_receive: bool,
    pub is_flashloan_callback: bool,
    pub contains_inline_assembly: bool,
    pub contains_unchecked_math: bool,
    pub emits_events: bool,
    pub is_constructor_or_init: bool,
    pub cyclomatic_complexity: f32,
    pub basic_block_count: usize,
    pub external_call_count: usize,
    pub storage_write_count: usize,
    pub opcode_histogram: [f32; 32],
}

/// Node data for `ConditionNode` (Shape: 48 features).
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionNodeData {
    pub guard_origin: u8, // 0: Require, 1: RevertCustomError, 2: Assert, 3: IfBranch, 4: ModifierCheck, 5: LoopBound
    pub operator_profile: [bool; 10], // EQ, NEQ, LT, LE, GT, GE, MUL_DIV, MOD, BITWISE, CALLER_CHECK
    pub constant_threshold_scale: f32,
    pub ast_depth: usize,
}

/// Node data for `ExternalCallNode` (Shape: 32 features).
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalCallNodeData {
    pub call_opcode: u8, // 0: CALL, 1: DELEGATECALL, 2: STATICCALL, 3: CALLCODE, 4: CREATE, 5: CREATE2
    pub target_classification: u8, // 0: Erc20Transfer, 1: Erc20TransferFrom, 2: UniswapPool, 3: Oracle, 4: Flashloan, 5: Callback, 6: KnownProtocol, 7: Arbitrary
    pub writes_storage_after_call: bool,
    pub forwards_all_gas: bool,
    pub sends_native_eth: bool,
    pub target_is_user_controlled: bool,
}

/// Heterogeneous node variants in HSDG.
#[derive(Debug, Clone, PartialEq)]
pub enum HsdgNode {
    Storage(StorageNodeData),
    Function(FunctionNodeData),
    Condition(ConditionNodeData),
    ExternalCall(ExternalCallNodeData),
}

/// Edge payload in HSDG containing relation and 16-dimensional edge attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct HsdgEdge {
    pub relation: EdgeRelation,
    pub attributes: [f32; 16],
}

/// In-memory graph representation of an entire project or multi-contract system.
pub struct ProjectCfgGraph {
    pub graph: StableDiGraph<HsdgNode, HsdgEdge>,
    pub storage_indices: Vec<NodeIndex>,
    pub function_indices: Vec<NodeIndex>,
    pub condition_indices: Vec<NodeIndex>,
    pub ext_call_indices: Vec<NodeIndex>,
}

impl Default for ProjectCfgGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectCfgGraph {
    /// Creates a new empty `ProjectCfgGraph`.
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            storage_indices: Vec::new(),
            function_indices: Vec::new(),
            condition_indices: Vec::new(),
            ext_call_indices: Vec::new(),
        }
    }

    /// Adds a `StorageNode` and tracks its typed index.
    pub fn add_storage(&mut self, data: StorageNodeData) -> NodeIndex {
        let idx = self.graph.add_node(HsdgNode::Storage(data));
        self.storage_indices.push(idx);
        idx
    }

    /// Adds a `FunctionNode` and tracks its typed index.
    pub fn add_function(&mut self, data: FunctionNodeData) -> NodeIndex {
        let idx = self.graph.add_node(HsdgNode::Function(data));
        self.function_indices.push(idx);
        idx
    }

    /// Adds a `ConditionNode` and tracks its typed index.
    pub fn add_condition(&mut self, data: ConditionNodeData) -> NodeIndex {
        let idx = self.graph.add_node(HsdgNode::Condition(data));
        self.condition_indices.push(idx);
        idx
    }

    /// Adds an `ExternalCallNode` and tracks its typed index.
    pub fn add_ext_call(&mut self, data: ExternalCallNodeData) -> NodeIndex {
        let idx = self.graph.add_node(HsdgNode::ExternalCall(data));
        self.ext_call_indices.push(idx);
        idx
    }

    /// Adds a bidirectional relation edge pair `(src -> dst)` and reverse `(dst -> src)`.
    pub fn add_semantic_edge(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        relation: EdgeRelation,
        attrs: [f32; 16],
    ) {
        self.graph.add_edge(
            src,
            dst,
            HsdgEdge {
                relation,
                attributes: attrs,
            },
        );
        self.graph.add_edge(
            dst,
            src,
            HsdgEdge {
                relation: relation.reverse(),
                attributes: attrs,
            },
        );
    }
}

/// Control Flow Graph Lifter converting compiler artifacts into a unified `ProjectCfgGraph`.
pub struct SsaCfgLifter;

impl SsaCfgLifter {
    /// Lifts a set of compiled contract artifacts into a `ProjectCfgGraph`.
    pub fn lift_artifacts(artifacts: &[FoundryContractArtifact]) -> GfnResult<ProjectCfgGraph> {
        let mut cfg_graph = ProjectCfgGraph::new();
        let mut contract_functions: std::collections::BTreeMap<String, Vec<([u8; 4], NodeIndex)>> =
            std::collections::BTreeMap::new();

        for artifact in artifacts {
            let fn_nodes = Self::lift_single_contract(artifact, &mut cfg_graph)?;
            contract_functions.insert(artifact.contract_name.clone(), fn_nodes);
        }

        let calls =
            crate::cross_contract::CrossContractResolver::resolve_inter_contract_calls(artifacts);
        let _ = crate::cross_contract::CrossContractResolver::link_cross_contract_graph_edges(
            &mut cfg_graph,
            &contract_functions,
            &calls,
        );

        // Enforce Non-Emptiness Rule from HSDG spec: N_storage >= 1, N_function >= 1
        if cfg_graph.storage_indices.is_empty() {
            cfg_graph.add_storage(StorageNodeData {
                slot_id: 0,
                solidity_type: SoliditySlotType::Uint256,
                semantic_role: StorageSemanticRole::GenericState,
                byte_offset: 0,
                byte_size: 32,
                is_dynamic_mapping: false,
                is_immutable: false,
                is_constant: false,
                is_proxy_implementation: false,
            });
        }

        if cfg_graph.function_indices.is_empty() {
            cfg_graph.add_function(FunctionNodeData {
                selector: [0x00, 0x00, 0x00, 0x00],
                visibility: 0, // External
                mutability: 1, // NonPayable
                has_only_owner_or_role: false,
                has_non_reentrant: false,
                is_fallback_or_receive: true,
                is_flashloan_callback: false,
                contains_inline_assembly: false,
                contains_unchecked_math: false,
                emits_events: false,
                is_constructor_or_init: false,
                cyclomatic_complexity: 1.0,
                basic_block_count: 1,
                external_call_count: 0,
                storage_write_count: 0,
                opcode_histogram: [0.0; 32],
            });
        }

        Ok(cfg_graph)
    }

    fn lift_single_contract(
        artifact: &FoundryContractArtifact,
        cfg_graph: &mut ProjectCfgGraph,
    ) -> GfnResult<Vec<([u8; 4], NodeIndex)>> {
        let mut contract_storage_nodes = Vec::new();
        let mut contract_function_nodes = Vec::new();

        // 1. Storage Nodes
        for slot in &artifact.storage_slots {
            let s_idx = cfg_graph.add_storage(StorageNodeData {
                slot_id: slot.slot_id,
                solidity_type: slot.solidity_type,
                semantic_role: slot.semantic_role,
                byte_offset: slot.byte_offset,
                byte_size: slot.byte_size,
                is_dynamic_mapping: slot.is_dynamic_mapping,
                is_immutable: false,
                is_constant: false,
                is_proxy_implementation: slot.slot_id == 0x360894a13ba1a321,
            });
            contract_storage_nodes.push((slot.slot_id, s_idx));
        }

        // 2. Disassemble deployed bytecode to discover basic blocks, functions, conditions, and calls
        let bytecode = &artifact.deployed_bytecode;
        let blocks = Self::decompose_basic_blocks(bytecode);

        // 3. Extract Function Selectors from Contract ABI
        let mut selectors = Vec::new();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&artifact.abi_json_str) {
            if let Some(items) = value.as_array() {
                for item in items {
                    if item.get("type").and_then(|t| t.as_str()) == Some("function") {
                        if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                            let mut param_types = Vec::new();
                            if let Some(inputs) = item.get("inputs").and_then(|i| i.as_array()) {
                                for input in inputs {
                                    if let Some(t) = input.get("type").and_then(|s| s.as_str()) {
                                        param_types.push(t);
                                    }
                                }
                            }
                            let sig = format!("{}({})", name, param_types.join(","));
                            let hash = alloy_primitives::keccak256(sig.as_bytes());
                            let mut sel = [0u8; 4];
                            sel.copy_from_slice(&hash[0..4]);
                            selectors.push((name.to_string(), sel));
                        }
                    }
                }
            }
        }

        if selectors.is_empty() {
            selectors.push(("fallback".to_string(), [0x00, 0x00, 0x00, 0x00]));
        }

        // Create FunctionNode for each selector
        for (func_name, sel) in &selectors {
            let lower_name = func_name.to_ascii_lowercase();
            let is_flashloan = lower_name.contains("callback")
                || lower_name.contains("executeoperation")
                || lower_name.contains("onflashloan");

            let f_idx = cfg_graph.add_function(FunctionNodeData {
                selector: *sel,
                visibility: 0, // External
                mutability: 1, // NonPayable
                has_only_owner_or_role: lower_name.contains("admin")
                    || lower_name.contains("owner"),
                has_non_reentrant: lower_name.contains("nonreentrant")
                    || lower_name.contains("lock"),
                is_fallback_or_receive: func_name == "fallback" || func_name == "receive",
                is_flashloan_callback: is_flashloan,
                contains_inline_assembly: false,
                contains_unchecked_math: false,
                emits_events: true,
                is_constructor_or_init: lower_name.contains("initialize")
                    || lower_name.contains("init"),
                cyclomatic_complexity: (blocks.len() as f32).clamp(1.0, 50.0),
                basic_block_count: blocks.len(),
                external_call_count: 0,
                storage_write_count: 0,
                opcode_histogram: Self::compute_opcode_histogram(bytecode),
            });
            contract_function_nodes.push((*sel, f_idx));

            // 4. Connect Function to Storage (ReadsFrom / WritesTo)
            for (_slot_id, s_idx) in &contract_storage_nodes {
                let mut edge_attrs = [0.0f32; 16];
                edge_attrs[0] = 1.0; // Basic block depth 1
                cfg_graph.add_semantic_edge(f_idx, *s_idx, EdgeRelation::ReadsFrom, edge_attrs);

                let mut write_attrs = [0.0f32; 16];
                write_attrs[0] = 1.0; // Assignment operator
                cfg_graph.add_semantic_edge(f_idx, *s_idx, EdgeRelation::WritesTo, write_attrs);
            }

            // 5. Create Condition Nodes for JUMPI instructions
            for (bb_idx, block) in blocks.iter().enumerate() {
                if block.has_jumpi {
                    let c_idx = cfg_graph.add_condition(ConditionNodeData {
                        guard_origin: 0, // Require
                        operator_profile: [
                            true, false, false, false, false, false, false, false, false, false,
                        ],
                        constant_threshold_scale: 0.5,
                        ast_depth: bb_idx.saturating_add(1),
                    });

                    let mut guard_attrs = [0.0f32; 16];
                    guard_attrs[0] = 1.0; // Dominates function exit
                    cfg_graph.add_semantic_edge(c_idx, f_idx, EdgeRelation::Guards, guard_attrs);
                }

                if block.has_call {
                    let ext_idx = cfg_graph.add_ext_call(ExternalCallNodeData {
                        call_opcode: block.call_opcode,
                        target_classification: 0, // Erc20Transfer
                        writes_storage_after_call: false,
                        forwards_all_gas: true,
                        sends_native_eth: false,
                        target_is_user_controlled: false,
                    });

                    let mut invoke_attrs = [0.0f32; 16];
                    invoke_attrs[0] = 1.0;
                    cfg_graph.add_semantic_edge(
                        f_idx,
                        ext_idx,
                        EdgeRelation::Invokes,
                        invoke_attrs,
                    );
                }
            }
        }

        Ok(contract_function_nodes)
    }

    fn decompose_basic_blocks(bytecode: &[u8]) -> Vec<BasicBlockInfo> {
        let mut blocks = Vec::new();
        let mut cur_has_call = false;
        let mut cur_call_opcode = 0u8;

        let mut i = 0;
        while i < bytecode.len() {
            let op = bytecode[i];
            if op == 0x57 {
                // JUMPI terminates a block
                blocks.push(BasicBlockInfo {
                    has_jumpi: true,
                    has_call: cur_has_call,
                    call_opcode: cur_call_opcode,
                });
                cur_has_call = false;
            } else if op == 0x56 || op == 0x00 || op == 0xF3 || op == 0xFD || op == 0xFE {
                // JUMP, STOP, RETURN, REVERT, INVALID terminates a block
                blocks.push(BasicBlockInfo {
                    has_jumpi: false,
                    has_call: cur_has_call,
                    call_opcode: cur_call_opcode,
                });
                cur_has_call = false;
            } else if op == 0xF1 || op == 0xF4 || op == 0xFA {
                // CALL, DELEGATECALL, STATICCALL
                cur_has_call = true;
                cur_call_opcode = match op {
                    0xF1 => 0,
                    0xF4 => 1,
                    0xFA => 2,
                    _ => 0,
                };
            }

            // Skip PUSH data
            if (0x60..=0x7F).contains(&op) {
                let push_size = (op - 0x5F) as usize;
                i = i.saturating_add(push_size);
            }
            i = i.saturating_add(1);
        }

        if cur_has_call {
            blocks.push(BasicBlockInfo {
                has_jumpi: false,
                has_call: true,
                call_opcode: cur_call_opcode,
            });
        }

        if blocks.is_empty() {
            blocks.push(BasicBlockInfo {
                has_jumpi: false,
                has_call: false,
                call_opcode: 0,
            });
        }

        blocks
    }

    fn compute_opcode_histogram(bytecode: &[u8]) -> [f32; 32] {
        let mut hist = [0.0f32; 32];
        if bytecode.is_empty() {
            return hist;
        }

        let mut op_count = 0usize;
        let mut i = 0usize;
        while i < bytecode.len() {
            let op = bytecode[i];
            let bucket = (op as usize) / 8;
            if bucket < 32 {
                hist[bucket] += 1.0;
                op_count = op_count.saturating_add(1);
            }
            if (0x60..=0x7F).contains(&op) {
                let push_size = (op - 0x5F) as usize;
                i = i.saturating_add(push_size);
            }
            i = i.saturating_add(1);
        }

        let total = (op_count.max(1)) as f32;
        for val in &mut hist {
            *val /= total;
        }
        hist
    }
}

struct BasicBlockInfo {
    has_jumpi: bool,
    has_call: bool,
    call_opcode: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfn_common::schemas::StorageSlotRecord;

    #[test]
    fn test_empty_artifacts_creates_valid_graph() -> GfnResult<()> {
        let graph = SsaCfgLifter::lift_artifacts(&[])?;
        assert_eq!(graph.storage_indices.len(), 1);
        assert_eq!(graph.function_indices.len(), 1);
        Ok(())
    }

    #[test]
    fn test_reverse_relations() {
        assert_eq!(
            EdgeRelation::ReadsFrom.reverse(),
            EdgeRelation::RevReadsFrom
        );
        assert_eq!(EdgeRelation::WritesTo.reverse(), EdgeRelation::RevWritesTo);
        assert_eq!(EdgeRelation::Guards.reverse(), EdgeRelation::RevGuards);
        assert_eq!(EdgeRelation::Invokes.reverse(), EdgeRelation::RevInvokes);
        assert_eq!(
            EdgeRelation::RevReadsFrom.reverse(),
            EdgeRelation::ReadsFrom
        );
    }

    #[test]
    fn test_lift_single_contract_with_storage_and_calls() -> GfnResult<()> {
        let bytecode = vec![
            0x60, 0x80, 0x60, 0x40, 0x52, // PUSH1 0x80 PUSH1 0x40 MSTORE
            0x60, 0x00, 0x54, // PUSH1 0x00 SLOAD
            0x60, 0x20, 0x57, // PUSH1 0x20 JUMPI
            0xF1, // CALL
            0x00, // STOP
        ];

        let artifact = FoundryContractArtifact {
            project_id: uuid::Uuid::nil(),
            contract_name: "TestVault".to_string(),
            source_path: "src/TestVault.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0u8; 32],
            deployed_bytecode: bytecode,
            abi_json_str: r#"[{"type":"function","name":"deposit","inputs":[{"type":"uint256"}]}]"#
                .to_string(),
            storage_slots: vec![StorageSlotRecord {
                slot_id: 0,
                byte_offset: 0,
                byte_size: 32,
                solidity_type: SoliditySlotType::Uint256,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AssetBalance,
            }],
        };

        let graph = SsaCfgLifter::lift_artifacts(&[artifact])?;
        assert!(!graph.storage_indices.is_empty());
        assert!(!graph.function_indices.is_empty());
        assert!(!graph.condition_indices.is_empty());
        assert!(!graph.ext_call_indices.is_empty());

        // Verify bidirectional edge relations
        let mut found_reads = false;
        let mut found_rev_reads = false;
        for edge in graph.graph.edge_weights() {
            if edge.relation == EdgeRelation::ReadsFrom {
                found_reads = true;
            }
            if edge.relation == EdgeRelation::RevReadsFrom {
                found_rev_reads = true;
            }
        }
        assert!(found_reads);
        assert!(found_rev_reads);

        Ok(())
    }
}
