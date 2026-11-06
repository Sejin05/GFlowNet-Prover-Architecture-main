#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! EIP-1967, UUPS, and EIP-2535 Diamond proxy resolver.
//!
//! Resolves implementation storage slots and merges implementation contract logic
//! (`FunctionNode`, `ConditionNode`, `ExternalCallNode`) directly onto proxy contract `StorageNode`s.

use crate::cfg_lifter::{EdgeRelation, ProjectCfgGraph};
use alloy_primitives::{b256, B256};
use gfn_common::error::GfnResult;
use petgraph::graph::NodeIndex;

/// EIP-1967 implementation slot: `bytes32(uint256(keccak256('eip1967.proxy.implementation')) - 1)`
pub const EIP1967_IMPLEMENTATION_SLOT: B256 =
    b256!("360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc");

/// EIP-1967 admin slot: `bytes32(uint256(keccak256('eip1967.proxy.admin')) - 1)`
pub const EIP1967_ADMIN_SLOT: B256 =
    b256!("b53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103");

/// EIP-1967 beacon slot: `bytes32(uint256(keccak256('eip1967.proxy.beacon')) - 1)`
pub const EIP1967_BEACON_SLOT: B256 =
    b256!("a3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50");

/// Proxy resolver responsible for detecting proxy storage layouts and merging graph nodes.
pub struct ProxySlotResolver;

impl ProxySlotResolver {
    /// Checks whether a 256-bit storage slot corresponds to any known proxy implementation slot.
    pub fn is_proxy_slot(slot: &B256) -> bool {
        *slot == EIP1967_IMPLEMENTATION_SLOT
            || *slot == EIP1967_ADMIN_SLOT
            || *slot == EIP1967_BEACON_SLOT
    }

    /// Checks whether a 64-bit truncated base slot matches the top 64-bits of EIP-1967 implementation slot.
    pub fn is_truncated_eip1967_slot(slot_id: u64) -> bool {
        let top64 = u64::from_be_bytes([
            EIP1967_IMPLEMENTATION_SLOT[0],
            EIP1967_IMPLEMENTATION_SLOT[1],
            EIP1967_IMPLEMENTATION_SLOT[2],
            EIP1967_IMPLEMENTATION_SLOT[3],
            EIP1967_IMPLEMENTATION_SLOT[4],
            EIP1967_IMPLEMENTATION_SLOT[5],
            EIP1967_IMPLEMENTATION_SLOT[6],
            EIP1967_IMPLEMENTATION_SLOT[7],
        ]);
        slot_id == top64
    }

    /// Resolves proxy relationships in `ProjectCfgGraph` and merges implementation function edges
    /// onto proxy storage nodes.
    pub fn merge_proxy_implementation_edges(
        graph: &mut ProjectCfgGraph,
        proxy_storage: &[NodeIndex],
        impl_functions: &[NodeIndex],
    ) -> GfnResult<()> {
        for &f_idx in impl_functions {
            for &s_idx in proxy_storage {
                // Connect implementation function directly to proxy state
                let mut read_attrs = [0.0f32; 16];
                read_attrs[0] = 1.0; // Proxy delegatecall read
                graph.add_semantic_edge(f_idx, s_idx, EdgeRelation::ReadsFrom, read_attrs);

                let mut write_attrs = [0.0f32; 16];
                write_attrs[0] = 1.0; // Proxy delegatecall write
                graph.add_semantic_edge(f_idx, s_idx, EdgeRelation::WritesTo, write_attrs);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg_lifter::{FunctionNodeData, StorageNodeData};
    use gfn_common::schemas::{SoliditySlotType, StorageSemanticRole};

    #[test]
    fn test_eip1967_slot_constants() {
        assert!(ProxySlotResolver::is_proxy_slot(
            &EIP1967_IMPLEMENTATION_SLOT
        ));
        assert!(ProxySlotResolver::is_proxy_slot(&EIP1967_ADMIN_SLOT));
        assert!(ProxySlotResolver::is_proxy_slot(&EIP1967_BEACON_SLOT));

        let arbitrary_slot =
            b256!("0000000000000000000000000000000000000000000000000000000000000001");
        assert!(!ProxySlotResolver::is_proxy_slot(&arbitrary_slot));
    }

    #[test]
    fn test_merge_proxy_implementation_edges() -> GfnResult<()> {
        let mut cfg = ProjectCfgGraph::new();

        let s_idx = cfg.add_storage(StorageNodeData {
            slot_id: 0,
            solidity_type: SoliditySlotType::Uint256,
            semantic_role: StorageSemanticRole::AssetBalance,
            byte_offset: 0,
            byte_size: 32,
            is_dynamic_mapping: false,
            is_immutable: false,
            is_constant: false,
            is_proxy_implementation: false,
        });

        let f_idx = cfg.add_function(FunctionNodeData {
            selector: [0x12, 0x34, 0x56, 0x78],
            visibility: 0,
            mutability: 1,
            has_only_owner_or_role: false,
            has_non_reentrant: false,
            is_fallback_or_receive: false,
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

        ProxySlotResolver::merge_proxy_implementation_edges(&mut cfg, &[s_idx], &[f_idx])?;

        let mut has_reads = false;
        let mut has_writes = false;
        for edge in cfg.graph.edge_weights() {
            if edge.relation == EdgeRelation::ReadsFrom {
                has_reads = true;
            }
            if edge.relation == EdgeRelation::WritesTo {
                has_writes = true;
            }
        }

        assert!(has_reads);
        assert!(has_writes);
        Ok(())
    }
}
