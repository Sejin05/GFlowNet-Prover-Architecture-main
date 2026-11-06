#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Metamorphic Perturbation Robustness Test Suite (`tests/metamorphic_robustness_test.rs`).
//!
//! Evaluates the 4 semantics-preserving metamorphic perturbation transformations:
//! 1. `ObfuscateNames`: Hashing contract identifiers, source paths, and ABI function names.
//! 2. `PermuteStorage`: Interleaving dummy storage slots and reordering physical slot IDs.
//! 3. `SplitBasicBlocks`: Rewriting compound EVM bytecode conditionals into split `JUMPI` basic blocks.
//! 4. `ScaleBalances1000x`: Scaling initial token endowments/reserves and `PUSH32` constants by 10^3 and 10^-3.
//!
//! Enforces the formal release gate criterion via property-based testing (`proptest`):
//! Semantic role & structural mode recall degradation across randomized perturbations must be <= 2.0%.

use alloy_primitives::U256;
use gfn_common::error::GfnResult;
use gfn_common::math::u256_to_log2_norm;
use gfn_common::schemas::{
    FoundryContractArtifact, SoliditySlotType, StorageSemanticRole, StorageSlotRecord,
};
use gfn_extractor::cfg_lifter::{HsdgNode, ProjectCfgGraph, SsaCfgLifter};
use gfn_extractor::compiler::ProjectExtractor;
use gfn_extractor::hsdg_builder::{HsdgBuilder, HsdgValidator};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestRunner};
use std::collections::BTreeSet;
use tch::Device;
use uuid::Uuid;

/// Helper creating a realistic 4-contract artifact suite with multi-block EVM bytecode.
fn create_synthetic_protocol_artifacts() -> Vec<FoundryContractArtifact> {
    let base_bytecode = vec![
        0x60, 0x80, 0x60, 0x40, 0x52, // PUSH1 0x80 PUSH1 0x40 MSTORE
        0x60, 0x00, 0x54, // PUSH1 0x00 SLOAD
        0x60, 0x18, 0x57, // PUSH1 0x18 JUMPI (Block 1 conditional guard)
        0x5B, // JUMPDEST
        0x60, 0x01, 0x55, // PUSH1 0x01 SSTORE
        0xF1, // CALL
        0x00, // STOP
    ];

    vec![
        FoundryContractArtifact {
            project_id: Uuid::nil(),
            contract_name: "VaultErc4626".to_string(),
            source_path: "src/VaultErc4626.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0x11u8; 32],
            deployed_bytecode: base_bytecode.clone(),
            abi_json_str: r#"[{"type":"function","name":"deposit","inputs":[{"type":"uint256"}],"outputs":[{"type":"uint256"}]},{"type":"function","name":"donateAssets","inputs":[{"type":"uint256"}],"outputs":[]}]"#
                .to_string(),
            storage_slots: vec![
                StorageSlotRecord {
                    slot_id: 0,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::TotalSupply,
                },
                StorageSlotRecord {
                    slot_id: 1,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::AssetBalance,
                },
            ],
        },
        FoundryContractArtifact {
            project_id: Uuid::nil(),
            contract_name: "AmmPool".to_string(),
            source_path: "src/AmmPool.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0x22u8; 32],
            deployed_bytecode: base_bytecode.clone(),
            abi_json_str: r#"[{"type":"function","name":"swap","inputs":[{"type":"uint256"},{"type":"uint256"}],"outputs":[]}]"#
                .to_string(),
            storage_slots: vec![
                StorageSlotRecord {
                    slot_id: 0,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::PoolReserve,
                },
                StorageSlotRecord {
                    slot_id: 1,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::PoolReserve,
                },
            ],
        },
        FoundryContractArtifact {
            project_id: Uuid::nil(),
            contract_name: "LendingMarket".to_string(),
            source_path: "src/LendingMarket.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0x33u8; 32],
            deployed_bytecode: base_bytecode.clone(),
            abi_json_str: r#"[{"type":"function","name":"borrow","inputs":[{"type":"uint256"}],"outputs":[]}]"#
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
                    semantic_role: StorageSemanticRole::DebtShare,
                },
            ],
        },
        FoundryContractArtifact {
            project_id: Uuid::nil(),
            contract_name: "GovTimelock".to_string(),
            source_path: "src/GovTimelock.sol".to_string(),
            solc_version: "0.8.24".to_string(),
            deployed_bytecode_hash: [0x44u8; 32],
            deployed_bytecode: base_bytecode,
            abi_json_str:
                r#"[{"type":"function","name":"executeProposal","inputs":[{"type":"uint256"}],"outputs":[]}]"#
                    .to_string(),
            storage_slots: vec![StorageSlotRecord {
                slot_id: 0,
                byte_offset: 0,
                byte_size: 20,
                solidity_type: SoliditySlotType::Address,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AdminRole,
            }],
        },
    ]
}

fn extract_non_generic_roles(graph: &ProjectCfgGraph) -> BTreeSet<u8> {
    let mut roles = BTreeSet::new();
    for &idx in &graph.storage_indices {
        if let Some(HsdgNode::Storage(s)) = graph.graph.node_weight(idx) {
            if s.semantic_role != StorageSemanticRole::GenericState {
                roles.insert(s.semantic_role as u8);
            }
        }
    }
    roles
}

#[test]
fn test_metamorphic_perturbation_1_obfuscate_names() -> GfnResult<()> {
    let base_artifacts = create_synthetic_protocol_artifacts();
    let base_graph = SsaCfgLifter::lift_artifacts(&base_artifacts)?;
    let base_tensors = HsdgBuilder::convert_cfg_to_tensors(&base_graph, Device::Cpu)?;
    HsdgValidator::validate(&base_tensors)?;

    // Obfuscate contract names, source paths, AND ABI function identifiers
    let mut obfuscated_artifacts = base_artifacts.clone();
    for (idx, artifact) in obfuscated_artifacts.iter_mut().enumerate() {
        let hex_tag = format!("{:06x}", (idx + 1) * 0x1a2b);
        artifact.contract_name = format!("C_{hex_tag}");
        artifact.source_path = format!("src/C_{hex_tag}.sol");
        if let Ok(mut abi_val) = serde_json::from_str::<serde_json::Value>(&artifact.abi_json_str) {
            if let Some(arr) = abi_val.as_array_mut() {
                for (f_idx, entry) in arr.iter_mut().enumerate() {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert(
                            "name".to_string(),
                            serde_json::Value::String(format!("fn_{hex_tag}_{f_idx}")),
                        );
                    }
                }
            }
            if let Ok(serialized) = serde_json::to_string(&abi_val) {
                artifact.abi_json_str = serialized;
            }
        }
    }

    let obf_graph = SsaCfgLifter::lift_artifacts(&obfuscated_artifacts)?;
    let obf_tensors = HsdgBuilder::convert_cfg_to_tensors(&obf_graph, Device::Cpu)?;
    HsdgValidator::validate(&obf_tensors)?;

    assert_eq!(
        base_tensors.storage_x.size(),
        obf_tensors.storage_x.size(),
        "Obfuscation must preserve storage node tensor dimension"
    );
    assert_eq!(
        base_tensors.function_x.size(),
        obf_tensors.function_x.size(),
        "Obfuscation must preserve function node tensor dimension"
    );
    assert_eq!(
        base_tensors.condition_x.size(),
        obf_tensors.condition_x.size(),
        "Obfuscation must preserve condition node tensor dimension"
    );
    assert_eq!(
        extract_non_generic_roles(&base_graph),
        extract_non_generic_roles(&obf_graph),
        "Obfuscation must preserve all storage semantic domain roles"
    );

    Ok(())
}

#[test]
fn test_metamorphic_perturbation_2_permute_storage() -> GfnResult<()> {
    let base_artifacts = create_synthetic_protocol_artifacts();
    let base_graph = SsaCfgLifter::lift_artifacts(&base_artifacts)?;

    // Interleave dummy storage slots and shift physical slot offsets
    let mut permuted_artifacts = base_artifacts.clone();
    for artifact in &mut permuted_artifacts {
        let mut new_layout = Vec::new();
        for (slot_idx, slot) in artifact.storage_slots.iter().enumerate() {
            let mut shifted_slot = slot.clone();
            shifted_slot.slot_id = (slot_idx * 3 + 1) as u64;
            new_layout.push(shifted_slot);

            for d in 0..2 {
                new_layout.push(StorageSlotRecord {
                    slot_id: (slot_idx * 3 + 2 + d) as u64,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::GenericState,
                });
            }
        }
        ProjectExtractor::validate_storage_layout(&new_layout)?;
        artifact.storage_slots = new_layout;
    }

    let perm_graph = SsaCfgLifter::lift_artifacts(&permuted_artifacts)?;
    let perm_tensors = HsdgBuilder::convert_cfg_to_tensors(&perm_graph, Device::Cpu)?;
    HsdgValidator::validate(&perm_tensors)?;

    assert_eq!(
        perm_tensors.storage_x.size()[0],
        base_graph.storage_indices.len() as i64 * 3,
        "Permuted storage layout must reflect exact interleaved slot count"
    );
    assert_eq!(
        extract_non_generic_roles(&base_graph),
        extract_non_generic_roles(&perm_graph),
        "Slot permutation must preserve every non-generic storage role"
    );
    Ok(())
}

#[test]
fn test_metamorphic_perturbation_3_split_basic_blocks() -> GfnResult<()> {
    let base_artifacts = create_synthetic_protocol_artifacts();
    let base_graph = SsaCfgLifter::lift_artifacts(&base_artifacts)?;
    let base_tensors = HsdgBuilder::convert_cfg_to_tensors(&base_graph, Device::Cpu)?;
    HsdgValidator::validate(&base_tensors)?;

    // Rewrite compound require guard in bytecode into two sequential JUMPI (0x57) basic blocks
    let mut split_artifacts = base_artifacts.clone();
    for artifact in &mut split_artifacts {
        let mut rewritten_bytecode = vec![
            0x60, 0x80, 0x60, 0x40, 0x52, // Prologue
            0x60, 0x00, 0x54, // SLOAD
            0x60, 0x12, 0x57, // First split JUMPI guard
            0x5B, // JUMPDEST
            0x60, 0x01, 0x54, // Second SLOAD check
            0x60, 0x20, 0x57, // Second split JUMPI guard
            0x5B, // JUMPDEST
            0x60, 0x01, 0x55, // SSTORE
            0xF1, // CALL
            0x00, // STOP
        ];
        std::mem::swap(&mut artifact.deployed_bytecode, &mut rewritten_bytecode);
    }

    let split_graph = SsaCfgLifter::lift_artifacts(&split_artifacts)?;
    let split_tensors = HsdgBuilder::convert_cfg_to_tensors(&split_graph, Device::Cpu)?;
    HsdgValidator::validate(&split_tensors)?;

    // Storage & function counts stay invariant, while condition nodes double from split JUMPI blocks!
    assert_eq!(
        split_tensors.storage_x.size()[0],
        base_tensors.storage_x.size()[0],
        "Basic block splitting must not alter storage node count"
    );
    assert_eq!(
        split_tensors.function_x.size()[0],
        base_tensors.function_x.size()[0],
        "Basic block splitting must not alter function node count"
    );
    assert!(
        split_tensors.condition_x.size()[0] > base_tensors.condition_x.size()[0],
        "Split JUMPI basic blocks must expand condition node count ({} vs {})",
        split_tensors.condition_x.size()[0],
        base_tensors.condition_x.size()[0]
    );
    Ok(())
}

#[test]
fn test_metamorphic_perturbation_4_scale_balances_1000x() -> GfnResult<()> {
    // 1. Verify log2 normalization monotonicity and Lipschitz bound under 10^3 / 10^-3 scaling
    let raw_val_1 = U256::from(1_000_000_000_000_000_000u64); // 1e18
    let raw_val_scale_up = raw_val_1.saturating_mul(U256::from(1000u64));
    let raw_val_scale_down = raw_val_1 / U256::from(1000u64);

    let norm_base = u256_to_log2_norm(raw_val_1);
    let norm_up = u256_to_log2_norm(raw_val_scale_up);
    let norm_down = u256_to_log2_norm(raw_val_scale_down);

    assert!(norm_base > 0.0 && norm_base <= 1.0);
    assert!(norm_up > norm_base && norm_up <= 1.0);
    assert!(norm_down < norm_base && norm_down >= 0.0);

    let delta = norm_up - norm_base;
    assert!(
        delta < 0.05,
        "Log2 normalized 1000x scale delta must remain strictly < 0.05: got {delta}"
    );

    // 2. Verify opcode histogram invariance when PUSH32 immediate constants are scaled by 1000x
    let mut artifacts_1x = create_synthetic_protocol_artifacts();
    let mut artifacts_1000x = artifacts_1x.clone();

    let mut push32_1x = vec![0x7F]; // PUSH32
    push32_1x.extend_from_slice(&raw_val_1.to_be_bytes::<32>());
    push32_1x.push(0x00); // STOP

    let mut push32_1000x = vec![0x7F]; // PUSH32
    push32_1000x.extend_from_slice(&raw_val_scale_up.to_be_bytes::<32>());
    push32_1000x.push(0x00); // STOP

    artifacts_1x[0].deployed_bytecode = push32_1x;
    artifacts_1000x[0].deployed_bytecode = push32_1000x;

    let g_1x = SsaCfgLifter::lift_artifacts(&artifacts_1x)?;
    let g_1000x = SsaCfgLifter::lift_artifacts(&artifacts_1000x)?;

    let f_1x = match g_1x.graph.node_weight(g_1x.function_indices[0]) {
        Some(HsdgNode::Function(f)) => f.opcode_histogram,
        _ => panic!("Expected function node"),
    };
    let f_1000x = match g_1000x.graph.node_weight(g_1000x.function_indices[0]) {
        Some(HsdgNode::Function(f)) => f.opcode_histogram,
        _ => panic!("Expected function node"),
    };
    assert_eq!(
        f_1x, f_1000x,
        "Opcode histogram must skip PUSH32 immediates and remain 100% invariant to constant scaling"
    );

    Ok(())
}

#[test]
fn test_metamorphic_perturbation_5_reentrancy_delegatecall_wrapper() -> GfnResult<()> {
    let base_artifacts = create_synthetic_protocol_artifacts();
    let base_graph = SsaCfgLifter::lift_artifacts(&base_artifacts)?;
    let base_tensors = HsdgBuilder::convert_cfg_to_tensors(&base_graph, Device::Cpu)?;
    HsdgValidator::validate(&base_tensors)?;

    // Wrap low-level call site into DELEGATECALL (0xF4) / STATICCALL (0xFA) proxy forwarding bytecode
    let mut delegatecall_artifacts = base_artifacts.clone();
    for artifact in &mut delegatecall_artifacts {
        let mut rewritten_bytecode = vec![
            0x60, 0x80, 0x60, 0x40, 0x52, // Prologue
            0x60, 0x00, 0x54,             // SLOAD
            0x60, 0x14, 0x57,             // JUMPI
            0x5B,                         // JUMPDEST
            0x60, 0x01, 0x55,             // SSTORE
            0xF4,                         // DELEGATECALL
            0xFA,                         // STATICCALL
            0x00,                         // STOP
        ];
        std::mem::swap(&mut artifact.deployed_bytecode, &mut rewritten_bytecode);
    }

    let del_graph = SsaCfgLifter::lift_artifacts(&delegatecall_artifacts)?;
    let del_tensors = HsdgBuilder::convert_cfg_to_tensors(&del_graph, Device::Cpu)?;
    HsdgValidator::validate(&del_tensors)?;

    assert_eq!(
        del_tensors.storage_x.size()[0],
        base_tensors.storage_x.size()[0],
        "Delegatecall wrapper must preserve storage node count"
    );
    assert_eq!(
        del_tensors.function_x.size()[0],
        base_tensors.function_x.size()[0],
        "Delegatecall wrapper must preserve function node count"
    );
    assert_eq!(
        extract_non_generic_roles(&base_graph),
        extract_non_generic_roles(&del_graph),
        "Delegatecall wrapper must preserve every non-generic storage role"
    );

    Ok(())
}

#[test]
fn test_proptest_metamorphic_mode_coverage_degradation_release_gate() {
    let base_artifacts = create_synthetic_protocol_artifacts();
    let base_graph = match SsaCfgLifter::lift_artifacts(&base_artifacts) {
        Ok(g) => g,
        Err(e) => panic!("Baseline lift failed: {e}"),
    };
    let baseline_roles = extract_non_generic_roles(&base_graph);
    assert!(
        baseline_roles.len() >= 5,
        "Expected 5 distinct semantic roles in baseline protocol"
    );

    let mut runner = TestRunner::new(ProptestConfig::with_cases(32));

    runner
        .run(&(1u64..500u64, 1usize..4usize), |(slot_stride, dummy_pad)| {
            let mut perturbed = base_artifacts.clone();
            for (c_idx, artifact) in perturbed.iter_mut().enumerate() {
                artifact.contract_name = format!("Obf_{c_idx}_{slot_stride:x}");
                let mut permuted_slots = Vec::new();
                for (s_idx, slot) in artifact.storage_slots.iter().enumerate() {
                    let mut moved = slot.clone();
                    moved.slot_id = ((s_idx + 1) as u64).saturating_mul(slot_stride + 4);
                    permuted_slots.push(moved);
                    for d in 0..dummy_pad {
                        permuted_slots.push(StorageSlotRecord {
                            slot_id: ((s_idx + 1) as u64)
                                .saturating_mul(slot_stride + 4)
                                .saturating_add((d + 1) as u64),
                            byte_offset: 0,
                            byte_size: 32,
                            solidity_type: SoliditySlotType::Uint256,
                            is_dynamic_mapping: false,
                            semantic_role: StorageSemanticRole::GenericState,
                        });
                    }
                }
                prop_assert!(ProjectExtractor::validate_storage_layout(&permuted_slots).is_ok());
                artifact.storage_slots = permuted_slots;
            }

            let pert_graph = SsaCfgLifter::lift_artifacts(&perturbed)
                .map_err(|e| TestCaseError::fail(format!("Lift failed: {e}")))?;
            let pert_tensors = HsdgBuilder::convert_cfg_to_tensors(&pert_graph, Device::Cpu)
                .map_err(|e| TestCaseError::fail(format!("Tensor build failed: {e}")))?;
            HsdgValidator::validate(&pert_tensors)
                .map_err(|e| TestCaseError::fail(format!("Tensor validation failed: {e}")))?;

            let perturbed_roles = extract_non_generic_roles(&pert_graph);
            let retained = baseline_roles.intersection(&perturbed_roles).count() as f64;
            let baseline_total = baseline_roles.len() as f64;
            let degradation = (baseline_total - retained) / baseline_total;

            prop_assert!(
                degradation <= 0.020,
                "Metamorphic role/mode degradation {degradation:.4} exceeds 2.0% release gate"
            );
            Ok(())
        })
        .unwrap_or_else(|e| panic!("Proptest metamorphic gate failed: {e}"));
}
