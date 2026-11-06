#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Inter-Contract Call Resolver and Cross-Contract Edge Linker.
//!
//! Resolves call targets across contracts, identifies Checks-Effects-Interactions
//! reentrancy risks, and links semantic invocation edges across contract boundaries in HSDG.

use crate::cfg_lifter::{EdgeRelation, ProjectCfgGraph};
use alloy_primitives::keccak256;
use gfn_common::error::GfnResult;
use gfn_common::schemas::{FoundryContractArtifact, InterContractCallRecord};
use petgraph::graph::NodeIndex;
use std::collections::BTreeMap;
use std::path::Path;

/// Cross-contract resolver matching call sites to candidate recipient contracts and selectors.
pub struct CrossContractResolver;

impl CrossContractResolver {
    fn is_test_or_script_source(source_path: &str) -> bool {
        let lower = source_path.replace('\\', "/").to_ascii_lowercase();
        lower.starts_with("test/")
            || lower.starts_with("tests/")
            || lower.contains("/test/")
            || lower.contains("/tests/")
            || lower.ends_with(".t.sol")
            || lower.ends_with(".s.sol")
    }

    fn canonicalize_sol_param_type(raw_param: &str) -> Option<String> {
        let tokens: Vec<&str> = raw_param.split_whitespace().collect();
        let raw_t = *tokens.first()?;
        let mapped = match raw_t {
            "uint" => "uint256",
            "int" => "int256",
            "byte" => "bytes1",
            other => other,
        };
        Some(mapped.to_string())
    }

    /// Supplements a selector catalog with `interface` function declarations scanned from `.sol` sources.
    pub fn augment_catalog_with_source_interfaces(
        project_root: &Path,
        catalog: &mut BTreeMap<[u8; 4], Vec<(String, String, String)>>,
    ) {
        for sub in ["src", "contracts"] {
            let dir = project_root.join(sub);
            if !dir.exists() {
                continue;
            }
            Self::scan_interfaces_dir(&dir, catalog);
        }
    }

    fn scan_interfaces_dir(
        dir: &Path,
        catalog: &mut BTreeMap<[u8; 4], Vec<(String, String, String)>>,
    ) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_interfaces_dir(&path, catalog);
            } else if path.extension().and_then(|e| e.to_str()) == Some("sol") {
                if let Ok(src) = std::fs::read_to_string(&path) {
                    Self::parse_interface_selectors_from_source(&src, catalog);
                }
            }
        }
    }

    fn parse_interface_selectors_from_source(
        source: &str,
        catalog: &mut BTreeMap<[u8; 4], Vec<(String, String, String)>>,
    ) {
        let mut current_iface: Option<String> = None;
        let mut brace_depth: usize = 0;
        let mut seen_open_brace = false;

        for raw_line in source.lines() {
            let line = raw_line.split("//").next().unwrap_or(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if current_iface.is_none() && line.starts_with("interface ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(name) = parts.get(1) {
                    let clean = name.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
                    if !clean.is_empty() {
                        current_iface = Some(clean.to_string());
                        brace_depth = 0;
                        seen_open_brace = false;
                    }
                }
            }

            if let Some(iface_name) = current_iface.clone() {
                if let Some(fn_pos) = line.find("function ") {
                    let after_fn = &line[fn_pos + "function ".len()..];
                    if let (Some(open), Some(close)) = (after_fn.find('('), after_fn.find(')')) {
                        if open < close {
                            let fn_name = after_fn[..open].trim();
                            let params_raw = &after_fn[open + 1..close];
                            let mut canonical_types = Vec::new();
                            if !params_raw.trim().is_empty() {
                                for p in params_raw.split(',') {
                                    if let Some(ct) = Self::canonicalize_sol_param_type(p) {
                                        canonical_types.push(ct);
                                    }
                                }
                            }
                            if !fn_name.is_empty() {
                                let sig = format!("{}({})", fn_name, canonical_types.join(","));
                                let hash = keccak256(sig.as_bytes());
                                let mut sel = [0u8; 4];
                                sel.copy_from_slice(&hash[0..4]);
                                let list = catalog.entry(sel).or_default();
                                let entry_tuple = (iface_name, fn_name.to_string(), sig);
                                if !list.contains(&entry_tuple) {
                                    list.push(entry_tuple);
                                }
                            }
                        }
                    }
                }

                let opens = line.chars().filter(|&c| c == '{').count();
                let closes = line.chars().filter(|&c| c == '}').count();
                if opens > 0 {
                    seen_open_brace = true;
                }
                brace_depth = brace_depth.saturating_add(opens);
                brace_depth = brace_depth.saturating_sub(closes);
                if seen_open_brace && brace_depth == 0 {
                    current_iface = None;
                    seen_open_brace = false;
                }
            }
        }
    }

    /// Builds a catalog of all public/external function selectors across artifacts.
    ///
    /// Key: 4-byte selector -> Value: List of `(contract_name, function_name, signature)`
    pub fn build_selector_catalog(
        artifacts: &[FoundryContractArtifact],
    ) -> BTreeMap<[u8; 4], Vec<(String, String, String)>> {
        let mut catalog: BTreeMap<[u8; 4], Vec<(String, String, String)>> = BTreeMap::new();

        for artifact in artifacts {
            if Self::is_test_or_script_source(&artifact.source_path) {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&artifact.abi_json_str) {
                if let Some(items) = value.as_array() {
                    for item in items {
                        if item.get("type").and_then(|t| t.as_str()) == Some("function") {
                            if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                                let mut param_types = Vec::new();
                                if let Some(inputs) = item.get("inputs").and_then(|i| i.as_array())
                                {
                                    for input in inputs {
                                        if let Some(t) = input.get("type").and_then(|s| s.as_str())
                                        {
                                            param_types.push(t);
                                        }
                                    }
                                }
                                let sig = format!("{}({})", name, param_types.join(","));
                                let hash = keccak256(sig.as_bytes());
                                let mut sel = [0u8; 4];
                                sel.copy_from_slice(&hash[0..4]);

                                catalog.entry(sel).or_default().push((
                                    artifact.contract_name.clone(),
                                    name.to_string(),
                                    sig,
                                ));
                            }
                        }
                    }
                }
            }
        }
        catalog
    }

    /// Resolves all inter-contract calls from compiled artifacts by inspecting bytecode and ABI calls.
    pub fn resolve_inter_contract_calls(
        artifacts: &[FoundryContractArtifact],
    ) -> Vec<InterContractCallRecord> {
        let catalog = Self::build_selector_catalog(artifacts);
        Self::resolve_with_catalog(artifacts, &catalog)
    }

    /// Resolves inter-contract calls using both compiled artifacts and interface declarations in `project_root`.
    pub fn resolve_inter_contract_calls_for_project(
        artifacts: &[FoundryContractArtifact],
        project_root: &Path,
    ) -> Vec<InterContractCallRecord> {
        let mut catalog = Self::build_selector_catalog(artifacts);
        Self::augment_catalog_with_source_interfaces(project_root, &mut catalog);
        Self::resolve_with_catalog(artifacts, &catalog)
    }

    fn is_ignored_compiler_selector(sel: [u8; 4]) -> bool {
        matches!(
            sel,
            [0xff, 0xff, 0xff, 0xff]
                | [0x00, 0x00, 0x00, 0x00]
                | [0x4e, 0x48, 0x7b, 0x71] // Panic(uint256)
                | [0x08, 0xc3, 0x79, 0xa0] // Error(string)
        )
    }

    fn has_post_call_sstore(bytecode: &[u8], call_pc: usize) -> bool {
        let mut pc = call_pc.saturating_add(1);
        while pc < bytecode.len() && pc.saturating_sub(call_pc) <= 96 {
            let op = bytecode[pc];
            if op == 0x55 {
                return true;
            }
            // Terminate scan at function-exit opcodes: STOP, RETURN, INVALID, SELFDESTRUCT.
            // Do not break on inline `REVERT` (0xFD) or ABI-helper `JUMP` (0x56), which solc 0.8
            // emits in the post-CALL success check right before continuing to `SSTORE`.
            if matches!(op, 0x00 | 0xF3 | 0xFE | 0xFF) {
                break;
            }
            if (0x60..=0x7F).contains(&op) {
                let push_size = (op - 0x5F) as usize;
                pc = pc.saturating_add(push_size);
            }
            pc = pc.saturating_add(1);
        }
        false
    }

    fn resolve_with_catalog(
        artifacts: &[FoundryContractArtifact],
        catalog: &BTreeMap<[u8; 4], Vec<(String, String, String)>>,
    ) -> Vec<InterContractCallRecord> {
        let has_non_test = artifacts
            .iter()
            .any(|a| !Self::is_test_or_script_source(&a.source_path));
        let mut merged_calls: BTreeMap<(String, u8, [u8; 4]), InterContractCallRecord> =
            BTreeMap::new();

        for artifact in artifacts {
            if has_non_test && Self::is_test_or_script_source(&artifact.source_path) {
                continue;
            }

            let caller_contract = &artifact.contract_name;
            let bytecode = &artifact.deployed_bytecode;

            // Scan bytecode for PUSH4 selectors followed by CALL/DELEGATECALL/STATICCALL opcodes.
            // Ignore Solidity's 0xffffffff/0x00000000 bitmasks and Panic/Error selectors so real
            // target selectors pushed during call setup are preserved across inline solc guards.
            let mut i = 0;
            let mut last_catalog_push4: Option<([u8; 4], usize)> = None;
            let mut last_nonmask_push4: Option<([u8; 4], usize)> = None;

            while i < bytecode.len() {
                let op = bytecode[i];

                if op == 0x63 && i + 4 < bytecode.len() {
                    let mut sel = [0u8; 4];
                    sel.copy_from_slice(&bytecode[i + 1..i + 5]);
                    if !Self::is_ignored_compiler_selector(sel) {
                        last_nonmask_push4 = Some((sel, i));
                        if catalog.contains_key(&sel) {
                            last_catalog_push4 = Some((sel, i));
                        }
                    }
                    i = i.saturating_add(4);
                } else if op == 0xF1 || op == 0xF4 || op == 0xFA {
                    // CALL (0xF1), DELEGATECALL (0xF4), STATICCALL (0xFA)
                    let call_opcode = match op {
                        0xF1 => 0,
                        0xF4 => 1,
                        0xFA => 2,
                        _ => 0,
                    };

                    // Only state-capable calls (CALL / DELEGATECALL) create direct CEI reentrancy risk
                    let writes_storage_after_call =
                        (op == 0xF1 || op == 0xF4) && Self::has_post_call_sstore(bytecode, i);

                    let nearby_sel = last_catalog_push4
                        .filter(|(_, pc)| i.saturating_sub(*pc) <= 128)
                        .map(|(sel, _)| sel)
                        .or_else(|| {
                            last_nonmask_push4
                                .filter(|(_, pc)| i.saturating_sub(*pc) <= 128)
                                .map(|(sel, _)| sel)
                        });

                    let resolved_sel = match nearby_sel {
                        Some(sel) => Some(sel),
                        None if op == 0xF1 || op == 0xF4 => Some([0u8; 4]),
                        None => None,
                    };

                    if let Some(sel) = resolved_sel {
                        let (hint_contract, hint_fn) = if sel == [0u8; 4] {
                            (None, Some("<arbitrary_calldata>".to_string()))
                        } else {
                            match catalog.get(&sel) {
                                Some(matches) => {
                                    let target_match = matches
                                        .iter()
                                        .find(|(c, _, _)| c != caller_contract)
                                        .or_else(|| matches.first());

                                    match target_match {
                                        Some((c, f, _)) => (Some(c.clone()), Some(f.clone())),
                                        None => (None, None),
                                    }
                                }
                                None => (None, None),
                            }
                        };

                        let key = (caller_contract.clone(), call_opcode, sel);
                        merged_calls
                            .entry(key)
                            .and_modify(|existing| {
                                existing.is_reentrant_risk |= writes_storage_after_call;
                            })
                            .or_insert(InterContractCallRecord {
                                caller_contract: caller_contract.clone(),
                                caller_function: "executeOperation".to_string(),
                                call_opcode,
                                target_contract_hint: hint_contract,
                                target_selector: sel,
                                target_function_name: hint_fn,
                                is_reentrant_risk: writes_storage_after_call,
                            });
                    }
                    last_catalog_push4 = None;
                    last_nonmask_push4 = None;
                } else if matches!(op, 0x00 | 0xF3 | 0xFE | 0xFF) {
                    last_catalog_push4 = None;
                    last_nonmask_push4 = None;
                } else if (0x60..=0x7F).contains(&op) {
                    let push_size = (op - 0x5F) as usize;
                    i = i.saturating_add(push_size);
                }

                i = i.saturating_add(1);
            }
        }

        merged_calls.into_values().collect()
    }

    /// Links cross-contract `EdgeRelation::Invokes` edges directly in `ProjectCfgGraph`.
    pub fn link_cross_contract_graph_edges(
        graph: &mut ProjectCfgGraph,
        contract_functions: &BTreeMap<String, Vec<([u8; 4], NodeIndex)>>,
        calls: &[InterContractCallRecord],
    ) -> GfnResult<usize> {
        let mut linked_edges = 0usize;

        for call in calls {
            if let Some(target_contract) = &call.target_contract_hint {
                // Find caller functions in caller_contract
                if let Some(caller_fns) = contract_functions.get(&call.caller_contract) {
                    // Find target functions matching selector in target_contract
                    if let Some(target_fns) = contract_functions.get(target_contract) {
                        for (_, caller_idx) in caller_fns {
                            for (t_sel, target_idx) in target_fns {
                                if *t_sel == call.target_selector || *t_sel == [0, 0, 0, 0] {
                                    let mut attrs = [0.0f32; 16];
                                    attrs[0] = 1.0; // Basic block depth 1
                                    attrs[1] = 1.0; // Cross-contract flag
                                    if call.is_reentrant_risk {
                                        attrs[2] = 1.0; // Reentrancy risk flag
                                    }

                                    graph.add_semantic_edge(
                                        *caller_idx,
                                        *target_idx,
                                        EdgeRelation::Invokes,
                                        attrs,
                                    );
                                    linked_edges = linked_edges.saturating_add(1);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(linked_edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_selector_catalog_and_call_resolution() {
        let artifacts = vec![
            FoundryContractArtifact {
                project_id: Uuid::nil(),
                contract_name: "AmmPool".to_string(),
                source_path: "src/AmmPool.sol".to_string(),
                solc_version: "0.8.24".to_string(),
                deployed_bytecode_hash: [0u8; 32],
                deployed_bytecode: vec![0x60, 0x80],
                abi_json_str: r#"[{"type":"function","name":"swap","inputs":[{"type":"uint256"},{"type":"uint256"}],"outputs":[]}]"#
                    .to_string(),
                storage_slots: vec![],
            },
            FoundryContractArtifact {
                project_id: Uuid::nil(),
                contract_name: "LendingMarket".to_string(),
                source_path: "src/LendingMarket.sol".to_string(),
                solc_version: "0.8.24".to_string(),
                deployed_bytecode_hash: [0u8; 32],
                // Bytecode pushes 4-byte selector then executes CALL (0xF1) followed by SSTORE (0x55)
                deployed_bytecode: vec![
                    0x63, 0x02, 0x2c, 0x0d, 0x62, // PUSH4
                    0xF1,                         // CALL
                    0x55,                         // SSTORE
                ],
                abi_json_str: r#"[{"type":"function","name":"borrow","inputs":[{"type":"uint256"}],"outputs":[]}]"#
                    .to_string(),
                storage_slots: vec![],
            },
        ];

        let catalog = CrossContractResolver::build_selector_catalog(&artifacts);
        assert!(!catalog.is_empty());

        let calls = CrossContractResolver::resolve_inter_contract_calls(&artifacts);
        assert!(!calls.is_empty());
        assert_eq!(calls[0].caller_contract, "LendingMarket");
        assert!(calls[0].is_reentrant_risk);
    }

    #[test]
    fn test_selector_bitmask_and_push_immediate_immunity() {
        let mut catalog = BTreeMap::new();
        CrossContractResolver::parse_interface_selectors_from_source(
            "interface IAmmCallback { function ammCallback(uint256 a0, uint256 a1, bytes calldata data) external; }",
            &mut catalog,
        );
        // IAmmCallback.ammCallback(uint256,uint256,bytes) -> 0x467af484
        assert!(catalog.contains_key(&[0x46, 0x7a, 0xf4, 0x84]));

        let artifacts = vec![
            FoundryContractArtifact {
                project_id: Uuid::nil(),
                contract_name: "Oracle".to_string(),
                source_path: "src/Oracle.sol".to_string(),
                solc_version: "0.8.24".to_string(),
                deployed_bytecode_hash: [0u8; 32],
                deployed_bytecode: vec![0x60, 0x80],
                abi_json_str: r#"[{"type":"function","name":"getPrice","inputs":[],"outputs":[{"type":"uint256"}]}]"#
                    .to_string(),
                storage_slots: vec![],
            },
            FoundryContractArtifact {
                project_id: Uuid::nil(),
                contract_name: "Caller".to_string(),
                source_path: "src/Caller.sol".to_string(),
                solc_version: "0.8.24".to_string(),
                deployed_bytecode_hash: [0u8; 32],
                deployed_bytecode: vec![
                    // PUSH4 0x98d5fdca (getPrice()), PUSH4 0xffffffff (mask), STATICCALL, PUSH2 0x5500 (immediate 0x55!), RETURN
                    0x63, 0x98, 0xd5, 0xfd, 0xca,
                    0x63, 0xff, 0xff, 0xff, 0xff,
                    0xFA,
                    0x61, 0x55, 0x00,
                    0xF3,
                    // Callback CALL with inline 0xFD revert check followed by real SSTORE (0x55)
                    0x63, 0x46, 0x7a, 0xf4, 0x84,
                    0xFD,
                    0xF1,
                    0xFD,
                    0x55,
                    0x00,
                ],
                abi_json_str: "[]".to_string(),
                storage_slots: vec![],
            },
            FoundryContractArtifact {
                project_id: Uuid::nil(),
                contract_name: "BenignUserTest".to_string(),
                source_path: "test/BenignUserTest.t.sol".to_string(),
                solc_version: "0.8.24".to_string(),
                deployed_bytecode_hash: [0u8; 32],
                deployed_bytecode: vec![0x63, 0x98, 0xd5, 0xfd, 0xca, 0xF1, 0x55],
                abi_json_str: "[]".to_string(),
                storage_slots: vec![],
            },
        ];

        for (k, v) in CrossContractResolver::build_selector_catalog(&artifacts) {
            catalog.entry(k).or_default().extend(v);
        }
        let calls = CrossContractResolver::resolve_with_catalog(&artifacts, &catalog);
        // Test contract `BenignUserTest` must be excluded when `src/` contracts are present
        assert!(calls.iter().all(|c| c.caller_contract == "Caller"));
        assert_eq!(calls.len(), 2);

        let oracle_call = calls
            .iter()
            .find(|c| c.target_selector == [0x98, 0xd5, 0xfd, 0xca]);
        assert!(oracle_call.is_some());
        if let Some(oc) = oracle_call {
            assert_eq!(oc.target_contract_hint.as_deref(), Some("Oracle"));
            assert!(!oc.is_reentrant_risk);
        }

        let cb_call = calls
            .iter()
            .find(|c| c.target_selector == [0x46, 0x7a, 0xf4, 0x84]);
        assert!(cb_call.is_some());
        if let Some(cb) = cb_call {
            assert_eq!(cb.target_contract_hint.as_deref(), Some("IAmmCallback"));
            assert_eq!(cb.target_function_name.as_deref(), Some("ammCallback"));
            assert!(cb.is_reentrant_risk);
        }
    }
}
