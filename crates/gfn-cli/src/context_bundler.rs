#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Multi-Contract Project Context Bundler and AI Prompt Generator.
//!
//! Synthesizes whole-project compilation artifacts, topological hierarchies,
//! inter-contract interaction matrices, and deployment wiring into a cohesive,
//! token-optimized prompt bundle enabling AI models to ingest and audit entire
//! protocols simultaneously without file-by-file fragmentation.

use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::ProtocolSemanticModel;
use std::path::Path;

/// Bundler synthesizing whole-project semantic models into structured AI prompt bundles.
pub struct ProjectContextBundler;

impl ProjectContextBundler {
    /// Serializes the `ProtocolSemanticModel` into structured JSON.
    pub fn export_json(model: &ProtocolSemanticModel) -> GfnResult<String> {
        serde_json::to_string_pretty(model).map_err(|e| GfnError::Serialization {
            details: format!("Failed to serialize ProtocolSemanticModel to JSON: {e}"),
        })
    }

    /// Generates a unified, token-optimized Markdown prompt bundle for whole-project AI comprehension.
    pub fn export_ai_prompt_bundle(
        model: &ProtocolSemanticModel,
        project_root: Option<&Path>,
    ) -> GfnResult<String> {
        let mut doc = String::with_capacity(16384);

        // 1. Title & Header
        doc.push_str("# Unified Smart Contract Project Semantic Context & AI Audit Bundle\n\n");
        doc.push_str("> **Autonomous AI Ingestion Manifest:** This document unifies the entire multi-contract repository into a single coherent system representation, including contract dependencies, deployment topology, inter-contract call graphs, and storage layouts.\n\n");

        // 2. Project Metadata Table
        doc.push_str("## 1. System Metadata & Compiler Specification\n\n");
        doc.push_str("| Attribute | Specification Value |\n");
        doc.push_str("| :--- | :--- |\n");
        doc.push_str(&format!(
            "| **Project Name** | `{}` |\n",
            model.project_name
        ));
        doc.push_str(&format!("| **Project UUID** | `{}` |\n", model.project_id));
        doc.push_str(&format!(
            "| **Workspace Root** | `{}` |\n",
            model.workspace_root
        ));
        doc.push_str(&format!(
            "| **Solidity Compiler** | `solc {}` |\n",
            model.solc_version
        ));
        doc.push_str(&format!("| **EVM Target** | `{}` |\n", model.evm_version));
        doc.push_str(&format!(
            "| **Total Contracts Discovered** | `{}` |\n",
            model.contracts.len()
        ));
        doc.push_str(&format!(
            "| **Compiled Artifacts** | `{}` |\n\n",
            model.artifacts.len()
        ));

        // 3. Topological Contract Hierarchy
        doc.push_str("## 2. Topological Contract Hierarchy & Roles\n\n");
        doc.push_str("Contracts are ordered below in **topological dependency sequence** (base interfaces and libraries first, followed by dependent business logic):\n\n");
        doc.push_str("| Order | Contract Name | Source Path | Type / Role | Inherited Bases | Libraries Used |\n");
        doc.push_str("| :---: | :--- | :--- | :--- | :--- | :--- |\n");

        for (idx, contract_name) in model.topological_order.iter().enumerate() {
            if let Some(c) = model
                .contracts
                .iter()
                .find(|n| &n.contract_name == contract_name)
            {
                let role = if c.is_test {
                    "Test Harness"
                } else if c.is_interface {
                    "Interface"
                } else if c.is_library {
                    "Library"
                } else {
                    "Core Contract"
                };

                let bases = if c.inherited_contracts.is_empty() {
                    "-".to_string()
                } else {
                    c.inherited_contracts.join(", ")
                };

                let libs = if c.libraries_used.is_empty() {
                    "-".to_string()
                } else {
                    c.libraries_used.join(", ")
                };

                doc.push_str(&format!(
                    "| {} | **{}** | `{}` | {} | {} | {} |\n",
                    idx + 1,
                    c.contract_name,
                    c.source_path,
                    role,
                    bases,
                    libs
                ));
            } else {
                doc.push_str(&format!(
                    "| {} | **{}** | - | External / Imported | - | - |\n",
                    idx + 1,
                    contract_name
                ));
            }
        }
        doc.push('\n');

        // 4. Test Harness Deployment & Initialization Topology
        if let Some(harness) = &model.harness_context {
            doc.push_str(
                "## 3. Protocol Deployment & Harness Initialization State (`setUp()`)\n\n",
            );
            doc.push_str(&format!(
                "Extracted from test harness **`{}`** (`{}`):\n\n",
                harness.harness_contract, harness.harness_source_path
            ));

            if !harness.address_bindings.is_empty() {
                doc.push_str("### Contract Instantiation & Variable Bindings\n\n");
                doc.push_str("| Variable Name | Target Contract / Address |\n");
                doc.push_str("| :--- | :--- |\n");
                for (var_name, binding) in &harness.address_bindings {
                    doc.push_str(&format!("| `{var_name}` | `{binding}` |\n"));
                }
                doc.push('\n');
            }

            if !harness.setup_actions_summary.is_empty() {
                doc.push_str("### Initial State & Liquidity Configuration Actions\n\n");
                for action in &harness.setup_actions_summary {
                    doc.push_str(&format!("- `{action}`\n"));
                }
                doc.push('\n');
            }
        }

        // 5. Inter-Contract Interaction & Reentrancy Risk Matrix
        doc.push_str("## 4. Cross-Contract Interaction Matrix & Reentrancy Risks\n\n");
        if model.inter_contract_calls.is_empty() {
            doc.push_str(
                "_No direct inter-contract call instructions detected in compiled bytecode._\n\n",
            );
        } else {
            doc.push_str("| Caller Contract | Opcode | Target Contract | Target Function | Selector | CEI Reentrancy Risk |\n");
            doc.push_str("| :--- | :---: | :--- | :--- | :---: | :---: |\n");

            for call in &model.inter_contract_calls {
                let op_str = match call.call_opcode {
                    0 => "CALL",
                    1 => "DELEGATECALL",
                    2 => "STATICCALL",
                    _ => "CALL",
                };

                let target_c = call
                    .target_contract_hint
                    .as_deref()
                    .unwrap_or("Dynamic / External");
                let target_f = call.target_function_name.as_deref().unwrap_or("Unknown");
                let sel_hex = format!(
                    "0x{:02x}{:02x}{:02x}{:02x}",
                    call.target_selector[0],
                    call.target_selector[1],
                    call.target_selector[2],
                    call.target_selector[3]
                );

                let risk_badge = if call.is_reentrant_risk {
                    "⚠️ **HIGH (State write follows call)**"
                } else {
                    "Normal"
                };

                doc.push_str(&format!(
                    "| **{}** | `{}` | **{}** | `{}` | `{}` | {} |\n",
                    call.caller_contract, op_str, target_c, target_f, sel_hex, risk_badge
                ));
            }
            doc.push('\n');
        }

        // 6. Global Storage Layout & Invariant Surface
        doc.push_str("## 5. Storage Layout & Domain Semantic Roles\n\n");
        for artifact in &model.artifacts {
            if !artifact.storage_slots.is_empty() {
                doc.push_str(&format!("### Contract: `{}`\n\n", artifact.contract_name));
                doc.push_str("| Slot ID | Offset | Size | Solidity Type | Dynamic Mapping | Semantic Role |\n");
                doc.push_str("| :---: | :---: | :---: | :--- | :---: | :--- |\n");

                for slot in &artifact.storage_slots {
                    doc.push_str(&format!(
                        "| `{}` | {} | {} | `{:?}` | {} | `{:?}` |\n",
                        slot.slot_id,
                        slot.byte_offset,
                        slot.byte_size,
                        slot.solidity_type,
                        if slot.is_dynamic_mapping { "Yes" } else { "No" },
                        slot.semantic_role
                    ));
                }
                doc.push('\n');
            }
        }

        // 7. Multi-Contract Source Code Catalog
        if let Some(root) = project_root {
            doc.push_str("## 6. Multi-Contract Source Code Repository\n\n");
            doc.push_str("Below are the complete source codes of the protocol contracts, presented in topological dependency order:\n\n");

            for contract_name in &model.topological_order {
                if let Some(node) = model
                    .contracts
                    .iter()
                    .find(|n| &n.contract_name == contract_name)
                {
                    let file_path = root.join(&node.source_path);
                    if file_path.exists() {
                        if let Ok(src) = std::fs::read_to_string(&file_path) {
                            doc.push_str(&format!(
                                "### Source: `{}` (`{}`)\n\n",
                                node.contract_name, node.source_path
                            ));
                            doc.push_str("```solidity\n");
                            doc.push_str(&src);
                            if !src.ends_with('\n') {
                                doc.push('\n');
                            }
                            doc.push_str("```\n\n");
                        }
                    }
                }
            }
        }

        // 8. Guidance for AI Auditor
        doc.push_str("## 7. AI System Audit Guidance\n\n");
        doc.push_str(
            "When analyzing this unified multi-contract protocol, systematically check:\n",
        );
        doc.push_str("1. **Cross-Contract Invariant Desynchronization**: Do pricing updates in pools instantaneously propagate to lending market collateral valuation?\n");
        doc.push_str("2. **Checks-Effects-Interactions (CEI) Violations**: Look for external calls flagged with reentrancy risks where storage is modified afterward.\n");
        doc.push_str("3. **Precision & Rounding Dust Invariants**: Can an attacker deposit zero or 1 wei of assets in vaults to inflate share prices?\n");
        doc.push_str("4. **Privilege & Governance Isolation**: Are administrative roles protected across all proxy and timelock interfaces?\n");

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfn_common::schemas::{ContractDependencyNode, InterContractCallRecord};
    use uuid::Uuid;

    #[test]
    fn test_export_ai_prompt_bundle() -> GfnResult<()> {
        let model = ProtocolSemanticModel {
            project_id: Uuid::nil(),
            project_name: "MockDeFi".to_string(),
            workspace_root: "/test/root".to_string(),
            solc_version: "0.8.24".to_string(),
            evm_version: "cancun".to_string(),
            contracts: vec![ContractDependencyNode {
                contract_name: "AmmPool".to_string(),
                source_path: "src/AmmPool.sol".to_string(),
                is_test: false,
                is_interface: false,
                is_library: false,
                inherited_contracts: vec![],
                libraries_used: vec![],
                constructor_params: vec![],
            }],
            inter_contract_calls: vec![InterContractCallRecord {
                caller_contract: "LendingMarket".to_string(),
                caller_function: "borrow".to_string(),
                call_opcode: 0,
                target_contract_hint: Some("AmmPool".to_string()),
                target_selector: [0x01, 0x02, 0x03, 0x04],
                target_function_name: Some("swap".to_string()),
                is_reentrant_risk: true,
            }],
            harness_context: None,
            topological_order: vec!["AmmPool".to_string()],
            artifacts: vec![],
        };

        let bundle = ProjectContextBundler::export_ai_prompt_bundle(&model, None)?;
        assert!(
            bundle.contains("# Unified Smart Contract Project Semantic Context & AI Audit Bundle")
        );
        assert!(bundle.contains("MockDeFi"));
        assert!(bundle.contains("AmmPool"));
        assert!(bundle.contains("HIGH (State write follows call)"));

        let json_str = ProjectContextBundler::export_json(&model)?;
        assert!(json_str.contains("MockDeFi"));

        Ok(())
    }
}
