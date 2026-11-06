#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Whole-Project Semantic Context Extractor.
//!
//! Orchestrates end-to-end multi-contract ingestion by unifying compilation artifacts,
//! AST topology, cross-contract calls, and harness deployment context into a cohesive
//! `ProtocolSemanticModel`.

use crate::compiler::ProjectExtractor;
use crate::cross_contract::CrossContractResolver;
use crate::topology::ProjectTopologyAnalyzer;
use gfn_common::error::GfnResult;
use gfn_common::schemas::{FoundryContractArtifact, ProtocolSemanticModel};
use std::path::{Path, PathBuf};

/// High-level project context extractor constructing a complete `ProtocolSemanticModel`.
pub struct ProjectContextExtractor {
    pub project_root: PathBuf,
    pub solc_version: String,
    pub evm_version: String,
    pub setup_contract: String,
}

impl ProjectContextExtractor {
    /// Creates a new `ProjectContextExtractor`.
    pub fn new(
        project_root: &Path,
        solc_version: &str,
        evm_version: &str,
        setup_contract: &str,
    ) -> Self {
        Self {
            project_root: project_root.to_path_buf(),
            solc_version: solc_version.to_string(),
            evm_version: evm_version.to_string(),
            setup_contract: setup_contract.to_string(),
        }
    }

    /// Extracts the complete whole-project `ProtocolSemanticModel`.
    pub fn extract_model(&self) -> GfnResult<ProtocolSemanticModel> {
        let project_id = ProjectExtractor::compute_project_id(&self.project_root)?;

        let project_name = self
            .project_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("SmartContractProject")
            .to_string();

        // 1. Programmatic compilation and artifact extraction
        let extractor = ProjectExtractor::new(&self.project_root, &self.solc_version);
        let artifacts = extractor.extract().unwrap_or_else(|_| Vec::new());

        // 2. Extract AST contract dependencies and inheritance tree
        let contracts = ProjectTopologyAnalyzer::extract_contract_dependencies(&self.project_root)?;

        // 3. Compute topological order
        let topological_order = ProjectTopologyAnalyzer::compute_topological_order(&contracts);

        // 4. Extract harness deployment context from setup contract
        let harness_context = ProjectTopologyAnalyzer::extract_harness_deployment_context(
            &self.project_root,
            &self.setup_contract,
        )?;

        // 5. Discover inter-contract calls and reentrancy risks (including source interfaces)
        let inter_contract_calls = CrossContractResolver::resolve_inter_contract_calls_for_project(
            &artifacts,
            &self.project_root,
        );

        Ok(ProtocolSemanticModel {
            project_id,
            project_name,
            workspace_root: self.project_root.to_string_lossy().to_string(),
            solc_version: self.solc_version.clone(),
            evm_version: self.evm_version.clone(),
            contracts,
            inter_contract_calls,
            harness_context,
            topological_order,
            artifacts,
        })
    }

    /// Helper to extract model with pre-compiled artifacts.
    pub fn extract_model_with_artifacts(
        &self,
        artifacts: Vec<FoundryContractArtifact>,
    ) -> GfnResult<ProtocolSemanticModel> {
        let project_id = ProjectExtractor::compute_project_id(&self.project_root)?;

        let project_name = self
            .project_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("SmartContractProject")
            .to_string();

        let contracts = ProjectTopologyAnalyzer::extract_contract_dependencies(&self.project_root)?;
        let topological_order = ProjectTopologyAnalyzer::compute_topological_order(&contracts);
        let harness_context = ProjectTopologyAnalyzer::extract_harness_deployment_context(
            &self.project_root,
            &self.setup_contract,
        )?;
        let inter_contract_calls = CrossContractResolver::resolve_inter_contract_calls_for_project(
            &artifacts,
            &self.project_root,
        );

        Ok(ProtocolSemanticModel {
            project_id,
            project_name,
            workspace_root: self.project_root.to_string_lossy().to_string(),
            solc_version: self.solc_version.clone(),
            evm_version: self.evm_version.clone(),
            contracts,
            inter_contract_calls,
            harness_context,
            topological_order,
            artifacts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_extractor_initialization() {
        let root = Path::new("tests/fixtures/multi_bug_defi_protocol");
        let extractor = ProjectContextExtractor::new(root, "0.8.24", "cancun", "ExploitHarness");
        assert_eq!(extractor.solc_version, "0.8.24");
        assert_eq!(extractor.evm_version, "cancun");
        assert_eq!(extractor.setup_contract, "ExploitHarness");
    }

    #[test]
    fn test_extract_multi_bug_model_topology() -> Result<(), Box<dyn std::error::Error>> {
        let p1 = PathBuf::from("tests/fixtures/multi_bug_defi_protocol");
        let root = if p1.exists() {
            p1
        } else {
            PathBuf::from("../../tests/fixtures/multi_bug_defi_protocol")
        };

        if root.exists() {
            let extractor =
                ProjectContextExtractor::new(&root, "0.8.24", "cancun", "ExploitHarness");
            let model = extractor.extract_model_with_artifacts(Vec::new())?;

            assert!(!model.contracts.is_empty());
            let names: Vec<&str> = model
                .contracts
                .iter()
                .map(|c| c.contract_name.as_str())
                .collect();
            assert!(names.contains(&"VaultErc4626"));
            assert!(names.contains(&"AmmPool"));
            assert!(names.contains(&"LendingMarket"));
            assert!(names.contains(&"GovTimelock"));

            assert!(!model.topological_order.is_empty());

            if let Some(harness) = &model.harness_context {
                assert_eq!(harness.harness_contract, "ExploitHarness");
                assert!(
                    !harness.instantiated_contracts.is_empty()
                        || !harness.address_bindings.is_empty()
                );
            }
        }
        Ok(())
    }
}
