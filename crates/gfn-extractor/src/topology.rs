#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Project Topology Analyzer and Harness Deployment Extractor.
//!
//! Analyzes multi-contract Solidity codebases to extract inheritance hierarchies,
//! library dependencies, constructor parameters, topological build orders,
//! and deployment context from test harnesses (`setUp()`).

use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::{ContractDependencyNode, HarnessDeploymentContext};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

/// Topology analyzer extracting multi-contract structural dependencies and harness context.
pub struct ProjectTopologyAnalyzer;

impl ProjectTopologyAnalyzer {
    /// Evaluates whether a file path matches zero-day exploit/attack leakage patterns.
    pub fn is_leakage_path(path: &Path) -> bool {
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => return false,
        };
        let lower = file_name.to_ascii_lowercase();
        lower.contains("exploit") || lower.contains("attack") || lower.contains("poc")
    }

    /// Scans a project directory and extracts all `ContractDependencyNode` definitions.
    pub fn extract_contract_dependencies(
        project_root: &Path,
    ) -> GfnResult<Vec<ContractDependencyNode>> {
        let mut nodes = Vec::new();
        let search_dirs = [
            project_root.join("src"),
            project_root.join("contracts"),
            project_root.join("test"),
            project_root.join("tests"),
        ];

        for dir in &search_dirs {
            if dir.exists() {
                Self::scan_directory_for_dependencies(dir, project_root, &mut nodes)?;
            }
        }

        // Sort nodes by contract name for determinism
        nodes.sort_by(|a, b| a.contract_name.cmp(&b.contract_name));
        Ok(nodes)
    }

    fn scan_directory_for_dependencies(
        dir: &Path,
        root: &Path,
        nodes: &mut Vec<ContractDependencyNode>,
    ) -> GfnResult<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| GfnError::CompilationArtifact {
            reason: format!("Failed to read directory {}: {e}", dir.display()),
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| GfnError::CompilationArtifact {
                reason: format!("Failed entry read: {e}"),
            })?;
            let path = entry.path();

            if path.is_dir() {
                // Avoid recursing into node_modules, .git, target, lib/forge-std
                let path_str = path.to_string_lossy();
                if !path_str.contains(".git")
                    && !path_str.contains("target")
                    && !path_str.contains("node_modules")
                {
                    Self::scan_directory_for_dependencies(&path, root, nodes)?;
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("sol") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let rel_path = path
                        .strip_prefix(root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();
                    let parsed_nodes = Self::parse_contracts_from_source(&content, &rel_path);
                    nodes.extend(parsed_nodes);
                }
            }
        }
        Ok(())
    }

    /// Parses contract, interface, and library declarations from a Solidity source string.
    pub fn parse_contracts_from_source(
        source: &str,
        source_path: &str,
    ) -> Vec<ContractDependencyNode> {
        let mut nodes = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Detect contract / interface / library declarations
            let is_contract =
                line.starts_with("contract ") || line.starts_with("abstract contract ");
            let is_interface = line.starts_with("interface ");
            let is_library = line.starts_with("library ");

            if is_contract || is_interface || is_library {
                // Collect full declaration header which might span multiple lines until '{'
                let mut decl_header = line.to_string();
                while !decl_header.contains('{') && i + 1 < lines.len() {
                    i = i.saturating_add(1);
                    decl_header.push(' ');
                    decl_header.push_str(lines[i].trim());
                }

                if let Some(brace_pos) = decl_header.find('{') {
                    decl_header.truncate(brace_pos);
                }

                let tokens: Vec<&str> = decl_header.split_whitespace().collect();
                let mut contract_name = String::new();
                let mut inherited_contracts = Vec::new();
                let mut has_is = false;

                for (idx, &token) in tokens.iter().enumerate() {
                    if (token == "contract" || token == "interface" || token == "library")
                        && idx + 1 < tokens.len()
                    {
                        contract_name = tokens[idx + 1].replace(['(', ')'], "");
                    } else if token == "is" {
                        has_is = true;
                    } else if has_is {
                        let clean = token.trim_matches(',').trim();
                        if !clean.is_empty() {
                            inherited_contracts.push(clean.to_string());
                        }
                    }
                }

                if !contract_name.is_empty() {
                    let is_test = contract_name.ends_with("Test")
                        || source_path.contains("test")
                        || inherited_contracts.contains(&"Test".to_string());

                    // Scan body for `using X for Y;` and `constructor(...)` with exact brace-depth tracking
                    let mut libraries_used = Vec::new();
                    let mut constructor_params = Vec::new();

                    let mut body_idx = i;
                    let mut depth: usize = 0;
                    while body_idx < lines.len() {
                        let raw_body_line = lines[body_idx];
                        let body_line = raw_body_line
                            .split("//")
                            .next()
                            .unwrap_or(raw_body_line)
                            .trim();
                        let opens = body_line.chars().filter(|&c| c == '{').count();
                        let closes = body_line.chars().filter(|&c| c == '}').count();
                        depth = depth.saturating_add(opens);

                        if body_line.starts_with("using ") {
                            let parts: Vec<&str> = body_line.split_whitespace().collect();
                            if parts.len() >= 2 {
                                libraries_used.push(parts[1].to_string());
                            }
                        }

                        if body_line.starts_with("constructor") {
                            if let Some(open) = body_line.find('(') {
                                if let Some(close) = body_line.find(')') {
                                    if open < close {
                                        let params_str = &body_line[open + 1..close];
                                        for p in params_str.split(',') {
                                            let trimmed = p.trim();
                                            if !trimmed.is_empty() {
                                                constructor_params.push(trimmed.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        depth = depth.saturating_sub(closes);
                        body_idx = body_idx.saturating_add(1);
                        if depth == 0 && opens + closes > 0 {
                            break;
                        }
                    }

                    nodes.push(ContractDependencyNode {
                        contract_name,
                        source_path: source_path.to_string(),
                        is_test,
                        is_interface,
                        is_library,
                        inherited_contracts,
                        libraries_used,
                        constructor_params,
                    });
                }
            }
            i = i.saturating_add(1);
        }

        nodes
    }

    /// Computes a deterministic topological dependency ordering of contracts.
    pub fn compute_topological_order(nodes: &[ContractDependencyNode]) -> Vec<String> {
        let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
        let mut adj_list: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let all_names: BTreeSet<&str> = nodes.iter().map(|n| n.contract_name.as_str()).collect();

        for name in &all_names {
            in_degree.insert(name, 0);
            adj_list.insert(name, Vec::new());
        }

        for node in nodes {
            let child = node.contract_name.as_str();
            for parent in &node.inherited_contracts {
                let parent_str = parent.as_str();
                if all_names.contains(parent_str) {
                    if let Some(list) = adj_list.get_mut(parent_str) {
                        list.push(child);
                    }
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg = deg.saturating_add(1);
                    }
                }
            }
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (name, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(name);
            }
        }

        let mut ordered = Vec::new();
        while let Some(u) = queue.pop_front() {
            ordered.push(u.to_string());
            if let Some(neighbors) = adj_list.get(u) {
                for &v in neighbors {
                    if let Some(deg) = in_degree.get_mut(v) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        // If cycles or remaining nodes exist, append them deterministically
        for node in nodes {
            if !ordered.contains(&node.contract_name) {
                ordered.push(node.contract_name.clone());
            }
        }

        ordered
    }

    /// Extracts test harness deployment context from `setUp()` in test files, including inherited base harnesses.
    pub fn extract_harness_deployment_context(
        project_root: &Path,
        harness_contract: &str,
    ) -> GfnResult<Option<HarnessDeploymentContext>> {
        let test_dirs = [
            project_root.join("test"),
            project_root.join("tests"),
            project_root.join("contracts"),
            project_root.join("src"),
        ];

        let mut sol_files: Vec<(std::path::PathBuf, String)> = Vec::new();
        for dir in &test_dirs {
            if !dir.exists() {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("sol") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            sol_files.push((path, content));
                        }
                    }
                }
            }
        }

        let mut candidate_names = vec![harness_contract.to_string()];
        if let Some(stripped) = harness_contract.strip_suffix("Test") {
            if !stripped.is_empty() {
                candidate_names.push(stripped.to_string());
            }
        } else {
            candidate_names.push(format!("{harness_contract}Test"));
        }

        let find_contract_file = |cname: &str| -> Option<(std::path::PathBuf, String)> {
            let pat_regular = format!("contract {cname} ");
            let pat_brace = format!("contract {cname}{{");
            let pat_eol = format!("contract {cname}\n");
            let pat_cr = format!("contract {cname}\r");
            sol_files
                .iter()
                .find(|(_, content)| {
                    content.contains(&pat_regular)
                        || content.contains(&pat_brace)
                        || content.contains(&pat_eol)
                        || content.contains(&pat_cr)
                })
                .cloned()
        };

        let mut resolved_name = harness_contract.to_string();
        let mut target_file = None;
        for cand in &candidate_names {
            if let Some(found) = find_contract_file(cand) {
                resolved_name = cand.clone();
                target_file = Some(found);
                break;
            }
        }

        let (path, content) = match target_file {
            Some(pair) => pair,
            None => return Ok(None),
        };

        let rel_path = path
            .strip_prefix(project_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        // Collect inheritance chain (base contracts first, then child harness) so `super.setUp()` actions are included
        let mut ordered_sources: Vec<(String, String)> = Vec::new();
        let parsed = Self::parse_contracts_from_source(&content, &rel_path);
        if let Some(node) = parsed.iter().find(|n| n.contract_name == resolved_name) {
            for base_name in &node.inherited_contracts {
                if base_name != "Test" {
                    if let Some((_, base_src)) = find_contract_file(base_name) {
                        ordered_sources.push((base_name.clone(), base_src));
                    }
                }
            }
        }
        ordered_sources.push((resolved_name.clone(), content));

        let mut instantiated_contracts = Vec::new();
        let mut address_bindings = Vec::new();
        let mut setup_actions_summary = Vec::new();

        for (cname, src_text) in &ordered_sources {
            Self::extract_setup_from_contract(
                src_text,
                cname,
                &mut instantiated_contracts,
                &mut address_bindings,
                &mut setup_actions_summary,
            );
        }

        Ok(Some(HarnessDeploymentContext {
            harness_contract: resolved_name,
            harness_source_path: rel_path,
            instantiated_contracts,
            address_bindings,
            setup_actions_summary,
        }))
    }

    fn extract_setup_from_contract(
        source: &str,
        contract_name: &str,
        instantiated_contracts: &mut Vec<String>,
        address_bindings: &mut Vec<(String, String)>,
        setup_actions_summary: &mut Vec<String>,
    ) {
        let lines: Vec<&str> = source.lines().collect();
        let mut in_target_contract = false;
        let mut contract_depth: usize = 0;
        let mut in_setup = false;
        let mut setup_depth: usize = 0;

        let pat_space = format!("contract {contract_name} ");
        let pat_brace = format!("contract {contract_name}{{");

        for raw_line in lines {
            let trimmed = raw_line.split("//").next().unwrap_or(raw_line).trim();
            if trimmed.is_empty() {
                continue;
            }
            let opens = trimmed.chars().filter(|&c| c == '{').count();
            let closes = trimmed.chars().filter(|&c| c == '}').count();

            if !in_target_contract {
                if trimmed.contains(&pat_space)
                    || trimmed.contains(&pat_brace)
                    || trimmed.ends_with(&format!("contract {contract_name}"))
                {
                    in_target_contract = true;
                    contract_depth = opens.saturating_sub(closes);
                }
                continue;
            }

            contract_depth = contract_depth.saturating_add(opens);

            if !in_setup && trimmed.contains("function setUp(") {
                in_setup = true;
                setup_depth = opens.saturating_sub(closes);
                contract_depth = contract_depth.saturating_sub(closes);
                continue;
            }

            if in_setup {
                setup_depth = setup_depth.saturating_add(opens);

                if trimmed.contains("= new ") {
                    let parts: Vec<&str> = trimmed.split("= new ").collect();
                    if parts.len() == 2 {
                        let lhs = parts[0].trim();
                        let rhs = parts[1].trim();

                        let var_name = lhs.split_whitespace().last().unwrap_or(lhs).trim();
                        let contract_type = rhs.split('(').next().unwrap_or(rhs).trim();

                        if !contract_type.is_empty() {
                            if !instantiated_contracts.contains(&contract_type.to_string()) {
                                instantiated_contracts.push(contract_type.to_string());
                            }
                            let pair = (var_name.to_string(), contract_type.to_string());
                            if !address_bindings.contains(&pair) {
                                address_bindings.push(pair);
                            }
                        }
                    }
                } else if trimmed.contains("address(") && trimmed.contains(");") {
                    let parts: Vec<&str> = trimmed.split('=').collect();
                    if parts.len() == 2 {
                        let var_name = parts[0]
                            .split_whitespace()
                            .last()
                            .unwrap_or(parts[0])
                            .trim();
                        let rhs = parts[1].trim().trim_end_matches(';').trim();
                        let pair = (var_name.to_string(), rhs.to_string());
                        if !address_bindings.contains(&pair) {
                            address_bindings.push(pair);
                        }
                    }
                }

                if trimmed.contains("addLiquidity")
                    || trimmed.contains("deal(")
                    || trimmed.contains("warp(")
                    || trimmed.contains("roll(")
                {
                    let action = trimmed.to_string();
                    if !setup_actions_summary.contains(&action) {
                        setup_actions_summary.push(action);
                    }
                }

                setup_depth = setup_depth.saturating_sub(closes);
                if setup_depth == 0 && closes > 0 {
                    in_setup = false;
                }
            }

            contract_depth = contract_depth.saturating_sub(closes);
            if contract_depth == 0 && closes > 0 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_contracts_from_source() {
        let source = r#"
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function totalSupply() external view returns (uint256);
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}

contract VaultErc4626 is IERC20 {
    using SafeMath for uint256;
    constructor(address _asset) {
    }
}
"#;
        let nodes = ProjectTopologyAnalyzer::parse_contracts_from_source(source, "src/Vault.sol");
        assert_eq!(nodes.len(), 3);

        assert_eq!(nodes[0].contract_name, "IERC20");
        assert!(nodes[0].is_interface);

        assert_eq!(nodes[1].contract_name, "SafeMath");
        assert!(nodes[1].is_library);

        assert_eq!(nodes[2].contract_name, "VaultErc4626");
        assert!(!nodes[2].is_interface);
        assert_eq!(nodes[2].inherited_contracts, vec!["IERC20"]);
        assert_eq!(nodes[2].libraries_used, vec!["SafeMath"]);
        assert_eq!(nodes[2].constructor_params, vec!["address _asset"]);
    }

    #[test]
    fn test_topological_sort() {
        let nodes = vec![
            ContractDependencyNode {
                contract_name: "Vault".to_string(),
                source_path: "src/Vault.sol".to_string(),
                is_test: false,
                is_interface: false,
                is_library: false,
                inherited_contracts: vec!["BaseVault".to_string()],
                libraries_used: vec![],
                constructor_params: vec![],
            },
            ContractDependencyNode {
                contract_name: "BaseVault".to_string(),
                source_path: "src/BaseVault.sol".to_string(),
                is_test: false,
                is_interface: false,
                is_library: false,
                inherited_contracts: vec!["IVault".to_string()],
                libraries_used: vec![],
                constructor_params: vec![],
            },
            ContractDependencyNode {
                contract_name: "IVault".to_string(),
                source_path: "src/IVault.sol".to_string(),
                is_test: false,
                is_interface: true,
                is_library: false,
                inherited_contracts: vec![],
                libraries_used: vec![],
                constructor_params: vec![],
            },
        ];

        let order = ProjectTopologyAnalyzer::compute_topological_order(&nodes);
        assert_eq!(order, vec!["IVault", "BaseVault", "Vault"]);
    }
}
