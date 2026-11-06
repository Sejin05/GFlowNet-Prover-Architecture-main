#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Programmatic Foundry compiler and Solidity artifact extractor for GFlowNet-Prover.
//!
//! Provides zero-leakage compilation, automatic extraction of AST, ABI, deployed bytecode,
//! and storage layouts, as well as 128-permutation MinHash signature generation for fork clustering.

use alloy_primitives::{keccak256, B256};
use foundry_compilers::{
    artifacts::{output_selection::OutputSelection, Settings},
    compilers::{multi::MultiCompilerSettings, solc::SolcSettings},
    Artifact, Project, ProjectPathsConfig,
};
use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::{
    FoundryContractArtifact, SoliditySlotType, StorageSemanticRole, StorageSlotRecord,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Namespaces for deterministic UUIDv5 generation.
const GFN_PROJECT_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

/// Project extractor responsible for compiling Solidity contracts and extracting artifacts.
#[derive(Debug, Clone)]
pub struct ProjectExtractor {
    pub project_root: PathBuf,
    pub target_solc: String,
}

impl ProjectExtractor {
    /// Creates a new `ProjectExtractor` for the given project directory and solc version string.
    pub fn new(project_root: &Path, target_solc: &str) -> Self {
        Self {
            project_root: project_root.to_path_buf(),
            target_solc: target_solc.to_string(),
        }
    }

    /// Evaluates whether a file path matches zero-day exploit/attack leakage patterns.
    ///
    /// Files matching `*Exploit*.t.sol`, `*Attack*.sol`, or test harness exploit traces
    /// are stripped during extraction to guarantee zero ground-truth leakage.
    pub fn is_leakage_path(path: &Path) -> bool {
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => return false,
        };

        let lower = file_name.to_ascii_lowercase();
        if lower.contains("exploit") || lower.contains("attack") {
            return true;
        }
        // Match standalone `poc` word/camelCase token without falsely matching `Epoch.sol`
        let stem = lower.split('.').next().unwrap_or(&lower);
        stem == "poc"
            || stem.starts_with("poc_")
            || stem.ends_with("_poc")
            || stem.contains("_poc_")
            || file_name.contains("PoC")
    }

    /// Validates that a contract identifier matches the EVM identifier regex `^[A-Za-z_][A-Za-z0-9_]*$`.
    pub fn is_valid_contract_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let mut chars = name.chars();
        match chars.next() {
            Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
            _ => return false,
        }
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Computes a deterministic UUIDv5 project ID by hashing the relative paths and contents
    /// of all sanitized Solidity source files.
    pub fn compute_project_id(project_root: &Path) -> GfnResult<Uuid> {
        let mut hasher = blake3::Hasher::new();
        let src_dir = project_root.join("src");
        let scan_dir = if src_dir.exists() {
            &src_dir
        } else {
            project_root
        };

        if scan_dir.exists() {
            Self::hash_directory_recursive(scan_dir, project_root, &mut hasher)?;
        } else {
            hasher.update(project_root.to_string_lossy().as_bytes());
        }

        let digest = hasher.finalize();
        Ok(Uuid::new_v5(&GFN_PROJECT_NAMESPACE, digest.as_bytes()))
    }

    fn hash_directory_recursive(
        dir: &Path,
        root: &Path,
        hasher: &mut blake3::Hasher,
    ) -> GfnResult<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| GfnError::CompilationArtifact {
            reason: format!("Failed to read directory {}: {e}", dir.display()),
        })?;

        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| GfnError::CompilationArtifact {
                reason: format!("Failed entry read: {e}"),
            })?;
            paths.push(entry.path());
        }
        paths.sort();

        for path in paths {
            if path.is_dir() {
                Self::hash_directory_recursive(&path, root, hasher)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("sol") {
                if Self::is_leakage_path(&path) {
                    continue;
                }
                if let Ok(rel) = path.strip_prefix(root) {
                    hasher.update(rel.to_string_lossy().as_bytes());
                }
                if let Ok(content) = std::fs::read(&path) {
                    hasher.update(&content);
                }
            }
        }
        Ok(())
    }

    /// Classifies an EVM storage variable type string into a normalized `SoliditySlotType`.
    pub fn classify_solidity_type(type_str: &str) -> SoliditySlotType {
        let clean = type_str.trim();
        if clean.starts_with("t_uint256") || clean == "uint256" || clean == "uint" {
            SoliditySlotType::Uint256
        } else if clean.starts_with("t_uint128") || clean == "uint128" {
            SoliditySlotType::Uint128
        } else if clean.starts_with("t_uint64") || clean == "uint64" {
            SoliditySlotType::Uint64
        } else if clean.starts_with("t_uint32") || clean == "uint32" {
            SoliditySlotType::Uint32
        } else if clean.starts_with("t_uint8") || clean == "uint8" {
            SoliditySlotType::Uint8
        } else if clean.starts_with("t_int256") || clean == "int256" || clean == "int" {
            SoliditySlotType::Int256
        } else if clean.starts_with("t_int128") || clean == "int128" {
            SoliditySlotType::Int128
        } else if clean.starts_with("t_address") || clean == "address" {
            SoliditySlotType::Address
        } else if clean.starts_with("t_bool") || clean == "bool" {
            SoliditySlotType::Bool
        } else if clean.starts_with("t_bytes32") || clean == "bytes32" {
            SoliditySlotType::Bytes32
        } else if clean.starts_with("t_bytes") && clean.chars().any(|c| c.is_ascii_digit()) {
            SoliditySlotType::BytesN
        } else if clean.contains("mapping") {
            if clean.contains("address") && clean.contains("uint") {
                SoliditySlotType::MappingAddressUint
            } else if clean.contains("address") && clean.contains("address") {
                SoliditySlotType::MappingAddressAddress
            } else {
                SoliditySlotType::MappingNested
            }
        } else if clean.starts_with("t_struct") || clean.contains("struct") {
            SoliditySlotType::Struct
        } else if clean.ends_with("[]") || clean.contains("dyn_array") {
            SoliditySlotType::DynamicArray
        } else if clean.contains('[') && clean.contains(']') {
            SoliditySlotType::StaticArray
        } else if clean.contains("string") || clean == "bytes" {
            SoliditySlotType::StringOrBytes
        } else {
            SoliditySlotType::Uint256
        }
    }

    /// Classifies the semantic domain role of a storage variable based on identifier name and type.
    pub fn classify_semantic_role(
        var_name: &str,
        slot_type: SoliditySlotType,
    ) -> StorageSemanticRole {
        let lower = var_name.to_ascii_lowercase();
        if lower.contains("balance") || lower.contains("userbalance") || lower.contains("ledger") {
            StorageSemanticRole::AssetBalance
        } else if lower.contains("totalsupply")
            || lower.contains("total_supply")
            || lower.contains("supply")
        {
            StorageSemanticRole::TotalSupply
        } else if lower.contains("reserve") || lower.contains("liquidity") || lower.contains("pool")
        {
            StorageSemanticRole::PoolReserve
        } else if lower.contains("debt") || lower.contains("borrow") || lower.contains("share") {
            StorageSemanticRole::DebtShare
        } else if lower.contains("price") || lower.contains("oracle") || lower.contains("rate") {
            StorageSemanticRole::OraclePrice
        } else if lower.contains("owner")
            || lower.contains("admin")
            || lower.contains("role")
            || lower.contains("authority")
        {
            StorageSemanticRole::AdminRole
        } else if lower.contains("lock")
            || lower.contains("status")
            || lower.contains("entered")
            || lower.contains("reentranc")
        {
            StorageSemanticRole::ReentrancyLock
        } else {
            match slot_type {
                SoliditySlotType::MappingAddressUint => StorageSemanticRole::AssetBalance,
                SoliditySlotType::Address => StorageSemanticRole::AdminRole,
                SoliditySlotType::Bool => StorageSemanticRole::ReentrancyLock,
                _ => StorageSemanticRole::GenericState,
            }
        }
    }

    /// Validates storage slot records according to the data pipeline packing invariant:
    /// For every packed slot `slot_id`, all intervals `[byte_offset, byte_offset + byte_size)`
    /// must be within `[0, 32)` and mutually disjoint.
    pub fn validate_storage_layout(records: &[StorageSlotRecord]) -> GfnResult<()> {
        let mut slots_map: std::collections::BTreeMap<u64, Vec<&StorageSlotRecord>> =
            std::collections::BTreeMap::new();
        for rec in records {
            slots_map.entry(rec.slot_id).or_default().push(rec);
        }

        for (slot_id, var_records) in slots_map {
            let mut covered_bytes = [false; 32];
            for r in var_records {
                let start = r.byte_offset as usize;
                let size = r.byte_size as usize;
                let end = start.saturating_add(size);

                if size == 0 || end > 32 {
                    return Err(GfnError::CompilationArtifact {
                        reason: format!(
                            "Invalid storage packing in slot {}: offset {} size {} exceeds 32 bytes",
                            slot_id, r.byte_offset, r.byte_size
                        ),
                    });
                }

                for b in covered_bytes.iter_mut().take(end).skip(start) {
                    if *b {
                        return Err(GfnError::CompilationArtifact {
                            reason: format!(
                                "Overlapping storage packing in slot {}: collision in byte range [{}, {})",
                                slot_id, start, end
                            ),
                        });
                    }
                    *b = true;
                }
            }
        }
        Ok(())
    }

    /// Computes a 128-permutation MinHash signature over 5-shingles of EVM/SSA bytecode opcodes.
    /// Used for fork clustering deduplication with Jaccard threshold tau_sim = 0.70.
    pub fn compute_minhash_signature(opcodes: &[u8]) -> [u64; 128] {
        let mut min_hashes = [u64::MAX; 128];
        if opcodes.len() < 5 {
            let mut shingle = [0u8; 5];
            for (i, &b) in opcodes.iter().enumerate() {
                shingle[i] = b;
            }
            for (seed, min_hash) in min_hashes.iter_mut().enumerate() {
                *min_hash = xxh3_64_with_seed(&shingle, seed as u64);
            }
            return min_hashes;
        }

        for window in opcodes.windows(5) {
            for (seed, min_hash) in min_hashes.iter_mut().enumerate() {
                let hash_val = xxh3_64_with_seed(window, seed as u64);
                if hash_val < *min_hash {
                    *min_hash = hash_val;
                }
            }
        }
        min_hashes
    }

    /// Computes the estimated Jaccard similarity between two 128-permutation MinHash signatures.
    pub fn minhash_jaccard_similarity(sig_a: &[u64; 128], sig_b: &[u64; 128]) -> f64 {
        let mut matches = 0usize;
        for i in 0..128 {
            if sig_a[i] == sig_b[i] {
                matches = matches.saturating_add(1);
            }
        }
        matches as f64 / 128.0
    }

    /// Programmatically compiles the project using `foundry-compilers` and extracts sanitized artifacts.
    pub fn extract(&self) -> GfnResult<Vec<FoundryContractArtifact>> {
        let project_id = Self::compute_project_id(&self.project_root)?;

        let mut paths = ProjectPathsConfig::builder()
            .root(&self.project_root)
            .build()
            .map_err(|e| GfnError::CompilationArtifact {
                reason: format!("Failed to configure project paths: {e}"),
            })?;

        // Filter out leakage paths
        let mut sanitized_sources = Vec::new();
        for src_path in &paths.input_files() {
            if !Self::is_leakage_path(src_path) {
                sanitized_sources.push(src_path.clone());
            }
        }

        let solc_settings = Settings {
            output_selection: OutputSelection::complete_output_selection(),
            ..Default::default()
        };
        let multi_settings = MultiCompilerSettings {
            solc: SolcSettings {
                settings: solc_settings,
                ..Default::default()
            },
            ..Default::default()
        };

        paths.sources = self.project_root.join("src");

        let project = Project::builder()
            .paths(paths)
            .settings(multi_settings)
            .ephemeral()
            .no_artifacts()
            .build(Default::default())
            .map_err(|e| GfnError::CompilationArtifact {
                reason: format!("Failed to build foundry-compilers Project: {e}"),
            })?;

        let output = match project.compile() {
            Ok(out) => out,
            Err(comp_err) => {
                // Check if forge is installed and can build the project
                let _ = std::process::Command::new("forge")
                    .arg("build")
                    .arg("--offline")
                    .current_dir(&self.project_root)
                    .output();

                let out_dir = self.project_root.join("out");
                if out_dir.exists() {
                    return self.extract_from_out_dir(&out_dir, project_id);
                }

                return Err(GfnError::CompilationArtifact {
                    reason: format!(
                        "Compilation failed for project {}: {comp_err}",
                        self.project_root.display()
                    ),
                });
            }
        };

        let mut artifacts = Vec::new();
        let mut processed_contracts = HashSet::new();

        for (artifact_id, artifact) in output.into_artifacts() {
            let path = artifact_id.source;
            let contract_name = artifact_id.name;

            if Self::is_leakage_path(path.as_path()) {
                continue;
            }

            if !Self::is_valid_contract_name(&contract_name) {
                continue;
            }

            if processed_contracts.contains(&contract_name) {
                continue;
            }

            let deployed_bytecode = match artifact.get_deployed_bytecode_bytes() {
                Some(bytes) if bytes.len() >= 2 => bytes.to_vec(),
                _ => continue,
            };

            let bytecode_hash: B256 = keccak256(&deployed_bytecode);

            let abi_json_str = match &artifact.abi {
                Some(abi) => serde_json::to_string(abi).map_err(|e| GfnError::Serialization {
                    details: format!("Failed to serialize ABI for {contract_name}: {e}"),
                })?,
                None => "[]".to_string(),
            };

            let mut storage_slots = Vec::new();
            if let Some(layout) = &artifact.storage_layout {
                for (idx, entry) in layout.storage.iter().enumerate() {
                    let slot_id = entry.slot.parse::<u64>().unwrap_or(idx as u64);
                    let byte_offset = entry.offset as u8;
                    let var_type = Self::classify_solidity_type(&entry.storage_type);
                    let semantic_role = Self::classify_semantic_role(&entry.label, var_type);
                    let byte_size = match var_type {
                        SoliditySlotType::Bool | SoliditySlotType::Uint8 => 1,
                        SoliditySlotType::Uint32 => 4,
                        SoliditySlotType::Uint64 => 8,
                        SoliditySlotType::Uint128 | SoliditySlotType::Int128 => 16,
                        SoliditySlotType::Address => 20,
                        SoliditySlotType::Bytes32
                        | SoliditySlotType::Uint256
                        | SoliditySlotType::Int256 => 32,
                        _ => 32,
                    };

                    storage_slots.push(StorageSlotRecord {
                        slot_id,
                        byte_offset,
                        byte_size,
                        solidity_type: var_type,
                        is_dynamic_mapping: matches!(
                            var_type,
                            SoliditySlotType::MappingAddressUint
                                | SoliditySlotType::MappingAddressAddress
                                | SoliditySlotType::MappingNested
                        ),
                        semantic_role,
                    });
                }
            }

            // If no storage layout, provide at least one baseline storage record if contract is non-empty
            if storage_slots.is_empty() {
                storage_slots.push(StorageSlotRecord {
                    slot_id: 0,
                    byte_offset: 0,
                    byte_size: 32,
                    solidity_type: SoliditySlotType::Uint256,
                    is_dynamic_mapping: false,
                    semantic_role: StorageSemanticRole::GenericState,
                });
            }

            Self::validate_storage_layout(&storage_slots)?;

            processed_contracts.insert(contract_name.clone());
            artifacts.push(FoundryContractArtifact {
                project_id,
                contract_name,
                source_path: path.to_string_lossy().to_string(),
                solc_version: self.target_solc.clone(),
                deployed_bytecode_hash: bytecode_hash.0,
                deployed_bytecode,
                abi_json_str,
                storage_slots,
            });
        }

        Ok(artifacts)
    }

    /// Fallback extractor when programmatic compilation fails, reading existing or `forge build` artifacts in `out/`.
    pub fn extract_from_out_dir(
        &self,
        out_dir: &Path,
        project_id: Uuid,
    ) -> GfnResult<Vec<FoundryContractArtifact>> {
        let mut artifacts = Vec::new();
        let mut processed_contracts = HashSet::new();

        let entries = match std::fs::read_dir(out_dir) {
            Ok(e) => e,
            Err(_) => return Ok(artifacts),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if dir_name == "build-info"
                    || dir_name.ends_with(".t.sol")
                    || dir_name.ends_with(".s.sol")
                    || Self::is_leakage_path(Path::new(dir_name))
                {
                    continue;
                }

                let sub_entries = match std::fs::read_dir(&path) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                for sub in sub_entries.flatten() {
                    let file_path = sub.path();
                    if file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                        let contract_name = match file_path.file_stem().and_then(|s| s.to_str()) {
                            Some(name) => name.to_string(),
                            None => continue,
                        };

                        if Self::is_leakage_path(Path::new(&contract_name)) {
                            continue;
                        }
                        if !Self::is_valid_contract_name(&contract_name) {
                            continue;
                        }
                        if processed_contracts.contains(&contract_name) {
                            continue;
                        }

                        let content = match std::fs::read_to_string(&file_path) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };

                        let val: serde_json::Value = match serde_json::from_str(&content) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        let deployed_bytecode_hex = val
                            .get("deployedBytecode")
                            .and_then(|d| d.get("object"))
                            .and_then(|o| o.as_str())
                            .or_else(|| {
                                val.get("bytecode")
                                    .and_then(|b| b.get("object"))
                                    .and_then(|o| o.as_str())
                            });

                        let hex_str = match deployed_bytecode_hex {
                            Some(h) if h.starts_with("0x") => &h[2..],
                            Some(h) => h,
                            None => continue,
                        };

                        let deployed_bytecode = match alloy_primitives::hex::decode(hex_str) {
                            Ok(b) if b.len() >= 2 => b,
                            _ => continue,
                        };

                        let bytecode_hash: B256 = keccak256(&deployed_bytecode);

                        let abi_json_str = val
                            .get("abi")
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "[]".to_string());

                        let types_table = val.get("storageLayout").and_then(|l| l.get("types"));
                        let mut storage_slots = Vec::new();
                        if let Some(storage) = val
                            .get("storageLayout")
                            .and_then(|l| l.get("storage"))
                            .and_then(|s| s.as_array())
                        {
                            for (idx, entry) in storage.iter().enumerate() {
                                let slot_id = entry
                                    .get("slot")
                                    .and_then(|s| s.as_str())
                                    .and_then(|s| s.parse::<u64>().ok())
                                    .unwrap_or(idx as u64);
                                let byte_offset =
                                    entry.get("offset").and_then(|o| o.as_u64()).unwrap_or(0) as u8;
                                let storage_type = entry
                                    .get("type")
                                    .or_else(|| entry.get("storage_type"))
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("uint256");
                                let label =
                                    entry.get("label").and_then(|l| l.as_str()).unwrap_or("");
                                let var_type = Self::classify_solidity_type(storage_type);
                                let semantic_role = Self::classify_semantic_role(label, var_type);
                                let exact_bytes = types_table
                                    .and_then(|tt| tt.get(storage_type))
                                    .and_then(|td| td.get("numberOfBytes"))
                                    .and_then(|nb| {
                                        nb.as_str()
                                            .and_then(|s| s.parse::<u64>().ok())
                                            .or_else(|| nb.as_u64())
                                    })
                                    .filter(|&n| (1..=32).contains(&n))
                                    .map(|n| n as u8);
                                let byte_size = exact_bytes.unwrap_or(match var_type {
                                    SoliditySlotType::Bool | SoliditySlotType::Uint8 => 1,
                                    SoliditySlotType::Uint32 => 4,
                                    SoliditySlotType::Uint64 => 8,
                                    SoliditySlotType::Uint128 | SoliditySlotType::Int128 => 16,
                                    SoliditySlotType::Address => 20,
                                    SoliditySlotType::Bytes32
                                    | SoliditySlotType::Uint256
                                    | SoliditySlotType::Int256 => 32,
                                    _ => 32,
                                });
                                storage_slots.push(StorageSlotRecord {
                                    slot_id,
                                    byte_offset,
                                    byte_size,
                                    solidity_type: var_type,
                                    is_dynamic_mapping: matches!(
                                        var_type,
                                        SoliditySlotType::MappingAddressUint
                                            | SoliditySlotType::MappingAddressAddress
                                            | SoliditySlotType::MappingNested
                                    ),
                                    semantic_role,
                                });
                            }
                        }

                        if storage_slots.is_empty() {
                            storage_slots.push(StorageSlotRecord {
                                slot_id: 0,
                                byte_offset: 0,
                                byte_size: 32,
                                solidity_type: SoliditySlotType::Uint256,
                                is_dynamic_mapping: false,
                                semantic_role: StorageSemanticRole::GenericState,
                            });
                        }

                        let source_path = val
                            .get("metadata")
                            .and_then(|m| m.get("settings"))
                            .and_then(|s| s.get("compilationTarget"))
                            .and_then(|ct| ct.as_object())
                            .and_then(|obj| obj.keys().next())
                            .cloned()
                            .unwrap_or_else(|| {
                                if dir_name.ends_with(".sol") {
                                    format!("src/{dir_name}")
                                } else {
                                    format!("src/{contract_name}.sol")
                                }
                            });

                        let norm_src = source_path.replace('\\', "/");
                        if Self::is_leakage_path(Path::new(&norm_src))
                            || norm_src.starts_with("test/")
                            || norm_src.starts_with("tests/")
                            || norm_src.starts_with("script/")
                            || norm_src.starts_with("scripts/")
                            || norm_src.starts_with("lib/")
                            || norm_src.ends_with(".t.sol")
                            || norm_src.ends_with(".s.sol")
                        {
                            continue;
                        }

                        Self::validate_storage_layout(&storage_slots)?;
                        processed_contracts.insert(contract_name.clone());

                        artifacts.push(FoundryContractArtifact {
                            project_id,
                            contract_name,
                            source_path,
                            solc_version: self.target_solc.clone(),
                            deployed_bytecode_hash: bytecode_hash.0,
                            deployed_bytecode,
                            abi_json_str,
                            storage_slots,
                        });
                    }
                }
            }
        }

        Ok(artifacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leakage_path_detection() {
        assert!(ProjectExtractor::is_leakage_path(Path::new(
            "test/ExploitPoC.t.sol"
        )));
        assert!(ProjectExtractor::is_leakage_path(Path::new(
            "src/AttackContract.sol"
        )));
        assert!(ProjectExtractor::is_leakage_path(Path::new(
            "contracts/poc_test.sol"
        )));
        assert!(!ProjectExtractor::is_leakage_path(Path::new(
            "src/LendingVault.sol"
        )));
        assert!(!ProjectExtractor::is_leakage_path(Path::new(
            "src/ERC20Token.sol"
        )));
    }

    #[test]
    fn test_valid_contract_identifiers() {
        assert!(ProjectExtractor::is_valid_contract_name("VaultV2"));
        assert!(ProjectExtractor::is_valid_contract_name("_InternalToken"));
        assert!(ProjectExtractor::is_valid_contract_name("SimpleAMM_Pool"));
        assert!(!ProjectExtractor::is_valid_contract_name(""));
        assert!(!ProjectExtractor::is_valid_contract_name("123BadStart"));
        assert!(!ProjectExtractor::is_valid_contract_name("Invalid-Hyphen"));
    }

    #[test]
    fn test_storage_slot_validation_success() -> GfnResult<()> {
        let valid_records = vec![
            StorageSlotRecord {
                slot_id: 0,
                byte_offset: 0,
                byte_size: 20, // address
                solidity_type: SoliditySlotType::Address,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AdminRole,
            },
            StorageSlotRecord {
                slot_id: 0,
                byte_offset: 20,
                byte_size: 1, // bool
                solidity_type: SoliditySlotType::Bool,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::ReentrancyLock,
            },
            StorageSlotRecord {
                slot_id: 1,
                byte_offset: 0,
                byte_size: 32, // uint256
                solidity_type: SoliditySlotType::Uint256,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::TotalSupply,
            },
        ];

        ProjectExtractor::validate_storage_layout(&valid_records)
    }

    #[test]
    fn test_storage_slot_validation_collision() {
        let colliding_records = vec![
            StorageSlotRecord {
                slot_id: 0,
                byte_offset: 0,
                byte_size: 20,
                solidity_type: SoliditySlotType::Address,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AdminRole,
            },
            StorageSlotRecord {
                slot_id: 0,
                byte_offset: 16, // Collides with [0..20)
                byte_size: 16,
                solidity_type: SoliditySlotType::Uint128,
                is_dynamic_mapping: false,
                semantic_role: StorageSemanticRole::AssetBalance,
            },
        ];

        let res = ProjectExtractor::validate_storage_layout(&colliding_records);
        assert!(res.is_err());
    }

    #[test]
    fn test_minhash_exact_match_and_difference() {
        let opcodes_a = vec![
            0x60, 0x80, 0x60, 0x40, 0x52, 0x34, 0x80, 0x15, 0x61, 0x00, 0x10,
        ];
        let opcodes_b = opcodes_a.clone();
        let opcodes_c = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

        let sig_a = ProjectExtractor::compute_minhash_signature(&opcodes_a);
        let sig_b = ProjectExtractor::compute_minhash_signature(&opcodes_b);
        let sig_c = ProjectExtractor::compute_minhash_signature(&opcodes_c);

        let sim_ab = ProjectExtractor::minhash_jaccard_similarity(&sig_a, &sig_b);
        let sim_ac = ProjectExtractor::minhash_jaccard_similarity(&sig_a, &sig_c);

        assert!((sim_ab - 1.0).abs() < 1e-6);
        assert!(sim_ac < 0.5);
    }
}
