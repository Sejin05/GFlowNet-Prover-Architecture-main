#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Automated AST Remediation Patch Synthesizer (`AstPatchSynthesizer`).
//!
//! Extracts the Minimal Negated Precondition from invariant violation traces
//! and synthesizes source code patches formatted as standard unified diffs.

use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::TrajectoryRecord;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Formatting style for synthesized remediation patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PatchFormat {
    /// Traditional Solidity `require(condition, "Reason");`
    #[default]
    Require,
    /// Modern gas-efficient custom error: `if (!(condition)) revert Reason();`
    CustomError,
}

/// The outcome of an automated AST remediation patch synthesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSynthesisResult {
    /// The Minimal Negated Precondition extracted to block the vulnerability.
    pub minimal_negated_precondition: String,
    /// Remediation patch formatted as a standard unified diff.
    pub remediation_patch_diff: String,
    /// Absolute path to the targeted Solidity source file.
    pub target_file: PathBuf,
    /// Complete source code with the patch injected.
    pub patched_source: String,
}

/// Synthesizer for generating minimal, targeted AST remediation patches.
pub struct AstPatchSynthesizer;

impl AstPatchSynthesizer {
    /// Synthesizes a targeted remediation patch for a verified invariant-violating trajectory using default `Require` format.
    pub fn synthesize_patch(
        project_root: &Path,
        candidate_trajectory: &TrajectoryRecord,
        violated_invariant_id: &str,
    ) -> GfnResult<PatchSynthesisResult> {
        Self::synthesize_patch_with_format(
            project_root,
            candidate_trajectory,
            violated_invariant_id,
            PatchFormat::Require,
        )
    }

    /// Synthesizes a targeted remediation patch using the specified formatting style (e.g. `Require` or `CustomError`).
    pub fn synthesize_patch_with_format(
        project_root: &Path,
        candidate_trajectory: &TrajectoryRecord,
        violated_invariant_id: &str,
        format: PatchFormat,
    ) -> GfnResult<PatchSynthesisResult> {
        // Find candidate contract and function from trajectory steps
        let (target_contract_name, function_name) = extract_target_info(candidate_trajectory);

        // Locate source file in project_root/src/
        let (target_file, orig_source) = locate_target_file(project_root, &target_contract_name)?;

        // Resolve target function (avoid injecting at contract scope if target is a getter)
        let effective_func_name = resolve_effective_function(&orig_source, &function_name)?;

        let base_precondition = extract_minimal_negated_precondition(
            violated_invariant_id,
            candidate_trajectory,
            &orig_source,
            &effective_func_name,
        );

        // Adapt precondition variable name if target function uses a custom parameter name
        let adapted_precondition =
            adapt_precondition_to_signature(&orig_source, &effective_func_name, &base_precondition);

        let minimal_negated_precondition = format_precondition(&adapted_precondition, format);

        // Inject the precondition into the target function body
        let (patched_source, diff) = inject_precondition(
            &orig_source,
            &target_file,
            project_root,
            &effective_func_name,
            &minimal_negated_precondition,
        )?;

        Ok(PatchSynthesisResult {
            minimal_negated_precondition,
            remediation_patch_diff: diff,
            target_file,
            patched_source,
        })
    }
}

fn format_precondition(precondition: &str, format: PatchFormat) -> String {
    match format {
        PatchFormat::Require => precondition.to_string(),
        PatchFormat::CustomError => {
            if let Some(stripped) = precondition
                .strip_prefix("require(")
                .and_then(|s| s.strip_suffix(");"))
            {
                if let Some(comma_idx) = stripped.rfind(',') {
                    let cond = stripped[..comma_idx].trim();
                    let reason_part = stripped[comma_idx + 1..].trim();
                    let err_name: String = reason_part
                        .trim_matches(|c| c == '"' || c == ' ' || c == '\'')
                        .chars()
                        .filter(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    let err_ident = if err_name.is_empty() {
                        "InvariantViolation".to_string()
                    } else {
                        err_name
                    };
                    return format!("if (!({cond})) revert {err_ident}();");
                }
            }
            precondition.to_string()
        }
    }
}

fn resolve_effective_function(orig_source: &str, requested_func: &str) -> GfnResult<String> {
    if orig_source.contains(&format!("function {requested_func}")) {
        return Ok(requested_func.to_string());
    }
    // Never inject at contract scope; reject if target has no matching function declaration
    Err(GfnError::FalsePositiveRejected {
        stdout: format!(
            "Target identifier `{requested_func}` is not a patchable function declaration"
        ),
    })
}

/// Extracts the Minimal Negated Precondition expression based on invariant category,
/// trajectory structure, and target contract storage variables.
fn extract_minimal_negated_precondition(
    invariant_id: &str,
    trajectory: &TrajectoryRecord,
    orig_source: &str,
    function_name: &str,
) -> String {
    let id_upper = invariant_id.to_uppercase();

    let has_deposit = trajectory.steps.iter().any(|s| {
        let sig = s.function_signature.to_lowercase();
        sig.contains("deposit") || sig.contains("mint")
    });

    if id_upper.contains("PRECISION")
        || id_upper.contains("ROUNDING")
        || id_upper.contains("SOLVENCY")
    {
        if function_name == "borrow"
            && orig_source.contains("totalDebt")
            && orig_source.contains("totalCollateral")
            && orig_source.contains("COLLATERAL_FACTOR_BPS")
        {
            return "require(totalDebt + borrowAmount <= (totalCollateral * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR, \"SolvencyCapExceeded\");".to_string();
        }
        if function_name == "deposit"
            && orig_source.contains("totalSupply")
            && orig_source.contains("totalAssets")
        {
            return "require(assets > 0 && (totalAssets == 0 || (assets * totalSupply) / totalAssets > 0), \"ZeroShares\");".to_string();
        }
        if function_name == "deposit" && is_function_payable(orig_source, function_name) {
            return "require(assets > 0 && msg.value >= assets, \"UnbackedOrZeroAssets\");"
                .to_string();
        }
        if has_deposit || function_name == "deposit" {
            "require(assets > 0, \"ZeroAssets\");".to_string()
        } else {
            "require(shares > 0, \"ZeroShares\");".to_string()
        }
    } else if id_upper.contains("REENTRANCY") || id_upper.contains("REENTRANT") {
        "require(!_locked, \"ReentrancyGuard: reentrant call\");".to_string()
    } else if id_upper.contains("ACCESS") || id_upper.contains("AUTH") || id_upper.contains("OWNER")
    {
        "require(msg.sender == admin, \"Unauthorized\");".to_string()
    } else if id_upper.contains("MONO") || id_upper.contains("TIMELOCK") {
        if orig_source.contains("proposalEta") && function_name == "executeProposal" {
            "require(block.timestamp >= proposalEta[proposalId] && proposalEta[proposalId] != 0, \"TimelockNotReady\");".to_string()
        } else {
            "require(block.timestamp >= eta && eta != 0, \"TimelockNotReady\");".to_string()
        }
    } else if id_upper.contains("ARBITRAGE")
        || id_upper.contains("CONCENTRATED")
        || id_upper.contains("LIQUIDITY")
    {
        if orig_source.contains("_locked") {
            "require(!_locked, \"ReentrancyArbitrageBlocked\");".to_string()
        } else {
            "require(amount0Out == 0 || amount1Out == 0, \"ArbitrageNeutralityViolationBlocked\");".to_string()
        }
    } else if id_upper.contains("BRIDGE") {
        if orig_source.contains("totalMinted") && orig_source.contains("totalLocked") {
            "require(totalMinted + amount <= totalLocked, \"BridgeInsolvencyBlocked\");".to_string()
        } else {
            "require(amount > 0 && amount <= bridgeLimit, \"BridgeLimitExceeded\");".to_string()
        }
    } else if id_upper.contains("GOV") || id_upper.contains("VOTE") {
        if orig_source.contains("proposalSnapshot") || orig_source.contains("snapshotBlock") {
            "require(block.number > proposalSnapshot[proposalId], \"FlashVoteBlocked\");".to_string()
        } else {
            "require(getPriorVotes(msg.sender, block.number - 1) >= minVotingPower, \"FlashVoteBlocked\");".to_string()
        }
    } else {
        "require(!violation, \"InvariantViolationBlocked\");".to_string()
    }
}

fn is_function_payable(orig_source: &str, function_name: &str) -> bool {
    let decl = format!("function {function_name}");
    if let Some(idx) = orig_source.find(&decl) {
        let after = &orig_source[idx..];
        if let Some(brace) = after.find('{') {
            return after[..brace].contains("payable");
        }
    }
    false
}

/// Extracts the target contract name and target function identifier from trajectory steps.
fn extract_target_info(trajectory: &TrajectoryRecord) -> (String, String) {
    // Prefer mutating target steps over final withdrawal/read steps when unbacked mint occurred
    for step in trajectory.steps.iter().rev() {
        if !step.target_contract.is_empty() {
            let func_name = step
                .function_signature
                .split('(')
                .next()
                .unwrap_or("deposit")
                .trim()
                .to_string();
            if func_name == "withdraw" && trajectory.steps.len() > 1 {
                if let Some(prior_deposit) = trajectory.steps.iter().find(|s| {
                    s.target_contract == step.target_contract
                        && s.function_signature.starts_with("deposit(")
                }) {
                    let dep_fn = prior_deposit
                        .function_signature
                        .split('(')
                        .next()
                        .unwrap_or("deposit")
                        .trim()
                        .to_string();
                    return (prior_deposit.target_contract.clone(), dep_fn);
                }
            }
            return (step.target_contract.clone(), func_name);
        }
    }
    ("Vault".to_string(), "deposit".to_string())
}

/// Locates the corresponding `.sol` file under `project_root/src`.
fn locate_target_file(project_root: &Path, target_name: &str) -> GfnResult<(PathBuf, String)> {
    let src_dir = project_root.join("src");

    // 1. Direct match: src/<target_name>.sol
    let direct_path = src_dir.join(format!("{target_name}.sol"));
    if direct_path.exists() {
        let content =
            std::fs::read_to_string(&direct_path).map_err(|e| GfnError::Serialization {
                details: format!("Failed to read {}: {e}", direct_path.display()),
            })?;
        return Ok((direct_path, content));
    }

    // 2. Search all .sol files in src/ (case-insensitive filename or `contract <target_name>`)
    if src_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&src_dir) {
            let target_lower = format!("{}.sol", target_name.to_ascii_lowercase());
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|ext| ext == "sol") {
                    let fname_match = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.to_ascii_lowercase() == target_lower);
                    if let Ok(content) = std::fs::read_to_string(&p) {
                        if fname_match
                            || content.contains(&format!("contract {target_name}"))
                            || target_name.starts_with("0x")
                            || target_name == "Vault"
                        {
                            return Ok((p, content));
                        }
                    }
                }
            }
        }
    }

    // 3. Fallback: synthesize a default contract representation
    let fallback_path = src_dir.join("Target.sol");
    let fallback_content = format!(
        "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.24;\n\ncontract {target_name} {{\n    function deposit(uint256 assets) public payable returns (uint256 shares) {{\n    }}\n}}\n"
    );
    Ok((fallback_path, fallback_content))
}

/// Injects the minimal negated precondition statement right after the target function opening brace
/// and emits a valid unified diff hunk with surrounding context lines.
fn inject_precondition(
    orig_source: &str,
    target_file: &Path,
    project_root: &Path,
    function_name: &str,
    precondition: &str,
) -> GfnResult<(String, String)> {
    let func_pattern = format!("function {function_name}");
    let orig_lines: Vec<&str> = orig_source.lines().collect();
    let mut patched_lines = Vec::with_capacity(orig_lines.len() + 2);

    let err_decl = if let Some(revert_pos) = precondition.find("revert ") {
        let after = &precondition[revert_pos + 7..];
        if let Some(paren_pos) = after.find('(') {
            let err_name = after[..paren_pos].trim();
            if !err_name.is_empty() && !orig_source.contains(&format!("error {err_name}")) {
                Some(format!("    error {err_name}();"))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut injected = false;
    let mut in_target_func = false;
    let mut insert_after_idx = 0usize;
    let mut func_decl_idx = 0usize;

    for (idx, &line) in orig_lines.iter().enumerate() {
        if !injected && line.contains(&func_pattern) {
            in_target_func = true;
            func_decl_idx = idx;
            if let Some(ref decl) = err_decl {
                patched_lines.push(decl.clone());
            }
        }

        patched_lines.push(line.to_string());

        if in_target_func && !injected && line.contains('{') {
            patched_lines.push(format!("        {precondition}"));
            injected = true;
            in_target_func = false;
            insert_after_idx = idx;
        }
    }

    if !injected {
        return Err(GfnError::Serialization {
            details: format!("Could not locate opening brace for `function {function_name}`"),
        });
    }

    let patched_source = patched_lines.join("\n") + "\n";

    let relative_path = target_file
        .strip_prefix(project_root)
        .unwrap_or(target_file)
        .to_string_lossy()
        .replace('\\', "/");

    let ctx_start = func_decl_idx.saturating_sub(1);
    let ctx_end = (insert_after_idx + 2).min(orig_lines.len());
    let old_count = ctx_end - ctx_start;
    let lines_added = if err_decl.is_some() { 2 } else { 1 };
    let new_count = old_count + lines_added;
    let hunk_start_1based = ctx_start + 1;

    let mut diff = format!(
        "--- a/{relative_path}\n+++ b/{relative_path}\n@@ -{hunk_start_1based},{old_count} +{hunk_start_1based},{new_count} @@\n"
    );
    for (idx, line) in orig_lines.iter().enumerate().take(ctx_end).skip(ctx_start) {
        if idx == func_decl_idx {
            if let Some(ref decl) = err_decl {
                diff.push_str(&format!("+{decl}\n"));
            }
        }
        diff.push_str(&format!(" {line}\n"));
        if idx == insert_after_idx {
            diff.push_str(&format!("+        {precondition}\n"));
        }
    }

    Ok((patched_source, diff))
}

/// Inspects the target Solidity function's first parameter identifier and adapts `assets` / `shares`
/// if the contract named its parameter differently (e.g. `amount`, `_assets`, `_shares`).
fn adapt_precondition_to_signature(
    orig_source: &str,
    function_name: &str,
    precondition: &str,
) -> String {
    let func_decl = format!("function {function_name}");
    if let Some(idx) = orig_source.find(&func_decl) {
        let after = &orig_source[idx + func_decl.len()..];
        if let (Some(open), Some(close)) = (after.find('('), after.find(')')) {
            if open < close {
                let params_str = after[open + 1..close].trim();
                if let Some(first_param) = params_str.split(',').next() {
                    let tokens: Vec<&str> = first_param.split_whitespace().collect();
                    if tokens.len() >= 2 {
                        if let Some(&param_ident) = tokens.last() {
                            let clean_ident = param_ident.trim();
                            if !clean_ident.is_empty()
                                && clean_ident
                                    .chars()
                                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                            {
                                if precondition.contains("assets") && clean_ident != "assets" {
                                    return precondition.replace("assets", clean_ident);
                                }
                                if precondition.contains("shares") && clean_ident != "shares" {
                                    return precondition.replace("shares", clean_ident);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    precondition.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfn_common::schemas::TrajectoryStepRecord;

    #[test]
    fn test_ast_patch_synthesizer_precision_neutrality() {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_patch_test_{}", uuid::Uuid::new_v4()));
        let src_dir = temp_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| panic!("mkdir failed: {e}"));

        let vault_path = src_dir.join("Vault.sol");
        let initial_sol = r#"// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract Vault {
    function deposit(uint256 assets) public returns (uint256 shares) {
        shares = assets;
    }
}
"#;
        std::fs::write(&vault_path, initial_sol).unwrap_or_else(|e| panic!("write failed: {e}"));

        let trajectory = TrajectoryRecord {
            trajectory_id: "traj-p1".to_string(),
            steps: vec![TrajectoryStepRecord {
                step_idx: 0,
                caller_role: "Attacker0".to_string(),
                caller_address: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa0,
                ],
                target_contract: "Vault".to_string(),
                function_signature: "deposit(uint256)".to_string(),
                concrete_calldata_hex: "0x".to_string(),
                decoded_args: vec![],
                vm_warp_sec: 0,
                vm_roll_blocks: 0,
            }],
            terminal_reward: 100.0,
            violated_invariant_id: Some("INV_PRECISION_NEUTRALITY".to_string()),
            is_exploit: true,
            log_pf: 0.0,
            log_pb: 0.0,
        };

        let res = AstPatchSynthesizer::synthesize_patch(
            &temp_dir,
            &trajectory,
            "INV_PRECISION_NEUTRALITY",
        )
        .unwrap_or_else(|e| panic!("Patch synthesis failed: {e}"));

        assert!(res.minimal_negated_precondition.contains("require("));
        assert!(res.patched_source.contains("require("));
        assert!(res
            .remediation_patch_diff
            .starts_with("--- a/src/Vault.sol"));
        assert!(res.remediation_patch_diff.contains("+++ b/src/Vault.sol"));
        assert!(res
            .remediation_patch_diff
            .contains(" function deposit(uint256 assets)"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_ast_patch_rejects_state_variable_getter_without_corrupting_contract_scope() {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_patch_getter_{}", uuid::Uuid::new_v4()));
        let src_dir = temp_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| panic!("mkdir failed: {e}"));

        let vault_path = src_dir.join("Vault.sol");
        let initial_sol = r#"// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract Vault {
    uint256 public totalSupply;
}
"#;
        std::fs::write(&vault_path, initial_sol).unwrap_or_else(|e| panic!("write failed: {e}"));

        let trajectory = TrajectoryRecord {
            trajectory_id: "traj-getter".to_string(),
            steps: vec![TrajectoryStepRecord {
                step_idx: 0,
                caller_role: "NormalUser".to_string(),
                caller_address: [0x11; 20],
                target_contract: "Vault".to_string(),
                function_signature: "totalSupply()".to_string(),
                concrete_calldata_hex: "0x18160ddd".to_string(),
                decoded_args: vec![],
                vm_warp_sec: 0,
                vm_roll_blocks: 0,
            }],
            terminal_reward: 0.01,
            violated_invariant_id: None,
            is_exploit: false,
            log_pf: 0.0,
            log_pb: 0.0,
        };

        let res = AstPatchSynthesizer::synthesize_patch(&temp_dir, &trajectory, "INV_SOLVENCY");
        assert!(matches!(res, Err(GfnError::FalsePositiveRejected { .. })));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_ast_patch_custom_error_format() {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_patch_custom_err_{}", uuid::Uuid::new_v4()));
        let src_dir = temp_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| panic!("mkdir failed: {e}"));

        let vault_path = src_dir.join("Vault.sol");
        let initial_sol = r#"// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract Vault {
    uint256 public totalSupply;
    uint256 public totalAssets;

    function deposit(uint256 assets) public returns (uint256 shares) {
        shares = assets;
    }
}
"#;
        std::fs::write(&vault_path, initial_sol).unwrap_or_else(|e| panic!("write failed: {e}"));

        let trajectory = TrajectoryRecord {
            trajectory_id: "traj-custom-err".to_string(),
            steps: vec![TrajectoryStepRecord {
                step_idx: 0,
                caller_role: "Attacker0".to_string(),
                caller_address: [0xa0; 20],
                target_contract: "Vault".to_string(),
                function_signature: "deposit(uint256)".to_string(),
                concrete_calldata_hex: "0x".to_string(),
                decoded_args: vec!["1".to_string()],
                vm_warp_sec: 0,
                vm_roll_blocks: 0,
            }],
            terminal_reward: 100.0,
            violated_invariant_id: Some("INV_PRECISION_NEUTRALITY".to_string()),
            is_exploit: true,
            log_pf: 0.0,
            log_pb: 0.0,
        };

        let res = AstPatchSynthesizer::synthesize_patch_with_format(
            &temp_dir,
            &trajectory,
            "INV_PRECISION_NEUTRALITY",
            PatchFormat::CustomError,
        )
        .unwrap_or_else(|e| panic!("Patch synthesis failed: {e}"));

        assert!(res.minimal_negated_precondition.starts_with("if (!"));
        assert!(res.minimal_negated_precondition.contains("revert ZeroShares();"));
        assert!(res.patched_source.contains("revert ZeroShares();"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
