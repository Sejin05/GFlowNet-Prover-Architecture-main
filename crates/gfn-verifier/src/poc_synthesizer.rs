#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Turnkey Foundry verification harness synthesizer (`PocSynthesizer`).
//!
//! Translates candidate adversarial trajectories and violated invariant specifications
//! into deterministic, self-contained Foundry test harnesses (`ExploitPoC.t.sol`) with
//! non-tautological pre/post-state EVM invariant assertions.

use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::TrajectoryRecord;
use std::collections::BTreeSet;

/// Canonical actor addresses used across the Foundry execution environment (EIP-55 checksummed).
pub const ATTACKER0_ADDR: &str = "0x00000000000000000000000000000000000000a0";
pub const ATTACKER1_SYBIL_ADDR: &str = "0x00000000000000000000000000000000000000A1";
pub const NORMAL_USER_ADDR: &str = "0x00000000000000000000000000000000000000b2";
pub const LIQUIDITY_PROVIDER_ADDR: &str = "0x00000000000000000000000000000000000000C3";
pub const LIQUIDATOR_KEEPER_ADDR: &str = "0x00000000000000000000000000000000000000D4";

pub const ATTACKER0_BYTES: [u8; 20] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa0,
];

/// Synthesizer for generating concrete Foundry proof-of-concept (`ExploitPoC.t.sol`) test files.
pub struct PocSynthesizer;

impl PocSynthesizer {
    /// Generates standalone Foundry test code (`ExploitPoC.t.sol`) from a candidate trajectory.
    pub fn synthesize_poc(
        trajectory: &TrajectoryRecord,
        violated_invariant_id: &str,
    ) -> GfnResult<String> {
        Self::synthesize_poc_with_options(trajectory, violated_invariant_id, None)
    }

    /// Generates standalone Foundry test code with optional harness inheritance or contract setup.
    pub fn synthesize_poc_with_options(
        trajectory: &TrajectoryRecord,
        violated_invariant_id: &str,
        harness_parent: Option<&str>,
    ) -> GfnResult<String> {
        if trajectory.steps.is_empty() {
            return Err(GfnError::Serialization {
                details: "Candidate trajectory contains zero transaction steps".to_string(),
            });
        }

        let mut out = String::with_capacity(6144);

        // 1. Pragma & standard imports
        out.push_str("// SPDX-License-Identifier: MIT\n");
        out.push_str("pragma solidity ^0.8.24;\n\n");
        out.push_str("import {Test} from \"forge-std/Test.sol\";\n");

        if let Some(parent) = harness_parent {
            out.push_str(&format!(
                "import {{{parent}}} from \"./{parent}.t.sol\";\n\n"
            ));
        } else {
            out.push('\n');
        }

        // 2. Identify unique non-address target contracts to declare
        let mut target_names = BTreeSet::new();
        for step in &trajectory.steps {
            if !step.target_contract.is_empty() && !step.target_contract.starts_with("0x") {
                target_names.insert(step.target_contract.clone());
            }
        }

        // 3. Contract definition
        let is_clause = if let Some(parent) = harness_parent {
            format!("is {parent}")
        } else {
            "is Test".to_string()
        };

        out.push_str(&format!("contract ExploitPoC {is_clause} {{\n"));

        // Actor address constants
        out.push_str(&format!(
            "    address internal constant ATTACKER0 = address({ATTACKER0_ADDR});\n"
        ));
        out.push_str(&format!(
            "    address internal constant ATTACKER1_SYBIL = address({ATTACKER1_SYBIL_ADDR});\n"
        ));
        out.push_str(&format!(
            "    address internal constant NORMAL_USER = address({NORMAL_USER_ADDR});\n"
        ));
        out.push_str(&format!(
            "    address internal constant LIQUIDITY_PROVIDER = address({LIQUIDITY_PROVIDER_ADDR});\n"
        ));
        out.push_str(&format!(
            "    address internal constant LIQUIDATOR_KEEPER = address({LIQUIDATOR_KEEPER_ADDR});\n\n"
        ));

        // Declare contract address variables if not using a parent harness
        if harness_parent.is_none() {
            for name in &target_names {
                let var_name = sanitize_var_name(name);
                out.push_str(&format!("    address internal {var_name};\n"));
            }
            if !target_names.is_empty() {
                out.push('\n');
            }
        }

        // Safe staticcall state inspection helpers (non-reverting on arbitrary contracts)
        out.push_str(
            "    function _readUint(address targetAddr, bytes memory callData) internal view returns (bool ok, uint256 val) {\n"
        );
        out.push_str("        bytes memory ret;\n");
        out.push_str("        (ok, ret) = targetAddr.staticcall(callData);\n");
        out.push_str("        if (ok && ret.length >= 32) {\n");
        out.push_str("            val = abi.decode(ret, (uint256));\n");
        out.push_str("        } else {\n");
        out.push_str("            ok = false;\n");
        out.push_str("        }\n");
        out.push_str("    }\n\n");

        out.push_str(
            "    function _readAddr(address targetAddr, bytes memory callData) internal view returns (bool ok, address val) {\n"
        );
        out.push_str("        bytes memory ret;\n");
        out.push_str("        (ok, ret) = targetAddr.staticcall(callData);\n");
        out.push_str("        if (ok && ret.length >= 32) {\n");
        out.push_str("            val = abi.decode(ret, (address));\n");
        out.push_str("        } else {\n");
        out.push_str("            ok = false;\n");
        out.push_str("        }\n");
        out.push_str("    }\n\n");

        // Universal DeFi & Flashloan Callback Hook Receivers
        out.push_str("    // Universal DeFi & Flashloan Callback Hook Receivers\n");
        out.push_str("    receive() external payable {}\n");
        out.push_str("    fallback() external payable {}\n\n");
        out.push_str("    function onFlashLoan(address, address, uint256, uint256, bytes calldata) external returns (bytes32) {\n");
        out.push_str("        return keccak256(\"ERC3156FlashBorrower.onFlashLoan\");\n");
        out.push_str("    }\n\n");
        out.push_str("    function executeOperation(address[] calldata, uint256[] calldata, uint256[] calldata, address, bytes calldata) external returns (bool) {\n");
        out.push_str("        return true;\n");
        out.push_str("    }\n\n");
        out.push_str("    function uniswapV2Call(address, uint256, uint256, bytes calldata) external {}\n\n");
        out.push_str("    function uniswapV3SwapCallback(int256, int256, bytes calldata) external {}\n\n");
        out.push_str("    function tokensReceived(address, address, address, uint256, bytes calldata, bytes calldata) external {}\n\n");

        let helper_body = synthesize_invariant_helper(violated_invariant_id);
        out.push_str(&helper_body);
        out.push('\n');

        // 4. setUp() block
        out.push_str("    function setUp() public virtual");
        if harness_parent.is_some() {
            out.push_str(" override");
        }
        out.push_str(" {\n");

        if harness_parent.is_some() {
            out.push_str("        super.setUp();\n");
        }

        out.push_str("        vm.roll(18_000_000);\n");
        out.push_str("        vm.warp(1_700_000_000);\n\n");

        out.push_str("        vm.deal(ATTACKER0, 100 ether);\n");
        out.push_str("        vm.deal(ATTACKER1_SYBIL, 100 ether);\n");
        out.push_str("        vm.deal(NORMAL_USER, 100 ether);\n");
        out.push_str("        vm.deal(LIQUIDITY_PROVIDER, 1000 ether);\n");
        out.push_str("        vm.deal(LIQUIDATOR_KEEPER, 100 ether);\n");

        out.push_str("    }\n\n");

        // Resolve primary target contract expression from final non-empty step
        let primary_target_expr = trajectory
            .steps
            .iter()
            .rev()
            .find(|s| !s.target_contract.is_empty())
            .map(|s| resolve_target_expr(&s.target_contract))
            .unwrap_or_else(|| "address(this)".to_string());

        // 5. test_ExploitPoC() block with pre-state snapshots
        out.push_str("    function test_ExploitPoC() public {\n");
        out.push_str(&format!(
            "        address primaryTarget = {primary_target_expr};\n"
        ));
        out.push_str("        uint256 initAttackerBal = ATTACKER0.balance;\n");
        out.push_str("        uint256 initTimestamp = block.timestamp;\n");
        out.push_str(
            "        (bool hasAdminPre, address adminPre) = _readAddr(primaryTarget, abi.encodeWithSignature(\"admin()\"));\n",
        );
        out.push_str(
            "        (bool hasOwnerPre, address ownerPre) = _readAddr(primaryTarget, abi.encodeWithSignature(\"owner()\"));\n\n",
        );

        for (idx, step) in trajectory.steps.iter().enumerate() {
            out.push_str(&format!(
                "        // Step {idx}: {} calls {}::{}\n",
                step.caller_role, step.target_contract, step.function_signature
            ));

            if step.vm_warp_sec > 0 {
                out.push_str(&format!(
                    "        vm.warp(block.timestamp + {});\n",
                    step.vm_warp_sec
                ));
            }
            if step.vm_roll_blocks > 0 {
                out.push_str(&format!(
                    "        vm.roll(block.number + {});\n",
                    step.vm_roll_blocks
                ));
            }

            // Caller prank
            let caller_expr = if step.caller_address == [0u8; 20] {
                "ATTACKER0".to_string()
            } else {
                let addr = alloy_primitives::Address::from(step.caller_address);
                format!("address({})", addr.to_checksum(None))
            };
            out.push_str(&format!("        vm.prank({caller_expr});\n"));

            let target_expr = resolve_target_expr(&step.target_contract);

            // Scoped calldata execution block prevents EVM stack-too-deep on deep trajectories
            let hex_raw = step
                .concrete_calldata_hex
                .strip_prefix("0x")
                .unwrap_or(&step.concrete_calldata_hex);

            out.push_str("        {\n");
            if !hex_raw.is_empty() {
                out.push_str(&format!(
                    "            (bool okStep, ) = {target_expr}.call(hex\"{hex_raw}\");\n"
                ));
                out.push_str(&format!(
                    "            require(okStep, \"Step {idx} execution failed\");\n"
                ));
            } else {
                out.push_str(&format!(
                    "            (bool okStep, ) = {target_expr}.call(\"\");\n"
                ));
                out.push_str(&format!(
                    "            require(okStep, \"Step {idx} execution failed\");\n"
                ));
            }
            out.push_str("        }\n\n");
        }

        // 6. Explicit Non-Tautological Foundry Invariant Assertions
        out.push_str(&format!(
            "        // Invariant Violation Assertion: {violated_invariant_id}\n"
        ));
        out.push_str(
            "        assertTrue(_isInvariantViolated(primaryTarget, initAttackerBal, initTimestamp, hasAdminPre, adminPre, hasOwnerPre, ownerPre), \"Invariant violation not reproduced\");\n",
        );

        out.push_str("    }\n");
        out.push_str("}\n");

        Ok(out)
    }
}

fn resolve_target_expr(target_contract: &str) -> String {
    if target_contract.starts_with("0x") {
        if let Ok(addr) = target_contract.parse::<alloy_primitives::Address>() {
            format!("address({})", addr.to_checksum(None))
        } else {
            format!("address({target_contract})")
        }
    } else if !target_contract.is_empty() {
        let var_name = sanitize_var_name(target_contract);
        format!("address({var_name})")
    } else {
        "address(this)".to_string()
    }
}

fn sanitize_var_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let sanitized: String = lower
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    if sanitized.is_empty() {
        "targetContract".to_string()
    } else {
        sanitized
    }
}

fn synthesize_invariant_helper(invariant_id: &str) -> String {
    let id_upper = invariant_id.to_uppercase();
    let mut s = String::from(
        "    function _isInvariantViolated(\n        address primaryTarget,\n        uint256 initAttackerBal,\n        uint256 initTimestamp,\n        bool hasAdminPre,\n        address adminPre,\n        bool hasOwnerPre,\n        address ownerPre\n    ) internal view returns (bool) {\n        (initTimestamp, hasAdminPre, adminPre, hasOwnerPre, ownerPre);\n        if (ATTACKER0.balance > initAttackerBal) { return true; }\n",
    );

    if id_upper.contains("SOLVENCY") {
        s.push_str(concat!(
            "        (, uint256 colPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalCollateral()\"));\n",
            "        (, uint256 debtPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalDebt()\"));\n",
            "        if (debtPost > colPost && debtPost > 0) { return true; }\n",
            "        (bool hasAssets, uint256 assetsPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalAssets()\"));\n",
            "        return hasAssets && assetsPost > primaryTarget.balance && primaryTarget.balance > 0;\n",
        ));
    } else if id_upper.contains("PRECISION") || id_upper.contains("ROUNDING") {
        s.push_str(concat!(
            "        (bool hasSup, uint256 supPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalSupply()\"));\n",
            "        (bool hasConv, uint256 unitShares) = _readUint(primaryTarget, abi.encodeWithSignature(\"convertToShares(uint256)\", 1 ether));\n",
            "        return hasSup && hasConv && supPost > 0 && unitShares == 0;\n",
        ));
    } else if id_upper.contains("ACCESS") || id_upper.contains("AUTH") || id_upper.contains("OWNER")
    {
        s.push_str(concat!(
            "        (bool hasAdmPost, address admPost) = _readAddr(primaryTarget, abi.encodeWithSignature(\"admin()\"));\n",
            "        if (hasAdminPre && hasAdmPost && adminPre != ATTACKER0 && admPost == ATTACKER0) { return true; }\n",
            "        (bool hasOwnPost, address ownPost) = _readAddr(primaryTarget, abi.encodeWithSignature(\"owner()\"));\n",
            "        return hasOwnerPre && hasOwnPost && ownerPre != ATTACKER0 && ownPost == ATTACKER0;\n",
        ));
    } else if id_upper.contains("MONO") || id_upper.contains("TIMELOCK") {
        s.push_str(concat!(
            "        (bool hasEta, uint256 propEta) = _readUint(primaryTarget, abi.encodeWithSignature(\"proposalEta(uint256)\", 99));\n",
            "        (bool hasExec, uint256 propExec) = _readUint(primaryTarget, abi.encodeWithSignature(\"proposalExecuted(uint256)\", 99));\n",
            "        return (hasEta && hasExec && propExec != 0 && block.timestamp < propEta) || (block.timestamp < initTimestamp);\n",
        ));
    } else if id_upper.contains("BRIDGE") {
        s.push_str(concat!(
            "        (, uint256 lockedPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalLocked()\"));\n",
            "        (, uint256 mintedPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalMinted()\"));\n",
            "        return mintedPost > lockedPost && mintedPost > 0;\n",
        ));
    } else if id_upper.contains("GOV") || id_upper.contains("VOTE") {
        s.push_str(concat!(
            "        (bool hasVotes, uint256 votesPost) = _readUint(primaryTarget, abi.encodeWithSignature(\"getVotes(address)\", ATTACKER0));\n",
            "        (, uint256 totalPower) = _readUint(primaryTarget, abi.encodeWithSignature(\"totalVotingPower()\"));\n",
            "        return hasVotes && totalPower > 0 && votesPost > (totalPower * 2000) / 10000;\n",
        ));
    } else {
        s.push_str("        (primaryTarget);\n        return false;\n");
    }

    s.push_str("    }\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfn_common::schemas::TrajectoryStepRecord;

    #[test]
    fn test_poc_synthesizer_generates_valid_non_tautological_solidity() {
        let trajectory = TrajectoryRecord {
            trajectory_id: "traj-001".to_string(),
            steps: vec![
                TrajectoryStepRecord {
                    step_idx: 0,
                    caller_role: "Attacker0".to_string(),
                    caller_address: ATTACKER0_BYTES,
                    target_contract: "0x1111111111111111111111111111111111111111".to_string(),
                    function_signature: "deposit(uint256)".to_string(),
                    concrete_calldata_hex:
                        "0xb6b55f250000000000000000000000000000000000000000000000000de0b6b3a7640000"
                            .to_string(),
                    decoded_args: vec!["1000000000000000000".to_string()],
                    vm_warp_sec: 60,
                    vm_roll_blocks: 5,
                },
                TrajectoryStepRecord {
                    step_idx: 1,
                    caller_role: "Attacker0".to_string(),
                    caller_address: ATTACKER0_BYTES,
                    target_contract: "0x1111111111111111111111111111111111111111".to_string(),
                    function_signature: "withdraw(uint256)".to_string(),
                    concrete_calldata_hex: "0x2e1a7d4d".to_string(),
                    decoded_args: vec![],
                    vm_warp_sec: 0,
                    vm_roll_blocks: 0,
                },
            ],
            terminal_reward: 100.0,
            violated_invariant_id: Some("INV_SOLVENCY_01".to_string()),
            is_exploit: true,
            log_pf: -2.5,
            log_pb: -3.0,
        };

        let poc = PocSynthesizer::synthesize_poc(&trajectory, "INV_SOLVENCY_01")
            .unwrap_or_else(|e| panic!("PoC synthesis failed: {e}"));

        assert!(poc.contains("// SPDX-License-Identifier: MIT"));
        assert!(poc.contains("pragma solidity ^0.8.24;"));
        assert!(poc.contains("import {Test} from \"forge-std/Test.sol\";"));
        assert!(poc.contains("contract ExploitPoC is Test {"));
        assert!(poc.contains("vm.roll(18_000_000);"));
        assert!(poc.contains("vm.warp(1_700_000_000);"));
        assert!(poc.contains("vm.deal(ATTACKER0, 100 ether);"));
        assert!(poc.contains("uint256 initAttackerBal = ATTACKER0.balance;"));
        assert!(poc.contains("ATTACKER0.balance > initAttackerBal"));
        assert!(poc.contains("function onFlashLoan("));
        assert!(poc.contains("function executeOperation("));
        assert!(poc.contains("function uniswapV2Call("));
        assert!(poc.contains("function uniswapV3SwapCallback("));
        assert!(poc.contains("function tokensReceived("));
        assert!(
            !poc.contains("assertGt(ATTACKER0.balance, 0"),
            "PoC must never emit tautological balance > 0 checks after vm.deal"
        );
    }

    #[test]
    fn test_poc_synthesizer_rejects_empty_trajectory() {
        let trajectory = TrajectoryRecord {
            trajectory_id: "traj-empty".to_string(),
            steps: vec![],
            terminal_reward: 0.0,
            violated_invariant_id: None,
            is_exploit: false,
            log_pf: 0.0,
            log_pb: 0.0,
        };

        let res = PocSynthesizer::synthesize_poc(&trajectory, "INV_01");
        assert!(res.is_err());
    }
}
