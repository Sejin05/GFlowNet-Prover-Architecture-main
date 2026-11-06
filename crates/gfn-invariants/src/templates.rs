#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Parameterized formal invariant templates across the 4 canonical domain categories:
//! 1. Conservation of Value & Solvency (`SOLVENCY`)
//! 2. State Machine Monotonicity & Lifecycle Integrity (`MONOTONICITY`)
//! 3. Privilege & Access Control Isolation (`ACCESS_CONTROL`)
//! 4. Rounding & Precision Neutrality (`PRECISION_NEUTRALITY`)

use alloy_primitives::U256;
use gfn_common::schemas::InvariantCategory;
use serde::{Deserialize, Serialize};

/// Parameterized formal invariant templates across the 4 canonical domain categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantTemplate {
    /// 1. Conservation of Value & Solvency (SOLVENCY):
    ///
    /// `\sum_{u \in Shares} slot(u) * P_{share} <= \sum_{v \in Assets} slot(v)`
    Solvency {
        /// Storage slots representing share tokens or liabilities.
        share_slots: Vec<u64>,
        /// Optional storage slot representing the share exchange rate or scaling factor.
        price_slot: Option<u64>,
        /// Storage slots representing backing reserve assets.
        asset_slots: Vec<u64>,
    },

    /// 2. State Machine Monotonicity & Lifecycle Integrity (MONOTONICITY):
    ///
    /// `slot_{state}(s_{t+1}) >= slot_{state}(s_t) \lor A_{valid}[state_t, state_{t+1}] == 1`
    Monotonicity {
        /// Storage slot tracking the protocol state or epoch counter.
        state_slot: u64,
        /// Explicit valid discrete state transition pairs `(from, to)`. If empty, monotonic non-decreasing is enforced.
        allowed_transitions: Vec<(U256, U256)>,
    },

    /// 3. Privilege & Access Control Isolation (ACCESS_CONTROL):
    ///
    /// `caller \notin AdminSet \implies slot_{critical}(s_{t+1}) == slot_{critical}(s_t)`
    AccessControl {
        /// Storage slot holding privileged addresses (owner, admin, beacon, implementation, oracle).
        critical_slot: u64,
        /// Personas permitted to mutate this slot (e.g. `1` for `ContractDeployerAdmin`).
        admin_personas: Vec<u8>,
    },

    /// 4. Rounding & Precision Neutrality (PRECISION_NEUTRALITY):
    ///
    /// `Balance_{attacker}(s_n) <= Balance_{attacker}(s_0) + \epsilon_{dust}` (where `\epsilon_{dust} = 0 wei`)
    PrecisionNeutrality {
        /// Storage slot tracking the attacker persona's asset or share balance.
        attacker_balance_slot: u64,
        /// Allowed rounding dust tolerance in wei (default 0).
        dust_tolerance: U256,
    },

    /// 5. AMM Multi-Pool Arbitrage & Sandwich Resistance (ARBITRAGE_NEUTRALITY):
    ///
    /// `k_pool(s_n) >= k_pool(s_0) && AttackerProfit <= MaxAllowedSlippage`
    ArbitrageNeutrality {
        /// Storage slot for pool reserve 0 (e.g. token0).
        reserve0_slot: u64,
        /// Storage slot for pool reserve 1 (e.g. token1).
        reserve1_slot: u64,
        /// Storage slot tracking attacker profit or net balance.
        attacker_balance_slot: u64,
        /// Max permitted profit from sandwich or cyclic arbitrage in wei.
        max_profit_tolerance: U256,
    },

    /// 6. Concentrated Liquidity Tick Boundary & Active Pool Invariant (CONCENTRATED_LIQUIDITY):
    ///
    /// `tick_lower <= tick_current < tick_upper ==> active_liquidity > 0`
    ConcentratedLiquidity {
        /// Storage slot tracking current pool tick.
        tick_current_slot: u64,
        /// Storage slot tracking lower bound of position tick range.
        tick_lower_slot: u64,
        /// Storage slot tracking upper bound of position tick range.
        tick_upper_slot: u64,
        /// Storage slot tracking active liquidity units.
        liquidity_slot: u64,
    },

    /// 7. Cross-Chain Bridge Backing & Asset Lock Solvency (BRIDGE_SOLVENCY):
    ///
    /// `\sum_{v \in Minted} slot(v) <= \sum_{u \in Locked} slot(u) + FeeReserve`
    BridgeSolvency {
        /// Storage slots tracking locked backing collateral on the native chain.
        locked_asset_slots: Vec<u64>,
        /// Storage slots tracking minted wrapped tokens on the destination chain.
        minted_wrapped_slots: Vec<u64>,
        /// Optional storage slot tracking accrued bridge fee reserve.
        fee_reserve_slot: Option<u64>,
    },

    /// 8. Governance Flash-Loan Vote Hijacking & Snapshot Isolation (GOVERNANCE_FLASH_VOTE):
    ///
    /// `votes(proposal) <= (totalVotingPower * maxVoteShareBps) / 10000`
    GovernanceFlashVote {
        /// Storage slot tracking current proposal vote count.
        proposal_votes_slot: u64,
        /// Storage slot tracking total eligible voting token supply.
        total_voting_power_slot: u64,
        /// Maximum allowed single-actor voting share threshold in basis points (e.g. 5000 = 50%).
        max_vote_share_bps: u32,
    },
}

impl InvariantTemplate {
    /// Returns the canonical category for this invariant template.
    #[must_use]
    pub fn category(&self) -> InvariantCategory {
        match self {
            Self::Solvency { .. } => InvariantCategory::Solvency,
            Self::Monotonicity { .. } => InvariantCategory::Monotonicity,
            Self::AccessControl { .. } => InvariantCategory::AccessControl,
            Self::PrecisionNeutrality { .. } => InvariantCategory::PrecisionNeutrality,
            Self::ArbitrageNeutrality { .. } => InvariantCategory::ArbitrageNeutrality,
            Self::ConcentratedLiquidity { .. } => InvariantCategory::ConcentratedLiquidity,
            Self::BridgeSolvency { .. } => InvariantCategory::BridgeSolvency,
            Self::GovernanceFlashVote { .. } => InvariantCategory::GovernanceFlashVote,
        }
    }

    /// Returns all storage slot IDs involved in this invariant template.
    #[must_use]
    pub fn involved_slots(&self) -> Vec<u64> {
        match self {
            Self::Solvency {
                share_slots,
                price_slot,
                asset_slots,
            } => {
                let mut slots = Vec::new();
                slots.extend_from_slice(share_slots);
                if let Some(p) = price_slot {
                    slots.push(*p);
                }
                slots.extend_from_slice(asset_slots);
                slots.sort_unstable();
                slots.dedup();
                slots
            }
            Self::Monotonicity { state_slot, .. } => vec![*state_slot],
            Self::AccessControl { critical_slot, .. } => vec![*critical_slot],
            Self::PrecisionNeutrality {
                attacker_balance_slot,
                ..
            } => vec![*attacker_balance_slot],
            Self::ArbitrageNeutrality {
                reserve0_slot,
                reserve1_slot,
                attacker_balance_slot,
                ..
            } => {
                let mut slots = vec![*reserve0_slot, *reserve1_slot, *attacker_balance_slot];
                slots.sort_unstable();
                slots.dedup();
                slots
            }
            Self::ConcentratedLiquidity {
                tick_current_slot,
                tick_lower_slot,
                tick_upper_slot,
                liquidity_slot,
            } => {
                let mut slots = vec![
                    *tick_current_slot,
                    *tick_lower_slot,
                    *tick_upper_slot,
                    *liquidity_slot,
                ];
                slots.sort_unstable();
                slots.dedup();
                slots
            }
            Self::BridgeSolvency {
                locked_asset_slots,
                minted_wrapped_slots,
                fee_reserve_slot,
            } => {
                let mut slots = Vec::new();
                slots.extend_from_slice(locked_asset_slots);
                slots.extend_from_slice(minted_wrapped_slots);
                if let Some(f) = fee_reserve_slot {
                    slots.push(*f);
                }
                slots.sort_unstable();
                slots.dedup();
                slots
            }
            Self::GovernanceFlashVote {
                proposal_votes_slot,
                total_voting_power_slot,
                ..
            } => {
                let mut slots = vec![*proposal_votes_slot, *total_voting_power_slot];
                slots.sort_unstable();
                slots.dedup();
                slots
            }
        }
    }

    /// Formats the formal mathematical expression for telemetry, SMT export, and reporting.
    #[must_use]
    pub fn formal_expression(&self) -> String {
        match self {
            Self::Solvency {
                share_slots,
                price_slot,
                asset_slots,
            } => {
                let shares_str = if share_slots.is_empty() {
                    "0".to_string()
                } else {
                    share_slots
                        .iter()
                        .map(|s| format!("slot({s})"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                let assets_str = if asset_slots.is_empty() {
                    "0".to_string()
                } else {
                    asset_slots
                        .iter()
                        .map(|s| format!("slot({s})"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                if let Some(p) = price_slot {
                    format!("({shares_str}) * slot({p}) <= ({assets_str})")
                } else {
                    format!("({shares_str}) <= ({assets_str})")
                }
            }
            Self::Monotonicity {
                state_slot,
                allowed_transitions,
            } => {
                if allowed_transitions.is_empty() {
                    format!("slot({state_slot})_next >= slot({state_slot})_prev")
                } else {
                    format!(
                        "slot({state_slot})_next >= slot({state_slot})_prev \
                         || is_valid_transition(slot({state_slot})_prev, slot({state_slot})_next)"
                    )
                }
            }
            Self::AccessControl {
                critical_slot,
                admin_personas,
            } => {
                let admins = admin_personas
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "caller not in [{admins}] ==> slot({critical_slot})_next == slot({critical_slot})_prev"
                )
            }
            Self::PrecisionNeutrality {
                attacker_balance_slot,
                dust_tolerance,
            } => {
                format!(
                    "slot({attacker_balance_slot})_n <= slot({attacker_balance_slot})_0 + {dust_tolerance}"
                )
            }
            Self::ArbitrageNeutrality {
                reserve0_slot,
                reserve1_slot,
                attacker_balance_slot,
                max_profit_tolerance,
            } => {
                format!(
                    "(slot({reserve0_slot})_n * slot({reserve1_slot})_n >= slot({reserve0_slot})_0 * slot({reserve1_slot})_0) \
                     && (slot({attacker_balance_slot})_n <= slot({attacker_balance_slot})_0 + {max_profit_tolerance})"
                )
            }
            Self::ConcentratedLiquidity {
                tick_current_slot,
                tick_lower_slot,
                tick_upper_slot,
                liquidity_slot,
            } => {
                format!(
                    "(slot({tick_lower_slot}) <= slot({tick_current_slot}) && slot({tick_current_slot}) < slot({tick_upper_slot})) \
                     ==> slot({liquidity_slot}) > 0"
                )
            }
            Self::BridgeSolvency {
                locked_asset_slots,
                minted_wrapped_slots,
                fee_reserve_slot,
            } => {
                let locked_str = if locked_asset_slots.is_empty() {
                    "0".to_string()
                } else {
                    locked_asset_slots
                        .iter()
                        .map(|s| format!("slot({s})"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                let minted_str = if minted_wrapped_slots.is_empty() {
                    "0".to_string()
                } else {
                    minted_wrapped_slots
                        .iter()
                        .map(|s| format!("slot({s})"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                if let Some(f) = fee_reserve_slot {
                    format!("({minted_str}) <= ({locked_str}) + slot({f})")
                } else {
                    format!("({minted_str}) <= ({locked_str})")
                }
            }
            Self::GovernanceFlashVote {
                proposal_votes_slot,
                total_voting_power_slot,
                max_vote_share_bps,
            } => {
                format!(
                    "slot({proposal_votes_slot}) <= (slot({total_voting_power_slot}) * {max_vote_share_bps}) / 10000"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solvency_template_metadata() {
        let template = InvariantTemplate::Solvency {
            share_slots: vec![1, 2],
            price_slot: Some(5),
            asset_slots: vec![3, 4],
        };

        assert_eq!(template.category(), InvariantCategory::Solvency);
        assert_eq!(template.involved_slots(), vec![1, 2, 3, 4, 5]);
        assert_eq!(
            template.formal_expression(),
            "(slot(1) + slot(2)) * slot(5) <= (slot(3) + slot(4))"
        );
    }

    #[test]
    fn test_monotonicity_template_metadata() {
        let template = InvariantTemplate::Monotonicity {
            state_slot: 10,
            allowed_transitions: vec![(U256::from(1), U256::from(2))],
        };

        assert_eq!(template.category(), InvariantCategory::Monotonicity);
        assert_eq!(template.involved_slots(), vec![10]);
        assert!(template.formal_expression().contains("slot(10)_next"));
    }

    #[test]
    fn test_access_control_template_metadata() {
        let template = InvariantTemplate::AccessControl {
            critical_slot: 7,
            admin_personas: vec![1],
        };

        assert_eq!(template.category(), InvariantCategory::AccessControl);
        assert_eq!(template.involved_slots(), vec![7]);
        assert_eq!(
            template.formal_expression(),
            "caller not in [1] ==> slot(7)_next == slot(7)_prev"
        );
    }

    #[test]
    fn test_precision_neutrality_template_metadata() {
        let template = InvariantTemplate::PrecisionNeutrality {
            attacker_balance_slot: 0,
            dust_tolerance: U256::ZERO,
        };

        assert_eq!(template.category(), InvariantCategory::PrecisionNeutrality);
        assert_eq!(template.involved_slots(), vec![0]);
        assert_eq!(template.formal_expression(), "slot(0)_n <= slot(0)_0 + 0");
    }

    #[test]
    fn test_arbitrage_neutrality_template_metadata() {
        let template = InvariantTemplate::ArbitrageNeutrality {
            reserve0_slot: 1,
            reserve1_slot: 2,
            attacker_balance_slot: 0,
            max_profit_tolerance: U256::from(100),
        };

        assert_eq!(template.category(), InvariantCategory::ArbitrageNeutrality);
        assert_eq!(template.involved_slots(), vec![0, 1, 2]);
        assert!(template.formal_expression().contains("slot(1)_n * slot(2)_n"));
    }

    #[test]
    fn test_concentrated_liquidity_template_metadata() {
        let template = InvariantTemplate::ConcentratedLiquidity {
            tick_current_slot: 10,
            tick_lower_slot: 11,
            tick_upper_slot: 12,
            liquidity_slot: 13,
        };

        assert_eq!(template.category(), InvariantCategory::ConcentratedLiquidity);
        assert_eq!(template.involved_slots(), vec![10, 11, 12, 13]);
        assert!(template.formal_expression().contains("slot(13) > 0"));
    }

    #[test]
    fn test_bridge_solvency_template_metadata() {
        let template = InvariantTemplate::BridgeSolvency {
            locked_asset_slots: vec![1, 2],
            minted_wrapped_slots: vec![3],
            fee_reserve_slot: Some(4),
        };

        assert_eq!(template.category(), InvariantCategory::BridgeSolvency);
        assert_eq!(template.involved_slots(), vec![1, 2, 3, 4]);
        assert!(template.formal_expression().contains("(slot(3)) <= (slot(1) + slot(2)) + slot(4)"));
    }

    #[test]
    fn test_governance_flash_vote_template_metadata() {
        let template = InvariantTemplate::GovernanceFlashVote {
            proposal_votes_slot: 20,
            total_voting_power_slot: 21,
            max_vote_share_bps: 5000,
        };

        assert_eq!(template.category(), InvariantCategory::GovernanceFlashVote);
        assert_eq!(template.involved_slots(), vec![20, 21]);
        assert!(template.formal_expression().contains("slot(20) <= (slot(21) * 5000) / 10000"));
    }
}
