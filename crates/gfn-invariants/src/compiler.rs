#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! SMT-LIBv2 `QF_BV` Invariant Compiler and Vectorized Evaluator.
//!
//! Compiles candidate `InvariantTemplate` definitions into executable `CompiledInvariant` instances
//! containing fast concrete state evaluators, normalized boundary proximity metric calculators,
//! and SMT-LIBv2 256-bit BitVector negated invariant encodings.

use alloy_primitives::U256;
use gfn_common::error::GfnResult;
use gfn_common::schemas::{InvariantCategory, SynthesizedInvariant};

use crate::templates::InvariantTemplate;

/// Compiled safety invariant with fast concrete evaluator, boundary proximity calculator, and SMT-LIBv2 generator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledInvariant {
    /// Schema metadata for telemetry, monitoring, and export.
    pub metadata: SynthesizedInvariant,
    /// Underlying parameterized AST template.
    pub template: InvariantTemplate,
}

impl CompiledInvariant {
    /// Evaluates whether the invariant holds across state transition `prev_slots -> curr_slots` with caller persona `caller_id`.
    ///
    /// Returns `true` if the invariant is satisfied, or `false` if violated
    /// ($\neg \mathcal{I}_m(s_n) = \text{true}$, which triggers the exploit reward $R_{\text{exploit}} = 1.0$).
    #[must_use]
    pub fn evaluate_satisfied(
        &self,
        prev_slots: &[U256],
        curr_slots: &[U256],
        caller_id: u8,
    ) -> bool {
        match &self.template {
            InvariantTemplate::Solvency {
                share_slots,
                price_slot,
                asset_slots,
            } => {
                // \sum_{u \in Shares} slot(u) * P_{share} <= \sum_{v \in Assets} slot(v)
                let mut sum_shares = U256::ZERO;
                for &s in share_slots {
                    sum_shares = sum_shares.saturating_add(slot_or_zero(curr_slots, s));
                }

                let price = match price_slot {
                    Some(p) => slot_or_zero(curr_slots, *p),
                    None => U256::from(1),
                };

                let (lhs, overflow) = sum_shares.overflowing_mul(price);
                if overflow {
                    // Overflow in U256 (> 1.15e77) exceeds any possible backing assets
                    return false;
                }

                let mut sum_assets = U256::ZERO;
                for &a in asset_slots {
                    sum_assets = sum_assets.saturating_add(slot_or_zero(curr_slots, a));
                }

                lhs <= sum_assets
            }

            InvariantTemplate::Monotonicity {
                state_slot,
                allowed_transitions,
            } => {
                let prev = slot_or_zero(prev_slots, *state_slot);
                let curr = slot_or_zero(curr_slots, *state_slot);

                if curr >= prev {
                    return true;
                }

                for &(from, to) in allowed_transitions {
                    if from == prev && to == curr {
                        return true;
                    }
                }

                false
            }

            InvariantTemplate::AccessControl {
                critical_slot,
                admin_personas,
            } => {
                if admin_personas.contains(&caller_id) {
                    return true;
                }

                let prev = slot_or_zero(prev_slots, *critical_slot);
                let curr = slot_or_zero(curr_slots, *critical_slot);

                curr == prev
            }

            InvariantTemplate::PrecisionNeutrality {
                attacker_balance_slot,
                dust_tolerance,
            } => {
                // Precision neutrality evaluates attacker balance growth across attacker trajectories (Personas 0 and 1).
                // Benign protocol users (Personas >= 2) depositing or trading do not violate attacker neutrality.
                if caller_id != 0 && caller_id != 1 {
                    return true;
                }

                let initial = slot_or_zero(prev_slots, *attacker_balance_slot);
                let current = slot_or_zero(curr_slots, *attacker_balance_slot);
                let allowed_limit = initial.saturating_add(*dust_tolerance);

                current <= allowed_limit
            }

            InvariantTemplate::ArbitrageNeutrality {
                reserve0_slot,
                reserve1_slot,
                attacker_balance_slot,
                max_profit_tolerance,
            } => {
                let r0_prev = slot_or_zero(prev_slots, *reserve0_slot);
                let r1_prev = slot_or_zero(prev_slots, *reserve1_slot);
                let r0_curr = slot_or_zero(curr_slots, *reserve0_slot);
                let r1_curr = slot_or_zero(curr_slots, *reserve1_slot);

                let (k_prev, _) = r0_prev.overflowing_mul(r1_prev);
                let (k_curr, _) = r0_curr.overflowing_mul(r1_curr);

                if k_curr < k_prev && !k_prev.is_zero() {
                    return false;
                }

                if caller_id == 0 || caller_id == 1 {
                    let bal_prev = slot_or_zero(prev_slots, *attacker_balance_slot);
                    let bal_curr = slot_or_zero(curr_slots, *attacker_balance_slot);
                    if bal_curr > bal_prev.saturating_add(*max_profit_tolerance) {
                        return false;
                    }
                }

                true
            }

            InvariantTemplate::ConcentratedLiquidity {
                tick_current_slot,
                tick_lower_slot,
                tick_upper_slot,
                liquidity_slot,
            } => {
                let tick = slot_or_zero(curr_slots, *tick_current_slot);
                let tick_lower = slot_or_zero(curr_slots, *tick_lower_slot);
                let tick_upper = slot_or_zero(curr_slots, *tick_upper_slot);
                let liq = slot_or_zero(curr_slots, *liquidity_slot);

                if tick >= tick_lower && tick < tick_upper {
                    !liq.is_zero()
                } else {
                    true
                }
            }

            InvariantTemplate::BridgeSolvency {
                locked_asset_slots,
                minted_wrapped_slots,
                fee_reserve_slot,
            } => {
                let mut sum_locked = U256::ZERO;
                for &l in locked_asset_slots {
                    sum_locked = sum_locked.saturating_add(slot_or_zero(curr_slots, l));
                }
                if let Some(f) = fee_reserve_slot {
                    sum_locked = sum_locked.saturating_add(slot_or_zero(curr_slots, *f));
                }

                let mut sum_minted = U256::ZERO;
                for &m in minted_wrapped_slots {
                    sum_minted = sum_minted.saturating_add(slot_or_zero(curr_slots, m));
                }

                sum_minted <= sum_locked
            }

            InvariantTemplate::GovernanceFlashVote {
                proposal_votes_slot,
                total_voting_power_slot,
                max_vote_share_bps,
            } => {
                let votes = slot_or_zero(curr_slots, *proposal_votes_slot);
                let total_power = slot_or_zero(curr_slots, *total_voting_power_slot);
                if total_power.is_zero() {
                    return votes.is_zero();
                }

                let (max_allowed, _) = total_power
                    .overflowing_mul(U256::from(*max_vote_share_bps));
                let cap = max_allowed / U256::from(10_000);

                if caller_id == 0 || caller_id == 1 {
                    votes <= cap
                } else {
                    true
                }
            }
        }
    }

    /// Computes the exact boundary proximity metric $R_{\text{boundary}, m}(s_n) \in [0.0, 1.0]$:
    ///
    /// $$R_{\text{boundary}, m}(s_n) = \exp\left( - \lambda_b \cdot \frac{\max(0, U_m(s_n) - L_m(s_n))}{|U_m(s_n)| + 1.0} \right), \quad \lambda_b = 5.0$$
    #[must_use]
    pub fn compute_boundary_proximity(&self, curr_slots: &[U256]) -> f32 {
        let (l_m, u_m) = match &self.template {
            InvariantTemplate::Solvency {
                share_slots,
                price_slot,
                asset_slots,
            } => {
                let mut sum_shares_f = 0.0f64;
                for &s in share_slots {
                    sum_shares_f += u256_to_f64(&slot_or_zero(curr_slots, s));
                }

                let price_f = match price_slot {
                    Some(p) => u256_to_f64(&slot_or_zero(curr_slots, *p)),
                    None => 1.0f64,
                };

                let mut sum_assets_f = 0.0f64;
                for &a in asset_slots {
                    sum_assets_f += u256_to_f64(&slot_or_zero(curr_slots, a));
                }

                let lhs = sum_shares_f * price_f;
                (lhs, sum_assets_f)
            }

            InvariantTemplate::Monotonicity { state_slot, .. } => {
                let curr = u256_to_f64(&slot_or_zero(curr_slots, *state_slot));
                (curr, curr + 1.0)
            }

            InvariantTemplate::AccessControl { critical_slot, .. } => {
                let curr = u256_to_f64(&slot_or_zero(curr_slots, *critical_slot));
                (curr, curr)
            }

            InvariantTemplate::PrecisionNeutrality {
                attacker_balance_slot,
                dust_tolerance,
            } => {
                let current = u256_to_f64(&slot_or_zero(curr_slots, *attacker_balance_slot));
                let limit = current + u256_to_f64(dust_tolerance);
                (current, limit)
            }

            InvariantTemplate::ArbitrageNeutrality {
                attacker_balance_slot,
                max_profit_tolerance,
                ..
            } => {
                let current = u256_to_f64(&slot_or_zero(curr_slots, *attacker_balance_slot));
                let limit = current + u256_to_f64(max_profit_tolerance);
                (current, limit)
            }

            InvariantTemplate::ConcentratedLiquidity {
                liquidity_slot,
                ..
            } => {
                let liq = u256_to_f64(&slot_or_zero(curr_slots, *liquidity_slot));
                (liq, liq + 1.0)
            }

            InvariantTemplate::BridgeSolvency {
                locked_asset_slots,
                minted_wrapped_slots,
                fee_reserve_slot,
            } => {
                let mut sum_locked_f = 0.0f64;
                for &l in locked_asset_slots {
                    sum_locked_f += u256_to_f64(&slot_or_zero(curr_slots, l));
                }
                if let Some(f) = fee_reserve_slot {
                    sum_locked_f += u256_to_f64(&slot_or_zero(curr_slots, *f));
                }

                let mut sum_minted_f = 0.0f64;
                for &m in minted_wrapped_slots {
                    sum_minted_f += u256_to_f64(&slot_or_zero(curr_slots, m));
                }

                (sum_minted_f, sum_locked_f)
            }

            InvariantTemplate::GovernanceFlashVote {
                proposal_votes_slot,
                total_voting_power_slot,
                max_vote_share_bps,
            } => {
                let votes_f = u256_to_f64(&slot_or_zero(curr_slots, *proposal_votes_slot));
                let total_power_f =
                    u256_to_f64(&slot_or_zero(curr_slots, *total_voting_power_slot));
                let cap_f = (total_power_f * (*max_vote_share_bps as f64)) / 10_000.0;
                (votes_f, cap_f)
            }
        };

        let diff = (u_m - l_m).max(0.0);
        let denom = u_m.abs() + 1.0;
        let ratio = diff / denom;
        let lambda_b = 5.0f64;
        let score = (-lambda_b * ratio).exp().clamp(0.0, 1.0);
        score as f32
    }

    /// Emits valid, self-contained SMT-LIBv2 256-bit BitVector (`(_ BitVec 256)`) assertions
    /// representing the negated invariant $\neg \mathcal{I}_m(s_n)$ to be solved in `gfn-symbolic`.
    #[must_use]
    pub fn to_negated_smt2(&self) -> String {
        let mut smt = String::with_capacity(512);
        smt.push_str("; --- GFlowNet-Prover Negated Safety Invariant ---\n");
        smt.push_str(&format!("; ID: {}\n", self.metadata.invariant_id));
        smt.push_str(&format!(
            "; Expression: {}\n",
            self.metadata.formal_expression
        ));

        match &self.template {
            InvariantTemplate::Solvency {
                share_slots,
                price_slot,
                asset_slots,
            } => {
                for &s in share_slots {
                    smt.push_str(&format!("(declare-const slot_{s} (_ BitVec 256))\n"));
                }
                if let Some(p) = price_slot {
                    smt.push_str(&format!("(declare-const slot_{p} (_ BitVec 256))\n"));
                }
                for &a in asset_slots {
                    smt.push_str(&format!("(declare-const slot_{a} (_ BitVec 256))\n"));
                }

                let shares_expr = build_bvadd_tree("slot", share_slots);
                let lhs_expr = match price_slot {
                    Some(p) => format!("(bvmul {shares_expr} slot_{p})"),
                    None => shares_expr,
                };
                let rhs_expr = build_bvadd_tree("slot", asset_slots);

                smt.push_str(&format!("(assert (bvugt {lhs_expr} {rhs_expr}))\n"));
            }

            InvariantTemplate::Monotonicity { state_slot, .. } => {
                smt.push_str(&format!(
                    "(declare-const slot_{state_slot}_prev (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{state_slot}_curr (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(assert (bvult slot_{state_slot}_curr slot_{state_slot}_prev))\n"
                ));
            }

            InvariantTemplate::AccessControl { critical_slot, .. } => {
                smt.push_str(&format!(
                    "(declare-const slot_{critical_slot}_prev (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{critical_slot}_curr (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(assert (distinct slot_{critical_slot}_curr slot_{critical_slot}_prev))\n"
                ));
            }

            InvariantTemplate::PrecisionNeutrality {
                attacker_balance_slot,
                dust_tolerance,
            } => {
                smt.push_str(&format!(
                    "(declare-const slot_{attacker_balance_slot}_0 (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{attacker_balance_slot}_n (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(assert (bvugt slot_{attacker_balance_slot}_n (bvadd slot_{attacker_balance_slot}_0 (_ bv{} 256))))\n",
                    dust_tolerance
                ));
            }

            InvariantTemplate::ArbitrageNeutrality {
                attacker_balance_slot,
                max_profit_tolerance,
                ..
            } => {
                smt.push_str(&format!(
                    "(declare-const slot_{attacker_balance_slot}_0 (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{attacker_balance_slot}_n (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(assert (bvugt slot_{attacker_balance_slot}_n (bvadd slot_{attacker_balance_slot}_0 (_ bv{} 256))))\n",
                    max_profit_tolerance
                ));
            }

            InvariantTemplate::ConcentratedLiquidity {
                tick_current_slot,
                tick_lower_slot,
                tick_upper_slot,
                liquidity_slot,
            } => {
                smt.push_str(&format!(
                    "(declare-const slot_{tick_current_slot} (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{tick_lower_slot} (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{tick_upper_slot} (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{liquidity_slot} (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(assert (and (bvuge slot_{tick_current_slot} slot_{tick_lower_slot}) (bvult slot_{tick_current_slot} slot_{tick_upper_slot}) (= slot_{liquidity_slot} #x0000000000000000000000000000000000000000000000000000000000000000)))\n"
                ));
            }

            InvariantTemplate::BridgeSolvency {
                locked_asset_slots,
                minted_wrapped_slots,
                fee_reserve_slot,
            } => {
                for &l in locked_asset_slots {
                    smt.push_str(&format!("(declare-const slot_{l} (_ BitVec 256))\n"));
                }
                if let Some(f) = fee_reserve_slot {
                    smt.push_str(&format!("(declare-const slot_{f} (_ BitVec 256))\n"));
                }
                for &m in minted_wrapped_slots {
                    smt.push_str(&format!("(declare-const slot_{m} (_ BitVec 256))\n"));
                }

                let minted_expr = build_bvadd_tree("slot", minted_wrapped_slots);
                let locked_expr = match fee_reserve_slot {
                    Some(f) => {
                        let mut slots = locked_asset_slots.clone();
                        slots.push(*f);
                        build_bvadd_tree("slot", &slots)
                    }
                    None => build_bvadd_tree("slot", locked_asset_slots),
                };

                smt.push_str(&format!("(assert (bvugt {minted_expr} {locked_expr}))\n"));
            }

            InvariantTemplate::GovernanceFlashVote {
                proposal_votes_slot,
                total_voting_power_slot,
                max_vote_share_bps,
            } => {
                smt.push_str(&format!(
                    "(declare-const slot_{proposal_votes_slot} (_ BitVec 256))\n"
                ));
                smt.push_str(&format!(
                    "(declare-const slot_{total_voting_power_slot} (_ BitVec 256))\n"
                ));
                let cap_expr = format!(
                    "(bvudiv (bvmul slot_{total_voting_power_slot} (_ bv{max_vote_share_bps} 256)) (_ bv10000 256))"
                );
                smt.push_str(&format!(
                    "(assert (bvugt slot_{proposal_votes_slot} {cap_expr}))\n"
                ));
            }
        }

        smt
    }
}

/// Automated Invariant Compiler converting parameterized `InvariantTemplate`s into `CompiledInvariant`s.
pub struct InvariantCompiler;

impl InvariantCompiler {
    /// Compiles an `InvariantTemplate` into a `CompiledInvariant` with assigned confidence and unique identifier.
    pub fn compile(template: InvariantTemplate, confidence: f64) -> GfnResult<CompiledInvariant> {
        let category = template.category();
        let formal_expression = template.formal_expression();
        let involved_slots = template.involved_slots();

        let prefix = match category {
            InvariantCategory::Solvency => "solv",
            InvariantCategory::Monotonicity => "mono",
            InvariantCategory::AccessControl => "ac",
            InvariantCategory::PrecisionNeutrality => "prec",
            InvariantCategory::ArbitrageNeutrality => "arb",
            InvariantCategory::ConcentratedLiquidity => "liq",
            InvariantCategory::BridgeSolvency => "bridge",
            InvariantCategory::GovernanceFlashVote => "gov_vote",
        };
        let invariant_id = format!("inv_{}_{}", prefix, uuid::Uuid::new_v4().simple());

        let metadata = SynthesizedInvariant {
            invariant_id,
            category,
            formal_expression,
            confidence: confidence.clamp(0.0, 1.0),
            involved_slots,
        };

        Ok(CompiledInvariant { metadata, template })
    }
}

#[inline]
fn slot_or_zero(slots: &[U256], idx: u64) -> U256 {
    match slots.get(idx as usize) {
        Some(&v) => v,
        None => U256::ZERO,
    }
}

/// Converts a `U256` value to a finite `f64` using direct limb arithmetic.
#[inline]
pub fn u256_to_f64(val: &U256) -> f64 {
    let limbs = val.as_limbs();
    const TWO_POW_64: f64 = 18446744073709551616.0;
    const TWO_POW_128: f64 = 340282366920938463463374607431768211456.0;
    const TWO_POW_192: f64 = 6277101735386680763835789423207666416102355444464034512896.0;

    (limbs[0] as f64)
        + (limbs[1] as f64) * TWO_POW_64
        + (limbs[2] as f64) * TWO_POW_128
        + (limbs[3] as f64) * TWO_POW_192
}

fn build_bvadd_tree(prefix: &str, slots: &[u64]) -> String {
    if slots.is_empty() {
        return "(_ bv0 256)".to_string();
    }
    if slots.len() == 1 {
        return match slots.first() {
            Some(s) => format!("{prefix}_{s}"),
            None => "(_ bv0 256)".to_string(),
        };
    }

    let mut expr = match slots.first() {
        Some(s) => format!("{prefix}_{s}"),
        None => "(_ bv0 256)".to_string(),
    };
    for &s in &slots[1..] {
        expr = format!("(bvadd {expr} {prefix}_{s})");
    }
    expr
}

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solvency_evaluation_and_boundary() -> GfnResult<()> {
        let template = InvariantTemplate::Solvency {
            share_slots: vec![0],
            price_slot: None,
            asset_slots: vec![1],
        };

        let compiled = InvariantCompiler::compile(template, 0.95)?;
        assert_eq!(compiled.metadata.category, InvariantCategory::Solvency);

        // Case 1: shares = 100, assets = 150 (holds)
        let curr = vec![U256::from(100), U256::from(150)];
        assert!(compiled.evaluate_satisfied(&[], &curr, 0));

        // Proximity should be strictly in (0.0, 1.0)
        let prox = compiled.compute_boundary_proximity(&curr);
        assert!((0.0..=1.0).contains(&prox));

        // Case 2: shares = 200, assets = 150 (violated)
        let curr_violated = vec![U256::from(200), U256::from(150)];
        assert!(!compiled.evaluate_satisfied(&[], &curr_violated, 0));

        // When violated, boundary proximity should be 1.0
        let prox_violated = compiled.compute_boundary_proximity(&curr_violated);
        assert!((prox_violated - 1.0).abs() < 1e-4);

        // Check SMT output
        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(declare-const slot_0 (_ BitVec 256))"));
        assert!(smt.contains("(declare-const slot_1 (_ BitVec 256))"));
        assert!(smt.contains("(assert (bvugt slot_0 slot_1))"));

        Ok(())
    }

    #[test]
    fn test_access_control_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::AccessControl {
            critical_slot: 2,
            admin_personas: vec![1],
        };

        let compiled = InvariantCompiler::compile(template, 0.99)?;

        let prev = vec![U256::ZERO, U256::ZERO, U256::from(0xbeef)];
        let curr_mutated = vec![U256::ZERO, U256::ZERO, U256::from(0xdead)];

        // Unprivileged attacker (0) cannot mutate critical slot
        assert!(!compiled.evaluate_satisfied(&prev, &curr_mutated, 0));

        // Admin (1) is allowed to mutate critical slot
        assert!(compiled.evaluate_satisfied(&prev, &curr_mutated, 1));

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(assert (distinct slot_2_curr slot_2_prev))"));

        Ok(())
    }

    #[test]
    fn test_monotonicity_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::Monotonicity {
            state_slot: 0,
            allowed_transitions: vec![(U256::from(10), U256::from(5))], // permitted reset
        };

        let compiled = InvariantCompiler::compile(template, 0.90)?;

        let prev = vec![U256::from(1)];
        let curr = vec![U256::from(2)];
        assert!(compiled.evaluate_satisfied(&prev, &curr, 0)); // 2 >= 1

        let prev_reset = vec![U256::from(10)];
        let curr_reset = vec![U256::from(5)];
        assert!(compiled.evaluate_satisfied(&prev_reset, &curr_reset, 0)); // permitted explicit transition

        let prev_bad = vec![U256::from(9)];
        let curr_bad = vec![U256::from(5)];
        assert!(!compiled.evaluate_satisfied(&prev_bad, &curr_bad, 0)); // illegal backward transition

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(assert (bvult slot_0_curr slot_0_prev))"));

        Ok(())
    }

    #[test]
    fn test_precision_neutrality_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::PrecisionNeutrality {
            attacker_balance_slot: 3,
            dust_tolerance: U256::ZERO,
        };

        let compiled = InvariantCompiler::compile(template, 0.95)?;

        let s0 = vec![U256::ZERO, U256::ZERO, U256::ZERO, U256::from(1000)];
        let sn_equal = vec![U256::ZERO, U256::ZERO, U256::ZERO, U256::from(1000)];
        let sn_profit = vec![U256::ZERO, U256::ZERO, U256::ZERO, U256::from(1001)];

        assert!(compiled.evaluate_satisfied(&s0, &sn_equal, 0));
        assert!(!compiled.evaluate_satisfied(&s0, &sn_profit, 0)); // profit detected

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(assert (bvugt slot_3_n (bvadd slot_3_0 (_ bv0 256))))"));

        Ok(())
    }

    #[test]
    fn test_arbitrage_neutrality_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::ArbitrageNeutrality {
            reserve0_slot: 0,
            reserve1_slot: 1,
            attacker_balance_slot: 2,
            max_profit_tolerance: U256::from(10),
        };

        let compiled = InvariantCompiler::compile(template, 0.96)?;
        assert_eq!(compiled.metadata.category, InvariantCategory::ArbitrageNeutrality);

        let s0 = vec![U256::from(100), U256::from(100), U256::from(50)];
        let sn_fair = vec![U256::from(105), U256::from(96), U256::from(55)]; // k_curr >= k_prev (10080 >= 10000), profit 5 <= 10
        let sn_depleted = vec![U256::from(80), U256::from(80), U256::from(55)]; // k depleted (6400 < 10000)
        let sn_excess_profit = vec![U256::from(110), U256::from(100), U256::from(70)]; // profit 20 > 10

        assert!(compiled.evaluate_satisfied(&s0, &sn_fair, 0));
        assert!(!compiled.evaluate_satisfied(&s0, &sn_depleted, 0));
        assert!(!compiled.evaluate_satisfied(&s0, &sn_excess_profit, 0));

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("slot_2_n"));

        Ok(())
    }

    #[test]
    fn test_concentrated_liquidity_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::ConcentratedLiquidity {
            tick_current_slot: 0,
            tick_lower_slot: 1,
            tick_upper_slot: 2,
            liquidity_slot: 3,
        };

        let compiled = InvariantCompiler::compile(template, 0.97)?;
        assert_eq!(compiled.metadata.category, InvariantCategory::ConcentratedLiquidity);

        // Tick = 5, Range = [0, 10), Liquidity = 1000 -> Active & Non-zero -> OK
        let s_active = vec![U256::from(5), U256::from(0), U256::from(10), U256::from(1000)];
        // Tick = 5, Range = [0, 10), Liquidity = 0 -> In range but zero liquidity -> Violation!
        let s_depleted = vec![U256::from(5), U256::from(0), U256::from(10), U256::ZERO];
        // Tick = 15, Range = [0, 10), Liquidity = 0 -> Out of range -> OK
        let s_out_of_range = vec![U256::from(15), U256::from(0), U256::from(10), U256::ZERO];

        assert!(compiled.evaluate_satisfied(&[], &s_active, 0));
        assert!(!compiled.evaluate_satisfied(&[], &s_depleted, 0));
        assert!(compiled.evaluate_satisfied(&[], &s_out_of_range, 0));

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("slot_3"));

        Ok(())
    }

    #[test]
    fn test_bridge_solvency_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::BridgeSolvency {
            locked_asset_slots: vec![0],
            minted_wrapped_slots: vec![1],
            fee_reserve_slot: Some(2),
        };

        let compiled = InvariantCompiler::compile(template, 0.98)?;
        assert_eq!(compiled.metadata.category, InvariantCategory::BridgeSolvency);

        // Case 1: locked = 100, minted = 105, fee_reserve = 10. Total backing = 110 >= 105 (holds)
        let s_solvent = vec![U256::from(100), U256::from(105), U256::from(10)];
        assert!(compiled.evaluate_satisfied(&[], &s_solvent, 0));

        let prox = compiled.compute_boundary_proximity(&s_solvent);
        assert!((0.0..=1.0).contains(&prox));

        // Case 2: locked = 100, minted = 120, fee_reserve = 10. Total backing = 110 < 120 (insolvency violation!)
        let s_insolvent = vec![U256::from(100), U256::from(120), U256::from(10)];
        assert!(!compiled.evaluate_satisfied(&[], &s_insolvent, 0));

        let prox_violated = compiled.compute_boundary_proximity(&s_insolvent);
        assert!((prox_violated - 1.0).abs() < 1e-4);

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(declare-const slot_0 (_ BitVec 256))"));
        assert!(smt.contains("(declare-const slot_1 (_ BitVec 256))"));
        assert!(smt.contains("(declare-const slot_2 (_ BitVec 256))"));
        assert!(smt.contains("(assert (bvugt slot_1 (bvadd slot_0 slot_2)))"));

        Ok(())
    }

    #[test]
    fn test_governance_flash_vote_evaluation() -> GfnResult<()> {
        let template = InvariantTemplate::GovernanceFlashVote {
            proposal_votes_slot: 0,
            total_voting_power_slot: 1,
            max_vote_share_bps: 2000, // 20%
        };

        let compiled = InvariantCompiler::compile(template, 0.99)?;
        assert_eq!(
            compiled.metadata.category,
            InvariantCategory::GovernanceFlashVote
        );

        // Total power = 10,000. Cap = 2,000 votes.
        // Case 1: votes = 1,500 <= 2,000 (holds)
        let s_normal = vec![U256::from(1500), U256::from(10000)];
        assert!(compiled.evaluate_satisfied(&[], &s_normal, 0));

        let prox = compiled.compute_boundary_proximity(&s_normal);
        assert!((0.0..=1.0).contains(&prox));

        // Case 2: votes = 2,500 > 2,000 (flash vote hijack violation!)
        let s_hijack = vec![U256::from(2500), U256::from(10000)];
        assert!(!compiled.evaluate_satisfied(&[], &s_hijack, 0));

        let smt = compiled.to_negated_smt2();
        assert!(smt.contains("(declare-const slot_0 (_ BitVec 256))"));
        assert!(smt.contains("(declare-const slot_1 (_ BitVec 256))"));
        assert!(smt.contains("(assert (bvugt slot_0 (bvudiv (bvmul slot_1 (_ bv2000 256)) (_ bv10000 256))))"));

        Ok(())
    }
}
