#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Property-Based Fuzzing & Numerical Boundary Verification Suite (`tests/fuzz_robustness_test.rs`).
//!
//! Validates formal invariants and mathematical guarantees across arbitrary inputs using `proptest`:
//! 1. 256-bit log2 normalization bounds $\in [0.0, 1.0]$ and monotonicity.
//! 2. 256-bit signed log2 delta antisymmetry $f(a, b) = -f(b, a)$ and bounds $\in [-1.0, 1.0]$.
//! 3. 16-channel sinusoidal chunk encoding bounds $\in [-1.0, 1.0]$ across arbitrary EVM words.
//! 4. Spearman rank correlation bounds $\in [-1.0, 1.0]$, identity preservation, and tie-breaking stability.
//! 5. SMT-LIBv2 S-expression AST parsing crash resistance on malformed/fuzzed inputs.
//! 6. Concrete invariant template evaluation stability across arbitrary slot tensors (Zero Panics).

use alloy_primitives::U256;
use gfn_common::math::{
    spearman_rank_correlation, u256_signed_log2_delta, u256_sinusoidal_chunk_encoding,
    u256_to_log2_norm,
};
use gfn_common::schemas::{InvariantCategory, SynthesizedInvariant};
use gfn_invariants::compiler::CompiledInvariant;
use gfn_invariants::templates::InvariantTemplate;
use gfn_symbolic::solver::parse_smt_ast;
use proptest::prelude::*;

fn u256_strategy() -> impl Strategy<Value = U256> {
    prop::array::uniform32(any::<u8>()).prop_map(U256::from_be_bytes)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // -------------------------------------------------------------------------
    // 1. U256 Log2 Normalization Invariant
    // -------------------------------------------------------------------------
    #[test]
    fn prop_u256_to_log2_norm_bounds_and_monotonicity(
        a in u256_strategy(),
        b in u256_strategy()
    ) {
        let norm_a = u256_to_log2_norm(a);
        let norm_b = u256_to_log2_norm(b);

        prop_assert!(norm_a.is_finite(), "norm_a not finite: {}", norm_a);
        prop_assert!(norm_b.is_finite(), "norm_b not finite: {}", norm_b);
        prop_assert!((0.0..=1.0).contains(&norm_a), "norm_a out of bounds: {}", norm_a);
        prop_assert!((0.0..=1.0).contains(&norm_b), "norm_b out of bounds: {}", norm_b);

        if a <= b {
            prop_assert!(norm_a <= norm_b + 1e-6, "Monotonicity violated: a <= b but norm_a > norm_b");
        } else {
            prop_assert!(norm_a + 1e-6 >= norm_b, "Monotonicity violated: a > b but norm_a < norm_b");
        }
    }

    // -------------------------------------------------------------------------
    // 2. U256 Signed Log2 Delta Invariant
    // -------------------------------------------------------------------------
    #[test]
    fn prop_u256_signed_log2_delta_antisymmetry(
        a in u256_strategy(),
        b in u256_strategy()
    ) {
        let delta_ab = u256_signed_log2_delta(a, b);
        let delta_ba = u256_signed_log2_delta(b, a);

        prop_assert!(delta_ab.is_finite());
        prop_assert!(delta_ba.is_finite());
        prop_assert!((-1.0..=1.0).contains(&delta_ab));
        prop_assert!((-1.0..=1.0).contains(&delta_ba));

        if a == b {
            prop_assert_eq!(delta_ab, 0.0);
            prop_assert_eq!(delta_ba, 0.0);
        } else {
            prop_assert!((delta_ab + delta_ba).abs() < 1e-6, "Antisymmetry violated: delta(a,b) != -delta(b,a)");
        }
    }

    // -------------------------------------------------------------------------
    // 3. 16-Channel Sinusoidal Chunk Encoding Bounds
    // -------------------------------------------------------------------------
    #[test]
    fn prop_u256_sinusoidal_encoding_bounds(
        val in u256_strategy()
    ) {
        let encoding = u256_sinusoidal_chunk_encoding(val);
        prop_assert_eq!(encoding.len(), 16);
        for &ch in &encoding {
            prop_assert!(ch.is_finite());
            prop_assert!((-1.0001..=1.0001).contains(&ch), "Sinusoidal channel out of bounds: {}", ch);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Spearman Rank Correlation Bounds & Invariants
    // -------------------------------------------------------------------------
    #[test]
    fn prop_spearman_rank_correlation_bounds(
        xs in prop::collection::vec(any::<f32>().prop_filter("finite", |v| v.is_finite()), 3..30),
        ys in prop::collection::vec(any::<f32>().prop_filter("finite", |v| v.is_finite()), 3..30),
    ) {
        let x64: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
        let min_len = x64.len().min(ys.len());
        let x_slice = &x64[..min_len];
        let y64: Vec<f64> = ys[..min_len].iter().map(|&v| v as f64).collect();

        if min_len >= 2 {
            if let Ok(rho) = spearman_rank_correlation(x_slice, &y64) {
                prop_assert!(rho.is_finite(), "rho must be finite: {}", rho);
                prop_assert!((-1.0001..=1.0001).contains(&rho), "rho out of [-1, 1]: {}", rho);
            }

            // Self-correlation with non-degenerate vector must be >= 0.999 or 0.0 if all elements tied
            if let Ok(self_rho) = spearman_rank_correlation(x_slice, x_slice) {
                prop_assert!(self_rho.is_finite());
                prop_assert!(self_rho >= 0.999 || self_rho == 0.0, "Self correlation failed: {}", self_rho);
            }
        }
    }


    // -------------------------------------------------------------------------
    // 5. SMT-LIBv2 S-Expression Parser Crash Resistance
    // -------------------------------------------------------------------------
    #[test]
    fn prop_smt_parser_crash_resistance(
        input in "(\\s*(\\([a-zA-Z0-9_# ]*\\))*)*"
    ) {
        // Must never panic on arbitrary parenthesized structures
        let _ast = parse_smt_ast(&input);
    }

    // -------------------------------------------------------------------------
    // 6. Vectorized Invariant Template Stability
    // -------------------------------------------------------------------------
    #[test]
    fn prop_invariant_evaluation_stability(
        s0 in u256_strategy(),
        s1 in u256_strategy(),
        s2 in u256_strategy(),
        s3 in u256_strategy(),
        caller_id in 0u8..=5u8,
    ) {
        let prev = vec![s0, s1, s2, s3];
        let curr = vec![s1, s0, s3, s2];

        let templates = vec![
            InvariantTemplate::Solvency {
                share_slots: vec![0],
                price_slot: Some(1),
                asset_slots: vec![2],
            },
            InvariantTemplate::Monotonicity {
                state_slot: 0,
                allowed_transitions: vec![(s0, s1)],
            },
            InvariantTemplate::AccessControl {
                critical_slot: 0,
                admin_personas: vec![1],
            },
            InvariantTemplate::PrecisionNeutrality {
                attacker_balance_slot: 2,
                dust_tolerance: U256::from(10u64),
            },
            InvariantTemplate::ArbitrageNeutrality {
                reserve0_slot: 0,
                reserve1_slot: 1,
                attacker_balance_slot: 2,
                max_profit_tolerance: U256::from(100u64),
            },
            InvariantTemplate::ConcentratedLiquidity {
                tick_current_slot: 0,
                tick_lower_slot: 1,
                tick_upper_slot: 2,
                liquidity_slot: 3,
            },
            InvariantTemplate::BridgeSolvency {
                locked_asset_slots: vec![0],
                minted_wrapped_slots: vec![1],
                fee_reserve_slot: Some(2),
            },
            InvariantTemplate::GovernanceFlashVote {
                proposal_votes_slot: 0,
                total_voting_power_slot: 1,
                max_vote_share_bps: 1000,
            },
        ];

        for tmpl in templates {
            let inv = CompiledInvariant {
                metadata: SynthesizedInvariant {
                    invariant_id: "FUZZ_INV".to_string(),
                    category: InvariantCategory::Solvency,
                    formal_expression: "fuzz".to_string(),
                    confidence: 0.99,
                    involved_slots: vec![0, 1, 2, 3],
                },
                template: tmpl,
            };

            // Evaluate satisfaction and boundary proximity: both must execute without panicking
            let _sat = inv.evaluate_satisfied(&prev, &curr, caller_id);
            let prox = inv.compute_boundary_proximity(&curr);
            prop_assert!(prox.is_finite());
            prop_assert!((0.0..=1.0001).contains(&prox));
        }
    }
}
