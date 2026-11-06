//! Criterion micro-benchmarks for vectorized invariant evaluation and boundary metrics.

use alloy_primitives::U256;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gfn_common::schemas::{InvariantCategory, SynthesizedInvariant};
use gfn_invariants::compiler::CompiledInvariant;
use gfn_invariants::templates::InvariantTemplate;

fn bench_invariant_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfn_invariants_eval");

    let prev_slots = vec![
        U256::from(1000u64),
        U256::from(2000u64),
        U256::from(3000u64),
        U256::from(4000u64),
    ];
    let curr_slots = vec![
        U256::from(1050u64),
        U256::from(2100u64),
        U256::from(3000u64),
        U256::from(4500u64),
    ];

    let solvency_inv = CompiledInvariant {
        metadata: SynthesizedInvariant {
            invariant_id: "BENCH_SOLVENCY".to_string(),
            category: InvariantCategory::Solvency,
            formal_expression: "shares * price <= assets".to_string(),
            confidence: 0.99,
            involved_slots: vec![0, 1],
        },
        template: InvariantTemplate::Solvency {
            share_slots: vec![0],
            price_slot: None,
            asset_slots: vec![1],
        },
    };

    let arb_inv = CompiledInvariant {
        metadata: SynthesizedInvariant {
            invariant_id: "BENCH_ARB".to_string(),
            category: InvariantCategory::ArbitrageNeutrality,
            formal_expression: "k_curr >= k_prev".to_string(),
            confidence: 0.98,
            involved_slots: vec![0, 1, 2],
        },
        template: InvariantTemplate::ArbitrageNeutrality {
            reserve0_slot: 0,
            reserve1_slot: 1,
            attacker_balance_slot: 2,
            max_profit_tolerance: U256::from(100u64),
        },
    };

    let conc_liq_inv = CompiledInvariant {
        metadata: SynthesizedInvariant {
            invariant_id: "BENCH_CONC_LIQ".to_string(),
            category: InvariantCategory::ConcentratedLiquidity,
            formal_expression: "in_range => liq > 0".to_string(),
            confidence: 0.97,
            involved_slots: vec![0, 1, 2, 3],
        },
        template: InvariantTemplate::ConcentratedLiquidity {
            tick_current_slot: 0,
            tick_lower_slot: 1,
            tick_upper_slot: 2,
            liquidity_slot: 3,
        },
    };

    group.bench_function("eval_solvency_satisfied", |b| {
        b.iter(|| solvency_inv.evaluate_satisfied(black_box(&prev_slots), black_box(&curr_slots), black_box(0)));
    });

    group.bench_function("eval_solvency_boundary_proximity", |b| {
        b.iter(|| solvency_inv.compute_boundary_proximity(black_box(&curr_slots)));
    });

    group.bench_function("eval_arbitrage_satisfied", |b| {
        b.iter(|| arb_inv.evaluate_satisfied(black_box(&prev_slots), black_box(&curr_slots), black_box(0)));
    });

    group.bench_function("eval_concentrated_liquidity_satisfied", |b| {
        b.iter(|| conc_liq_inv.evaluate_satisfied(black_box(&prev_slots), black_box(&curr_slots), black_box(0)));
    });

    group.finish();
}

criterion_group!(benches, bench_invariant_evaluation);
criterion_main!(benches);
