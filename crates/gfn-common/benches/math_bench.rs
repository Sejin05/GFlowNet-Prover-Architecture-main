//! Criterion micro-benchmarks for 256-bit EVM math and correlation utilities.

use alloy_primitives::U256;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gfn_common::math::{
    spearman_rank_correlation, u256_signed_log2_delta, u256_sinusoidal_chunk_encoding,
    u256_to_log2_norm,
};

fn bench_math_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfn_common_math");

    let val_zero = U256::ZERO;
    let val_mid = U256::from(1_000_000_000_000_000_000u64);
    let val_max = U256::MAX;

    group.bench_function("u256_to_log2_norm_zero", |b| {
        b.iter(|| u256_to_log2_norm(black_box(val_zero)));
    });

    group.bench_function("u256_to_log2_norm_mid", |b| {
        b.iter(|| u256_to_log2_norm(black_box(val_mid)));
    });

    group.bench_function("u256_to_log2_norm_max", |b| {
        b.iter(|| u256_to_log2_norm(black_box(val_max)));
    });

    group.bench_function("u256_signed_log2_delta", |b| {
        b.iter(|| u256_signed_log2_delta(black_box(val_mid), black_box(val_max)));
    });

    group.bench_function("u256_sinusoidal_chunk_encoding", |b| {
        b.iter(|| u256_sinusoidal_chunk_encoding(black_box(val_mid)));
    });

    let x_vals = vec![1.0, 2.5, 3.0, 4.2, 5.0, 6.1, 7.8, 9.0];
    let y_vals = vec![1.2, 2.3, 3.1, 4.0, 5.5, 6.0, 8.0, 8.9];

    group.bench_function("spearman_rank_correlation_n8", |b| {
        b.iter(|| spearman_rank_correlation(black_box(&x_vals), black_box(&y_vals)));
    });

    group.finish();
}

criterion_group!(benches, bench_math_primitives);
criterion_main!(benches);
