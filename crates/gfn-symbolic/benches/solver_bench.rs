//! Criterion micro-benchmarks for SMT S-expression parsing and two-tier cache lookups.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gfn_symbolic::solver::parse_smt_ast;
use gfn_symbolic::unsat_cache::{SmtResolution, SmtStatus, TwoTierUnsatCache};

fn bench_symbolic_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfn_symbolic_solver");

    let smt_query = r#"
        (set-logic QF_BV)
        (declare-const x (_ BitVec 256))
        (declare-const y (_ BitVec 256))
        (assert (bvugt x #x0000000000000000000000000000000000000000000000000000000000000005))
        (assert (bvult y #x0000000000000000000000000000000000000000000000000000000000000064))
        (assert (= (bvadd x y) #x0000000000000000000000000000000000000000000000000000000000000042))
        (check-sat)
    "#;

    group.bench_function("parse_smt_ast_medium", |b| {
        b.iter(|| parse_smt_ast(black_box(smt_query)));
    });

    let temp_cache_dir = std::env::temp_dir().join(format!("bench_cache_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_cache_dir);
    let cache = match TwoTierUnsatCache::new(&temp_cache_dir) {
        Ok(c) => c,
        Err(e) => panic!("Failed to init TwoTierUnsatCache for bench: {e:?}"),
    };

    let sample_key = [0xabu8; 16];
    let resolution = SmtResolution {
        status: SmtStatus::Unsat,
        concrete_witness_calldata: None,
        unsat_core_hash: None,
        m_unsat_multiplier: 0.001,
        solve_time_ms: 1.25,
    };
    if let Err(e) = cache.insert(sample_key, resolution) {
        panic!("Failed to insert sample into cache: {e:?}");
    }

    group.bench_function("unsat_cache_l1_lookup", |b| {
        b.iter(|| cache.get(black_box(&sample_key)));
    });

    group.finish();
}

criterion_group!(benches, bench_symbolic_components);
criterion_main!(benches);
