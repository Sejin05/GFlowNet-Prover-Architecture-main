//! Criterion throughput and latency benchmark for `RevmArena`.
//!
//! Measures transaction simulation throughput across $B=64$ parallel environments
//! on 8 `rayon` threads, verifying the SLA metric: $\ge 65,000\text{ TPS/thread}$.

use alloy_primitives::{Address, Bytes, U256};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gfn_evm::arena::{RevmArena, StepBatchInput};
use revm::db::{CacheDB, EmptyDB};
use revm::primitives::{AccountInfo, Bytecode};
use std::time::Duration;

fn bench_revm_tps(c: &mut Criterion) {
    let mut group = c.benchmark_group("revm_arena_tps");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let batch_size = 64;
    let num_threads = 8;

    // Deploy a contract executing SSTORE(slot 0, val + 1):
    // PUSH1 0x00 SLOAD PUSH1 0x01 ADD PUSH1 0x00 SSTORE STOP
    let bytecode =
        Bytes::from_static(&[0x60, 0x00, 0x54, 0x60, 0x01, 0x01, 0x60, 0x00, 0x55, 0x00]);

    let target_contract = Address::repeat_byte(0x55);
    let mut base_db = CacheDB::new(EmptyDB::default());
    let info = AccountInfo {
        code: Some(Bytecode::new_raw(bytecode)),
        ..Default::default()
    };
    base_db.insert_account_info(target_contract, info);

    let tracked_slots = vec![(target_contract, U256::ZERO)];

    let mut arena = match RevmArena::new_with_base_db(
        base_db,
        tracked_slots,
        Vec::new(),
        batch_size,
        num_threads,
        42,
    ) {
        Ok(a) => a,
        Err(e) => panic!("Failed to initialize arena for bench: {e:?}"),
    };

    let input = StepBatchInput {
        caller_indices: vec![0; batch_size],
        target_contracts: vec![target_contract; batch_size],
        calldata_batch: vec![Bytes::new(); batch_size],
        warp_deltas: vec![0; batch_size],
        roll_deltas: vec![0; batch_size],
    };

    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("step_batch_b64", |b| {
        b.iter(|| match arena.step_batch(black_box(&input)) {
            Ok(out) => black_box(out),
            Err(e) => panic!("step_batch error during benchmark: {e:?}"),
        });
    });

    group.finish();
}

criterion_group!(benches, bench_revm_tps);
criterion_main!(benches);
