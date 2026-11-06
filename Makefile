.PHONY: all build check test test-fast test-release test-py test-all review clippy fmt fmt-check bench audit docker-build clean

all: check test test-py review

build:
	cargo build --workspace --release

check:
	cargo check --workspace

test:
	cargo test --workspace --lib --bins --tests

test-fast:
	python scripts/run_tests.py --fast

test-release:
	cargo test --workspace --release --lib --bins --tests

test-py:
	pytest -q

test-all:
	python scripts/run_tests.py --full

review:
	python scripts/code_review.py

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

bench:
	cargo bench -p gfn-evm

audit:
	cargo run -p gfn-cli --release -- audit --project ./tests/fixtures/multi_bug_defi_protocol

docker-build:
	docker build -t gflownet-prover:latest -f Dockerfile .

clean:
	cargo clean
