# Deployment & Serving Specification: GFlowNet-Prover (Pure Rust / Foundry Edition)

## 1. Inference Topology & Serving Modes

GFlowNet-Prover provides four cohesive operational serving and execution topologies:
1. **Mode A: Local/CI Autonomous Native CLI Auditor (`gfn-forge audit`)**:
   - Invoked directly inside a developer's repository or GitHub Actions CI runner to audit a Foundry project synchronously, streaming live discovered invariant violations and writing `ExploitPoC.t.sol` and remediation patches to disk.
2. **Mode B: Asynchronous HTTP/SSE & gRPC Microservice (`gfn-forge serve`)**:
   - Built natively on `axum = "0.7.9"` + `tokio = "1.41.1"` + `tonic = "0.12.3"` (HTTP/JSON, Server-Sent Events streaming, and gRPC). Manages stateful, multi-minute Phase 3 Online Adaptation audit jobs via an internal `tokio::sync::mpsc` bounded work queue backed by GPU worker actors, while exposing a real-time **Step-Level Action Scoring endpoint** ($p_{99} \le 9.5\text{ ms}$) for interactive IDE/fuzzer co-pilots.
3. **Mode C: Unified Headless Python CLI, Training & Reporting Suite (`python main.py` / `python app.py`)**:
   - Provides pure command-line execution for multi-phase gradient training (Phase 1 MSTM, Phase 2 Curriculum TB, Phase 3 Continual EWC), generating SafeTensors binary checkpoints, publication-grade 6-panel dashboards (`plots/training_progress.png`), in-terminal ASCII progress curves, semantic AI context extraction (`PROJECT_AI_CONTEXT.md`), and automated OASIS SARIF v2.1.0 / GitHub Security Advisory export.
4. **Mode D: High-Efficiency Automated Verification & Benchmark Runners (`scripts/`)**:
   - Orchestrates <150 ms deterministic static & semantic code-review gates (`scripts/code_review.py`), high-precision Criterion performance & memory RSS SLA scorecards (`scripts/run_benchmarks.py`), and unified multi-stage test execution (`scripts/run_tests.py`).

### 1.1 Real-Time Batching, Concurrency & Caching Thresholds
- **Dynamic Micro-Batching Window**: For the interactive `/v1/policy/sample-step` endpoint, incoming state-graph requests are accumulated over a `2.0 ms` `tokio::time::sleep` batching window up to `max_batch_size = 64` environments per GPU actor thread.
- **Worker Concurrency Limits**:
  - 1 Dedicated OS Thread per physical GPU owning the `tch::nn::VarStore` and CUDA stream (`cudarc` / `libtorch`).
  - 1 `RevmArena` instance per active audit job pinned to 8 CPU hardware threads via `rayon::ThreadPoolBuilder::new().num_threads(8)`.
  - Up to 6 concurrent `Bitwuzla`/`Z3` solver threads managed via a dedicated solver `rayon::ThreadPool` with hard Bitwuzla `Terminator` deadlines (`5,000 ms`).
- **Two-Tier Native Rust Caching Layer**:
  1. **L1 In-Memory Unsat Core Cache (`DashMap<[u8; 16], SmtCacheEntry>`, 65,536 entries, `< 0.01 ms` lock-free lookup)**: Maps canonical symbolic branch constraint hashes (`xxhash_rust::xxh3::xxh3_128(normalized_smt2_bytes)`) to `SmtStatus::Unsat` or `SmtStatus::Sat(Vec<U256>)` so identical arithmetic guards encountered across parallel trajectories never re-invoke Bitwuzla/Z3.
  2. **L2 Persistent LMDB Graph & Invariant Cache (`heed = "0.20.5"` at `/var/cache/gfn-prover/lmdb`, 32 GB max)**: Caches zero-copy `rkyv`-archived `HeteroGraphTensors` and verified HGT safety invariants keyed by `blake3(foundry_build_artifacts)`.

---

## 2. Input & Output Payload JSON Schemas

All REST API payloads are deserialized and validated with zero-copy where possible using `serde = "1.0.215"` and `serde_json = "1.0.132"`, and documented via `utoipa = "5.2.0"` (OpenAPI 3.1 / JSON Schema Draft 2020-12).

### 2.1 Audit Submission Request Schema (`POST /v1/audits`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AuditSubmissionRequest",
  "type": "object",
  "required": ["project_source", "foundry_profile", "search_budget"],
  "additionalProperties": false,
  "properties": {
    "project_source": {
      "type": "object",
      "required": ["workspace_path", "setup_contract", "setup_function"],
      "properties": {
        "workspace_path": { "type": "string", "minLength": 1, "description": "Absolute path or mounted volume to Foundry project root." },
        "setup_contract": { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$", "default": "ExploitHarnessTest" },
        "setup_function": { "type": "string", "default": "setUp" },
        "target_contracts": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1,
          "description": "Explicit list of in-scope contract names to audit."
        }
      }
    },
    "foundry_profile": {
      "type": "object",
      "required": ["solc_version", "evm_version"],
      "properties": {
        "solc_version": { "type": "string", "pattern": "^0\\.8\\.[0-9]{1,2}$", "default": "0.8.24" },
        "evm_version": { "type": "string", "enum": ["paris", "shanghai", "cancun"], "default": "cancun" },
        "fork_block_number": { "type": ["integer", "null"], "minimum": 1, "default": null }
      }
    },
    "search_budget": {
      "type": "object",
      "required": ["max_trajectories", "max_depth", "wall_clock_timeout_sec", "smt_timeout_ms"],
      "properties": {
        "max_trajectories": { "type": "integer", "minimum": 64, "maximum": 1000000, "default": 100000 },
        "max_depth": { "type": "integer", "minimum": 2, "maximum": 25, "default": 15 },
        "batch_size": { "type": "integer", "enum": [8, 16, 32, 64, 128], "default": 64 },
        "wall_clock_timeout_sec": { "type": "integer", "minimum": 30, "maximum": 7200, "default": 900 },
        "smt_timeout_ms": { "type": "integer", "minimum": 250, "maximum": 30000, "default": 5000 },
        "enable_online_adaptation": { "type": "boolean", "default": true }
      }
    }
  }
}
```

### 2.2 Audit Result & Verified Exploit Response Schema (`GET /v1/audits/{audit_id}`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AuditResultResponse",
  "type": "object",
  "required": ["audit_id", "status", "telemetry", "synthesized_invariants", "verified_findings"],
  "additionalProperties": false,
  "properties": {
    "audit_id": { "type": "string", "format": "uuid" },
    "status": { "type": "string", "enum": ["QUEUED", "EXTRACTING_GRAPH", "EXPLORING_GFLOWNET", "VERIFYING_POC", "COMPLETED", "FAILED"] },
    "telemetry": {
      "type": "object",
      "required": ["trajectories_evaluated", "revm_tps", "smt_queries_total", "smt_sat_hits", "smt_unsat_hits", "elapsed_sec", "log_z_estimate"],
      "properties": {
        "trajectories_evaluated": { "type": "integer" },
        "revm_tps": { "type": "number" },
        "smt_queries_total": { "type": "integer" },
        "smt_sat_hits": { "type": "integer" },
        "smt_unsat_hits": { "type": "integer" },
        "elapsed_sec": { "type": "number" },
        "log_z_estimate": { "type": "number" }
      }
    },
    "synthesized_invariants": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["invariant_id", "category", "formal_expression", "confidence"],
        "properties": {
          "invariant_id": { "type": "string" },
          "category": { "type": "string", "enum": ["SOLVENCY", "MONOTONICITY", "ACCESS_CONTROL", "PRECISION_NEUTRALITY"] },
          "formal_expression": { "type": "string" },
          "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
        }
      }
    },
    "verified_findings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["finding_id", "violated_invariant_id", "severity", "trajectory_steps", "foundry_poc_path", "foundry_poc_solidity_code", "forge_test_stdout", "remediation_patch_diff"],
        "properties": {
          "finding_id": { "type": "string" },
          "violated_invariant_id": { "type": "string" },
          "severity": { "type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM"] },
          "trajectory_steps": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["step_idx", "caller_role", "caller_address", "target_contract", "function_signature", "concrete_calldata_hex", "decoded_args", "vm_warp_sec", "vm_roll_blocks"],
              "properties": {
                "step_idx": { "type": "integer" },
                "caller_role": { "type": "string" },
                "caller_address": { "type": "string", "pattern": "^0x[a-fA-F0-9]{40}$" },
                "target_contract": { "type": "string" },
                "function_signature": { "type": "string" },
                "concrete_calldata_hex": { "type": "string", "pattern": "^0x[a-fA-F0-9]*$" },
                "decoded_args": { "type": "array", "items": { "type": "string" } },
                "vm_warp_sec": { "type": "integer" },
                "vm_roll_blocks": { "type": "integer" }
              }
            }
          },
          "foundry_poc_path": { "type": "string" },
          "foundry_poc_solidity_code": { "type": "string" },
          "forge_test_stdout": { "type": "string" },
          "minimal_negated_precondition": { "type": "string" },
          "remediation_patch_diff": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 3. Model Serialization, Compilation & Quantization Standards

1. **Weight Serialization Standard (`safetensors`)**:
   - All `tch::nn::VarStore` checkpoints (`HgtInvariantSynthesizer`, `StateGraphRgatEncoder`, `ForwardPolicyHead`, `BackwardPolicyHead`, `RndPredictor`) are saved and loaded via `safetensors = "0.4.5"` (`VarStore::load_safetensors` / `save_safetensors`) paired with a `manifest.json` recording `blake3` checksums and exact tensor dimensions.
2. **C++ LibTorch JIT & CUDA Graph Capture**:
   - Because Phase 3 Online Auditing requires backpropagating Trajectory Balance gradients $\nabla_\theta \mathcal{L}_{TB}$ in real time to adapt to new protocol topologies, static inference-only runtimes cannot be used for the mutable policy head.
   - Instead, `gfn-engine` uses `tch::Cuda::cudnn_set_benchmark(true)` and captures fixed-shape dense MLP forward/backward passes via `ATen` CUDA Graphs while executing dynamic-topology sparse scatter-attention directly in `Kind::BFloat16` cuBLAS kernels, achieving single-batch ($B=64$) forward+backward latency of `7.2 ms` on NVIDIA L40S/A100.
3. **Mixed-Precision & Quantization Policy**:
   - **RGAT Backbone & HGT Invariant Synthesizer**: Executed in native hardware `Kind::BFloat16` with `Kind::Float` (`f32`) master weights during online adaptation. For read-only zero-shot inference on GPUs with $\le 16\text{ GB}$ VRAM, linear layer weights are dynamically quantized to weight-only `Kind::QInt8` with `< 0.4%` degradation in Spearman flow correlation $\rho_{TB}$.
   - **Log-Probability & Partition Function Head ($\log P_F, \log P_B, \log Z_\theta$)**: Strictly pinned to `Kind::Float` (`f32`) to preserve exact log-space flow conservation across 15-step trajectories.

---

## 4. Container Runtime Configuration (`Dockerfile` & Security Hardening)

The production container is a pure Rust + LibTorch CUDA 12.4 + Foundry image with zero Python runtime dependencies:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    LIBTORCH=/opt/libtorch \
    LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/cargo/bin:/root/.foundry/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl unzip pkg-config \
    libgmp-dev libmpfr-dev liblmdb-dev libclang-dev clang z3 libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download official C++ LibTorch 2.5.1+cu124 for tch-rs
RUN curl -L https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -o /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# Install Rust 1.82.0 toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.82.0

# Install Foundry 1.0.0-stable
RUN curl -L https://foundry.paradigm.xyz | bash && foundryup --version nightly-e3f9a1c8a9fb3b2e7f480e8f4b4b9c2f1a8d2e1b

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Compile pure-Rust workspace with LTO and AVX2/SIMD
RUN RUSTFLAGS="-C target-cpu=haswell -C lto=thin" cargo build --release --workspace --bin gfn-forge

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgmp10 libmpfr6 liblmdb0 libz3-4 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 gfnuser && useradd -u 10001 -g gfnuser -m -s /bin/bash gfnuser

COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib
COPY --from=builder /root/.foundry/bin/forge /usr/local/bin/forge
COPY --from=builder /root/.foundry/bin/cast /usr/local/bin/cast
COPY --from=builder /root/.foundry/bin/anvil /usr/local/bin/anvil
COPY --from=builder /workspace/target/release/gfn-forge /usr/local/bin/gfn-forge
COPY configs/ /app/configs/

ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/bin:${PATH}" \
    RAYON_NUM_THREADS=8 \
    RUST_LOG="info,gfn_prover=debug"

USER gfnuser
WORKDIR /app
EXPOSE 8080 9090
ENTRYPOINT ["gfn-forge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
```

### 4.1 Kubernetes / Runtime Resource & Security Constraints
- `shm_size`: Minimum `16Gi` mounted at `/dev/shm` (required for tmpfs Foundry PoC compilation and `heed` LMDB memory-mapped caches).
- `ulimits`: `memlock: -1`, `stack: 67108864`.
- Network Isolation during Audit Execution: Target contract execution runs with `FOUNDRY_OFFLINE=true` unless an explicit `--fork-url` pinned block archive is configured.
