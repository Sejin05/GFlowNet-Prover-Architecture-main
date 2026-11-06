# Autonomous Agent Execution Roadmap: GFlowNet-Prover (Pure Rust / Foundry Edition)

> **Canonical Document Reference:** [agent_prompt_execution_plan.md](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md)

This document provides the complete, context-window-optimized **21-Step Prompt Execution Plan** for an AI coding agent to implement the entire GFlowNet-Prover architecture from scratch.

---

## Quick Navigation

1. [Executive Summary & Context-Window Feasibility Analysis](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md#1-executive-summary--context-window-feasibility-analysis)
2. [Compilation Dependency DAG](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md#2-compilation-dependency-dag--system-architecture)
3. [The 21-Step Prompt Roadmap](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md#3-the-21-step-prompt-roadmap)
   - **Phase 0:** Workspace Foundation (`Steps 1–2`)
   - **Phase 1:** High-Speed In-Memory EVM Engine (`Steps 3–4`)
   - **Phase 2:** Static Analysis & HSDG Builder (`Steps 5–6`)
   - **Phase 3:** Automated Invariant Synthesis (`Steps 7–8`)
   - **Phase 4:** Native Concolic EVM & SMT Solver Bridge (`Steps 9–10`)
   - **Phase 5:** Trajectory GFlowNet Engine (`Steps 11–14`)
   - **Phase 6:** Concrete Verification & Exploit Synthesis (`Steps 15–16`)
   - **Phase 7:** Operational Monitoring & Drift Detection (`Step 17`)
   - **Phase 8:** Production Serving & CLI Binary (`Steps 18–19`)
   - **Phase 9:** Multi-Bug DeFi Benchmark & Full Integration Tests (`Steps 20–21`)
4. [Execution Summary Table & Token Budgets](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md#4-execution-summary-table)
5. [Failure Remediation & Error Handling Protocol](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md#5-failure-remediation--error-handling-protocol)

---

## 21-Step Execution Summary Table

| Step | Phase | Target Crate / Scope | Key Deliverables | Est. Output Tokens | Verification Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | Phase 0 | Root Workspace | Cargo.toml, Dockerfile, Makefile, configs | ~3,200 | `cargo check --workspace` |
| **02** | Phase 0 | `gfn-common` | Error hierarchy, schemas, math utilities | ~3,800 | `cargo test -p gfn-common` |
| **03** | Phase 1 | `gfn-evm` | Inspector (JUMPI distance), cheatcodes | ~4,200 | `cargo check -p gfn-evm` |
| **04** | Phase 1 | `gfn-evm` | RevmArena CoW batching, 65k TPS bench | ~4,600 | `cargo bench -p gfn-evm` |
| **05** | Phase 2 | `gfn-extractor` | Foundry compiler, Solar SSA CFG lifter | ~4,800 | `cargo check -p gfn-extractor` |
| **06** | Phase 2 | `gfn-extractor` | HsdgBuilder & HeteroGraphTensors | ~4,500 | `cargo test -p gfn-extractor` |
| **07** | Phase 3 | `gfn-invariants` | 4 Invariant templates, SMT compiler | ~4,200 | `cargo check -p gfn-invariants` |
| **08** | Phase 3 | `gfn-invariants` | 4-layer HGT module & slot matcher | ~4,400 | `cargo test -p gfn-invariants` |
| **09** | Phase 4 | `gfn-symbolic` | L1/L2 Unsat cache (DashMap+heed), concretizer | ~4,100 | `cargo check -p gfn-symbolic` |
| **10** | Phase 4 | `gfn-symbolic` | Concolic EVM & Bitwuzla/Z3 solver pool | ~5,000 | `cargo test -p gfn-symbolic` |
| **11** | Phase 5 | `gfn-engine` | 6-layer StateGraphRgatEncoder in BFloat16 | ~4,600 | `cargo check -p gfn-engine` |
| **12** | Phase 5 | `gfn-engine` | Autoregressive forward/backward policy heads | ~4,800 | `cargo check -p gfn-engine` |
| **13** | Phase 5 | `gfn-engine` | RND curiosity reward, Huber TB loss in f32 | ~4,400 | `cargo check -p gfn-engine` |
| **14** | Phase 5 | `gfn-engine` | Fenwick replay buffer & rollout worker | ~5,100 | `cargo test -p gfn-engine` |
| **15** | Phase 6 | `gfn-verifier` | Turnkey ExploitPoC.t.sol synthesizer | ~3,900 | `cargo check -p gfn-verifier` |
| **16** | Phase 6 | `gfn-verifier` | Concrete forge test gate (0% FPR) & patcher | ~4,700 | `cargo test -p gfn-verifier` |
| **17** | Phase 7 | `gfn-monitor` | MMD, PSI, KS drift & Prometheus telemetry | ~4,200 | `cargo test -p gfn-monitor` |
| **18** | Phase 8 | `gfn-cli` | Axum HTTP/SSE server, micro-batching | ~4,800 | `cargo check -p gfn-cli` |
| **19** | Phase 8 | `gfn-cli` | Clap CLI (`gfn-forge`) binary entrypoint | ~3,800 | `cargo build -p gfn-cli --release` |
| **20** | Phase 9 | `tests/fixtures` | 4-contract Multi-Bug DeFi protocol suite | ~4,200 | `forge test` on fixture suite |
| **21** | Phase 9 | `tests/` | End-to-end audit & metamorphic tests | ~4,500 | `cargo test --workspace --release` |

For the verbatim, production-ready prompts, parameter contracts, and deterministic verification gates, consult the full document: [agent_prompt_execution_plan.md](file:///c:/Users/sowae/Downloads/GFlowNet-Prover%20Architecture/docs/08_agent_execution_roadmap/agent_prompt_execution_plan.md).
