# Model Development Specification: GFlowNet-Prover (Pure Rust / Foundry Edition)

## 1. Baseline Benchmarks & Comparative Reference Architectures

All neural models and baselines are implemented in Rust (`crates/gfn-engine` and `crates/gfn-invariants`) using `tch-rs` (`tch = "0.18.0"`, backed by C++ `libtorch 2.5.1+cu124`). Before evaluating the full GFlowNet-Prover neuro-symbolic architecture, three baselines are benchmarked on the identical Foundry evaluation harness:
1. **Baseline 1: Standalone Foundry Stateless & Stateful Fuzzer (`forge test --fuzz-runs 50000`)**:
   - Standard random mutation and coverage-guided invariant fuzzing without neural action planning or SMT branch solving.
2. **Baseline 2: Standalone Symbolic Execution (`gfn-symbolic` bounded depth $n=3$, 5,000 ms timeout)**:
   - Pure bounded symbolic execution up to transaction depth $n=3$ without GFlowNet macro-step guidance.
3. **Baseline 3: Proximal Policy Optimization (`PpoRgatAgent` in `crates/gfn-engine`)**:
   - Standard Reinforcement Learning agent using the identical 6-layer RGAT encoder (`tch::nn`) and composite reward $R(s_n)$, trained with clipped PPO surrogate loss ($\epsilon = 0.2$, entropy bonus $c_2 = 0.01$). Used specifically to quantify **mode-collapse mitigation** (PPO converges to a single easiest exploit path, whereas GFlowNet discovers all distinct exploit modes proportional to $R(\tau)$).

---

## 2. Primary Neural Architectures in Rust (`tch::nn`)

### 2.1 Crate `gfn-invariants`: Heterogeneous Graph Transformer (`HgtInvariantSynthesizer`)
To automatically synthesize protocol safety invariants $\mathcal{I}(s)$ across the 4 domain categories without manual annotations, a 4-layer **Heterogeneous Graph Transformer** in `crates/gfn-invariants/src/hgt.rs` processes the static HSDG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- **Input Projections**: Node-type specific `tch::nn::Linear` (`in_dim -> 512`) followed by `tch::nn::LayerNorm` (`[512]`) and `.gelu("none")`.
- **Backbone**: 4 stacked `HgtConvLayer` modules (`in_dim = 512`, `out_dim = 512`, `num_heads = 8`, `num_node_types = 4`, `num_edge_types = 8`) implemented via segmented `Tensor::index_select` and `Tensor::scatter_add_` with residual connections and pre-norm.
- **Template Classification & Slot Binding Head**:
  - Evaluates candidate pairs/tuples of `StorageNode` embeddings $(h_{u}, h_{v})$ via a bilinear attention classifier (`BilinearSlotMatcher`) to emit parameterized formal invariants across the 4 canonical templates:
    1. *Conservation of Value & Solvency*: $\sum_{u \in \text{Shares}} \text{slot}(u) \times P_{\text{share}} \le \sum_{v \in \text{Assets}} \text{slot}(v)$
    2. *State Machine Monotonicity*: $\text{slot}_{\text{state}}(s_{t+1}) \ge \text{slot}_{\text{state}}(s_t)$ or valid finite-state transition mask $A_{\text{valid}}[\text{state}_t, \text{state}_{t+1}] = 1$
    3. *Privilege & Access Control Isolation*: $\text{caller} \notin \text{AdminSet} \implies \text{slot}_{\text{critical}}(s_{t+1}) == \text{slot}_{\text{critical}}(s_t)$
    4. *Rounding & Precision Neutrality*: $\forall \tau_{\text{roundtrip}}, \text{Balance}_{\text{attacker}}(s_n) \le \text{Balance}_{\text{attacker}}(s_0) + \epsilon_{\text{dust}}$ (where $\epsilon_{\text{dust}} = 0\text{ wei}$)
  - Only candidate invariants holding with $100\%$ satisfaction across 1,000 benign random walk transactions in `RevmArena` during `setUp()` warm-up are locked into the active invariant set `Vec<CompiledInvariant>`.

### 2.2 Crate `gfn-engine`: 6-Layer State-Graph Relational GAT (`StateGraphRgatEncoder`)
At every transaction step $t$, `RevmArena` updates the dynamic columns of `storage_x` (`[.., 32..64]`) and `condition_x` (`[.., 24..32]`). The updated `HeteroGraphBatch` is encoded by a 6-layer **Relational Graph Attention Network** (`crates/gfn-engine/src/rgat.rs`):

- **Hidden Dimension ($d$)**: `512`
- **Number of Layers ($L$)**: `6`
- **Attention Heads ($H$)**: `8` (head dimension $d_k = 64$)
- **Edge Attribute Dimension ($d_e$)**: `16` projected to `64` per relation type $r \in \{0..7\}$.
- **Sparse Scatter-Attention Layer Update Equation** (for node $i$ at layer $\ell \in \{1..6\}$):
  $$h_i^{(\ell)} = \text{LayerNorm}\left( h_i^{(\ell-1)} + \text{Dropout}_{0.1}\left( \text{GELU}\left( \sum_{r=0}^{7} \sum_{j \in \mathcal{N}_r(i)} \alpha_{i,j}^{(r, \ell)} W_V^{(r, \ell)} h_j^{(\ell-1)} \right) \right) \right)$$
  where segment-wise sparse softmax $\alpha_{i,j}^{(r, \ell)}$ is computed in `tch-rs` using `scatter_max` (for numerical stability) followed by `exp()` and `scatter_add_` normalization over target node indices `edge_index[r][1]`:
  $$\alpha_{i,j}^{(r, \ell)} = \text{SparseSoftmax}_{j \in \mathcal{N}_r(i)} \left( \frac{\left(W_Q^{(r, \ell)} h_i^{(\ell-1)}\right)^\top \left(W_K^{(r, \ell)} h_j^{(\ell-1)} + W_E^{(r, \ell)} e_{i,j}\right)}{\sqrt{64.0}} \right)$$
- **Global Context Pooling ($g_t$)**:
  Aggregates node representations across each graph in the batch (`batch_idx: Tensor`) alongside the global environment context $c_t \in \mathbb{R}^{B \times 32}$ (actor token balances, `block.timestamp`, `block.number`, step index $t / T_{\max}$):
  $$g_t = \text{MLP}_{\text{ctx}}\left( \left[ \text{SegmentMean}(H_{\text{storage}}^{(6)}) \,\|\, \text{SegmentMax}(H_{\text{cond}}^{(6)}) \,\|\, \text{SegmentMean}(H_{\text{func}}^{(6)}) \,\|\, W_{\text{env}} c_t \right] \right) \in \mathbb{R}^{B \times 512}$$

### 2.3 Autoregressive Hierarchical Forward Policy Head ($P_F(a_t \mid s_t; \theta)$)
Implemented in `crates/gfn-engine/src/policy.rs`. A transaction action $a_t = (a_t^{\text{caller}}, a_t^{\text{func}}, a_t^{\text{domain}}, a_t^{\text{env}})$ is factored autoregressively:
$$\log P_F(a_t \mid s_t; \theta) = \log P_F(a_t^{\text{caller}} \mid g_t) + \log P_F(a_t^{\text{func}} \mid g_t, a_t^{\text{caller}}) + \log P_F(a_t^{\text{domain}} \mid g_t, h_{a_t^{\text{func}}}^{(6)}, a_t^{\text{caller}}) + \log P_F(a_t^{\text{env}} \mid g_t, h_{a_t^{\text{func}}}^{(6)})$$

1. **Sub-Head 1: Caller Selection ($a_t^{\text{caller}} \in \{0..5\}$)**:
   - 3-layer MLP (`512 -> 256 -> 6`) over $g_t$ choosing from the 5 canonical actor personas (`0: Attacker0`, `1: Attacker1Sybil`, `2: NormalUser`, `3: LiquidityProvider`, `4: LiquidatorKeeper`) or terminating the trajectory (`5: StopTrajectory` if $t \ge 1$).
2. **Sub-Head 2: Target Function Node Selection ($a_t^{\text{func}} \in \{0..N_{\text{func}}-1\}$)**:
   - Pointer-network scaled dot-product attention between query $q_{\text{func}} = W_{q}\left([g_t \,\|\, \text{Emb}(a_t^{\text{caller}})]\right) \in \mathbb{R}^{B \times 512}$ and padded per-graph `FunctionNode` embeddings $H_{\text{func}}^{(6)} \in \mathbb{R}^{B \times N_{\text{func}} \times 512}$:
     $$\text{logits}_{\text{func}}[b, f] = \frac{q_{\text{func}}[b]^\top H_{\text{func}}^{(6)}[b, f]}{\sqrt{512.0}} + \text{Mask}_{\text{callable}}[b, f]$$
3. **Sub-Head 3: Symbolic Input Domain Partitioning ($a_t^{\text{domain}} \in \{0..7\}^{8}$)**:
   - For each of up to $K_{\text{args}} = 8$ function parameters, outputs logits of shape $[B, 8, 8]$ selecting one of 8 symbolic/concrete domain partitions (`0: Zero`, `1: OneWei`, `2: MaxUint256`, `3: CallerFullBalance`, `4: PoolReserveFraction50Pct`, `5: PoolReservePlusOneWei`, `6: FlashloanMaxLiquidity`, `7: SymbolicBv256Solve`).
4. **Sub-Head 4: Environmental Manipulation ($a_t^{\text{env}} \in \{0..5\}$)**:
   - Selects Foundry time/block cheatcode deltas (`0: NoOp`, `1: Warp1Block12s`, `2: Warp1Hour`, `3: Warp7DaysTimelock`, `4: Warp365DaysInterestAccrual`, `5: Roll100Blocks`).

### 2.4 Backward Policy Head ($P_B(s_{t-1} \mid s_t; \theta)$) & Partition Function ($\log Z_\theta$)
- **Backward Policy Head (`BackwardPolicyHead`)**: A 3-layer MLP (`1024 -> 512 -> max_parent_edges`) taking $[g_t \,\|\, h_{a_{t-1}^{\text{func}}}^{(6)}]$ to predict logits over observed parent transitions into state hash $s_t$.
- **Learnable Log-Partition Scalar (`LogPartitionNet`)**: Parameterized as $\log Z_\theta(\mathcal{G}, s_0) = \text{MLP}_{Z}(g_0) \in \mathbb{R}^{B}$ in `Kind::Float`, optimized with a dedicated learning rate $\eta_Z = 5 \times 10^{-3}$.

---

## 3. Mathematical Loss Functions & Multi-Objective Reward Formulation

### 3.1 Composite Neuro-Symbolic Reward Function $R(s_n)$
Implemented in `crates/gfn-engine/src/reward.rs`. Every terminal state $s_n$ of trajectory $\tau = (s_0, a_1, s_1, \dots, s_n)$ receives strictly positive reward $R(s_n) \ge R_{\min} = 10^{-6}$:

$$R(s_n) = \max\left(R_{\min},\; \left( \alpha \cdot R_{\text{boundary}}(s_n) + \beta \cdot R_{\text{novelty}}(s_n) + \gamma \cdot R_{\text{exploit}}(s_n) \right) \times M_{\text{unsat}}(\tau) \right)$$

with default mixing coefficients $\alpha = 1.0$, $\beta = 0.5$, $\gamma = 100.0$:

1. **Boundary Proximity Reward ($R_{\text{boundary}}(s_n) \in [0, 1]$)**:
   For each active safety invariant $\mathcal{I}_m(s_n)$ expressed in normalized inequality form $L_m(s_n) \le U_m(s_n)$:
   $$R_{\text{boundary}}(s_n) = \max_{m \in \{1 \dots M\}} \exp\left( - \lambda_b \cdot \frac{\max(0, U_m(s_n) - L_m(s_n))}{|U_m(s_n)| + 1.0} \right), \quad \lambda_b = 5.0$$
2. **Storage Novelty Entropy via Random Network Distillation ($R_{\text{novelty}}(s_n) \in [0, 1]$)**:
   - Let $f_{\text{target}}: \mathbb{R}^{512} \to \mathbb{R}^{128}$ be a frozen, orthogonally initialized 3-layer `tch::nn::Sequential` MLP (`set_requires_grad(false)`), and $\hat{f}_{\text{pred}; \phi}: \mathbb{R}^{512} \to \mathbb{R}^{128}$ be a trainable predictor MLP over pooled `StorageNode` embeddings $h_{\text{storage}}(s_n)$.
   - Normalized via Welford online running variance $(\mu_{\text{RND}}, \sigma_{\text{RND}})$:
     $$R_{\text{novelty}}(s_n) = \text{ clamp}\left( \frac{\| \hat{f}_{\text{pred}; \phi}(h_{\text{storage}}(s_n)) - f_{\text{target}}(h_{\text{storage}}(s_n)) \|_2^2 - \mu_{\text{RND}}}{\sigma_{\text{RND}} + 10^{-5}}, \; 0.0, \; 1.0 \right)$$
3. **Exploit Realization Bonus ($R_{\text{exploit}}(s_n) \in \{0.0, 1.0\}$)**:
   Equals $1.0$ if and only if at least one invariant $\mathcal{I}_m(s_n)$ is violated ($\neg \mathcal{I}_m(s_n) = \text{true}$) during concrete `revm` execution; otherwise $0.0$.
4. **Unsat Penalty Multiplier ($M_{\text{unsat}}(\tau) \in \{10^{-3}, 0.5, 1.0\}$)**:
   - $M_{\text{unsat}}(\tau) = 10^{-3}$ if `gfn-symbolic` (`Bitwuzla` / `Z3`) proves `SmtStatus::Unsat` on the symbolic branch target at the terminal transition of $\tau$ (or matches a cached Unsat Core prefix in `heed`).
   - $M_{\text{unsat}}(\tau) = 0.5$ if the SMT solver times out (`SmtStatus::Unknown`).
   - $M_{\text{unsat}}(\tau) = 1.0$ if concrete execution succeeds or SMT returns `SmtStatus::Sat`.

### 3.2 Primary Objective: Trajectory Balance Loss ($\mathcal{L}_{TB}$)
Implemented in `crates/gfn-engine/src/tb_loss.rs`. All log-probability sums are explicitly cast via `.to_kind(tch::Kind::Float)` (`f32`) before computing the Huber-clipped residual at $\delta_{TB} = 10.0$:

$$\Delta_{TB}(\tau; \theta) = \log Z_\theta(s_0) + \sum_{t=0}^{n-1} \log P_F(s_{t+1} \mid s_t; \theta) - \log R(s_n) - \sum_{t=1}^{n} \log P_B(s_{t-1} \mid s_t; \theta)$$

$$\mathcal{L}_{TB}(\tau; \theta) = \begin{cases} \frac{1}{2} \left(\Delta_{TB}(\tau; \theta)\right)^2 & \text{if } |\Delta_{TB}(\tau; \theta)| \le \delta_{TB} \\ \delta_{TB} \cdot \left(|\Delta_{TB}(\tau; \theta)| - \frac{1}{2}\delta_{TB}\right) & \text{otherwise} \end{cases}$$

### 3.3 Phase 1 Pre-Training Objective: Masked State-Transition Modeling ($\mathcal{L}_{MSTM}$)
During Phase 1 self-supervised pre-training on 52,400 contracts, $30\%$ of mutated `StorageNode` targets in a single transaction step $(s_t, a_t) \to s_{t+1}$ are masked:
$$\mathcal{L}_{MSTM} = \mathcal{L}_{\text{BCE}}\left(\hat{y}_{\text{mutated}}, y_{\text{mutated}}\right) + \mathcal{L}_{\text{CE}}\left(\hat{y}_{\text{direction}}, y_{\text{direction}} \in \{\text{Dec}, \text{Unchanged}, \text{Inc}, \text{Zeroed}\}\right) + \mathcal{L}_{\text{Huber}}\left(\hat{\phi}_{\text{delta}}, \phi_{\text{delta}}\right)$$

---

## 4. Optimization Strategies & Hyperparameter Search Spaces

### 4.1 Optimizer & Precision Configuration in `tch-rs`
- **Optimizer**: `tch::nn::AdamW` (`wd = 1e-4`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`).
- **Dual Parameter Group Learning Rates**:
  - Group 0 (`VarStore` variables under `"rgat/*"` and `"policy/*"`): $\eta_{\theta} = 3 \times 10^{-4}$ with linear warm-up over 2,000 steps followed by Cosine Annealing to $1 \times 10^{-5}$.
  - Group 1 (`VarStore` variables under `"log_z/*"`): `opt.set_lr_group(1, 5e-3)` (16.6x faster adaptation rate to stabilize flow magnitude early in training).
- **Mixed Precision & Gradient Clipping**:
  - RGAT message-passing tensors operate in `tch::Kind::BFloat16`; master parameters in `VarStore` and Trajectory Balance log-probabilities $\sum \log P_F$, $\sum \log P_B$, $\log Z_\theta$, and $\log R(s_n)$ are maintained in `tch::Kind::Float` (`f32`).
  - Global gradient norm clipping: `opt.clip_grad_norm(1.0)`.
- **Lock-Free Prioritized Trajectory Replay Buffer**:
  - Implemented in `crates/gfn-engine/src/replay_buffer.rs` using a lock-free ring buffer + Fenwick sum-tree (`parking_lot::RwLock<SumTree>`) storing up to $100,000$ trajectories prioritized by $p_i = \left(\max(R(\tau_i), |\Delta_{TB}(\tau_i)|) + 10^{-5}\right)^{\alpha_{\text{PER}}}$ with $\alpha_{\text{PER}} = 0.6$.

### 4.2 Hyperparameter Configuration Table (`configs/curriculum_tb.toml`)

| Hyperparameter | TOML Key | Search Space / Distribution | Selected Production Default |
| :--- | :--- | :--- | :--- |
| RGAT Hidden Dimension | `model.rgat_hidden_dim` | `{256, 512, 768}` | `512` |
| RGAT Layers | `model.rgat_num_layers` | `[4, 8]` | `6` |
| RGAT Attention Heads | `model.rgat_heads` | `{4, 8, 16}` | `8` |
| Policy Learning Rate | `optim.lr_policy` | $[5 \times 10^{-5}, 1 \times 10^{-3}]$ | `3e-4` |
| Log-Z Learning Rate | `optim.lr_log_z` | $[1 \times 10^{-3}, 5 \times 10^{-2}]$ | `5e-3` |
| Reward Boundary Weight ($\alpha$) | `reward.alpha_boundary` | $[0.2, 5.0]$ | `1.0` |
| Reward Novelty Weight ($\beta$) | `reward.beta_novelty` | $[0.1, 2.0]$ | `0.5` |
| Reward Exploit Spike ($\gamma$) | `reward.gamma_exploit` | $[20.0, 500.0]$ | `100.0` |
| Unsat Penalty Multiplier ($M_{\text{unsat}}$) | `reward.m_unsat_penalty` | $[10^{-5}, 10^{-2}]$ | `1e-3` |
| Max Trajectory Depth ($T_{\max}$) | `rollout.max_steps` | `{10, 15, 20}` | `15` |
| SMT Handover Boundary Threshold | `symbolic.boundary_trigger` | $[0.70, 0.95]$ | `0.85` |

---

## 5. Experiment Tracking & Reproducibility Protocols

1. **Structured Telemetry & Prometheus/OpenTelemetry**:
   Every training and online-audit run emits structured metrics every 50 gradient steps via `tracing` and `metrics-exporter-prometheus`:
   - `gfn_loss_trajectory_balance`, `gfn_loss_mstm_pretrain`, `gfn_loss_rnd_predictor`
   - `gfn_flow_log_z_mean`, `gfn_flow_spearman_rho_logpf_logr`, `gfn_flow_trajectory_entropy`
   - `gfn_reward_mean_total`, `gfn_reward_max_boundary_proximity`, `gfn_reward_rnd_novelty_mean`, `gfn_unique_exploits_found`
   - `gfn_smt_queries_per_sec`, `gfn_smt_sat_rate`, `gfn_smt_unsat_cache_hit_rate`, `gfn_smt_p95_solve_ms`
   - `gfn_revm_steps_per_sec_per_thread`, `gfn_revm_cow_memory_mb`
2. **Deterministic Seeding**:
   All runs enforce `tch::manual_seed(seed as i64)`, `tch::Cuda::manual_seed_all(seed as usize)`, and initialize per-worker `rand_chacha::ChaCha8Rng::seed_from_u64(seed)` inside `RevmArena`. Model checkpoints are exported via `VarStore::save_safetensors("model.safetensors")` paired with a `manifest.json` containing SHA-256 digests, `rustc --version`, and `forge --version`.

---

## 6. Training Stability Refinements (`TrainingStabilityController` & Numerical Guards)

To guarantee monotonic flow convergence without `NaN`/`Inf` gradient spikes across deep trajectories:

1. **Importance-Weighted & Clamped Trajectory Balance (`TrajectoryBalanceLoss::forward_weighted`)**:
   * Pre-clamps $\log Z_\theta \in [-50, 50]$, $\sum \log P_F \in [-200, 0]$, $\sum \log P_B \in [-200, 0]$, and $|\Delta_{TB}| \le \Delta_{\max} = 100.0$ prior to Huber evaluation.
   * Weights per-trajectory Huber losses by normalized Prioritized Experience Replay (PER) Importance Sampling weights $w_i \in [10^{-4}, 10.0]$.
2. **Smooth Piecewise Huber Loss in MSTM Pretraining (`MstmPretrainLoss`)**:
   * Replaces $L_1$ storage delta loss with smooth piecewise quadratic/linear Huber loss ($\delta = 1.0$) and clamps mutation/direction logits to $[-30.0, 30.0]$.
3. **Fisher Diagonal Clamping in Continual EWC (`compute_ewc_penalty`)**:
   * Clamps diagonal Fisher information entries $F_i \in [0.0, 10^4]$ and weight diffs $(\theta_i - \theta_i^*) \in [-10.0, 10.0]$ in `Kind::Float` precision.
4. **Autoregressive Policy Logit Soft-Clamping & 4-Head Entropy (`AutoregressiveForwardPolicyHead`)**:
   * Clamps sub-head logits to $[-30.0, 30.0]$ and evaluates composite Shannon entropy across all 4 autoregressive heads (`caller`, `func`, `domain`, and `env`).
5. **Winsorized Welford & EMA RND Normalization (`WelfordRunningStats`)**:
   * Applies $z_{\max} = 5.0$ Winsorized outlier clipping after warm-up ($N \ge 10$) and tracks EMA mean/variance ($\beta = 0.99$).
6. **Centralized `TrainingStabilityController`**:
   * Enforces Linear Warmup + Cosine Annealing over $(\eta_\theta(t), \eta_Z(t))$, global L2 gradient norm clipping ($\|\nabla_\theta \mathcal{L}\|_2 \le G_{\max}$), EMA loss smoothing ($\beta_{\text{EMA}} = 0.95$), $0.5\times$ step damping on $>2.5\times$ residual spikes, and bidirectional temperature recovery in `OnlineAdaptationController`. See `docs/09_system_analysis_and_stability_refinements.md` for full details.
