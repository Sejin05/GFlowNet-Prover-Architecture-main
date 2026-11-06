#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Autoregressive Forward Policy Heads, Backward Policy Head, and Log-Partition Network.
//!
//! Decomposes transaction generation into 4 autoregressive sub-actions:
//! 1. `caller_id` $\in \{0..5\}$ (5 actor personas or StopTrajectory)
//! 2. `func_node_id` $\in \{0..N_{\text{func}}-1\}$ (Pointer-network attention)
//! 3. `arg_domains` $\in \{0..7\}^8$ (Symbolic & concrete domain partitions)
//! 4. `env_op` $\in \{0..5\}$ (Time/block cheatcode manipulation)
//!
//! All log-probabilities ($\log P_F$, $\log P_B$, and $\log Z_\theta$) are strictly evaluated
//! in `tch::Kind::Float` (`f32`) precision.

use gfn_common::error::GfnResult;
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

/// Batched actions sampled or evaluated along a trajectory step.
#[derive(Debug)]
pub struct ActionBatchTensor {
    /// Actor persona index in `0..=5` (5 = `StopTrajectory`). Shape: `[B]`, `Kind::Int64`.
    pub caller_id: Tensor,
    /// Target function node index in `0..N_func`. Shape: `[B]`, `Kind::Int64`.
    pub func_node_id: Tensor,
    /// 8 domain categories across up to 8 parameters in `0..=7`. Shape: `[B, 8]`, `Kind::Int64`.
    pub arg_domains: Tensor,
    /// Time/block cheatcode manipulation in `0..=5`. Shape: `[B]`, `Kind::Int64`.
    pub env_op: Tensor,
    /// Exact `f32` sum of conditional log-probabilities $\log P_F(a_t \mid s_t)$. Shape: `[B]`, `Kind::Float`.
    pub log_pf: Tensor,
}

/// Autoregressive Forward Policy Head factoring $P_F(a_t \mid s_t; \theta)$.
pub struct AutoregressiveForwardPolicyHead {
    caller_mlp: nn::Sequential,
    caller_embed: nn::Embedding,
    func_query_proj: nn::Linear,
    domain_mlp: nn::Sequential,
    env_mlp: nn::Sequential,
    hidden_dim: i64,
    logit_clip_bound: f64,
}

impl AutoregressiveForwardPolicyHead {
    /// Creates a new `AutoregressiveForwardPolicyHead` module within the given VarStore path.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        // Sub-Head 1: Caller selection MLP (512 -> 256 -> 6)
        let caller_mlp = nn::seq()
            .add(nn::linear(
                vs / "caller_mlp_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "caller_mlp_2",
                hidden_dim / 2,
                6,
                Default::default(),
            ));

        // Caller embedding table (6 -> hidden_dim)
        let caller_embed = nn::embedding(vs / "caller_embed", 6, hidden_dim, Default::default());

        // Sub-Head 2: Function query projection (hidden_dim * 2 -> hidden_dim)
        let func_query_proj = nn::linear(
            vs / "func_query_proj",
            hidden_dim * 2,
            hidden_dim,
            Default::default(),
        );

        // Sub-Head 3: Domain partitioning MLP (hidden_dim * 3 -> hidden_dim -> 64)
        let domain_mlp = nn::seq()
            .add(nn::linear(
                vs / "domain_mlp_1",
                hidden_dim * 3,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "domain_mlp_2",
                hidden_dim,
                64,
                Default::default(),
            ));

        // Sub-Head 4: Environmental manipulation MLP (hidden_dim * 2 -> hidden_dim / 2 -> 6)
        let env_mlp = nn::seq()
            .add(nn::linear(
                vs / "env_mlp_1",
                hidden_dim * 2,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "env_mlp_2",
                hidden_dim / 2,
                6,
                Default::default(),
            ));

        Self {
            caller_mlp,
            caller_embed,
            func_query_proj,
            domain_mlp,
            env_mlp,
            hidden_dim,
            logit_clip_bound: 30.0,
        }
    }

    /// Configures the symmetric logit clamping bound $[-B, B]$ for numerical stability.
    #[must_use]
    pub fn with_logit_clip(mut self, bound: f64) -> Self {
        self.logit_clip_bound = bound.clamp(5.0, 100.0);
        self
    }

    /// Computes per-row uniform exploration weights strictly over callable (`mask > -1e3`) functions,
    /// falling back to full-row uniform only when an entire row is masked.
    pub fn uniform_valid_function_distribution(callable_mask: &Tensor) -> Tensor {
        let dims = callable_mask.size();
        if dims.len() != 2 || dims[0] <= 0 || dims[1] <= 0 {
            return Tensor::zeros_like(callable_mask);
        }
        let b_sz = dims[0] as usize;
        let n_fn = dims[1] as usize;
        let numel = b_sz * n_fn;
        let cpu_mask = callable_mask
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Float);
        let mut mask_vals = vec![0.0f32; numel];
        cpu_mask.copy_data(&mut mask_vals, numel);

        let mut out = vec![0.0f32; numel];

        for b in 0..b_sz {
            let row_offset = b * n_fn;
            let mut valid_indices = Vec::with_capacity(n_fn);
            for j in 0..n_fn {
                let m_val = mask_vals[row_offset + j];
                if m_val.is_finite() && m_val > -1_000.0 {
                    valid_indices.push(j);
                }
            }
            if valid_indices.is_empty() {
                let fallback_p = 1.0f32 / (n_fn as f32);
                for j in 0..n_fn {
                    out[row_offset + j] = fallback_p;
                }
            } else {
                let valid_p = 1.0f32 / (valid_indices.len() as f32);
                for j in valid_indices {
                    out[row_offset + j] = valid_p;
                }
            }
        }

        Tensor::from_slice(&out)
            .view([b_sz as i64, n_fn as i64])
            .to_device(callable_mask.device())
    }

    /// Samples a batch of actions autoregressively given current graph representations.
    pub fn sample_actions(
        &self,
        global_ctx: &Tensor,
        func_embeddings: &Tensor,
        callable_mask: &Tensor,
        temperature: f64,
        epsilon_uniform: f64,
    ) -> GfnResult<ActionBatchTensor> {
        let temp = if temperature <= 0.0 {
            1.0
        } else {
            temperature.clamp(0.1, 10.0)
        };
        let eps = epsilon_uniform.clamp(0.0, 0.5);
        let clip_b = self.logit_clip_bound;
        let batch_size = global_ctx.size()[0];

        // 1. Sub-Head 1: Caller selection
        let caller_logits = self.caller_mlp.forward(global_ctx).clamp(-clip_b, clip_b);
        let scaled_caller = &caller_logits / temp;
        let mut p_caller = scaled_caller.softmax(-1, Kind::Float);
        if eps > 0.0 {
            p_caller = (1.0 - eps) * &p_caller + (eps / 6.0);
        }
        let caller_id = p_caller.multinomial(1, true).squeeze_dim(-1);
        let log_p_caller = p_caller
            .clamp_min(1e-12)
            .log()
            .gather(-1, &caller_id.unsqueeze(-1), false)
            .squeeze_dim(-1);

        // 2. Sub-Head 2: Function pointer network
        let caller_emb = self.caller_embed.forward(&caller_id);
        let concat_caller = Tensor::cat(&[global_ctx, &caller_emb], 1);
        let q_func = self.func_query_proj.forward(&concat_caller);

        let scale = (self.hidden_dim as f64).sqrt();
        let func_logits = ((q_func.unsqueeze(1) * func_embeddings).sum_dim_intlist(
            Some([-1].as_slice()),
            false,
            Kind::Float,
        ) / scale)
            .clamp(-clip_b, clip_b);

        let masked_func_logits = (func_logits + callable_mask).clamp(-1e4, clip_b);
        let scaled_func = &masked_func_logits / temp;
        let mut p_func = scaled_func.softmax(-1, Kind::Float);
        if eps > 0.0 {
            let uniform_valid = Self::uniform_valid_function_distribution(callable_mask);
            p_func = (1.0 - eps) * &p_func + eps * uniform_valid;
        }
        let func_node_id = p_func.multinomial(1, true).squeeze_dim(-1);
        let log_p_func = p_func
            .clamp_min(1e-12)
            .log()
            .gather(-1, &func_node_id.unsqueeze(-1), false)
            .squeeze_dim(-1);

        // 3. Sub-Head 3: Domain partitioning across 8 arguments
        let h_func = func_embeddings
            .gather(
                1,
                &func_node_id
                    .view([-1, 1, 1])
                    .expand([-1, 1, self.hidden_dim], false),
                false,
            )
            .squeeze_dim(1);

        let concat_domain = Tensor::cat(&[global_ctx, &h_func, &caller_emb], 1);
        let domain_logits = self
            .domain_mlp
            .forward(&concat_domain)
            .view([batch_size, 8, 8])
            .clamp(-clip_b, clip_b);
        let scaled_domain = &domain_logits / temp;
        let mut p_domain = scaled_domain.softmax(-1, Kind::Float);
        if eps > 0.0 {
            p_domain = (1.0 - eps) * &p_domain + (eps / 8.0);
        }

        let p_domain_flat = p_domain.view([batch_size * 8, 8]);
        let sampled_domain_flat = p_domain_flat.multinomial(1, true);
        let arg_domains = sampled_domain_flat.view([batch_size, 8]);

        let log_p_domain = p_domain
            .clamp_min(1e-12)
            .log()
            .gather(-1, &arg_domains.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        // 4. Sub-Head 4: Environmental manipulation
        let concat_env = Tensor::cat(&[global_ctx, &h_func], 1);
        let env_logits = self.env_mlp.forward(&concat_env).clamp(-clip_b, clip_b);
        let scaled_env = &env_logits / temp;
        let mut p_env = scaled_env.softmax(-1, Kind::Float);
        if eps > 0.0 {
            p_env = (1.0 - eps) * &p_env + (eps / 6.0);
        }
        let env_op = p_env.multinomial(1, true).squeeze_dim(-1);
        let log_p_env = p_env
            .clamp_min(1e-12)
            .log()
            .gather(-1, &env_op.unsqueeze(-1), false)
            .squeeze_dim(-1);

        // Total log P_F = sum of conditional log-probabilities
        let log_pf = (&log_p_caller + &log_p_func + &log_p_domain + &log_p_env)
            .to_kind(Kind::Float)
            .clamp(-200.0, 0.0);

        Ok(ActionBatchTensor {
            caller_id,
            func_node_id,
            arg_domains,
            env_op,
            log_pf,
        })
    }

    /// Evaluates exact conditional log-probabilities $\log P_F(a_t \mid s_t; \theta)$ for given actions.
    pub fn evaluate_log_pf(
        &self,
        global_ctx: &Tensor,
        func_embeddings: &Tensor,
        callable_mask: &Tensor,
        actions: &ActionBatchTensor,
    ) -> GfnResult<Tensor> {
        let batch_size = global_ctx.size()[0];
        let clip_b = self.logit_clip_bound;

        // 1. Caller log-probability
        let caller_logits = self.caller_mlp.forward(global_ctx).clamp(-clip_b, clip_b);
        let log_p_caller = caller_logits
            .log_softmax(-1, Kind::Float)
            .gather(-1, &actions.caller_id.unsqueeze(-1), false)
            .squeeze_dim(-1);

        // 2. Function pointer-network log-probability
        let caller_emb = self.caller_embed.forward(&actions.caller_id);
        let concat_caller = Tensor::cat(&[global_ctx, &caller_emb], 1);
        let q_func = self.func_query_proj.forward(&concat_caller);

        let scale = (self.hidden_dim as f64).sqrt();
        let func_logits = ((q_func.unsqueeze(1) * func_embeddings).sum_dim_intlist(
            Some([-1].as_slice()),
            false,
            Kind::Float,
        ) / scale)
            .clamp(-clip_b, clip_b);

        let masked_func_logits = (func_logits + callable_mask).clamp(-1e4, clip_b);
        let log_p_func = masked_func_logits
            .log_softmax(-1, Kind::Float)
            .gather(-1, &actions.func_node_id.unsqueeze(-1), false)
            .squeeze_dim(-1);

        // 3. Domain partitioning log-probability
        let h_func = func_embeddings
            .gather(
                1,
                &actions
                    .func_node_id
                    .view([-1, 1, 1])
                    .expand([-1, 1, self.hidden_dim], false),
                false,
            )
            .squeeze_dim(1);

        let concat_domain = Tensor::cat(&[global_ctx, &h_func, &caller_emb], 1);
        let domain_logits = self
            .domain_mlp
            .forward(&concat_domain)
            .view([batch_size, 8, 8])
            .clamp(-clip_b, clip_b);
        let log_p_domain = domain_logits
            .log_softmax(-1, Kind::Float)
            .gather(-1, &actions.arg_domains.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        // 4. Environmental manipulation log-probability
        let concat_env = Tensor::cat(&[global_ctx, &h_func], 1);
        let env_logits = self.env_mlp.forward(&concat_env).clamp(-clip_b, clip_b);
        let log_p_env = env_logits
            .log_softmax(-1, Kind::Float)
            .gather(-1, &actions.env_op.unsqueeze(-1), false)
            .squeeze_dim(-1);

        let total_log_pf = (&log_p_caller + &log_p_func + &log_p_domain + &log_p_env)
            .to_kind(Kind::Float)
            .clamp(-200.0, 0.0);
        Ok(total_log_pf)
    }

    /// Computes composite Shannon entropy $\bar{H}(P_F)$ across all 4 autoregressive policy sub-heads
    /// for exploration monitoring and entropy regularization.
    pub fn compute_policy_entropy(
        &self,
        global_ctx: &Tensor,
        func_embeddings: &Tensor,
        callable_mask: &Tensor,
    ) -> GfnResult<Tensor> {
        let batch_size = global_ctx.size()[0];
        let clip_b = self.logit_clip_bound;

        // 1. Caller head entropy
        let caller_logits = self.caller_mlp.forward(global_ctx).clamp(-clip_b, clip_b);
        let p_caller = caller_logits.softmax(-1, Kind::Float);
        let log_p_caller = caller_logits.log_softmax(-1, Kind::Float);
        let h_caller =
            -(&p_caller * &log_p_caller).sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        // 2. Function head entropy with reference caller 0
        let dummy_caller = Tensor::zeros([batch_size], (Kind::Int64, global_ctx.device()));
        let caller_emb = self.caller_embed.forward(&dummy_caller);
        let concat_caller = Tensor::cat(&[global_ctx, &caller_emb], 1);
        let q_func = self.func_query_proj.forward(&concat_caller);
        let scale = (self.hidden_dim as f64).sqrt();
        let func_logits = ((q_func.unsqueeze(1) * func_embeddings).sum_dim_intlist(
            Some([-1].as_slice()),
            false,
            Kind::Float,
        ) / scale)
            .clamp(-clip_b, clip_b);
        let masked_func_logits = (func_logits + callable_mask).clamp(-1e4, clip_b);
        let p_func = masked_func_logits.softmax(-1, Kind::Float);
        let log_p_func = masked_func_logits.log_softmax(-1, Kind::Float);
        let h_func =
            -(&p_func * &log_p_func).sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        // 3. Domain and Environment head entropy using mean function embedding context
        let mean_func_emb = func_embeddings.mean_dim(Some([1].as_slice()), false, Kind::Float);
        let concat_domain = Tensor::cat(&[global_ctx, &mean_func_emb, &caller_emb], 1);
        let domain_logits = self
            .domain_mlp
            .forward(&concat_domain)
            .view([batch_size, 8, 8])
            .clamp(-clip_b, clip_b);
        let p_domain = domain_logits.softmax(-1, Kind::Float);
        let log_p_domain = domain_logits.log_softmax(-1, Kind::Float);
        let h_domain = -(&p_domain * &log_p_domain)
            .sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float)
            .mean_dim(Some([-1].as_slice()), false, Kind::Float);

        let concat_env = Tensor::cat(&[global_ctx, &mean_func_emb], 1);
        let env_logits = self.env_mlp.forward(&concat_env).clamp(-clip_b, clip_b);
        let p_env = env_logits.softmax(-1, Kind::Float);
        let log_p_env = env_logits.log_softmax(-1, Kind::Float);
        let h_env =
            -(&p_env * &log_p_env).sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        let aux_entropy: Tensor = (&h_domain + &h_env) * 0.25;
        let raw_entropy: Tensor = &h_caller + &h_func + aux_entropy;
        let total_entropy = raw_entropy.to_kind(Kind::Float).clamp_min(0.0);
        Ok(total_entropy)
    }
}

/// Backward Policy Head predicting transition probabilities $P_B(s_{t-1} \mid s_t; \theta)$.
pub struct BackwardPolicyHead {
    mlp: nn::Sequential,
    hidden_dim: i64,
    max_parent_edges: i64,
}

impl BackwardPolicyHead {
    /// Creates a new `BackwardPolicyHead` predicting logits over up to `max_parent_edges`.
    pub fn new(vs: &nn::Path, hidden_dim: i64, max_parent_edges: i64) -> Self {
        let mlp = nn::seq()
            .add(nn::linear(
                vs / "bp_1",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "bp_2",
                hidden_dim,
                max_parent_edges,
                Default::default(),
            ));

        Self {
            mlp,
            hidden_dim,
            max_parent_edges,
        }
    }

    /// Returns the hidden dimension of the backward policy head.
    #[must_use]
    pub fn hidden_dim(&self) -> i64 {
        self.hidden_dim
    }

    /// Returns the maximum number of parent transitions handled.
    #[must_use]
    pub fn max_parent_edges(&self) -> i64 {
        self.max_parent_edges
    }

    /// Evaluates exact log-probability $\log P_B(s_{t-1} \mid s_t; \theta)$ in `Kind::Float`.
    pub fn evaluate_log_pb(
        &self,
        global_ctx: &Tensor,
        prev_func_emb: &Tensor,
        parent_mask: &Tensor,
        chosen_parent_idx: &Tensor,
    ) -> GfnResult<Tensor> {
        let concat = Tensor::cat(&[global_ctx, prev_func_emb], 1);
        let raw_logits = self.mlp.forward(&concat).clamp(-30.0, 30.0);
        let logits = (raw_logits + parent_mask).clamp(-1e4, 30.0);
        let log_pb = logits
            .log_softmax(-1, Kind::Float)
            .gather(-1, &chosen_parent_idx.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .to_kind(Kind::Float)
            .clamp(-100.0, 0.0);

        Ok(log_pb)
    }
}

/// Learnable Log-Partition Function Network $\log Z_\theta(\mathcal{G}, s_0)$.
pub struct LogPartitionNet {
    mlp: nn::Sequential,
}

impl LogPartitionNet {
    /// Constructs a new `LogPartitionNet` module.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let mlp = nn::seq()
            .add(nn::linear(
                vs / "log_z_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "log_z_2",
                hidden_dim / 2,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs / "log_z_3", 128, 1, Default::default()));

        Self { mlp }
    }

    /// Computes predicted $\log Z_\theta(s_0) \in \mathbb{R}^B$ in `Kind::Float`, clamped to $[-50.0, 50.0]$.
    pub fn forward(&self, global_ctx_s0: &Tensor) -> GfnResult<Tensor> {
        let log_z = self
            .mlp
            .forward(global_ctx_s0)
            .squeeze_dim(-1)
            .to_kind(Kind::Float)
            .clamp(-50.0, 50.0);
        Ok(log_z)
    }
}

/// Proximal Policy Optimization (PPO) Baseline Agent (`PpoRgatAgent`).
///
/// Implemented to benchmark and empirically demonstrate mode-collapse mitigation:
/// standard PPO collapses to the single easiest exploit mode, whereas GFlowNet
/// samples all distinct terminal bug modes proportionally.
pub struct PpoRgatAgent {
    policy_mlp: nn::Sequential,
    value_mlp: nn::Sequential,
    clip_epsilon: f64,
    entropy_coeff: f64,
}

impl PpoRgatAgent {
    /// Creates a new `PpoRgatAgent` baseline.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_actions: i64) -> Self {
        let policy_mlp = nn::seq()
            .add(nn::linear(
                vs / "ppo_p_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "ppo_p_2",
                hidden_dim / 2,
                num_actions,
                Default::default(),
            ));

        let value_mlp = nn::seq()
            .add(nn::linear(
                vs / "ppo_v_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "ppo_v_2",
                hidden_dim / 2,
                1,
                Default::default(),
            ));

        Self {
            policy_mlp,
            value_mlp,
            clip_epsilon: 0.2,
            entropy_coeff: 0.01,
        }
    }

    /// Evaluates the clipped PPO surrogate objective:
    /// $\mathcal{L}_{\text{PPO}} = -\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) - c_2 H + c_1 (V - R)^2$.
    pub fn compute_loss(
        &self,
        global_ctx: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> GfnResult<Tensor> {
        let logits = self.policy_mlp.forward(global_ctx);
        let p = logits.softmax(-1, Kind::Float);
        let log_p = logits.log_softmax(-1, Kind::Float);

        let curr_log_probs = log_p
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1);

        let ratios = (curr_log_probs - old_log_probs).exp();
        let surr1 = &ratios * advantages;
        let surr2 = ratios.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages;
        let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);

        let values = self.value_mlp.forward(global_ctx).squeeze_dim(-1);
        let value_loss = (values - returns).pow_tensor_scalar(2.0).mean(Kind::Float);

        let entropy = -(&p * &log_p)
            .sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float)
            .mean(Kind::Float);

        let total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy;
        Ok(total_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_forward_policy_sample_and_evaluate() {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = AutoregressiveForwardPolicyHead::new(&vs.root(), 64);

        let global_ctx = Tensor::randn([2, 64], (Kind::Float, Device::Cpu));
        let func_embeddings = Tensor::randn([2, 5, 64], (Kind::Float, Device::Cpu));
        let callable_mask = Tensor::zeros([2, 5], (Kind::Float, Device::Cpu));

        let actions = head
            .sample_actions(&global_ctx, &func_embeddings, &callable_mask, 1.0, 0.1)
            .unwrap_or_else(|e| panic!("Sampling failed: {e}"));

        assert_eq!(actions.caller_id.size(), vec![2]);
        assert_eq!(actions.func_node_id.size(), vec![2]);
        assert_eq!(actions.arg_domains.size(), vec![2, 8]);
        assert_eq!(actions.env_op.size(), vec![2]);
        assert_eq!(actions.log_pf.size(), vec![2]);
        assert_eq!(actions.log_pf.kind(), Kind::Float);

        let evaluated_log_pf = head
            .evaluate_log_pf(&global_ctx, &func_embeddings, &callable_mask, &actions)
            .unwrap_or_else(|e| panic!("Evaluation failed: {e}"));

        assert_eq!(evaluated_log_pf.size(), vec![2]);
        let diff = (&evaluated_log_pf - &actions.log_pf)
            .abs()
            .max()
            .double_value(&[]);
        assert!(diff < 1e-4);

        let entropy = head
            .compute_policy_entropy(&global_ctx, &func_embeddings, &callable_mask)
            .unwrap_or_else(|e| panic!("Entropy computation failed: {e}"));
        assert_eq!(entropy.size(), vec![2]);
        assert!(entropy.double_value(&[0]) >= 0.0);
    }

    #[test]
    fn test_backward_policy_and_log_partition() {
        let vs = nn::VarStore::new(Device::Cpu);
        let bp_head = BackwardPolicyHead::new(&(&vs.root() / "bp"), 64, 8);
        let log_z_net = LogPartitionNet::new(&(&vs.root() / "z"), 64);

        let global_ctx = Tensor::randn([2, 64], (Kind::Float, Device::Cpu));
        let prev_func = Tensor::randn([2, 64], (Kind::Float, Device::Cpu));
        let parent_mask = Tensor::zeros([2, 8], (Kind::Float, Device::Cpu));
        let chosen_parent = Tensor::from_slice(&[1i64, 3]);

        let log_pb = bp_head
            .evaluate_log_pb(&global_ctx, &prev_func, &parent_mask, &chosen_parent)
            .unwrap_or_else(|e| panic!("log_pb failed: {e}"));
        assert_eq!(log_pb.size(), vec![2]);
        assert_eq!(log_pb.kind(), Kind::Float);

        let log_z = log_z_net
            .forward(&global_ctx)
            .unwrap_or_else(|e| panic!("log_z failed: {e}"));
        assert_eq!(log_z.size(), vec![2]);
        assert_eq!(log_z.kind(), Kind::Float);
    }

    #[test]
    fn test_masked_functions_receive_zero_exploration_mass() {
        // Row 0: functions 0 and 2 callable (0.0), functions 1, 3, 4 masked (-1e4)
        // Row 1: all functions masked (-1e4) -> safe uniform fallback
        let mask_data: [f32; 10] = [
            0.0, -10_000.0, 0.0, -10_000.0, -10_000.0, -10_000.0, -10_000.0, -10_000.0, -10_000.0,
            -10_000.0,
        ];
        let callable_mask = Tensor::from_slice(&mask_data).view([2, 5]);
        let uniform_valid =
            AutoregressiveForwardPolicyHead::uniform_valid_function_distribution(&callable_mask);

        assert_eq!(uniform_valid.size(), vec![2, 5]);
        assert!((uniform_valid.double_value(&[0, 0]) - 0.5).abs() < 1e-6);
        assert_eq!(uniform_valid.double_value(&[0, 1]), 0.0);
        assert!((uniform_valid.double_value(&[0, 2]) - 0.5).abs() < 1e-6);
        assert_eq!(uniform_valid.double_value(&[0, 3]), 0.0);
        assert_eq!(uniform_valid.double_value(&[0, 4]), 0.0);

        for j in 0..5 {
            assert!((uniform_valid.double_value(&[1, j]) - 0.2).abs() < 1e-6);
        }
    }
}
