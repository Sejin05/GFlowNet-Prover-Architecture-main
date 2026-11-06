#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Random Network Distillation (RND) Curiosity Module and Composite Reward Formulation.
//!
//! Evaluates intrinsic novelty over storage state embeddings $h_{\text{storage}}(s_n)$ normalized
//! via Welford online running variance. Combines boundary proximity, RND storage novelty,
//! concrete invariant violation bonus, and SMT Unsat penalty multipliers into the final
//! terminal reward $R(s_n) \ge R_{\min} = 10^{-6}$.

use gfn_common::error::GfnResult;
use parking_lot::RwLock;
use std::sync::Arc;
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

/// Thread-safe Welford online running mean, variance, and EMA tracker for intrinsic RND normalization.
#[derive(Debug, Clone)]
pub struct WelfordRunningStats {
    count: u64,
    mean: f64,
    m2: f64,
    ema_mean: f64,
    ema_var: f64,
    ema_decay: f64,
}

impl Default for WelfordRunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl WelfordRunningStats {
    /// Creates a new empty Welford statistics tracker with EMA tracking ($\beta = 0.99$).
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            ema_mean: 0.0,
            ema_var: 1.0,
            ema_decay: 0.99,
        }
    }

    /// Observes a new scalar sample and updates running mean and variance in $O(1)$.
    /// Ignores non-finite (`NaN`/`Inf`) samples to preserve numerical stability.
    pub fn update(&mut self, val: f64) {
        if !val.is_finite() {
            return;
        }
        self.count = self.count.saturating_add(1);
        let delta = val - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = val - self.mean;
        self.m2 += delta * delta2;

        if self.count == 1 {
            self.ema_mean = val;
            self.ema_var = 1.0;
        } else {
            let diff = val - self.ema_mean;
            self.ema_mean = self.ema_decay * self.ema_mean + (1.0 - self.ema_decay) * val;
            self.ema_var = self.ema_decay * self.ema_var
                + (1.0 - self.ema_decay) * (diff * (val - self.ema_mean));
        }
    }

    /// Observes a sample with Winsorized $z$-score outlier clipping after warm-up ($N \ge 10$).
    pub fn update_winsorized(&mut self, val: f64, max_z_score: f64) {
        if !val.is_finite() {
            return;
        }
        let clipped_val = if self.count >= 10 {
            let sd = self.std_dev();
            let bound = max_z_score.max(1.0) * sd;
            val.clamp(self.mean - bound, self.mean + bound)
        } else {
            val.clamp(-1e4, 1e4)
        };
        self.update(clipped_val);
    }

    /// Returns the current running mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the exponentially weighted moving average (EMA) mean.
    #[must_use]
    pub fn ema_mean(&self) -> f64 {
        self.ema_mean
    }

    /// Returns the current running variance (sample variance if $N > 1$).
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / ((self.count - 1) as f64)
        } else {
            1.0
        }
    }

    /// Returns the current standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().max(1e-8).sqrt()
    }
}

/// Random Network Distillation (RND) Curiosity Module.
///
/// Features a frozen orthogonally initialized target network $f_{\text{target}}: \mathbb{R}^{512} \to \mathbb{R}^{128}$
/// and a trainable predictor network $\hat{f}_{\text{pred}}: \mathbb{R}^{512} \to \mathbb{R}^{128}$.
pub struct RndCuriosityModule {
    target_mlp: nn::Sequential,
    predictor_mlp: nn::Sequential,
    stats: Arc<RwLock<WelfordRunningStats>>,
}

impl RndCuriosityModule {
    /// Constructs a new `RndCuriosityModule` with target and predictor MLPs.
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let target_mlp = nn::seq()
            .add(nn::linear(
                vs / "rnd_target_1",
                hidden_dim,
                256,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(
                vs / "rnd_target_2",
                256,
                128,
                Default::default(),
            ));

        let predictor_mlp = nn::seq()
            .add(nn::linear(
                vs / "rnd_pred_1",
                hidden_dim,
                256,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::linear(vs / "rnd_pred_2", 256, 128, Default::default()));

        Self {
            target_mlp,
            predictor_mlp,
            stats: Arc::new(RwLock::new(WelfordRunningStats::new())),
        }
    }

    /// Computes normalized novelty reward $R_{\text{novelty}}(s_n) \in [0.0, 1.0]$ for pooled storage embeddings.
    pub fn compute_novelty(&self, storage_embeddings: &Tensor) -> GfnResult<Tensor> {
        let target_out = tch::no_grad(|| self.target_mlp.forward(storage_embeddings));
        let pred_out = self.predictor_mlp.forward(storage_embeddings);

        // Squared L2 error per batch element: || f_pred - f_target ||_2^2, clamped against spikes
        let raw_errors = (&pred_out - &target_out)
            .clamp(-50.0, 50.0)
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(Some([-1].as_slice()), false, Kind::Float);

        let flat_errors = raw_errors.view([-1]);
        let num_elements = flat_errors.size()[0];
        let mut stats = self.stats.write();
        for i in 0..num_elements {
            let err = flat_errors.double_value(&[i]);
            stats.update_winsorized(err, 5.0);
        }

        let mean = stats.mean();
        let std_dev = stats.std_dev();
        drop(stats);

        // Normalized intrinsic novelty clamped to [0.0, 1.0]
        let normalized = (raw_errors - mean) / (std_dev + 1e-5);
        let clamped = normalized.clamp(0.0, 1.0).to_kind(Kind::Float);

        Ok(clamped)
    }

    /// Computes MSE loss for training the predictor MLP towards the target MLP.
    pub fn compute_predictor_loss(&self, storage_embeddings: &Tensor) -> GfnResult<Tensor> {
        let target_out = tch::no_grad(|| self.target_mlp.forward(storage_embeddings));
        let pred_out = self.predictor_mlp.forward(storage_embeddings);

        let loss = (&pred_out - &target_out)
            .clamp(-50.0, 50.0)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);

        Ok(loss)
    }

    /// Returns the current running statistics tracker.
    #[must_use]
    pub fn stats(&self) -> Arc<RwLock<WelfordRunningStats>> {
        Arc::clone(&self.stats)
    }
}

/// Composite Neuro-Symbolic Reward Evaluator.
///
/// Implements:
/// $$R(s_n) = \max\left(R_{\min},\; \left( \alpha \cdot R_{\text{boundary}} + \beta \cdot R_{\text{novelty}} + \gamma \cdot R_{\text{exploit}} \right) \times M_{\text{unsat}}(\tau) \right)$$
#[derive(Debug, Clone)]
pub struct CompositeRewardEvaluator {
    pub alpha_boundary: f64,
    pub beta_novelty: f64,
    pub gamma_exploit: f64,
    pub reward_min: f64,
}

impl Default for CompositeRewardEvaluator {
    fn default() -> Self {
        Self::new(1.0, 0.5, 100.0, 1e-6)
    }
}

impl CompositeRewardEvaluator {
    /// Creates a new `CompositeRewardEvaluator` with specified weighting coefficients.
    #[must_use]
    pub fn new(
        alpha_boundary: f64,
        beta_novelty: f64,
        gamma_exploit: f64,
        reward_min: f64,
    ) -> Self {
        Self {
            alpha_boundary,
            beta_novelty,
            gamma_exploit,
            reward_min,
        }
    }

    /// Evaluates composite terminal reward across a batch of trajectories.
    pub fn evaluate_rewards(
        &self,
        r_boundary: &Tensor,
        r_novelty: &Tensor,
        r_exploit: &Tensor,
        m_unsat: &Tensor,
    ) -> GfnResult<Tensor> {
        let weighted_sum = self.alpha_boundary * r_boundary
            + self.beta_novelty * r_novelty
            + self.gamma_exploit * r_exploit;

        let scaled = weighted_sum * m_unsat;
        let clamped = scaled.clamp_min(self.reward_min).to_kind(Kind::Float);

        Ok(clamped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_welford_running_stats() {
        let mut stats = WelfordRunningStats::new();
        for &val in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(val);
        }
        assert_eq!(stats.count, 5);
        assert!((stats.mean() - 3.0).abs() < 1e-6);
        assert!((stats.variance() - 2.5).abs() < 1e-6);
        assert!((stats.std_dev() - 2.5_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_rnd_novelty_and_predictor_loss() {
        let vs = nn::VarStore::new(Device::Cpu);
        let rnd = RndCuriosityModule::new(&vs.root(), 64);

        let storage_emb = Tensor::randn([4, 64], (Kind::Float, Device::Cpu));
        let novelty = rnd
            .compute_novelty(&storage_emb)
            .unwrap_or_else(|e| panic!("Novelty failed: {e}"));

        assert_eq!(novelty.size(), vec![4]);
        assert_eq!(novelty.kind(), Kind::Float);

        let min_v = novelty.min().double_value(&[]);
        let max_v = novelty.max().double_value(&[]);
        assert!(min_v >= 0.0);
        assert!(max_v <= 1.0);

        let pred_loss = rnd
            .compute_predictor_loss(&storage_emb)
            .unwrap_or_else(|e| panic!("Predictor loss failed: {e}"));
        assert!(pred_loss.double_value(&[]) >= 0.0);
    }

    #[test]
    fn test_composite_reward_evaluator() {
        let evaluator = CompositeRewardEvaluator::default();
        let r_boundary = Tensor::from_slice(&[0.8f32, 0.2f32]);
        let r_novelty = Tensor::from_slice(&[0.5f32, 0.9f32]);
        let r_exploit = Tensor::from_slice(&[0.0f32, 1.0f32]);
        let m_unsat = Tensor::from_slice(&[1.0f32, 1e-3f32]);

        let rewards = evaluator
            .evaluate_rewards(&r_boundary, &r_novelty, &r_exploit, &m_unsat)
            .unwrap_or_else(|e| panic!("Reward evaluation failed: {e}"));

        assert_eq!(rewards.size(), vec![2]);
        assert_eq!(rewards.kind(), Kind::Float);
        assert!(rewards.double_value(&[0]) > 0.0);
        assert!(rewards.double_value(&[1]) > 0.0);
    }
}
