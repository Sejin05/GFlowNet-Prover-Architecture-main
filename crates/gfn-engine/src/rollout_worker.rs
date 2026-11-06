#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Closed-Loop Rollout Worker & Online Adaptation Controller.
//!
//! Orchestrates the neuro-symbolic search cycle:
//! 1. State encoding via `StateGraphRgatEncoder`
//! 2. Autoregressive action sampling via `AutoregressiveForwardPolicyHead`
//! 3. SMT branch solver handover on narrow arithmetic guards (`branch_distance > 0.85`)
//! 4. Concrete `RevmArena` execution step
//! 5. RND curiosity & composite reward evaluation
//! 6. Huber-clipped Trajectory Balance loss updates with Prioritized Replay
//! 7. GPU OOM resilience and entropy-driven online temperature adaptation.

use tch::nn;
use tch::Device;

use crate::policy::{AutoregressiveForwardPolicyHead, BackwardPolicyHead, LogPartitionNet};
use crate::replay_buffer::PrioritizedReplayBuffer;
use crate::reward_rnd::{CompositeRewardEvaluator, RndCuriosityModule};
use crate::rgat::StateGraphRgatEncoder;
use crate::tb_loss::TrajectoryBalanceLoss;

/// Online Adaptation Controller tracking policy entropy and $\log Z_\theta$ stationarity.
#[derive(Debug)]
pub struct OnlineAdaptationController {
    pub temperature: f64,
    pub novelty_weight: f64,
    pub entropy_threshold: f64,
    pub log_z_history: std::collections::VecDeque<f64>,
    pub novelty_history: std::collections::VecDeque<f64>,
    pub window_size: usize,
}

impl Default for OnlineAdaptationController {
    fn default() -> Self {
        Self::new(1.0, 0.5, 0.35, 100)
    }
}

impl OnlineAdaptationController {
    /// Constructs a new `OnlineAdaptationController`.
    #[must_use]
    pub fn new(
        initial_temp: f64,
        initial_novelty: f64,
        entropy_threshold: f64,
        window_size: usize,
    ) -> Self {
        Self {
            temperature: initial_temp,
            novelty_weight: initial_novelty,
            entropy_threshold,
            log_z_history: std::collections::VecDeque::with_capacity(window_size),
            novelty_history: std::collections::VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Evaluates the entropy collapse guard: if $\bar{H}(P_F) < 0.35\text{ nats}$ before 20% budget,
    /// boosts temperature $\tau \leftarrow \min(2.5, 1.5 \times \tau)$ and $\beta \leftarrow 2.0 \times \beta$.
    /// If entropy recovers above $1.2\text{ nats}$, smoothly decays temperature back toward $1.0$.
    pub fn check_entropy_guard(
        &mut self,
        mean_entropy: f64,
        step: usize,
        max_steps: usize,
    ) -> bool {
        let budget_fraction = (step as f64) / ((max_steps.max(1)) as f64);
        if budget_fraction < 0.20 && mean_entropy < self.entropy_threshold {
            self.temperature = (self.temperature * 1.5).min(2.5);
            self.novelty_weight = (self.novelty_weight * 2.0).min(5.0);
            true
        } else {
            if mean_entropy > 1.2 && self.temperature > 1.0 {
                self.temperature = (self.temperature * 0.98).max(1.0);
                self.novelty_weight = (self.novelty_weight * 0.98).max(0.5);
            }
            false
        }
    }

    /// Records observations of $\log Z_\theta$ and $R_{\text{novelty}}$ into the sliding window.
    pub fn record_telemetry(&mut self, log_z: f64, novelty: f64) {
        if !log_z.is_finite() || !novelty.is_finite() {
            return;
        }
        if self.log_z_history.len() >= self.window_size {
            let _ = self.log_z_history.pop_front();
        }
        self.log_z_history.push_back(log_z);

        if self.novelty_history.len() >= self.window_size {
            let _ = self.novelty_history.pop_front();
        }
        self.novelty_history.push_back(novelty);
    }

    /// Checks the stationarity criterion:
    /// $\frac{\text{StdDev}(\log Z)}{|\text{Mean}(\log Z)| + 1.0} < 0.01 \quad \land \quad \bar{R}_{\text{novelty}} < 0.05$.
    #[must_use]
    pub fn is_stationary(&self) -> bool {
        if self.log_z_history.len() < self.window_size {
            return false;
        }

        let n = self.log_z_history.len() as f64;
        let mean_z = self.log_z_history.iter().sum::<f64>() / n;
        let var_z = self
            .log_z_history
            .iter()
            .map(|z| (z - mean_z).powi(2))
            .sum::<f64>()
            / n;
        let std_z = var_z.sqrt();

        let rel_std = std_z / (mean_z.abs() + 1.0);
        let mean_novelty = self.novelty_history.iter().sum::<f64>() / n;

        rel_std < 0.01 && mean_novelty < 0.05
    }
}

/// Telemetry report emitted per optimization step by `TrainingStabilityController`.
#[derive(Debug, Clone)]
pub struct StabilityStepReport {
    /// Scheduled forward/backward policy learning rate $\eta_\theta(t)$.
    pub lr_policy: f64,
    /// Scheduled log-partition learning rate $\eta_Z(t)$.
    pub lr_log_z: f64,
    /// Exponentially weighted moving average of Trajectory Balance loss.
    pub ema_tb_loss: f64,
    /// Raw L2 gradient norm $\|\nabla_\theta \mathcal{L}\|_2$ before clipping.
    pub raw_grad_norm: f64,
    /// Effective L2 gradient norm after `clip_grad_norm` projection.
    pub clipped_grad_norm: f64,
    /// Whether gradient norm clipping was triggered on this step.
    pub grad_was_clipped: bool,
    /// Fraction of trajectories in the batch entering the linear Huber regime ($|\Delta_{TB}| > \delta_{TB}$).
    pub huber_clipped_frac: f64,
    /// Multiplicative step scaling factor ($0.0$ on NaN guard, $0.5$ on spike damping, $1.0$ normal).
    pub step_scale: f64,
    /// Human-readable stability status badge (`"STABLE"`, `"WARMUP"`, `"GRAD_CLIPPED"`, `"SPIKE_DAMPED"`, `"NAN_GUARD"`).
    pub status_label: &'static str,
}

/// Training Stability Controller managing Cosine Annealing + Linear Warmup learning rates,
/// global L2 gradient norm clipping, EMA residual smoothing, and spike damping.
#[derive(Debug, Clone)]
pub struct TrainingStabilityController {
    /// Maximum L2 gradient norm threshold $G_{\max}$ (default `1.0`).
    pub clip_grad_norm: f64,
    /// Number of linear warmup steps $T_{\text{warmup}}$ (default `100`).
    pub lr_warmup_steps: usize,
    /// Peak policy learning rate $\eta_{\theta,\max}$ (default `3e-4`).
    pub lr_policy_max: f64,
    /// Peak log-partition learning rate $\eta_{Z,\max}$ (default `1e-2`).
    pub lr_log_z_max: f64,
    /// Minimum cosine floor ratio $\eta_{\min} / \eta_{\max}$ (default `0.10`).
    pub lr_min_ratio: f64,
    /// Exponential moving average decay $\beta_{\text{EMA}} \in (0, 1)$ (default `0.95`).
    pub ema_decay: f64,
    /// Running EMA of Trajectory Balance loss.
    pub ema_tb_loss: f64,
    /// Running EMA of L2 gradient norm.
    pub ema_grad_norm: f64,
    /// Total count of steps where gradient norm exceeded `clip_grad_norm`.
    pub grad_clip_events: usize,
    /// Total count of steps where `NaN`/`Inf` was intercepted and skipped.
    pub nan_backoff_events: usize,
    /// Total count of steps where loss spike damping was applied.
    pub spike_damp_events: usize,
    initialized: bool,
}

impl Default for TrainingStabilityController {
    fn default() -> Self {
        Self::new(1.0, 100, 3e-4, 1e-2, 0.95)
    }
}

impl TrainingStabilityController {
    /// Constructs a new `TrainingStabilityController`.
    #[must_use]
    pub fn new(
        clip_grad_norm: f64,
        lr_warmup_steps: usize,
        lr_policy_max: f64,
        lr_log_z_max: f64,
        ema_decay: f64,
    ) -> Self {
        Self {
            clip_grad_norm: clip_grad_norm.max(0.05),
            lr_warmup_steps,
            lr_policy_max: lr_policy_max.max(1e-7),
            lr_log_z_max: lr_log_z_max.max(1e-6),
            lr_min_ratio: 0.10,
            ema_decay: ema_decay.clamp(0.5, 0.999),
            ema_tb_loss: 0.0,
            ema_grad_norm: 0.0,
            grad_clip_events: 0,
            nan_backoff_events: 0,
            spike_damp_events: 0,
            initialized: false,
        }
    }

    /// Computes Linear Warmup + Cosine Annealing learning rates $(\eta_\theta(t), \eta_Z(t))$ at `step`.
    #[must_use]
    pub fn compute_lr_schedule(&self, step: usize, total_steps: usize) -> (f64, f64) {
        let total = total_steps.max(1);
        let warmup = self.lr_warmup_steps.min(total / 2).max(1);
        let mult = if step < warmup {
            ((step + 1) as f64) / (warmup as f64)
        } else {
            let progress = ((step.saturating_sub(warmup)) as f64)
                / ((total.saturating_sub(warmup)).max(1) as f64);
            let clamped_prog = progress.clamp(0.0, 1.0);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * clamped_prog).cos());
            self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine
        };
        (self.lr_policy_max * mult, self.lr_log_z_max * mult)
    }

    /// Evaluates step stability guards, updates EMA trackers, and returns the step telemetry report.
    pub fn observe_and_clip_step(
        &mut self,
        raw_tb_loss: f64,
        raw_grad_norm: f64,
        huber_clipped_frac: f64,
        step: usize,
        total_steps: usize,
    ) -> StabilityStepReport {
        let (lr_policy, lr_log_z) = self.compute_lr_schedule(step, total_steps);
        let clamped_huber_frac = huber_clipped_frac.clamp(0.0, 1.0);

        if !raw_tb_loss.is_finite() || !raw_grad_norm.is_finite() || raw_grad_norm < 0.0 {
            self.nan_backoff_events = self.nan_backoff_events.saturating_add(1);
            return StabilityStepReport {
                lr_policy: lr_policy * 0.5,
                lr_log_z: lr_log_z * 0.5,
                ema_tb_loss: self.ema_tb_loss,
                raw_grad_norm: 0.0,
                clipped_grad_norm: 0.0,
                grad_was_clipped: true,
                huber_clipped_frac: clamped_huber_frac,
                step_scale: 0.0,
                status_label: "NAN_GUARD",
            };
        }

        let mut step_scale = 1.0;
        let mut spike_damped = false;

        if !self.initialized {
            self.ema_tb_loss = raw_tb_loss;
            self.ema_grad_norm = raw_grad_norm.min(self.clip_grad_norm);
            self.initialized = true;
        } else {
            if step >= self.lr_warmup_steps && raw_tb_loss > 2.5 * self.ema_tb_loss.max(0.25) {
                step_scale = 0.5;
                spike_damped = true;
                self.spike_damp_events = self.spike_damp_events.saturating_add(1);
            }
            self.ema_tb_loss =
                self.ema_decay * self.ema_tb_loss + (1.0 - self.ema_decay) * raw_tb_loss;
            self.ema_grad_norm = self.ema_decay * self.ema_grad_norm
                + (1.0 - self.ema_decay) * raw_grad_norm.min(self.clip_grad_norm * 2.0);
        }

        let grad_was_clipped = raw_grad_norm > self.clip_grad_norm;
        if grad_was_clipped {
            self.grad_clip_events = self.grad_clip_events.saturating_add(1);
        }
        let clipped_grad_norm = raw_grad_norm.min(self.clip_grad_norm);

        let status_label = if spike_damped {
            "SPIKE_DAMPED"
        } else if grad_was_clipped {
            "GRAD_CLIPPED"
        } else if step < self.lr_warmup_steps {
            "WARMUP"
        } else {
            "STABLE"
        };

        StabilityStepReport {
            lr_policy: lr_policy * step_scale,
            lr_log_z: lr_log_z * step_scale,
            ema_tb_loss: self.ema_tb_loss,
            raw_grad_norm,
            clipped_grad_norm,
            grad_was_clipped,
            huber_clipped_frac: clamped_huber_frac,
            step_scale,
            status_label,
        }
    }
}

/// Closed-loop neuro-symbolic rollout worker.
pub struct RolloutWorker {
    pub encoder: StateGraphRgatEncoder,
    pub forward_head: AutoregressiveForwardPolicyHead,
    pub backward_head: BackwardPolicyHead,
    pub log_z_net: LogPartitionNet,
    pub tb_loss: TrajectoryBalanceLoss,
    pub replay_buffer: PrioritizedReplayBuffer,
    pub rnd: RndCuriosityModule,
    pub reward_evaluator: CompositeRewardEvaluator,
    pub controller: OnlineAdaptationController,
    pub stability: TrainingStabilityController,
    pub batch_size: usize,
    pub device: Device,
}

impl RolloutWorker {
    /// Constructs a new `RolloutWorker` initializing all neural sub-modules.
    pub fn new(
        vs: &nn::Path,
        hidden_dim: i64,
        num_layers: usize,
        num_heads: i64,
        batch_size: usize,
        replay_capacity: usize,
        device: Device,
    ) -> Self {
        let encoder = StateGraphRgatEncoder::new(&(vs / "rgat"), hidden_dim, num_layers, num_heads);
        let forward_head =
            AutoregressiveForwardPolicyHead::new(&(vs / "forward_policy"), hidden_dim);
        let backward_head = BackwardPolicyHead::new(&(vs / "backward_policy"), hidden_dim, 16);
        let log_z_net = LogPartitionNet::new(&(vs / "log_z"), hidden_dim);
        let tb_loss = TrajectoryBalanceLoss::new(10.0, 1e-6);
        let replay_buffer = PrioritizedReplayBuffer::new(replay_capacity);
        let rnd = RndCuriosityModule::new(&(vs / "rnd"), hidden_dim);
        let reward_evaluator = CompositeRewardEvaluator::default();
        let controller = OnlineAdaptationController::default();
        let stability = TrainingStabilityController::default();

        Self {
            encoder,
            forward_head,
            backward_head,
            log_z_net,
            tb_loss,
            replay_buffer,
            rnd,
            reward_evaluator,
            controller,
            stability,
            batch_size,
            device,
        }
    }

    /// Handles GPU out-of-memory condition safely: dynamically halves batch size.
    pub fn handle_gpu_oom(&mut self) {
        self.batch_size = (self.batch_size / 2).max(8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_adaptation_entropy_and_recovery() {
        let mut ctrl = OnlineAdaptationController::default();
        // Low entropy early in budget boosts temperature and novelty
        let triggered = ctrl.check_entropy_guard(0.20, 10, 1000);
        assert!(triggered);
        assert!(ctrl.temperature > 1.0);
        assert!(ctrl.novelty_weight > 0.5);

        let boosted_temp = ctrl.temperature;
        // High entropy later smoothly decays temperature toward 1.0
        let triggered_later = ctrl.check_entropy_guard(1.60, 300, 1000);
        assert!(!triggered_later);
        assert!(ctrl.temperature < boosted_temp);
    }

    #[test]
    fn test_training_stability_controller_warmup_cosine_and_guards() {
        let mut stab = TrainingStabilityController::new(1.0, 10, 3e-4, 1e-2, 0.90);

        // Step 0 warmup
        let (lr_p0, lr_z0) = stab.compute_lr_schedule(0, 100);
        assert!((lr_p0 - 3e-5).abs() < 1e-9);
        assert!((lr_z0 - 1e-3).abs() < 1e-9);

        // Peak at warmup end (step 9)
        let (lr_p9, _lr_z9) = stab.compute_lr_schedule(9, 100);
        assert!((lr_p9 - 3e-4).abs() < 1e-9);

        // End of cosine schedule (step 100 -> 10% floor)
        let (lr_p100, _lr_z100) = stab.compute_lr_schedule(100, 100);
        assert!((lr_p100 - 3e-5).abs() < 1e-9);

        // Observe normal step
        let rep1 = stab.observe_and_clip_step(1.2, 0.65, 0.05, 12, 100);
        assert_eq!(rep1.status_label, "STABLE");
        assert!(!rep1.grad_was_clipped);
        assert!((rep1.clipped_grad_norm - 0.65).abs() < 1e-9);

        // Observe gradient spike step
        let rep2 = stab.observe_and_clip_step(1.3, 4.2, 0.12, 13, 100);
        assert_eq!(rep2.status_label, "GRAD_CLIPPED");
        assert!(rep2.grad_was_clipped);
        assert!((rep2.clipped_grad_norm - 1.0).abs() < 1e-9);
        assert_eq!(stab.grad_clip_events, 1);

        // Observe loss spike step (> 2.5x EMA)
        let rep3 = stab.observe_and_clip_step(10.0, 0.8, 0.30, 14, 100);
        assert_eq!(rep3.status_label, "SPIKE_DAMPED");
        assert!((rep3.step_scale - 0.5).abs() < 1e-9);

        // Observe NaN guard
        let rep4 = stab.observe_and_clip_step(f64::NAN, 0.8, 0.0, 15, 100);
        assert_eq!(rep4.status_label, "NAN_GUARD");
        assert!((rep4.step_scale - 0.0).abs() < 1e-9);
        assert_eq!(stab.nan_backoff_events, 1);
    }
}
