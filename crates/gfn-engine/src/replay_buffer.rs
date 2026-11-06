#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Lock-free / RWLock Prioritized Replay Buffer with Fenwick / Binary Sum-Tree.
//!
//! Stores multi-step transaction trajectories with priority $p_i = \left(\max(R(\tau_i), |\Delta_{TB}(\tau_i)|) + 10^{-5}\right)^{\alpha_{\text{PER}}}$.
//! Supports $O(\log N)$ proportional stratified batch sampling and $O(\log N)$ priority updates.

use gfn_common::error::{GfnError, GfnResult};
use gfn_common::schemas::TrajectoryRecord;
use parking_lot::RwLock;
use rand::Rng;

/// Binary Sum-Tree for $O(\log N)$ proportional priority sampling.
#[derive(Debug, Clone)]
pub struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
}

impl SumTree {
    /// Constructs a new `SumTree` with the given leaf capacity (rounded up to power of two).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(1);
        Self {
            capacity: cap,
            tree: vec![0.0; 2 * cap],
        }
    }

    /// Updates the priority of the leaf at `index` in $O(\log N)$ using exact child summation
    /// to prevent cumulative floating-point cancellation drift over long training runs.
    pub fn update(&mut self, index: usize, priority: f64) {
        if index >= self.capacity {
            return;
        }
        let safe_p = if priority.is_finite() {
            priority.clamp(0.0, 1e6)
        } else {
            1e-5
        };
        let mut tree_idx = self.capacity + index;
        self.tree[tree_idx] = safe_p;

        while tree_idx > 1 {
            tree_idx /= 2;
            let left = self.tree[2 * tree_idx];
            let right = self.tree[2 * tree_idx + 1];
            self.tree[tree_idx] = (left + right).max(0.0);
        }
    }

    /// Returns the total accumulated priority sum at the root.
    #[must_use]
    pub fn total(&self) -> f64 {
        if self.tree.len() > 1 {
            self.tree[1].max(0.0)
        } else {
            0.0
        }
    }

    /// Finds the leaf index corresponding to cumulative priority value `value` in $O(\log N)$.
    #[must_use]
    pub fn find_prefix_sum(&self, mut value: f64) -> usize {
        if self.capacity == 0 || self.total() <= 0.0 {
            return 0;
        }
        let mut idx = 1;
        while idx < self.capacity {
            let left_child = 2 * idx;
            let left_val = self.tree[left_child];
            if value <= left_val {
                idx = left_child;
            } else {
                value -= left_val;
                idx = left_child + 1;
            }
        }
        (idx - self.capacity).min(self.capacity - 1)
    }

    /// Returns the priority stored at `index`.
    #[must_use]
    pub fn get_priority(&self, index: usize) -> f64 {
        if index < self.capacity {
            self.tree[self.capacity + index]
        } else {
            0.0
        }
    }
}

/// Thread-safe Prioritized Experience Replay Buffer for trajectory rollouts.
pub struct PrioritizedReplayBuffer {
    capacity: usize,
    storage: RwLock<Vec<Option<TrajectoryRecord>>>,
    tree: RwLock<SumTree>,
    write_idx: RwLock<usize>,
    size: RwLock<usize>,
    alpha_per: f64,
    beta_is: RwLock<f64>,
}

impl PrioritizedReplayBuffer {
    /// Creates a new `PrioritizedReplayBuffer` with the specified trajectory capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(16);
        Self {
            capacity: cap,
            storage: RwLock::new(vec![None; cap]),
            tree: RwLock::new(SumTree::new(cap)),
            write_idx: RwLock::new(0),
            size: RwLock::new(0),
            alpha_per: 0.6,
            beta_is: RwLock::new(0.4),
        }
    }

    /// Returns the current number of trajectories stored in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        *self.size.read()
    }

    /// Returns whether the replay buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Computes priority $p = \left(\max(R, |\Delta_{TB}|) + 10^{-5}\right)^{\alpha_{\text{PER}}}$
    /// with finite-value and upper-bound clamping to prevent SumTree priority skew.
    #[must_use]
    pub fn compute_priority(&self, reward: f64, abs_residual: f64) -> f64 {
        let safe_r = if reward.is_finite() {
            reward.clamp(0.0, 1e4)
        } else {
            0.0
        };
        let safe_res = if abs_residual.is_finite() {
            abs_residual.clamp(0.0, 100.0)
        } else {
            0.0
        };
        let base = safe_r.max(safe_res) + 1e-5;
        base.powf(self.alpha_per)
    }

    /// Inserts a new trajectory with its initial absolute TB residual into the buffer.
    pub fn push(&self, trajectory: TrajectoryRecord, abs_residual: f64) {
        let priority = self.compute_priority(trajectory.terminal_reward, abs_residual);

        let mut write_guard = self.write_idx.write();
        let idx = *write_guard;
        *write_guard = (*write_guard + 1) % self.capacity;
        drop(write_guard);

        let mut size_guard = self.size.write();
        if *size_guard < self.capacity {
            *size_guard += 1;
        }
        drop(size_guard);

        self.storage.write()[idx] = Some(trajectory);
        self.tree.write().update(idx, priority);
    }

    /// Samples a stratified batch of trajectories with normalized importance sampling weights.
    pub fn sample_batch<R: Rng>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> GfnResult<(Vec<usize>, Vec<TrajectoryRecord>, Vec<f32>)> {
        let cur_size = self.len();
        if cur_size == 0 {
            return Err(GfnError::MathDomainError {
                details: "Cannot sample from an empty PrioritizedReplayBuffer".to_string(),
            });
        }

        let actual_batch_size = batch_size.min(cur_size);
        let tree_guard = self.tree.read();
        let total_p = tree_guard.total().max(1e-8);
        let segment = total_p / (actual_batch_size as f64);

        let mut sampled_indices = Vec::with_capacity(actual_batch_size);
        let mut sampled_trajectories = Vec::with_capacity(actual_batch_size);
        let mut raw_weights = Vec::with_capacity(actual_batch_size);

        let beta = *self.beta_is.read();
        let storage_guard = self.storage.read();

        for i in 0..actual_batch_size {
            let lower = (i as f64) * segment;
            let upper = ((i + 1) as f64) * segment;
            let val = rng.gen_range(lower..upper);

            let leaf_idx = tree_guard.find_prefix_sum(val).min(self.capacity - 1);
            let bounded_idx = if leaf_idx < cur_size {
                leaf_idx
            } else {
                leaf_idx % cur_size
            };

            if let Some(record) = &storage_guard[bounded_idx] {
                sampled_indices.push(bounded_idx);
                sampled_trajectories.push(record.clone());

                let p = tree_guard.get_priority(bounded_idx) / total_p;
                let prob = p.max(1e-8);
                let w = ((cur_size as f64) * prob).powf(-beta);
                raw_weights.push(w as f32);
            }
        }

        drop(storage_guard);
        drop(tree_guard);

        if sampled_trajectories.is_empty() {
            return Err(GfnError::MathDomainError {
                details: "PrioritizedReplayBuffer failed to extract any trajectories".to_string(),
            });
        }

        let max_w = raw_weights
            .iter()
            .copied()
            .fold(0.0f32, |acc, w| acc.max(w))
            .max(1e-8);

        let norm_weights = raw_weights.into_iter().map(|w| w / max_w).collect();

        Ok((sampled_indices, sampled_trajectories, norm_weights))
    }

    /// Updates priorities for a batch of sampled indices after re-evaluating residuals.
    pub fn update_priorities(
        &self,
        indices: &[usize],
        new_residuals: &[f64],
        new_rewards: Option<&[f64]>,
    ) {
        let storage_guard = self.storage.read();
        let mut tree_guard = self.tree.write();

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.capacity {
                continue;
            }
            let res = new_residuals.get(i).copied().unwrap_or(0.0);
            let rew = if let Some(rewards) = new_rewards {
                rewards.get(i).copied().unwrap_or(0.0)
            } else if let Some(rec) = &storage_guard[idx] {
                rec.terminal_reward
            } else {
                0.0
            };

            let new_p = self.compute_priority(rew, res);
            tree_guard.update(idx, new_p);
        }
    }

    /// Anneals the importance sampling exponent $\beta \leftarrow \min(1.0, \beta + \Delta\beta)$.
    pub fn step_beta(&self, delta: f64) {
        let mut beta = self.beta_is.write();
        *beta = (*beta + delta).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_sum_tree_prefix_sum() {
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert_eq!(tree.total(), 10.0);
        assert_eq!(tree.find_prefix_sum(0.5), 0);
        assert_eq!(tree.find_prefix_sum(1.5), 1);
        assert_eq!(tree.find_prefix_sum(3.5), 2);
        assert_eq!(tree.find_prefix_sum(7.0), 3);
    }

    #[test]
    fn test_prioritized_replay_buffer_sampling_and_update() {
        let buffer = PrioritizedReplayBuffer::new(16);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for i in 0..5 {
            let record = TrajectoryRecord {
                trajectory_id: format!("traj_{i}"),
                steps: vec![],
                terminal_reward: (i as f64) * 10.0 + 1.0,
                violated_invariant_id: None,
                is_exploit: i == 4,
                log_pf: -2.0,
                log_pb: 0.0,
            };
            buffer.push(record, 1.5);
        }

        assert_eq!(buffer.len(), 5);

        let sample_res = buffer.sample_batch(3, &mut rng);
        assert!(sample_res.is_ok());
        if let Ok((indices, records, weights)) = sample_res {
            assert_eq!(indices.len(), 3);
            assert_eq!(records.len(), 3);
            assert_eq!(weights.len(), 3);
            for w in weights {
                assert!(w > 0.0 && w <= 1.0);
            }
            buffer.update_priorities(&indices, &[0.5, 0.2, 0.1], None);
        }
    }
}
