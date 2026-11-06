#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Operational data drift and concept drift detection for GFlowNet-Prover.
//!
//! Implements quantitative statistical distance tests:
//! 1. Spectral Maximum Mean Discrepancy ($\text{MMD}^2$) over graph embeddings.
//! 2. Population Stability Index ($\text{PSI}$) over opcode frequency histograms.
//! 3. Jensen-Shannon Divergence ($D_{JS}$) over storage slot types.
//! 4. Two-Sample Kolmogorov-Smirnov Test ($D_{KS}$) over step-0 RND prediction errors.
//!
//! Also implements concept drift detection:
//! - Trajectory Balance residual surge ($|\Delta_{TB}| > 2.5$)
//! - Spearman flow-reward rank correlation degradation ($\rho_{TB} < 0.72 / 0.60$)
//! - SMT solver timeout and saturation efficacy monitoring
//! - Revm-to-Forge concrete verification gate rejection tracking (strict zero-tolerance)

use gfn_common::error::{GfnError, GfnResult};
use serde::{Deserialize, Serialize};

/// High-priority operational alerting events emitted by drift monitors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TracingAlert {
    /// Spectral MMD exceeds warning threshold (> 0.08).
    DataDriftMmdWarning { mmd_squared: f64 },
    /// Spectral MMD exceeds critical retraining threshold (> 0.15).
    DataDriftMmdCritical { mmd_squared: f64 },
    /// Opcode PSI exceeds warning threshold (> 0.15).
    DataDriftPsiWarning { psi: f64 },
    /// Opcode PSI exceeds critical retraining threshold (> 0.25).
    DataDriftPsiCritical { psi: f64 },
    /// Storage slot JS divergence exceeds warning threshold (> 0.10).
    DataDriftJsDivergenceWarning { js_divergence: f64 },
    /// Storage slot JS divergence exceeds critical retraining threshold (> 0.20).
    DataDriftJsDivergenceCritical { js_divergence: f64 },
    /// RND KS-statistic exceeds warning threshold (> 0.18).
    DataDriftKsWarning { ks_statistic: f64 },
    /// RND KS-statistic exceeds critical retraining threshold (> 0.30).
    DataDriftKsCritical { ks_statistic: f64 },
    /// Trajectory Balance residual moving median exceeds 2.5 after adaptation.
    ConceptDriftFlowMisalignment {
        median_abs_tb: f64,
        adaptation_steps: usize,
    },
    /// Spearman rank correlation rho degraded below warning threshold (< 0.72).
    ConceptDriftSpearmanWarning { rho: f64 },
    /// Spearman rank correlation rho degraded below critical threshold (< 0.60).
    ConceptDriftSpearmanCritical { rho: f64 },
    /// SMT solver handover efficacy outside normal bounds (> 18% timeout or < 3% sat).
    SymbolicConcretizerCalibration { timeout_rate: f64, sat_rate: f64 },
    /// Revm vs Forge specification mismatch: Candidate passed Revm but failed concrete Forge gate.
    P1RevmForgeSpecMismatch { rejections: u64 },
}

/// Comprehensive report containing statistical distance metrics and active alerts for data drift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDriftReport {
    pub mmd_squared: f64,
    pub psi: f64,
    pub js_divergence: f64,
    pub ks_statistic: f64,
    pub is_warning: bool,
    pub requires_retraining: bool,
    pub alerts: Vec<TracingAlert>,
}

/// Comprehensive report containing concept drift and neuro-symbolic alignment indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptDriftReport {
    pub median_abs_tb: f64,
    pub spearman_rho: f64,
    pub smt_timeout_rate: f64,
    pub smt_sat_rate: f64,
    pub revm_forge_rejections: u64,
    pub is_warning: bool,
    pub is_critical: bool,
    pub alerts: Vec<TracingAlert>,
}

/// Operational drift detector evaluating mathematical statistical shifts against reference distributions.
pub struct DriftDetector;

impl DriftDetector {
    /// Computes Maximum Mean Discrepancy ($\text{MMD}^2$) with Radial Basis Function (RBF) kernel:
    /// $$k(x, y) = \exp\left(-\frac{\|x - y\|_2^2}{2\sigma^2}\right)$$
    ///
    /// Evaluated over multidimensional feature/degree vectors $P$ (production) vs $Q$ (reference).
    pub fn compute_mmd_rbf(
        samples_p: &[Vec<f64>],
        samples_q: &[Vec<f64>],
        sigma: f64,
    ) -> GfnResult<f64> {
        if samples_p.is_empty() || samples_q.is_empty() {
            return Err(GfnError::MathDomainError {
                details: "Cannot compute MMD over empty sample sets".to_string(),
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(GfnError::MathDomainError {
                details: format!(
                    "RBF bandwidth sigma must be finite and strictly positive, got {sigma}"
                ),
            });
        }

        let m = samples_p.len();
        let n = samples_q.len();
        let two_sigma_sq = 2.0 * sigma * sigma;

        // Sum over k(x_i, x_j)
        let mut k_pp_sum = 0.0;
        for pi in samples_p {
            for pj in samples_p {
                k_pp_sum += rbf_kernel(pi, pj, two_sigma_sq)?;
            }
        }
        let term_pp = k_pp_sum / (m as f64 * m as f64);

        // Sum over k(y_i, y_j)
        let mut k_qq_sum = 0.0;
        for qi in samples_q {
            for qj in samples_q {
                k_qq_sum += rbf_kernel(qi, qj, two_sigma_sq)?;
            }
        }
        let term_qq = k_qq_sum / (n as f64 * n as f64);

        // Sum over k(x_i, y_j)
        let mut k_pq_sum = 0.0;
        for pi in samples_p {
            for qj in samples_q {
                k_pq_sum += rbf_kernel(pi, qj, two_sigma_sq)?;
            }
        }
        let term_pq = 2.0 * k_pq_sum / (m as f64 * n as f64);

        let mmd_sq = term_pp + term_qq - term_pq;
        // Clamp numerical precision artifacts near 0.0
        Ok(mmd_sq.max(0.0))
    }

    /// Computes Population Stability Index ($\text{PSI}$) between probability distributions $P$ and $Q$:
    /// $$\text{PSI}(P, Q) = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b + \epsilon}{Q_b + \epsilon}\right)$$
    pub fn compute_psi(dist_p: &[f64], dist_q: &[f64]) -> GfnResult<f64> {
        if dist_p.len() != dist_q.len() || dist_p.is_empty() {
            return Err(GfnError::MathDomainError {
                details: format!(
                    "Mismatched or empty distribution lengths for PSI: P={}, Q={}",
                    dist_p.len(),
                    dist_q.len()
                ),
            });
        }

        if dist_p.iter().any(|x| !x.is_finite() || *x < 0.0)
            || dist_q.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(GfnError::MathDomainError {
                details: "PSI distribution bins must be finite and non-negative".to_string(),
            });
        }

        let sum_p: f64 = dist_p.iter().sum();
        let sum_q: f64 = dist_q.iter().sum();

        if !sum_p.is_finite() || !sum_q.is_finite() || sum_p <= 0.0 || sum_q <= 0.0 {
            return Err(GfnError::MathDomainError {
                details: "Distribution sum must be strictly positive".to_string(),
            });
        }

        let eps = 1e-6;
        let mut psi = 0.0;

        for (p_raw, q_raw) in dist_p.iter().zip(dist_q.iter()) {
            let p_norm = p_raw / sum_p;
            let q_norm = q_raw / sum_q;

            let diff = p_norm - q_norm;
            let ratio = (p_norm + eps) / (q_norm + eps);
            psi += diff * ratio.ln();
        }

        Ok(psi.max(0.0))
    }

    /// Computes Jensen-Shannon Divergence ($D_{JS}$) between distributions $P$ and $Q$:
    /// $$M = \frac{1}{2}(P + Q)$$
    /// $$D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
    pub fn compute_js_divergence(dist_p: &[f64], dist_q: &[f64]) -> GfnResult<f64> {
        if dist_p.len() != dist_q.len() || dist_p.is_empty() {
            return Err(GfnError::MathDomainError {
                details: format!(
                    "Mismatched or empty distribution lengths for JS Divergence: P={}, Q={}",
                    dist_p.len(),
                    dist_q.len()
                ),
            });
        }

        if dist_p.iter().any(|x| !x.is_finite() || *x < 0.0)
            || dist_q.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(GfnError::MathDomainError {
                details: "PSI distribution bins must be finite and non-negative".to_string(),
            });
        }

        let sum_p: f64 = dist_p.iter().sum();
        let sum_q: f64 = dist_q.iter().sum();

        if !sum_p.is_finite() || !sum_q.is_finite() || sum_p <= 0.0 || sum_q <= 0.0 {
            return Err(GfnError::MathDomainError {
                details: "Distribution sum must be strictly positive".to_string(),
            });
        }

        let n = dist_p.len();
        let mut p_norm = Vec::with_capacity(n);
        let mut q_norm = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);

        for i in 0..n {
            let p_val = (dist_p[i] / sum_p).max(0.0);
            let q_val = (dist_q[i] / sum_q).max(0.0);
            p_norm.push(p_val);
            q_norm.push(q_val);
            m.push(0.5 * (p_val + q_val));
        }

        let kl_pm = kl_divergence(&p_norm, &m)?;
        let kl_qm = kl_divergence(&q_norm, &m)?;

        let js = 0.5 * kl_pm + 0.5 * kl_qm;
        Ok(js.max(0.0))
    }

    /// Computes Two-Sample Kolmogorov-Smirnov Test ($D_{KS}$):
    /// $$D_{KS} = \sup_x | F_P(x) - F_Q(x) |$$
    pub fn compute_two_sample_ks(samples_p: &[f64], samples_q: &[f64]) -> GfnResult<f64> {
        if samples_p.is_empty() || samples_q.is_empty() {
            return Err(GfnError::MathDomainError {
                details: "Cannot compute KS-test over empty samples".to_string(),
            });
        }
        if samples_p.iter().any(|x| !x.is_finite()) || samples_q.iter().any(|x| !x.is_finite()) {
            return Err(GfnError::MathDomainError {
                details: "KS-test samples must contain only finite f64 values".to_string(),
            });
        }

        let mut p_sorted = samples_p.to_vec();
        let mut q_sorted = samples_q.to_vec();

        p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let m = p_sorted.len();
        let n = q_sorted.len();

        let mut max_diff: f64 = 0.0;
        let mut idx_p = 0;
        let mut idx_q = 0;

        while idx_p < m || idx_q < n {
            let x = if idx_p < m && idx_q < n {
                if p_sorted[idx_p] <= q_sorted[idx_q] {
                    p_sorted[idx_p]
                } else {
                    q_sorted[idx_q]
                }
            } else if idx_p < m {
                p_sorted[idx_p]
            } else {
                q_sorted[idx_q]
            };

            while idx_p < m && p_sorted[idx_p] <= x {
                idx_p += 1;
            }
            while idx_q < n && q_sorted[idx_q] <= x {
                idx_q += 1;
            }

            let cdf_p = idx_p as f64 / m as f64;
            let cdf_q = idx_q as f64 / n as f64;
            let diff = (cdf_p - cdf_q).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        Ok(max_diff)
    }

    /// Evaluates complete Data Drift suite across graph degrees, opcodes, storage slots, and RND errors.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_data_drift(
        prod_degrees: &[Vec<f64>],
        ref_degrees: &[Vec<f64>],
        prod_opcodes: &[f64],
        ref_opcodes: &[f64],
        prod_slots: &[f64],
        ref_slots: &[f64],
        prod_rnd: &[f64],
        ref_rnd: &[f64],
    ) -> GfnResult<DataDriftReport> {
        let mmd_squared = Self::compute_mmd_rbf(prod_degrees, ref_degrees, 1.0)?;
        let psi = Self::compute_psi(prod_opcodes, ref_opcodes)?;
        let js_divergence = Self::compute_js_divergence(prod_slots, ref_slots)?;
        let ks_statistic = Self::compute_two_sample_ks(prod_rnd, ref_rnd)?;

        let mut alerts = Vec::new();
        let mut is_warning = false;
        let mut requires_retraining = false;

        // 1. MMD Thresholds: Warning > 0.08, Critical > 0.15
        if mmd_squared > 0.15 {
            requires_retraining = true;
            alerts.push(TracingAlert::DataDriftMmdCritical { mmd_squared });
        } else if mmd_squared > 0.08 {
            is_warning = true;
            alerts.push(TracingAlert::DataDriftMmdWarning { mmd_squared });
        }

        // 2. PSI Thresholds: Warning > 0.15, Critical > 0.25
        if psi > 0.25 {
            requires_retraining = true;
            alerts.push(TracingAlert::DataDriftPsiCritical { psi });
        } else if psi > 0.15 {
            is_warning = true;
            alerts.push(TracingAlert::DataDriftPsiWarning { psi });
        }

        // 3. JS Divergence Thresholds: Warning > 0.10, Critical > 0.20
        if js_divergence > 0.20 {
            requires_retraining = true;
            alerts.push(TracingAlert::DataDriftJsDivergenceCritical { js_divergence });
        } else if js_divergence > 0.10 {
            is_warning = true;
            alerts.push(TracingAlert::DataDriftJsDivergenceWarning { js_divergence });
        }

        // 4. KS Statistic Thresholds: Warning > 0.18, Critical > 0.30
        if ks_statistic > 0.30 {
            requires_retraining = true;
            alerts.push(TracingAlert::DataDriftKsCritical { ks_statistic });
        } else if ks_statistic > 0.18 {
            is_warning = true;
            alerts.push(TracingAlert::DataDriftKsWarning { ks_statistic });
        }

        Ok(DataDriftReport {
            mmd_squared,
            psi,
            js_divergence,
            ks_statistic,
            is_warning,
            requires_retraining,
            alerts,
        })
    }

    /// Evaluates Concept Drift indicators across Trajectory Balance residuals, Spearman correlation,
    /// SMT solver handovers, and concrete verification gate rejections.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_concept_drift(
        abs_tb_residuals: &[f64],
        adaptation_steps: usize,
        log_pfs: &[f64],
        log_rewards: &[f64],
        smt_queries: u64,
        smt_sat: u64,
        _smt_unsat: u64,
        smt_timeout: u64,
        revm_forge_rejections: u64,
    ) -> GfnResult<ConceptDriftReport> {
        let mut alerts = Vec::new();
        let mut is_warning = false;
        let mut is_critical = false;

        // 1. Trajectory Balance Residual Surge
        let median_abs_tb = if abs_tb_residuals.is_empty() {
            0.0
        } else {
            let mut sorted = abs_tb_residuals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                0.5 * (sorted[mid - 1] + sorted[mid])
            } else {
                sorted[mid]
            }
        };

        if adaptation_steps >= 1_000 && median_abs_tb > 2.5 {
            is_critical = true;
            alerts.push(TracingAlert::ConceptDriftFlowMisalignment {
                median_abs_tb,
                adaptation_steps,
            });
        }

        // 2. Spearman Flow-Reward Rank Correlation
        let spearman_rho = if log_pfs.len() >= 2 && log_pfs.len() == log_rewards.len() {
            gfn_common::math::spearman_rank_correlation(log_pfs, log_rewards)?
        } else {
            1.0
        };

        if spearman_rho < 0.60 {
            is_critical = true;
            alerts.push(TracingAlert::ConceptDriftSpearmanCritical { rho: spearman_rho });
        } else if spearman_rho < 0.72 {
            is_warning = true;
            alerts.push(TracingAlert::ConceptDriftSpearmanWarning { rho: spearman_rho });
        }

        // 3. SMT Solver Handover Efficacy
        let smt_timeout_rate = if smt_queries > 0 {
            smt_timeout as f64 / smt_queries as f64
        } else {
            0.0
        };
        let smt_sat_rate = if smt_queries > 0 {
            smt_sat as f64 / smt_queries as f64
        } else {
            1.0
        };

        if smt_queries >= 20 && (smt_timeout_rate > 0.18 || smt_sat_rate < 0.03) {
            is_warning = true;
            alerts.push(TracingAlert::SymbolicConcretizerCalibration {
                timeout_rate: smt_timeout_rate,
                sat_rate: smt_sat_rate,
            });
        }

        // 4. Revm vs Forge Specification Mismatch (Strict Zero Tolerance)
        if revm_forge_rejections > 0 {
            is_critical = true;
            alerts.push(TracingAlert::P1RevmForgeSpecMismatch {
                rejections: revm_forge_rejections,
            });
        }

        Ok(ConceptDriftReport {
            median_abs_tb,
            spearman_rho,
            smt_timeout_rate,
            smt_sat_rate,
            revm_forge_rejections,
            is_warning,
            is_critical,
            alerts,
        })
    }
}

/// Helper function computing scalar RBF kernel between vectors $x$ and $y$.
fn rbf_kernel(x: &[f64], y: &[f64], two_sigma_sq: f64) -> GfnResult<f64> {
    if x.len() != y.len() {
        return Err(GfnError::MathDomainError {
            details: format!("Mismatched feature dimensions: {} != {}", x.len(), y.len()),
        });
    }
    let mut dist_sq = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(GfnError::MathDomainError {
                details: "RBF kernel feature vectors must contain only finite values".to_string(),
            });
        }
        let d = xi - yi;
        dist_sq += d * d;
    }
    Ok((-dist_sq / two_sigma_sq).exp())
}

/// Helper function computing Kullback-Leibler divergence $D_{KL}(P \parallel Q)$.
fn kl_divergence(p: &[f64], q: &[f64]) -> GfnResult<f64> {
    let eps = 1e-12;
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            let ratio = (pi + eps) / (qi + eps);
            kl += pi * ratio.ln();
        }
    }
    Ok(kl)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_identical_distributions_yield_zero_drift() {
        let p_degrees = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let q_degrees = p_degrees.clone();

        let p_opcodes = vec![0.1, 0.2, 0.3, 0.4];
        let q_opcodes = p_opcodes.clone();

        let p_slots = vec![0.5, 0.25, 0.25];
        let q_slots = p_slots.clone();

        let p_rnd = vec![0.1, 0.5, 1.2, 2.0];
        let q_rnd = p_rnd.clone();

        let report = DriftDetector::evaluate_data_drift(
            &p_degrees, &q_degrees, &p_opcodes, &q_opcodes, &p_slots, &q_slots, &p_rnd, &q_rnd,
        )
        .unwrap_or_else(|e| panic!("evaluate_data_drift failed: {e}"));

        assert!(
            report.mmd_squared < 1e-6,
            "MMD^2 must be zero for identical sets"
        );
        assert!(
            report.psi < 1e-5,
            "PSI must be zero for identical distributions"
        );
        assert!(
            report.js_divergence < 1e-5,
            "JS divergence must be zero for identical distributions"
        );
        assert!(
            report.ks_statistic < 1e-5,
            "KS statistic must be zero for identical samples"
        );
        assert!(!report.is_warning);
        assert!(!report.requires_retraining);
        assert!(report.alerts.is_empty());
    }

    #[test]
    fn test_shifted_distributions_trigger_critical_alerts() {
        // Shifted degrees
        let p_degrees = vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]];
        let q_degrees = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        // Heavily shifted opcodes
        let p_opcodes = vec![0.9, 0.03, 0.04, 0.03];
        let q_opcodes = vec![0.05, 0.8, 0.1, 0.05];

        // Shifted storage slots
        let p_slots = vec![0.95, 0.03, 0.02];
        let q_slots = vec![0.05, 0.45, 0.50];

        // Shifted RND prediction errors
        let p_rnd = vec![10.0, 12.0, 15.0, 20.0];
        let q_rnd = vec![0.1, 0.2, 0.3, 0.4];

        let report = DriftDetector::evaluate_data_drift(
            &p_degrees, &q_degrees, &p_opcodes, &q_opcodes, &p_slots, &q_slots, &p_rnd, &q_rnd,
        )
        .unwrap_or_else(|e| panic!("evaluate_data_drift failed: {e}"));

        assert!(
            report.mmd_squared > 0.15,
            "Expected critical MMD^2: got {}",
            report.mmd_squared
        );
        assert!(
            report.psi > 0.25,
            "Expected critical PSI: got {}",
            report.psi
        );
        assert!(
            report.js_divergence > 0.20,
            "Expected critical JS: got {}",
            report.js_divergence
        );
        assert!(
            report.ks_statistic > 0.30,
            "Expected critical KS: got {}",
            report.ks_statistic
        );
        assert!(report.requires_retraining);
        assert!(report.alerts.len() >= 4);
    }

    #[test]
    fn test_concept_drift_monitor_alerts() {
        // High TB residual after 1,000 adaptation steps
        let abs_residuals = vec![3.0, 3.5, 2.8, 4.0, 3.2];
        let log_pfs = vec![-1.0, -2.0, -3.0, -4.0];
        // Inverted reward ranking (negative correlation)
        let log_rewards = vec![-4.0, -3.0, -2.0, -1.0];

        let report = DriftDetector::evaluate_concept_drift(
            &abs_residuals,
            1_200,
            &log_pfs,
            &log_rewards,
            100, // smt queries
            2,   // smt sat (< 3%)
            70,
            25, // smt timeout (> 18%)
            1,  // revm forge rejection (> 0)
        )
        .unwrap_or_else(|e| panic!("evaluate_concept_drift failed: {e}"));

        assert!(report.is_critical);
        assert!(report
            .alerts
            .iter()
            .any(|a| matches!(a, TracingAlert::ConceptDriftFlowMisalignment { .. })));
        assert!(report
            .alerts
            .iter()
            .any(|a| matches!(a, TracingAlert::ConceptDriftSpearmanCritical { .. })));
        assert!(report
            .alerts
            .iter()
            .any(|a| matches!(a, TracingAlert::SymbolicConcretizerCalibration { .. })));
        assert!(report
            .alerts
            .iter()
            .any(|a| matches!(a, TracingAlert::P1RevmForgeSpecMismatch { .. })));
    }

    #[test]
    fn test_nan_and_negative_inputs_rejected_without_hanging() {
        assert!(DriftDetector::compute_two_sample_ks(&[f64::NAN, 1.0], &[1.0, 2.0]).is_err());
        assert!(DriftDetector::compute_two_sample_ks(&[1.0, 2.0], &[f64::INFINITY]).is_err());
        assert!(DriftDetector::compute_mmd_rbf(&[vec![f64::NAN]], &[vec![1.0]], 1.0).is_err());
        assert!(DriftDetector::compute_mmd_rbf(&[vec![1.0]], &[vec![1.0]], f64::NAN).is_err());
        assert!(DriftDetector::compute_psi(&[-0.2, 1.2], &[0.5, 0.5]).is_err());
        assert!(DriftDetector::compute_psi(&[f64::NAN, 1.0], &[0.5, 0.5]).is_err());
        assert!(DriftDetector::compute_js_divergence(&[-0.1, 1.1], &[0.5, 0.5]).is_err());
    }
}
