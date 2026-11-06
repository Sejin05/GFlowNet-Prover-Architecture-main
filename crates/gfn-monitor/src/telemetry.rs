#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Prometheus telemetry and structured tracing initialization for GFlowNet-Prover.
//!
//! Provides the operational `TelemetryRegistry` with strongly-typed metric emission
//! and Prometheus exposition on `:9090/metrics` or custom socket addresses.

use gfn_common::error::{GfnError, GfnResult};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::net::SocketAddr;
use tracing_subscriber::prelude::*;

/// Handle for the initialized telemetry and Prometheus metrics exporter.
#[derive(Clone)]
pub struct TelemetryHandle {
    prometheus_handle: PrometheusHandle,
}

impl TelemetryHandle {
    /// Renders current in-memory metrics in standard Prometheus text format.
    pub fn render(&self) -> String {
        self.prometheus_handle.render()
    }
}

/// Initializes structured logging (tracing) and the Prometheus metrics exporter.
pub fn init_telemetry(
    prometheus_addr: Option<SocketAddr>,
    json_logs: bool,
) -> GfnResult<TelemetryHandle> {
    // 1. Initialize tracing subscriber if not already set globally
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if json_logs {
        let fmt_layer = tracing_subscriber::fmt::layer().json();
        let _ = tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init();
    } else {
        let fmt_layer = tracing_subscriber::fmt::layer();
        let _ = tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init();
    }

    // 2. Initialize Prometheus metrics exporter
    let mut builder = PrometheusBuilder::new();
    if let Some(addr) = prometheus_addr {
        builder = builder.with_http_listener(addr);
    }

    let prometheus_handle = builder
        .install_recorder()
        .map_err(|e| GfnError::Serialization {
            details: format!("Failed to install Prometheus metrics recorder: {e}"),
        })?;

    Ok(TelemetryHandle { prometheus_handle })
}

/// Strongly-typed static registry for recording SLA, resource, and learning metrics.
pub struct TelemetryRegistry;

impl TelemetryRegistry {
    /// Records HSDG extraction duration in seconds (`Histogram`).
    pub fn record_hsdg_extraction_duration(
        seconds: f64,
        solc_version: &str,
        contract_count_bucket: &str,
    ) {
        metrics::histogram!(
            "gfn_hsdg_extraction_duration_seconds",
            "solc_version" => solc_version.to_string(),
            "contract_count_bucket" => contract_count_bucket.to_string()
        )
        .record(seconds);
    }

    /// Records RGAT policy step forward-pass latency in milliseconds (`Histogram`).
    pub fn record_rgat_policy_step_latency(ms: f64, batch_size: usize, device: &str) {
        metrics::histogram!(
            "gfn_rgat_policy_step_latency_ms",
            "batch_size" => batch_size.to_string(),
            "device" => device.to_string()
        )
        .record(ms);
    }

    /// Sets RevmArena batch execution throughput in TPS (`Gauge`).
    pub fn set_revm_batch_throughput(tps: f64, worker_id: usize, cow_depth: usize) {
        metrics::gauge!(
            "gfn_revm_batch_throughput_tps",
            "worker_id" => worker_id.to_string(),
            "cow_depth" => cow_depth.to_string()
        )
        .set(tps);
        metrics::gauge!(
            "gfn_revm_steps_per_sec_per_thread",
            "worker" => worker_id.to_string()
        )
        .set(tps);
    }

    /// Records SMT solver query duration in milliseconds (`Histogram`).
    pub fn record_smt_solve_duration(ms: f64, solver: &str, result: &str) {
        metrics::histogram!(
            "gfn_smt_solve_duration_ms",
            "solver" => solver.to_string(),
            "result" => result.to_string()
        )
        .record(ms);
    }

    /// Sets Unsat core cache hit ratio (`Gauge`).
    pub fn set_unsat_cache_hit_ratio(ratio: f64, cache_tier: &str) {
        metrics::gauge!(
            "gfn_unsat_cache_hit_ratio",
            "cache_tier" => cache_tier.to_string()
        )
        .set(ratio);
        metrics::gauge!("gfn_smt_unsat_cache_hit_rate").set(ratio);
    }

    /// Sets GPU physical VRAM allocated in bytes (`Gauge`).
    pub fn set_gpu_vram_allocated(bytes: f64, gpu_id: usize, phase: &str) {
        metrics::gauge!(
            "gfn_gpu_vram_allocated_bytes",
            "gpu_id" => gpu_id.to_string(),
            "phase" => phase.to_string()
        )
        .set(bytes);
    }

    /// Sets current Trajectory Balance loss (`Gauge`).
    pub fn set_loss_trajectory_balance(loss: f64) {
        metrics::gauge!("gfn_loss_trajectory_balance").set(loss);
    }

    /// Sets moving mean of the Log Partition Network $\log Z_\theta$ (`Gauge`).
    pub fn set_flow_log_z_mean(log_z: f64) {
        metrics::gauge!("gfn_flow_log_z_mean").set(log_z);
    }

    /// Sets Spearman rank correlation between $\log P_F$ and $\log R$ (`Gauge`).
    pub fn set_flow_spearman_rho(rho: f64) {
        metrics::gauge!("gfn_flow_spearman_rho_logpf_logr").set(rho);
    }

    /// Sets trajectory policy entropy in nats (`Gauge`).
    pub fn set_flow_trajectory_entropy(entropy: f64) {
        metrics::gauge!("gfn_flow_trajectory_entropy").set(entropy);
    }

    /// Increments the cumulative number of unique verified exploit trajectories discovered (`Counter`).
    pub fn increment_unique_exploits(count: u64) {
        metrics::counter!("gfn_unique_exploits_found").increment(count);
    }

    /// Increments the cumulative count of candidate trajectories rejected by concrete Forge testing (`Counter`).
    pub fn increment_forge_verification_rejected(count: u64) {
        metrics::counter!("gfn_forge_verification_rejected_total").increment(count);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_telemetry_initialization_and_metric_emission() {
        let handle =
            init_telemetry(None, false).unwrap_or_else(|e| panic!("init_telemetry failed: {e}"));

        TelemetryRegistry::record_hsdg_extraction_duration(3.2, "0.8.24", "1-5");
        TelemetryRegistry::record_rgat_policy_step_latency(4.5, 32, "cpu");
        TelemetryRegistry::set_revm_batch_throughput(72_000.0, 0, 1);
        TelemetryRegistry::record_smt_solve_duration(150.0, "bitwuzla", "sat");
        TelemetryRegistry::set_unsat_cache_hit_ratio(0.75, "l1_dashmap");
        TelemetryRegistry::set_gpu_vram_allocated(1_000_000_000.0, 0, "adaptation");
        TelemetryRegistry::set_loss_trajectory_balance(0.42);
        TelemetryRegistry::set_flow_log_z_mean(12.5);
        TelemetryRegistry::set_flow_spearman_rho(0.85);
        TelemetryRegistry::set_flow_trajectory_entropy(1.42);
        TelemetryRegistry::increment_unique_exploits(1);
        TelemetryRegistry::increment_forge_verification_rejected(0);

        let output = handle.render();
        assert!(output.contains("gfn_revm_batch_throughput_tps"));
        assert!(output.contains("gfn_loss_trajectory_balance 0.42"));
        assert!(output.contains("gfn_flow_log_z_mean 12.5"));
        assert!(output.contains("gfn_flow_spearman_rho_logpf_logr 0.85"));
        assert!(output.contains("gfn_flow_trajectory_entropy 1.42"));
        assert!(output.contains("gfn_unique_exploits_found 1"));
    }
}
