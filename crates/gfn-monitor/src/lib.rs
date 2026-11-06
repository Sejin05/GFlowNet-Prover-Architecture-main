#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! GFlowNet-Prover Monitor: Distribution drift metrics and Prometheus telemetry.

pub mod drift;
pub mod telemetry;

pub use drift::{ConceptDriftReport, DataDriftReport, DriftDetector, TracingAlert};
pub use telemetry::{init_telemetry, TelemetryHandle, TelemetryRegistry};
