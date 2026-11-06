#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! GFlowNet-Prover Common Foundation: Error protocols, data schemas, and mathematical utilities.

pub mod error;
pub mod math;
pub mod schemas;

pub use error::{GfnError, GfnResult};
pub use math::{
    spearman_rank_correlation, u256_signed_log2_delta, u256_sinusoidal_chunk_encoding,
    u256_to_log2_norm,
};
pub use schemas::{
    AuditResultResponse, AuditStatus, AuditSubmissionRequest, AuditTelemetry,
    ContractDependencyNode, EvaluationErrorKind, FoundryContractArtifact, FoundryProfileConfig,
    HarnessDeploymentContext, InterContractCallRecord, InvariantCategory, ProjectSourceConfig,
    ProtocolSemanticModel, SearchBudgetConfig, SoliditySlotType, StorageSemanticRole,
    StorageSlotRecord, SynthesizedInvariant, TrajectoryRecord, TrajectoryStepRecord,
    VerifiedFinding,
};
