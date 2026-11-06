#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! GFlowNet-Prover CLI & Axum Production Microservice.

pub mod auditor;
pub mod context_bundler;
pub mod server;
pub mod trainer;

pub use auditor::{finding_to_dto, CandidatePolicyScore, ProjectAuditExecution, ProjectAuditor};
pub use context_bundler::ProjectContextBundler;
pub use server::{
    build_router, AuditJob, AuditResultResponse, AuditServerState, AuditStatus,
    AuditSubmissionRequest, StepScoringRequest, StepScoringResponse,
};
pub use trainer::{
    extract_archetype_features, CorpusTrainer, TrainedGfnPolicy, TrainingPhaseSummary,
    VulnerabilityArchetype,
};
