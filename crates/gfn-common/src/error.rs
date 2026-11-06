#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Unified error hierarchy and result alias for GFlowNet-Prover.

use thiserror::Error;

/// Domain error enumeration covering all subsystems of GFlowNet-Prover.
#[derive(Debug, Error)]
pub enum GfnError {
    #[error("[GFN_ERR_101] Foundry compilation or artifact extraction failed: {reason}")]
    CompilationArtifact { reason: String },

    #[error("[GFN_ERR_102] Invalid HSDG topology or non-finite feature tensor: {details}")]
    InvalidGraphTopology { details: String },

    #[error("[GFN_ERR_103] Mathematical domain error or invalid numerical input: {details}")]
    MathDomainError { details: String },

    #[error("[GFN_ERR_104] Data serialization or deserialization failure: {details}")]
    Serialization { details: String },

    #[error("[GFN_ERR_201] RevmArena execution error on env_idx={env_idx}: {message}")]
    RevmExecution { env_idx: usize, message: String },

    #[error("[GFN_ERR_202] Worker thread panicked on env_idx={env_idx}: {msg}")]
    WorkerPanicked { env_idx: usize, msg: String },

    #[error("[GFN_ERR_301] SMT solver timed out after {timeout_ms} ms on branch_pc={branch_pc}")]
    SmtSolverTimeout { branch_pc: usize, timeout_ms: u64 },

    #[error("[GFN_ERR_302] SMT solver execution failure: {details}")]
    SmtSolverError { details: String },

    #[error("[GFN_ERR_401] Candidate exploit rejected by concrete `forge test` gate: {stdout}")]
    FalsePositiveRejected { stdout: String },

    #[error("[GFN_ERR_501] LibTorch tensor error: {message}")]
    Tensor { message: String },

    #[error("[GFN_ERR_502] LMDB persistence error: {message}")]
    Storage { message: String },
}

/// Specialized `Result` type for GFlowNet-Prover operations.
pub type GfnResult<T> = Result<T, GfnError>;
