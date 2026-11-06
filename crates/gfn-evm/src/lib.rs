#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! GFlowNet-Prover EVM: High-speed in-memory RevmArena and Foundry cheatcodes.

pub mod arena;
pub mod cheatcodes;
pub mod inspector;

pub use arena::{
    PinnedStateBuffer, RevmArena, StepBatchInput, StepBatchOutput, WorkerEnvironment,
    CANCUN_CHAIN_ID, CANCUN_INITIAL_BLOCK_NUMBER, CANCUN_INITIAL_BLOCK_TIMESTAMP,
};
pub use cheatcodes::{ActorPersona, CheatcodeState, PrankRecord, CHEATCODE_ADDRESS};
pub use inspector::{BranchTelemetry, GfnEvmInspector, KeccakPreimageRecord, StorageAccessRecord};
