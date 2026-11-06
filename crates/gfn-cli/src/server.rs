#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Production HTTP, Server-Sent Events (SSE), and micro-batching server for GFlowNet-Prover.
//!
//! Provides:
//! - `POST /v1/audits`: Submit asynchronous audit request.
//! - `GET /v1/audits/:id`: Retrieve live status, invariants, and verified findings.
//! - `GET /v1/audits/:id/stream`: SSE real-time event stream.
//! - `POST /v1/policy/sample-step`: Dynamic 2.0ms micro-batching policy scoring.
//! - `GET /openapi.json`: OpenAPI 3.1 specification.
//! - `GET /health`: Liveness probe.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use gfn_common::error::{GfnError, GfnResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use utoipa::{OpenApi, ToSchema};
use uuid::Uuid;

// --- DTO & Schema Definitions ---

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ProjectSourceDto {
    pub workspace_path: String,
    #[serde(default = "default_setup_contract")]
    pub setup_contract: String,
    #[serde(default = "default_setup_function")]
    pub setup_function: String,
    #[serde(default)]
    pub target_contracts: Option<Vec<String>>,
}

fn default_setup_contract() -> String {
    "ExploitHarnessTest".to_string()
}

fn default_setup_function() -> String {
    "setUp".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FoundryProfileDto {
    #[serde(default = "default_solc_version")]
    pub solc_version: String,
    #[serde(default = "default_evm_version")]
    pub evm_version: String,
    #[serde(default)]
    pub fork_block_number: Option<u64>,
}

fn default_solc_version() -> String {
    "0.8.24".to_string()
}

fn default_evm_version() -> String {
    "cancun".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchBudgetDto {
    #[serde(default = "default_max_trajectories")]
    pub max_trajectories: usize,
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_wall_clock_timeout")]
    pub wall_clock_timeout_sec: u64,
    #[serde(default = "default_smt_timeout")]
    pub smt_timeout_ms: u64,
    #[serde(default = "default_true")]
    pub enable_online_adaptation: bool,
}

fn default_max_trajectories() -> usize {
    100_000
}

fn default_max_depth() -> usize {
    15
}

fn default_batch_size() -> usize {
    64
}

fn default_wall_clock_timeout() -> u64 {
    900
}

fn default_smt_timeout() -> u64 {
    5000
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuditSubmissionRequest {
    pub project_source: ProjectSourceDto,
    #[serde(default = "default_foundry_profile")]
    pub foundry_profile: FoundryProfileDto,
    #[serde(default = "default_search_budget")]
    pub search_budget: SearchBudgetDto,
}

fn default_foundry_profile() -> FoundryProfileDto {
    FoundryProfileDto {
        solc_version: default_solc_version(),
        evm_version: default_evm_version(),
        fork_block_number: None,
    }
}

fn default_search_budget() -> SearchBudgetDto {
    SearchBudgetDto {
        max_trajectories: default_max_trajectories(),
        max_depth: default_max_depth(),
        batch_size: default_batch_size(),
        wall_clock_timeout_sec: default_wall_clock_timeout(),
        smt_timeout_ms: default_smt_timeout(),
        enable_online_adaptation: true,
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub enum AuditStatus {
    QUEUED,
    EXTRACTING_GRAPH,
    EXPLORING_GFLOWNET,
    VERIFYING_POC,
    COMPLETED,
    FAILED,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuditTelemetryDto {
    pub trajectories_evaluated: u64,
    pub revm_tps: f64,
    pub smt_queries_total: u64,
    pub smt_sat_hits: u64,
    pub smt_unsat_hits: u64,
    pub elapsed_sec: f64,
    pub log_z_estimate: f64,
}

impl From<&gfn_common::schemas::AuditTelemetry> for AuditTelemetryDto {
    fn from(t: &gfn_common::schemas::AuditTelemetry) -> Self {
        AuditTelemetryDto {
            trajectories_evaluated: t.trajectories_evaluated as u64,
            revm_tps: t.revm_tps,
            smt_queries_total: t.smt_queries_total as u64,
            smt_sat_hits: t.smt_sat_hits as u64,
            smt_unsat_hits: t.smt_unsat_hits as u64,
            elapsed_sec: t.elapsed_sec,
            log_z_estimate: t.log_z_estimate,
        }
    }
}

impl From<gfn_common::schemas::AuditTelemetry> for AuditTelemetryDto {
    fn from(t: gfn_common::schemas::AuditTelemetry) -> Self {
        AuditTelemetryDto::from(&t)
    }
}

impl From<ProjectSourceDto> for gfn_common::schemas::ProjectSourceConfig {
    fn from(dto: ProjectSourceDto) -> Self {
        gfn_common::schemas::ProjectSourceConfig {
            workspace_path: dto.workspace_path,
            setup_contract: dto.setup_contract,
            setup_function: dto.setup_function,
            target_contracts: dto.target_contracts.unwrap_or_default(),
        }
    }
}

impl From<FoundryProfileDto> for gfn_common::schemas::FoundryProfileConfig {
    fn from(dto: FoundryProfileDto) -> Self {
        gfn_common::schemas::FoundryProfileConfig {
            solc_version: dto.solc_version,
            evm_version: dto.evm_version,
            fork_block_number: dto.fork_block_number,
        }
    }
}

impl From<SearchBudgetDto> for gfn_common::schemas::SearchBudgetConfig {
    fn from(dto: SearchBudgetDto) -> Self {
        gfn_common::schemas::SearchBudgetConfig {
            max_trajectories: dto.max_trajectories,
            max_depth: dto.max_depth,
            batch_size: dto.batch_size,
            wall_clock_timeout_sec: dto.wall_clock_timeout_sec,
            smt_timeout_ms: dto.smt_timeout_ms,
            enable_online_adaptation: dto.enable_online_adaptation,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SynthesizedInvariantDto {
    pub invariant_id: String,
    pub category: String,
    pub formal_expression: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrajectoryStepRecordDto {
    pub step_idx: usize,
    pub caller_role: String,
    pub caller_address: String,
    pub target_contract: String,
    pub function_signature: String,
    pub concrete_calldata_hex: String,
    pub decoded_args: Vec<String>,
    pub vm_warp_sec: u64,
    pub vm_roll_blocks: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct VerifiedFindingDto {
    pub finding_id: String,
    pub violated_invariant_id: String,
    pub severity: String,
    pub trajectory_steps: Vec<TrajectoryStepRecordDto>,
    pub foundry_poc_path: String,
    pub foundry_poc_solidity_code: String,
    pub forge_test_stdout: String,
    pub minimal_negated_precondition: String,
    pub remediation_patch_diff: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuditResultResponse {
    pub audit_id: Uuid,
    pub status: AuditStatus,
    pub telemetry: AuditTelemetryDto,
    pub synthesized_invariants: Vec<SynthesizedInvariantDto>,
    pub verified_findings: Vec<VerifiedFindingDto>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StepScoringRequest {
    pub session_id: String,
    pub step_idx: usize,
    pub state_features: Vec<f32>,
    pub candidate_functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StepScoringResponse {
    pub action_caller: String,
    pub action_function: String,
    pub action_domain: String,
    pub log_probability: f32,
    pub batch_size: usize,
}

// --- Micro-Batching Internal Types ---

struct MicroBatchItem {
    request: StepScoringRequest,
    responder: oneshot::Sender<StepScoringResponse>,
}

pub struct AuditJob {
    pub audit_id: Uuid,
    pub request: AuditSubmissionRequest,
}

/// Thread-safe shared server state.
pub struct AuditServerState {
    pub audits: RwLock<HashMap<Uuid, AuditResultResponse>>,
    pub broadcasters: RwLock<HashMap<Uuid, Vec<mpsc::Sender<String>>>>,
    pub job_tx: mpsc::Sender<AuditJob>,
    batch_tx: mpsc::Sender<MicroBatchItem>,
}

impl AuditServerState {
    pub fn new(job_tx: mpsc::Sender<AuditJob>) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        let (batch_tx, batch_rx) = mpsc::channel::<MicroBatchItem>(1024);

        let state = Arc::new(Self {
            audits: RwLock::new(HashMap::new()),
            broadcasters: RwLock::new(HashMap::new()),
            job_tx,
            batch_tx,
        });

        // Spawn dynamic micro-batching background actor (2.0 ms accumulation window, max 64 items)
        let batch_handle = tokio::spawn(run_micro_batching_loop(batch_rx));

        (state, batch_handle)
    }

    pub fn emit_event(&self, audit_id: &Uuid, event_json: &str) {
        let mut broadcasters = self.broadcasters.write();
        if let Some(senders) = broadcasters.get_mut(audit_id) {
            senders.retain(|tx| tx.try_send(event_json.to_string()).is_ok());
        }
    }

    pub fn emit_typed_event(&self, audit_id: &Uuid, event_name: &str, data_json: &str) {
        let framed = serde_json::json!({
            "event": event_name,
            "data": serde_json::from_str::<serde_json::Value>(data_json)
                .unwrap_or_else(|_| serde_json::json!({ "raw": data_json }))
        })
        .to_string();
        self.emit_event(audit_id, &framed);
    }

    pub fn update_status(&self, audit_id: &Uuid, status: AuditStatus) {
        let mut audits = self.audits.write();
        if let Some(audit) = audits.get_mut(audit_id) {
            audit.status = status;
        }
    }

    pub fn update_telemetry(&self, audit_id: &Uuid, telemetry: AuditTelemetryDto) {
        let mut audits = self.audits.write();
        if let Some(audit) = audits.get_mut(audit_id) {
            audit.telemetry = telemetry;
        }
    }

    pub fn set_invariants(&self, audit_id: &Uuid, invariants: Vec<SynthesizedInvariantDto>) {
        let mut audits = self.audits.write();
        if let Some(audit) = audits.get_mut(audit_id) {
            audit.synthesized_invariants = invariants;
        }
    }

    pub fn set_findings(&self, audit_id: &Uuid, findings: Vec<VerifiedFindingDto>) {
        let mut audits = self.audits.write();
        if let Some(audit) = audits.get_mut(audit_id) {
            audit.verified_findings = findings;
        }
    }

    pub fn set_error(&self, audit_id: &Uuid, error_msg: String) {
        let mut audits = self.audits.write();
        if let Some(audit) = audits.get_mut(audit_id) {
            audit.status = AuditStatus::FAILED;
            audit.error_message = Some(error_msg);
        }
    }

    pub fn close_stream(&self, audit_id: &Uuid) {
        let mut broadcasters = self.broadcasters.write();
        if let Some(senders) = broadcasters.get_mut(audit_id) {
            senders.clear();
        }
    }
}

/// Actor loop for dynamic micro-batching up to 64 items within 2.0 ms.
async fn run_micro_batching_loop(mut rx: mpsc::Receiver<MicroBatchItem>) {
    let batch_window = Duration::from_millis(2);
    let max_batch_size = 64;

    loop {
        // Wait for first item
        let first = match rx.recv().await {
            Some(item) => item,
            None => break, // Channel closed
        };

        let mut batch = Vec::with_capacity(max_batch_size);
        batch.push(first);

        // Accumulate until window expires or batch reaches max_batch_size
        let deadline = tokio::time::Instant::now() + batch_window;
        while batch.len() < max_batch_size {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    break;
                }
                item = rx.recv() => {
                    match item {
                        Some(it) => batch.push(it),
                        None => break,
                    }
                }
            }
        }

        // Dispatch batched GPU forward-pass scoring simulation
        let batch_len = batch.len();
        for item in batch {
            let num_cands = item.request.candidate_functions.len().max(1);
            let feat_sum: f32 = item
                .request
                .state_features
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(|v| v.clamp(-30.0, 30.0))
                .sum();
            let cand_idx = (item.request.step_idx + (feat_sum.abs() as usize)) % num_cands;
            let selected_fn = item
                .request
                .candidate_functions
                .get(cand_idx)
                .or_else(|| item.request.candidate_functions.first())
                .cloned()
                .unwrap_or_else(|| "fallback()".to_string());

            let callers = [
                "ATTACKER0",
                "ATTACKER1_SYBIL",
                "NORMAL_USER",
                "LIQUIDITY_PROVIDER",
                "LIQUIDATOR_KEEPER",
            ];
            let caller = callers[item.request.step_idx % callers.len()].to_string();
            let log_prob = (-0.15 - (feat_sum.abs() * 0.01)).clamp(-10.0, -0.01);

            let response = StepScoringResponse {
                action_caller: caller,
                action_function: selected_fn,
                action_domain: "SOLVENCY".to_string(),
                log_probability: log_prob,
                batch_size: batch_len,
            };

            let _ = item.responder.send(response);
        }
    }
}

// --- Validation Logic ---

fn validate_submission(req: &AuditSubmissionRequest) -> GfnResult<()> {
    if req.project_source.workspace_path.trim().is_empty() {
        return Err(GfnError::Serialization {
            details: "workspace_path must not be empty".to_string(),
        });
    }

    let contract = &req.project_source.setup_contract;
    if contract.is_empty()
        || !contract.chars().all(|c| c.is_alphanumeric() || c == '_')
        || contract.chars().next().is_some_and(|c| c.is_ascii_digit())
    {
        return Err(GfnError::Serialization {
            details: format!("setup_contract '{contract}' contains invalid characters"),
        });
    }

    let solc = &req.foundry_profile.solc_version;
    if !solc.starts_with("0.8.") {
        return Err(GfnError::Serialization {
            details: format!("solc_version '{solc}' must match '^0.8.[0-9]+$'"),
        });
    }

    let evm = req.foundry_profile.evm_version.to_lowercase();
    if evm != "paris" && evm != "shanghai" && evm != "cancun" {
        return Err(GfnError::Serialization {
            details: format!("evm_version '{evm}' must be one of paris, shanghai, cancun"),
        });
    }

    let max_t = req.search_budget.max_trajectories;
    if !(64..=1_000_000).contains(&max_t) {
        return Err(GfnError::Serialization {
            details: format!("max_trajectories {max_t} must be within 64..=1_000_000"),
        });
    }

    let depth = req.search_budget.max_depth;
    if !(2..=25).contains(&depth) {
        return Err(GfnError::Serialization {
            details: format!("max_depth {depth} must be within 2..=25"),
        });
    }

    let bsz = req.search_budget.batch_size;
    if bsz != 8 && bsz != 16 && bsz != 32 && bsz != 64 && bsz != 128 {
        return Err(GfnError::Serialization {
            details: format!("batch_size {bsz} must be in [8, 16, 32, 64, 128]"),
        });
    }

    Ok(())
}

// --- HTTP Route Handlers ---

/// `POST /v1/audits`: Submit an audit request.
pub async fn submit_audit(
    State(state): State<Arc<AuditServerState>>,
    Json(payload): Json<AuditSubmissionRequest>,
) -> Result<(StatusCode, Json<AuditResultResponse>), (StatusCode, Json<serde_json::Value>)> {
    if let Err(e) = validate_submission(&payload) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "ValidationFailed",
                "details": e.to_string()
            })),
        ));
    }

    let audit_id = Uuid::new_v4();

    let initial_response = AuditResultResponse {
        audit_id,
        status: AuditStatus::QUEUED,
        telemetry: AuditTelemetryDto {
            trajectories_evaluated: 0,
            revm_tps: 0.0,
            smt_queries_total: 0,
            smt_sat_hits: 0,
            smt_unsat_hits: 0,
            elapsed_sec: 0.0,
            log_z_estimate: 0.0,
        },
        synthesized_invariants: Vec::new(),
        verified_findings: Vec::new(),
        error_message: None,
    };

    // Store in audit state
    state
        .audits
        .write()
        .insert(audit_id, initial_response.clone());

    // Initialize streaming listeners list
    state.broadcasters.write().insert(audit_id, Vec::new());

    // Enqueue job onto worker channel
    let job = AuditJob {
        audit_id,
        request: payload,
    };

    if let Err(e) = state.job_tx.send(job).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "QueueFullOrClosed",
                "details": e.to_string()
            })),
        ));
    }

    Ok((StatusCode::ACCEPTED, Json(initial_response)))
}

/// `GET /v1/audits/:id`: Retrieve current audit status and findings.
pub async fn get_audit_status(
    State(state): State<Arc<AuditServerState>>,
    Path(audit_id): Path<Uuid>,
) -> Result<Json<AuditResultResponse>, (StatusCode, Json<serde_json::Value>)> {
    let audits = state.audits.read();
    if let Some(audit) = audits.get(&audit_id) {
        Ok(Json(audit.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "NotFound",
                "details": format!("Audit '{audit_id}' does not exist")
            })),
        ))
    }
}

pub struct MpscEventStream {
    rx: mpsc::Receiver<String>,
}

impl futures_core::stream::Stream for MpscEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.rx.poll_recv(cx) {
            std::task::Poll::Ready(Some(msg)) => {
                std::task::Poll::Ready(Some(Ok(Event::default().data(msg))))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

/// `GET /v1/audits/:id/stream`: Stream live Server-Sent Events for an audit.
pub async fn stream_audit_events(
    State(state): State<Arc<AuditServerState>>,
    Path(audit_id): Path<Uuid>,
) -> Result<Sse<MpscEventStream>, StatusCode> {
    let current_status = {
        let audits = state.audits.read();
        audits.get(&audit_id).map(|a| a.status)
    };
    let Some(status) = current_status else {
        return Err(StatusCode::NOT_FOUND);
    };

    let (tx, rx) = mpsc::channel(64);
    let snapshot = serde_json::json!({
        "event": "status_change",
        "data": { "status": status, "stage": status }
    })
    .to_string();
    let _ = tx.try_send(snapshot);

    if status != AuditStatus::COMPLETED && status != AuditStatus::FAILED {
        let mut broadcasters = state.broadcasters.write();
        if let Some(senders) = broadcasters.get_mut(&audit_id) {
            senders.push(tx);
        }
    } else {
        drop(tx);
    }

    let stream = MpscEventStream { rx };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// `POST /v1/policy/sample-step`: Interactive step-level action scoring with dynamic micro-batching.
pub async fn sample_policy_step(
    State(state): State<Arc<AuditServerState>>,
    Json(request): Json<StepScoringRequest>,
) -> Result<Json<StepScoringResponse>, (StatusCode, Json<serde_json::Value>)> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let item = MicroBatchItem {
        request,
        responder: resp_tx,
    };

    if let Err(e) = state.batch_tx.send(item).await {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "BatchQueueExhausted",
                "details": e.to_string()
            })),
        ));
    }

    match resp_rx.await {
        Ok(response) => Ok(Json(response)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "BatchCanceled",
                "details": "Batch worker dropped responder"
            })),
        )),
    }
}

/// `GET /openapi.json`: OpenAPI 3.1 specification.
pub async fn get_openapi_spec() -> impl IntoResponse {
    #[derive(OpenApi)]
    #[openapi(
        paths(),
        components(schemas(
            ProjectSourceDto,
            FoundryProfileDto,
            SearchBudgetDto,
            AuditSubmissionRequest,
            AuditStatus,
            AuditTelemetryDto,
            SynthesizedInvariantDto,
            TrajectoryStepRecordDto,
            VerifiedFindingDto,
            AuditResultResponse,
            StepScoringRequest,
            StepScoringResponse,
        ))
    )]
    struct ApiDoc;

    let doc = ApiDoc::openapi();
    let json_str = doc.to_json().unwrap_or_else(|_| "{}".to_string());

    Response::builder()
        .header("Content-Type", "application/json")
        .body(json_str)
        .unwrap_or_else(|_| Response::new("{}".to_string()))
}

/// Query parameters for project context endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectContextQuery {
    pub workspace_path: Option<String>,
    pub setup_contract: Option<String>,
    pub solc_version: Option<String>,
    pub evm_version: Option<String>,
}

/// `GET /v1/projects/context`: Retrieve full ProtocolSemanticModel JSON.
pub async fn get_project_context_handler(
    Query(params): Query<ProjectContextQuery>,
) -> Result<Json<gfn_common::schemas::ProtocolSemanticModel>, (StatusCode, Json<serde_json::Value>)>
{
    let ws = params.workspace_path.unwrap_or_else(|| ".".to_string());
    let setup = params
        .setup_contract
        .unwrap_or_else(|| "ExploitHarness".to_string());
    let solc = params.solc_version.unwrap_or_else(|| "0.8.24".to_string());
    let evm = params.evm_version.unwrap_or_else(|| "cancun".to_string());

    let extractor =
        gfn_extractor::ProjectContextExtractor::new(std::path::Path::new(&ws), &solc, &evm, &setup);
    match extractor.extract_model() {
        Ok(model) => Ok(Json(model)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "ExtractionFailed",
                "details": e.to_string()
            })),
        )),
    }
}

/// `GET /v1/projects/ai-prompt`: Retrieve structured Markdown whole-project context bundle for AI models.
pub async fn get_project_ai_prompt_handler(
    Query(params): Query<ProjectContextQuery>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let ws = params.workspace_path.unwrap_or_else(|| ".".to_string());
    let setup = params
        .setup_contract
        .unwrap_or_else(|| "ExploitHarness".to_string());
    let solc = params.solc_version.unwrap_or_else(|| "0.8.24".to_string());
    let evm = params.evm_version.unwrap_or_else(|| "cancun".to_string());

    let project_path = std::path::Path::new(&ws);
    let extractor = gfn_extractor::ProjectContextExtractor::new(project_path, &solc, &evm, &setup);
    match extractor.extract_model() {
        Ok(model) => match crate::context_bundler::ProjectContextBundler::export_ai_prompt_bundle(
            &model,
            Some(project_path),
        ) {
            Ok(markdown) => {
                let resp = Response::builder()
                    .header("Content-Type", "text/markdown; charset=utf-8")
                    .body(axum::body::Body::from(markdown))
                    .unwrap_or_else(|_| Response::new(axum::body::Body::empty()));
                Ok(resp)
            }
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "BundleFailed", "details": e.to_string() })),
            )),
        },
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "ExtractionFailed", "details": e.to_string() })),
        )),
    }
}

/// `GET /`: Serve Web Verification Studio HTML interface.
pub async fn serve_web_studio() -> impl IntoResponse {
    let index_paths = [
        "ui-desktop/web/static/index.html",
        "../ui-desktop/web/static/index.html",
        "../../ui-desktop/web/static/index.html",
    ];
    for p in index_paths {
        if let Ok(content) = std::fs::read_to_string(p) {
            return Response::builder()
                .header("Content-Type", "text/html; charset=utf-8")
                .body(axum::body::Body::from(content))
                .unwrap_or_else(|_| Response::new(axum::body::Body::empty()));
        }
    }
    Response::builder()
        .header("Content-Type", "text/html; charset=utf-8")
        .body(axum::body::Body::from(
            "<h1>GFlowNet-Prover Web Studio</h1><p>UI assets not found on disk.</p>",
        ))
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// `GET /static/:file`: Serve Web Verification Studio static assets with strict traversal protection.
pub async fn serve_static_asset(Path(file): Path<String>) -> impl IntoResponse {
    let is_safe_single_name = !file.is_empty()
        && !file.contains("..")
        && !file.contains('/')
        && !file.contains('\\')
        && !file.contains(':')
        && !file.contains('\0')
        && {
            let mut comps = std::path::Path::new(&file).components();
            matches!(comps.next(), Some(std::path::Component::Normal(_))) && comps.next().is_none()
        };

    if !is_safe_single_name {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(axum::body::Body::from("Invalid static asset filename"))
            .unwrap_or_else(|_| Response::new(axum::body::Body::empty()));
    }

    let content_type = if file.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if file.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else {
        "application/octet-stream"
    };

    let candidates = [
        format!("ui-desktop/web/static/{file}"),
        format!("../ui-desktop/web/static/{file}"),
        format!("../../ui-desktop/web/static/{file}"),
    ];

    for p in candidates {
        if let Ok(bytes) = std::fs::read(p) {
            return Response::builder()
                .header("Content-Type", content_type)
                .body(axum::body::Body::from(bytes))
                .unwrap_or_else(|_| Response::new(axum::body::Body::empty()));
        }
    }
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(axum::body::Body::from("Asset not found"))
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// `GET /health`: Liveness probe.
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "gfn-forge",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Builds the Axum router for the GFlowNet-Prover HTTP/SSE microservice.
pub fn build_router(state: Arc<AuditServerState>) -> Router {
    Router::new()
        .route("/", get(serve_web_studio))
        .route("/static/:file", get(serve_static_asset))
        .route("/health", get(health_check))
        .route("/openapi.json", get(get_openapi_spec))
        .route("/v1/audits", post(submit_audit))
        .route("/v1/audits/:id", get(get_audit_status))
        .route("/v1/audits/:id/stream", get(stream_audit_events))
        .route("/v1/policy/sample-step", post(sample_policy_step))
        .route("/v1/projects/context", get(get_project_context_handler))
        .route("/v1/projects/ai-prompt", get(get_project_ai_prompt_handler))
        .with_state(state)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn create_test_state() -> (Arc<AuditServerState>, mpsc::Receiver<AuditJob>) {
        let (job_tx, job_rx) = mpsc::channel(16);
        let (state, _) = AuditServerState::new(job_tx);
        (state, job_rx)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let (state, _rx) = create_test_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_openapi_endpoint() {
        let (state, _rx) = create_test_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_audit_submission_validation() {
        let (state, mut rx) = create_test_state();
        let app = build_router(state);

        // Valid payload
        let valid_payload = serde_json::json!({
            "project_source": {
                "workspace_path": "/path/to/project",
                "setup_contract": "ExploitHarnessTest",
                "setup_function": "setUp"
            },
            "foundry_profile": {
                "solc_version": "0.8.24",
                "evm_version": "cancun"
            },
            "search_budget": {
                "max_trajectories": 1000,
                "max_depth": 10,
                "batch_size": 32
            }
        });

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audits")
                    .header("Content-Type", "application/json")
                    .body(Body::from(valid_payload.to_string()))
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let job = rx.try_recv().unwrap_or_else(|e| panic!("recv failed: {e}"));
        assert_eq!(
            job.request.project_source.setup_contract,
            "ExploitHarnessTest"
        );

        // Invalid payload: empty workspace_path
        let invalid_payload = serde_json::json!({
            "project_source": {
                "workspace_path": "",
                "setup_contract": "ExploitHarnessTest"
            }
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audits")
                    .header("Content-Type", "application/json")
                    .body(Body::from(invalid_payload.to_string()))
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_micro_batching_sample_step() {
        let (state, _rx) = create_test_state();
        let app = build_router(state);

        let payload = serde_json::json!({
            "session_id": "sess-01",
            "step_idx": 0,
            "state_features": [0.1, 0.2, 0.3],
            "candidate_functions": ["deposit(uint256)", "withdraw(uint256)"]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/policy/sample-step")
                    .header("Content-Type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_static_asset_rejects_path_traversal() {
        let (state, _rx) = create_test_state();
        let app = build_router(state);

        for malicious_uri in [
            "/static/..%5C..%5CCargo.toml",
            "/static/..%2FCargo.toml",
            "/static/C%3A%5CWindows%5Cwin.ini",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(malicious_uri)
                        .body(Body::empty())
                        .unwrap_or_else(|e| panic!("build request failed: {e}")),
                )
                .await
                .unwrap_or_else(|e| panic!("request failed: {e}"));

            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "Expected BAD_REQUEST for traversal URI {malicious_uri}"
            );
        }
    }

    #[tokio::test]
    async fn test_sse_terminal_audit_stream_closes_cleanly() {
        let (state, mut rx) = create_test_state();
        let app = build_router(state.clone());

        let payload = serde_json::json!({
            "project_source": {
                "workspace_path": "tests/fixtures/multi_bug_defi_protocol",
                "setup_contract": "ExploitHarness"
            }
        });

        let submit_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audits")
                    .header("Content-Type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap_or_else(|e| panic!("build request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("request failed: {e}"));
        assert_eq!(submit_resp.status(), StatusCode::ACCEPTED);

        let job = rx.try_recv().unwrap_or_else(|e| panic!("recv failed: {e}"));
        state.update_status(&job.audit_id, AuditStatus::COMPLETED);
        state.close_stream(&job.audit_id);

        let sse_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/audits/{}/stream", job.audit_id))
                    .body(Body::empty())
                    .unwrap_or_else(|e| panic!("build sse request failed: {e}")),
            )
            .await
            .unwrap_or_else(|e| panic!("sse request failed: {e}"));
        assert_eq!(sse_resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(sse_resp.into_body(), 65536)
            .await
            .unwrap_or_else(|e| panic!("sse body read failed: {e}"));
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("COMPLETED"));
    }
}
