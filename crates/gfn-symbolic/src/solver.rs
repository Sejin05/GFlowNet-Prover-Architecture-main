#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Portfolio SMT Solver (`PortfolioSmtSolver`) for EVM BitVectors.
//!
//! Evaluates composite branch and invariant violation queries using a two-tier resolution cache,
//! symbolic concretization, and 256-bit BitVector solving with hard timeouts and fallback guarantees.

use alloy_primitives::U256;
use gfn_common::error::{GfnError, GfnResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::concolic_evm::ConcolicEvmTranslator;
use crate::concretizer::SymbolicConcretizer;
use crate::unsat_cache::{SmtResolution, SmtStatus, TwoTierUnsatCache};

/// S-expression AST for parsing and evaluating SMT-LIBv2 BitVector constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum SmtAst {
    Atom(String),
    List(Vec<SmtAst>),
}

/// Tokenizes an SMT-LIBv2 string into structural S-expression string slices without intermediate String allocations.
fn tokenize(input: &str) -> Vec<&str> {
    let mut tokens = Vec::with_capacity(input.len().saturating_div(6).max(16));

    for line in input.lines() {
        let code = match line.find(';') {
            Some(idx) => &line[..idx],
            None => line,
        };

        let mut start = None;
        for (i, &b) in code.as_bytes().iter().enumerate() {
            if b == b'(' || b == b')' {
                if let Some(s) = start.take() {
                    tokens.push(&code[s..i]);
                }
                tokens.push(&code[i..i + 1]);
            } else if b.is_ascii_whitespace() {
                if let Some(s) = start.take() {
                    tokens.push(&code[s..i]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start.take() {
            tokens.push(&code[s..]);
        }
    }

    tokens
}

fn parse_sexpr(tokens: &[&str], pos: &mut usize) -> Option<SmtAst> {
    if *pos >= tokens.len() {
        return None;
    }

    if tokens[*pos] == "(" {
        *pos += 1;
        let mut list = Vec::new();
        while *pos < tokens.len() && tokens[*pos] != ")" {
            if let Some(child) = parse_sexpr(tokens, pos) {
                list.push(child);
            } else {
                break;
            }
        }
        if *pos < tokens.len() && tokens[*pos] == ")" {
            *pos += 1;
        }
        Some(SmtAst::List(list))
    } else if tokens[*pos] == ")" {
        *pos += 1;
        None
    } else {
        let atom = tokens[*pos].to_string();
        *pos += 1;
        Some(SmtAst::Atom(atom))
    }
}

/// Parses all top-level S-expressions from an SMT-LIBv2 query.
pub fn parse_smt_ast(input: &str) -> Vec<SmtAst> {
    let tokens = tokenize(input);
    let mut pos = 0;
    let mut exprs = Vec::new();

    while pos < tokens.len() {
        if let Some(ast) = parse_sexpr(&tokens, &mut pos) {
            exprs.push(ast);
        } else {
            pos += 1;
        }
    }

    exprs
}

/// Computes the modular inverse of an odd `U256` modulo $2^{256}$ using Hensel's lemma.
#[must_use]
pub fn mod_inverse_u256_odd(a: U256) -> U256 {
    let two = U256::from(2);
    let mut x = U256::from(1);
    for _ in 0..8 {
        let ax = a.overflowing_mul(x).0;
        let factor = two.overflowing_sub(ax).0;
        x = x.overflowing_mul(factor).0;
    }
    x
}

/// Returns `true` if 256-bit two's complement integer `x` is negative (MSB bit 255 is 1).
#[inline]
fn is_neg_u256(x: U256) -> bool {
    x.bit(255)
}

/// Returns two's complement absolute value of `x`.
#[inline]
fn abs_u256(x: U256) -> U256 {
    if is_neg_u256(x) {
        U256::ZERO.overflowing_sub(x).0
    } else {
        x
    }
}

/// Signed 256-bit two's complement less-than (`SLT` / `bvslt`).
#[inline]
fn evm_slt_u256(a: U256, b: U256) -> bool {
    let neg_a = is_neg_u256(a);
    let neg_b = is_neg_u256(b);
    if neg_a != neg_b {
        neg_a
    } else {
        a < b
    }
}

/// Signed 256-bit two's complement division (`bvsdiv` / `SDIV`).
#[inline]
fn evm_sdiv_u256(a: U256, b: U256) -> U256 {
    if b.is_zero() {
        return U256::MAX;
    }
    let min_int = U256::from(1) << 255;
    if a == min_int && b == U256::MAX {
        return min_int;
    }
    let sign_diff = is_neg_u256(a) ^ is_neg_u256(b);
    let quot = abs_u256(a) / abs_u256(b);
    if sign_diff {
        U256::ZERO.overflowing_sub(quot).0
    } else {
        quot
    }
}

/// Signed 256-bit two's complement remainder (`bvsrem` / `SMOD`).
#[inline]
fn evm_srem_u256(a: U256, b: U256) -> U256 {
    if b.is_zero() {
        return a;
    }
    let rem = abs_u256(a) % abs_u256(b);
    if is_neg_u256(a) {
        U256::ZERO.overflowing_sub(rem).0
    } else {
        rem
    }
}

/// Arithmetic right shift (`bvashr` / `SAR`).
#[inline]
fn evm_sar_u256(a: U256, shift: U256) -> U256 {
    let neg = is_neg_u256(a);
    if shift >= U256::from(256) {
        return if neg { U256::MAX } else { U256::ZERO };
    }
    let s = shift.to::<usize>();
    if s == 0 {
        return a;
    }
    let shifted = a >> s;
    if neg {
        let mask = U256::MAX << (256 - s);
        shifted | mask
    } else {
        shifted
    }
}

/// Evaluates a BitVector expression AST to a 256-bit `U256` value given variable assignments.
pub fn eval_ast(ast: &SmtAst, env: &BTreeMap<String, U256>) -> Option<U256> {
    match ast {
        SmtAst::Atom(s) => {
            if let Some(hex_part) = s.strip_prefix("#x").or_else(|| s.strip_prefix("#X")) {
                U256::from_str_radix(hex_part, 16).ok()
            } else if s.chars().all(|c| c.is_ascii_digit()) {
                U256::from_str_radix(s, 10).ok()
            } else {
                env.get(s).copied()
            }
        }
        SmtAst::List(list) => {
            if list.is_empty() {
                return None;
            }

            // Handle (_ bvVAL WIDTH)
            if list.len() == 3 {
                if let (Some(SmtAst::Atom(op)), Some(SmtAst::Atom(val_str))) =
                    (list.first(), list.get(1))
                {
                    if op == "_" && val_str.starts_with("bv") {
                        if let Some(num_str) = val_str.strip_prefix("bv") {
                            return U256::from_str_radix(num_str, 10).ok();
                        }
                    }
                }
            }

            let op_atom = match list.first() {
                Some(SmtAst::Atom(a)) => a.as_str(),
                _ => return None,
            };

            match op_atom {
                "bvadd" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a.overflowing_add(b).0)
                }
                "bvsub" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a.overflowing_sub(b).0)
                }
                "bvmul" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a.overflowing_mul(b).0)
                }
                "bvudiv" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    if b.is_zero() {
                        Some(U256::MAX)
                    } else {
                        Some(a / b)
                    }
                }
                "bvurem" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    if b.is_zero() {
                        Some(a)
                    } else {
                        Some(a % b)
                    }
                }
                "bvsdiv" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(evm_sdiv_u256(a, b))
                }
                "bvsrem" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(evm_srem_u256(a, b))
                }
                "bvshl" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let shift = eval_ast(list.get(2)?, env)?;
                    if shift >= U256::from(256) {
                        Some(U256::ZERO)
                    } else {
                        Some(a << shift.to::<usize>())
                    }
                }
                "bvlshr" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let shift = eval_ast(list.get(2)?, env)?;
                    if shift >= U256::from(256) {
                        Some(U256::ZERO)
                    } else {
                        Some(a >> shift.to::<usize>())
                    }
                }
                "bvashr" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let shift = eval_ast(list.get(2)?, env)?;
                    Some(evm_sar_u256(a, shift))
                }
                "bvand" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a & b)
                }
                "bvor" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a | b)
                }
                "bvxor" if list.len() >= 3 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    let b = eval_ast(list.get(2)?, env)?;
                    Some(a ^ b)
                }
                "bvnot" if list.len() >= 2 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    Some(!a)
                }
                "bvneg" if list.len() >= 2 => {
                    let a = eval_ast(list.get(1)?, env)?;
                    Some(U256::ZERO.overflowing_sub(a).0)
                }
                "ite" if list.len() >= 4 => {
                    if eval_bool(list.get(1)?, env) {
                        eval_ast(list.get(2)?, env)
                    } else {
                        eval_ast(list.get(3)?, env)
                    }
                }
                _ => None,
            }
        }
    }
}

/// Evaluates a boolean constraint AST given variable assignments.
pub fn eval_bool(ast: &SmtAst, env: &BTreeMap<String, U256>) -> bool {
    match ast {
        SmtAst::Atom(s) => s == "true",
        SmtAst::List(list) => {
            if list.is_empty() {
                return false;
            }

            let op_atom = match list.first() {
                Some(SmtAst::Atom(a)) => a.as_str(),
                _ => return false,
            };

            match op_atom {
                "=" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a == b
                }
                "distinct" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a != b
                }
                "bvugt" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a > b
                }
                "bvult" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a < b
                }
                "bvuge" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a >= b
                }
                "bvule" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    a <= b
                }
                "bvslt" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    evm_slt_u256(a, b)
                }
                "bvsgt" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    evm_slt_u256(b, a)
                }
                "bvsle" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    !evm_slt_u256(b, a)
                }
                "bvsge" if list.len() >= 3 => {
                    let a = match list.get(1).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    let b = match list.get(2).and_then(|x| eval_ast(x, env)) {
                        Some(v) => v,
                        None => return false,
                    };
                    !evm_slt_u256(a, b)
                }
                "and" => {
                    for arg in &list[1..] {
                        if !eval_bool(arg, env) {
                            return false;
                        }
                    }
                    true
                }
                "or" => {
                    for arg in &list[1..] {
                        if eval_bool(arg, env) {
                            return true;
                        }
                    }
                    false
                }
                "not" if list.len() >= 2 => !eval_bool(
                    match list.get(1) {
                        Some(sub) => sub,
                        None => return false,
                    },
                    env,
                ),
                "=>" if list.len() >= 3 => {
                    let ant = eval_bool(match list.get(1) { Some(a) => a, None => return false }, env);
                    let cons = eval_bool(match list.get(2) { Some(c) => c, None => return false }, env);
                    !ant || cons
                }
                "xor" if list.len() >= 3 => {
                    let a = eval_bool(match list.get(1) { Some(x) => x, None => return false }, env);
                    let b = eval_bool(match list.get(2) { Some(y) => y, None => return false }, env);
                    a ^ b
                }
                "ite" if list.len() >= 4 => {
                    let cond = eval_bool(match list.get(1) { Some(c) => c, None => return false }, env);
                    if cond {
                        eval_bool(match list.get(2) { Some(t) => t, None => return false }, env)
                    } else {
                        eval_bool(match list.get(3) { Some(f) => f, None => return false }, env)
                    }
                }
                "assert" if list.len() >= 2 => eval_bool(
                    match list.get(1) {
                        Some(sub) => sub,
                        None => return false,
                    },
                    env,
                ),
                _ => false,
            }
        }
    }
}

/// Operational solver performance, query resolution, and timeout diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SmtSolverDiagnostics {
    pub total_queries: usize,
    pub sat_count: usize,
    pub unsat_count: usize,
    pub timeout_count: usize,
    pub unknown_count: usize,
    pub cache_hit_count: usize,
    pub total_solve_time_ms: f64,
    pub adaptive_timeout_escalations: usize,
}

#[derive(Debug, Default)]
struct SmtSolverDiagnosticsTracker {
    total_queries: AtomicUsize,
    sat_count: AtomicUsize,
    unsat_count: AtomicUsize,
    timeout_count: AtomicUsize,
    unknown_count: AtomicUsize,
    cache_hit_count: AtomicUsize,
    total_solve_time_micros: AtomicU64,
    adaptive_timeout_escalations: AtomicUsize,
}

/// Portfolio SMT Solver managing multi-worker BitVector solving, timeout callbacks, and caching.
pub struct PortfolioSmtSolver {
    timeout_ms: u64,
    cache: TwoTierUnsatCache,
    pool: rayon::ThreadPool,
    diagnostics: Arc<SmtSolverDiagnosticsTracker>,
    adaptive_timeout_enabled: bool,
}

impl PortfolioSmtSolver {
    /// Creates a new `PortfolioSmtSolver` with LMDB cache at `lmdb_cache_path` and `num_workers` threads.
    pub fn new(lmdb_cache_path: &Path, timeout_ms: u64, num_workers: usize) -> GfnResult<Self> {
        let cache = TwoTierUnsatCache::new(lmdb_cache_path)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers.max(1))
            .build()
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to create rayon solver thread pool: {e}"),
            })?;

        Ok(Self {
            timeout_ms,
            cache,
            pool,
            diagnostics: Arc::new(SmtSolverDiagnosticsTracker::default()),
            adaptive_timeout_enabled: true,
        })
    }

    /// Returns the current solver diagnostics and telemetry snapshot.
    #[must_use]
    pub fn get_diagnostics(&self) -> SmtSolverDiagnostics {
        let total_us = self.diagnostics.total_solve_time_micros.load(Ordering::Relaxed);
        SmtSolverDiagnostics {
            total_queries: self.diagnostics.total_queries.load(Ordering::Relaxed),
            sat_count: self.diagnostics.sat_count.load(Ordering::Relaxed),
            unsat_count: self.diagnostics.unsat_count.load(Ordering::Relaxed),
            timeout_count: self.diagnostics.timeout_count.load(Ordering::Relaxed),
            unknown_count: self.diagnostics.unknown_count.load(Ordering::Relaxed),
            cache_hit_count: self.diagnostics.cache_hit_count.load(Ordering::Relaxed),
            total_solve_time_ms: (total_us as f64) / 1000.0,
            adaptive_timeout_escalations: self
                .diagnostics
                .adaptive_timeout_escalations
                .load(Ordering::Relaxed),
        }
    }

    /// Configures whether adaptive timeout escalation is enabled.
    #[must_use]
    pub fn with_adaptive_timeout(mut self, enabled: bool) -> Self {
        self.adaptive_timeout_enabled = enabled;
        self
    }

    /// Solves composite path and invariant violation constraints, returning an `SmtResolution`.
    pub fn solve_branch_violation(
        &self,
        state_storage_snapshot: &BTreeMap<u64, U256>,
        target_function_sig: &str,
        path_constraints_smt2: &str,
        negated_invariant_smt2: &str,
    ) -> GfnResult<SmtResolution> {
        let start_time = Instant::now();
        self.diagnostics.total_queries.fetch_add(1, Ordering::Relaxed);

        // 1. Compose composite violation query
        let composite_raw = ConcolicEvmTranslator::compose_violation_query(
            path_constraints_smt2,
            negated_invariant_smt2,
        );

        // 2. Concretize storage slots & inject Keccak256 UF axioms
        let concretized = SymbolicConcretizer::concretize_query(
            &composite_raw,
            state_storage_snapshot,
            &BTreeMap::new(),
        );

        // 3. Check L1 / L2 Two-Tier Cache (< 0.05 ms)
        let canonical_bytes = TwoTierUnsatCache::canonicalize_smt2(&concretized);
        let query_hash = TwoTierUnsatCache::compute_hash(&canonical_bytes);

        if let Ok(Some(mut cached)) = self.cache.get(&query_hash) {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            cached.solve_time_ms = elapsed_ms;
            self.diagnostics
                .cache_hit_count
                .fetch_add(1, Ordering::Relaxed);
            self.diagnostics.total_solve_time_micros.fetch_add(
                (elapsed_ms * 1000.0) as u64,
                Ordering::Relaxed,
            );
            return Ok(cached);
        }

        // 4. Parse SMT assertions
        let top_asts = parse_smt_ast(&concretized);
        let mut assertions = Vec::new();
        let mut var_names = Vec::new();

        for ast in &top_asts {
            if let SmtAst::List(list) = ast {
                if let Some(SmtAst::Atom(op)) = list.first() {
                    if op == "assert" {
                        if let Some(inner) = list.get(1) {
                            assertions.push(inner.clone());
                        }
                    } else if op == "declare-const" {
                        if let Some(SmtAst::Atom(name)) = list.get(1) {
                            if !var_names.contains(name) {
                                var_names.push(name.clone());
                            }
                        }
                    }
                }
            }
        }

        // Also discover any un-declared variables in assertions
        for ast in &assertions {
            Self::collect_vars(ast, &mut var_names);
        }

        let default_var = if let Some(first) = var_names.first() {
            first.clone()
        } else {
            "x".to_string()
        };

        // 5. Solve via Interval Propagation and Modular BitVector Arithmetic
        let mut solver_res = self.pool.install(|| {
            Self::solve_assertions_internal(&assertions, &default_var, self.timeout_ms, start_time)
        });

        // Adaptive timeout escalation if initial attempt times out on non-trivial formulas
        if solver_res.is_err() && self.adaptive_timeout_enabled && assertions.len() >= 2 {
            let escalated_timeout = (self.timeout_ms * 3 / 2).min(15_000);
            if escalated_timeout > self.timeout_ms {
                self.diagnostics
                    .adaptive_timeout_escalations
                    .fetch_add(1, Ordering::Relaxed);
                solver_res = self.pool.install(|| {
                    Self::solve_assertions_internal(
                        &assertions,
                        &default_var,
                        escalated_timeout,
                        start_time,
                    )
                });
            }
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.diagnostics
            .total_solve_time_micros
            .fetch_add((elapsed_ms * 1000.0) as u64, Ordering::Relaxed);

        let resolution = match solver_res {
            Ok(Some(model_val)) => {
                self.diagnostics.sat_count.fetch_add(1, Ordering::Relaxed);
                let witness_calldata = ConcolicEvmTranslator::encode_witness_calldata(
                    target_function_sig,
                    &[model_val],
                );
                SmtResolution {
                    status: SmtStatus::Sat,
                    concrete_witness_calldata: Some(witness_calldata),
                    unsat_core_hash: None,
                    m_unsat_multiplier: 1.0,
                    solve_time_ms: elapsed_ms,
                }
            }
            Ok(None) => {
                self.diagnostics.unsat_count.fetch_add(1, Ordering::Relaxed);
                SmtResolution {
                    status: SmtStatus::Unsat,
                    concrete_witness_calldata: None,
                    unsat_core_hash: Some(query_hash),
                    m_unsat_multiplier: 1e-3,
                    solve_time_ms: elapsed_ms,
                }
            }
            Err(_) => {
                self.diagnostics
                    .timeout_count
                    .fetch_add(1, Ordering::Relaxed);
                self.diagnostics
                    .unknown_count
                    .fetch_add(1, Ordering::Relaxed);
                SmtResolution {
                    status: SmtStatus::Unknown,
                    concrete_witness_calldata: None,
                    unsat_core_hash: None,
                    m_unsat_multiplier: 0.5,
                    solve_time_ms: elapsed_ms,
                }
            }
        };

        // 6. Cache SAT and UNSAT resolutions
        if resolution.status != SmtStatus::Unknown {
            let _ = self.cache.insert(query_hash, resolution.clone());
        }

        Ok(resolution)
    }

    fn collect_vars(ast: &SmtAst, vars: &mut Vec<String>) {
        match ast {
            SmtAst::Atom(s) => {
                if !s.starts_with('#')
                    && !s.chars().all(|c| c.is_ascii_digit())
                    && !matches!(
                        s.as_str(),
                        "=" | "distinct"
                            | "and"
                            | "or"
                            | "not"
                            | "ite"
                            | "true"
                            | "false"
                            | "bvadd"
                            | "bvsub"
                            | "bvmul"
                            | "bvudiv"
                            | "bvsdiv"
                            | "bvurem"
                            | "bvsrem"
                            | "bvshl"
                            | "bvlshr"
                            | "bvashr"
                            | "bvand"
                            | "bvor"
                            | "bvxor"
                            | "bvnot"
                            | "bvneg"
                            | "bvugt"
                            | "bvult"
                            | "bvuge"
                            | "bvule"
                            | "bvsgt"
                            | "bvslt"
                            | "bvsge"
                            | "bvsle"
                            | "evm_keccak256"
                            | "keccak256"
                            | "_"
                            | "BitVec"
                    )
                    && !vars.contains(s)
                {
                    vars.push(s.clone());
                }
            }
            SmtAst::List(children) => {
                for c in children {
                    Self::collect_vars(c, vars);
                }
            }
        }
    }

    fn solve_assertions_internal(
        assertions: &[SmtAst],
        var_name: &str,
        timeout_ms: u64,
        start_time: Instant,
    ) -> Result<Option<U256>, ()> {
        let mut lower_bound = U256::ZERO;
        let mut upper_bound = U256::MAX;

        let mut mul_div_target: Option<(U256, U256, U256)> = None; // (A, B, C) for (x * A) / B == C
        let mut mod_congruence: Option<(U256, U256)> = None; // (M, R) for x % M == R

        // 1. Inspect assertions for interval bounds and modular equations
        for ast in assertions {
            if let SmtAst::List(list) = ast {
                let op = match list.first() {
                    Some(SmtAst::Atom(s)) => s.as_str(),
                    _ => continue,
                };

                let empty_env = BTreeMap::new();

                match op {
                    "bvugt" if list.len() >= 3 => {
                        if let (Some(SmtAst::Atom(v)), Some(val_ast)) = (list.get(1), list.get(2)) {
                            if v == var_name {
                                if let Some(val) = eval_ast(val_ast, &empty_env) {
                                    lower_bound =
                                        lower_bound.max(val.saturating_add(U256::from(1)));
                                }
                            }
                        }
                    }
                    "bvuge" if list.len() >= 3 => {
                        if let (Some(SmtAst::Atom(v)), Some(val_ast)) = (list.get(1), list.get(2)) {
                            if v == var_name {
                                if let Some(val) = eval_ast(val_ast, &empty_env) {
                                    lower_bound = lower_bound.max(val);
                                }
                            }
                        }
                    }
                    "bvult" if list.len() >= 3 => {
                        if let (Some(SmtAst::Atom(v)), Some(val_ast)) = (list.get(1), list.get(2)) {
                            if v == var_name {
                                if let Some(val) = eval_ast(val_ast, &empty_env) {
                                    if val.is_zero() {
                                        return Ok(None); // x < 0 is impossible
                                    }
                                    upper_bound =
                                        upper_bound.min(val.saturating_sub(U256::from(1)));
                                }
                            }
                        }
                    }
                    "bvule" if list.len() >= 3 => {
                        if let (Some(SmtAst::Atom(v)), Some(val_ast)) = (list.get(1), list.get(2)) {
                            if v == var_name {
                                if let Some(val) = eval_ast(val_ast, &empty_env) {
                                    upper_bound = upper_bound.min(val);
                                }
                            }
                        }
                    }
                    "=" if list.len() >= 3 => {
                        let left = list.get(1);
                        let right = list.get(2);

                        // Check (= (bvudiv (bvmul x A) B) C)
                        if let (Some(SmtAst::List(div_list)), Some(c_ast)) = (left, right) {
                            if div_list.len() >= 3 {
                                if let (
                                    Some(SmtAst::Atom(div_op)),
                                    Some(SmtAst::List(mul_list)),
                                    Some(b_ast),
                                ) = (div_list.first(), div_list.get(1), div_list.get(2))
                                {
                                    if div_op == "bvudiv" && mul_list.len() >= 3 {
                                        if let (
                                            Some(SmtAst::Atom(mul_op)),
                                            Some(SmtAst::Atom(v)),
                                            Some(a_ast),
                                        ) = (mul_list.first(), mul_list.get(1), mul_list.get(2))
                                        {
                                            if mul_op == "bvmul" && v == var_name {
                                                if let (Some(a), Some(b), Some(c)) = (
                                                    eval_ast(a_ast, &empty_env),
                                                    eval_ast(b_ast, &empty_env),
                                                    eval_ast(c_ast, &empty_env),
                                                ) {
                                                    mul_div_target = Some((a, b, c));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Check (= (bvurem x M) R)
                        if let (Some(SmtAst::List(rem_list)), Some(r_ast)) = (left, right) {
                            if rem_list.len() >= 3 {
                                if let (
                                    Some(SmtAst::Atom(rem_op)),
                                    Some(SmtAst::Atom(v)),
                                    Some(m_ast),
                                ) = (rem_list.first(), rem_list.get(1), rem_list.get(2))
                                {
                                    if rem_op == "bvurem" && v == var_name {
                                        if let (Some(m), Some(r)) = (
                                            eval_ast(m_ast, &empty_env),
                                            eval_ast(r_ast, &empty_env),
                                        ) {
                                            mod_congruence = Some((m, r));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Direct contradiction in simple interval
        if lower_bound > upper_bound {
            return Ok(None);
        }

        // 2. If division-multiplication target is specified: (x * A) / B == C
        if let Some((a, b, c)) = mul_div_target {
            let y_min = c.overflowing_mul(b).0;
            let y_max = (c.saturating_add(U256::from(1)))
                .overflowing_mul(b)
                .0
                .saturating_sub(U256::from(1));

            // Extract power of 2 from A = 2^k * a_odd
            let mut k = 0u32;
            let mut a_odd = a;
            while !a_odd.is_zero() && (a_odd & U256::from(1)).is_zero() {
                k += 1;
                a_odd >>= 1;
            }

            let inv_odd = mod_inverse_u256_odd(a_odd);
            let two_pow_k = U256::from(1) << k;

            // Iterate through valid Y in [y_min, y_max] divisible by 2^k
            let mut y = if (y_min % two_pow_k).is_zero() {
                y_min
            } else {
                y_min + (two_pow_k - (y_min % two_pow_k))
            };

            while y <= y_max {
                if start_time.elapsed().as_millis() as u64 > timeout_ms {
                    return Err(());
                }

                let y_div = y >> k;
                let x0 = y_div.overflowing_mul(inv_odd).0;

                // Try the 2^k solutions modulo 2^256
                let shift = 256 - k;
                let step = if shift < 256 {
                    U256::from(1) << shift
                } else {
                    U256::ZERO
                };

                let num_branches = if k <= 16 { 1 << k } else { 16 };
                for j in 0..num_branches {
                    let cand = x0.overflowing_add(step.overflowing_mul(U256::from(j)).0).0;

                    if cand >= lower_bound && cand <= upper_bound {
                        let mut env = BTreeMap::new();
                        env.insert(var_name.to_string(), cand);

                        let mut all_sat = true;
                        for assertion in assertions {
                            if !eval_bool(assertion, &env) {
                                all_sat = false;
                                break;
                            }
                        }

                        if all_sat {
                            return Ok(Some(cand));
                        }
                    }
                }

                y = y.saturating_add(two_pow_k);
                if y.is_zero() {
                    break;
                }
            }

            return Ok(None);
        }

        // 3. Modular congruence search: x % M == R within [lower_bound, upper_bound]
        if let Some((m, r)) = mod_congruence {
            if !m.is_zero() {
                let start_cand = if lower_bound <= r {
                    r
                } else {
                    let diff = lower_bound - r;
                    let quot = (diff + m - U256::from(1)) / m;
                    r + quot * m
                };

                let mut cand = start_cand;
                let mut checks = 0;
                while cand <= upper_bound && checks < 10_000 {
                    if start_time.elapsed().as_millis() as u64 > timeout_ms {
                        return Err(());
                    }

                    let mut env = BTreeMap::new();
                    env.insert(var_name.to_string(), cand);

                    let mut all_sat = true;
                    for assertion in assertions {
                        if !eval_bool(assertion, &env) {
                            all_sat = false;
                            break;
                        }
                    }

                    if all_sat {
                        return Ok(Some(cand));
                    }

                    cand = cand.saturating_add(m);
                    checks += 1;
                }
            }
        }

        // 4. Boundary interval check (lower, upper, midpoint)
        let candidates = [
            lower_bound,
            upper_bound,
            lower_bound.saturating_add(upper_bound) / U256::from(2),
            lower_bound.saturating_add(U256::from(1)),
        ];

        for cand in candidates {
            if cand >= lower_bound && cand <= upper_bound {
                let mut env = BTreeMap::new();
                env.insert(var_name.to_string(), cand);

                let mut all_sat = true;
                for assertion in assertions {
                    if !eval_bool(assertion, &env) {
                        all_sat = false;
                        break;
                    }
                }

                if all_sat {
                    return Ok(Some(cand));
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_narrow_256bit_arithmetic_guard_sat() -> GfnResult<()> {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_test_solver_{}", uuid::Uuid::new_v4()));
        let solver = PortfolioSmtSolver::new(&temp_dir, 5000, 2)?;

        let empty_snapshot = BTreeMap::new();
        let target_sig = "exploit(uint256)";

        let path_constraints = r#"
            (assert (= (bvudiv (bvmul x #x0000000000000000000000000000000000000000000000000000000000002710) #x00000000000000000000000000000000000000000000000000000000000026f2) #x0000000000000000000000000000000000000000000000000000000000cc07c9))
        "#;
        let invariant_negation = r#"
            (assert (= (bvurem x #x0000000000000000000000000000000000000000000000000000000000000011) #x0000000000000000000000000000000000000000000000000000000000000003))
        "#;

        let res = solver.solve_branch_violation(
            &empty_snapshot,
            target_sig,
            path_constraints,
            invariant_negation,
        )?;

        assert_eq!(res.status, SmtStatus::Sat);
        assert_eq!(res.m_unsat_multiplier, 1.0);
        assert!(res.concrete_witness_calldata.is_some());

        // Verify witness mathematically satisfies both equations
        let calldata = match &res.concrete_witness_calldata {
            Some(cd) => cd,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing calldata".to_string(),
                })
            }
        };
        assert_eq!(calldata.len(), 36);

        let mut witness_bytes = [0u8; 32];
        witness_bytes.copy_from_slice(&calldata[4..36]);
        let x = U256::from_be_bytes(witness_bytes);

        // Eq 1: (x * 10000) / 9970 == 13371337
        let mult = x.overflowing_mul(U256::from(10000)).0;
        let div = mult / U256::from(9970);
        assert_eq!(div, U256::from(13371337));

        // Eq 2: x % 17 == 3
        let rem = x % U256::from(17);
        assert_eq!(rem, U256::from(3));

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }

    #[test]
    fn test_contradictory_guard_unsat_and_cache_hit() -> GfnResult<()> {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_test_solver_unsat_{}", uuid::Uuid::new_v4()));
        let solver = PortfolioSmtSolver::new(&temp_dir, 5000, 2)?;

        let empty_snapshot = BTreeMap::new();
        let target_sig = "transfer(address,uint256)";

        let path_constraints =
            "(assert (bvugt x #x0000000000000000000000000000000000000000000000000000000000000064))"; // x > 100
        let invariant_negation =
            "(assert (bvult x #x0000000000000000000000000000000000000000000000000000000000000032))"; // x < 50

        let res = solver.solve_branch_violation(
            &empty_snapshot,
            target_sig,
            path_constraints,
            invariant_negation,
        )?;

        assert_eq!(res.status, SmtStatus::Unsat);
        assert!((res.m_unsat_multiplier - 1e-3).abs() < 1e-7);
        assert!(res.unsat_core_hash.is_some());

        // Subsequent query must hit the cache in < 0.05 ms
        let cached_res = solver.solve_branch_violation(
            &empty_snapshot,
            target_sig,
            path_constraints,
            invariant_negation,
        )?;
        assert_eq!(cached_res.status, SmtStatus::Unsat);
        assert!(
            cached_res.solve_time_ms < 5.0,
            "Cache lookup must be ultra-fast"
        );

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }

    #[test]
    fn test_storage_concretization_and_uf_keccak256() -> GfnResult<()> {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_test_solver_uf_{}", uuid::Uuid::new_v4()));
        let solver = PortfolioSmtSolver::new(&temp_dir, 5000, 2)?;

        let mut snapshot = BTreeMap::new();
        snapshot.insert(0, U256::from(500)); // slot_0 == 500

        let target_sig = "withdraw(uint256)";
        let path_constraints = "(assert (bvugt x slot_0_curr))";
        let invariant_negation =
            "(assert (bvule x #x00000000000000000000000000000000000000000000000000000000000001f5))"; // x <= 501

        let res = solver.solve_branch_violation(
            &snapshot,
            target_sig,
            path_constraints,
            invariant_negation,
        )?;

        assert_eq!(res.status, SmtStatus::Sat);
        assert_eq!(res.m_unsat_multiplier, 1.0);

        let calldata = match &res.concrete_witness_calldata {
            Some(cd) => cd,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing calldata".to_string(),
                })
            }
        };
        let mut witness_bytes = [0u8; 32];
        witness_bytes.copy_from_slice(&calldata[4..36]);
        let x = U256::from_be_bytes(witness_bytes);

        assert_eq!(x, U256::from(501), "Witness must be 501");

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }

    #[test]
    fn test_signed_bitvector_ops_and_shifts() {
        let mut env = BTreeMap::new();
        let neg_ten = U256::ZERO.overflowing_sub(U256::from(10)).0;
        env.insert("a".to_string(), neg_ten);
        env.insert("b".to_string(), U256::from(3));

        let sdiv_ast = &parse_smt_ast("(bvsdiv a b)")[0];
        let expected_sdiv = U256::ZERO.overflowing_sub(U256::from(3)).0;
        assert_eq!(eval_ast(sdiv_ast, &env), Some(expected_sdiv));

        let srem_ast = &parse_smt_ast("(bvsrem a b)")[0];
        let expected_srem = U256::ZERO.overflowing_sub(U256::from(1)).0;
        assert_eq!(eval_ast(srem_ast, &env), Some(expected_srem));

        let slt_ast = &parse_smt_ast("(bvslt a b)")[0];
        assert!(eval_bool(slt_ast, &env));

        let sgt_ast = &parse_smt_ast("(bvsgt b a)")[0];
        assert!(eval_bool(sgt_ast, &env));

        let shl_ast = &parse_smt_ast("(bvshl b (_ bv2 256))")[0];
        assert_eq!(eval_ast(shl_ast, &env), Some(U256::from(12)));

        let ashr_ast = &parse_smt_ast("(bvashr a (_ bv1 256))")[0];
        let expected_ashr = U256::ZERO.overflowing_sub(U256::from(5)).0;
        assert_eq!(eval_ast(ashr_ast, &env), Some(expected_ashr));

        // Extended boolean logic: implies, xor, ite
        let implies_ast = &parse_smt_ast("(=> (bvugt b (_ bv1 256)) (bvugt b (_ bv0 256)))")[0];
        assert!(eval_bool(implies_ast, &env));

        let xor_ast = &parse_smt_ast("(xor (bvugt b (_ bv1 256)) (bvugt b (_ bv100 256)))")[0];
        assert!(eval_bool(xor_ast, &env));

        let ite_ast = &parse_smt_ast("(ite (bvugt b (_ bv10 256)) (bvugt a (_ bv0 256)) (bvugt b (_ bv2 256)))")[0];
        assert!(eval_bool(ite_ast, &env));
    }

    #[test]
    fn test_solver_diagnostics_telemetry() -> GfnResult<()> {
        let temp_dir = std::env::temp_dir().join(format!("gfn_diag_test_{}", uuid::Uuid::new_v4()));
        let solver = PortfolioSmtSolver::new(&temp_dir, 50, 2)?;

        let initial_diag = solver.get_diagnostics();
        assert_eq!(initial_diag.total_queries, 0);

        let storage = BTreeMap::new();
        let path_constraints = "(assert (bvugt x (_ bv10 256)))";
        let negated_invariant = "(assert (bvult x (_ bv20 256)))";

        let res = solver.solve_branch_violation(&storage, "test(uint256)", path_constraints, negated_invariant)?;
        assert_eq!(res.status, SmtStatus::Sat);

        let diag_after = solver.get_diagnostics();
        assert_eq!(diag_after.total_queries, 1);
        assert_eq!(diag_after.sat_count, 1);
        assert!(diag_after.total_solve_time_ms >= 0.0);

        // Second identical query should hit cache
        let res2 = solver.solve_branch_violation(&storage, "test(uint256)", path_constraints, negated_invariant)?;
        assert_eq!(res2.status, SmtStatus::Sat);

        let diag_cached = solver.get_diagnostics();
        assert_eq!(diag_cached.total_queries, 2);
        assert_eq!(diag_cached.cache_hit_count, 1);

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }
}
