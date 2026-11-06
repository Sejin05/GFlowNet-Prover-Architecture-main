#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! High-speed two-tier SMT resolution cache (`TwoTierUnsatCache`).
//!
//! Tier 1: In-memory `DashMap<[u8; 16], SmtResolution>` (65,536 entries, `< 0.01 ms` lookup).
//! Tier 2: Persistent LMDB store via `heed` keyed by `xxh3_128` of canonicalized SMT constraints.

use alloy_primitives::Bytes;
use dashmap::DashMap;
use gfn_common::error::{GfnError, GfnResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Result status of an SMT query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtStatus {
    Sat,
    Unsat,
    Unknown,
}

/// SMT query resolution containing witness calldata or unsat core metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtResolution {
    pub status: SmtStatus,
    pub concrete_witness_calldata: Option<Bytes>,
    pub unsat_core_hash: Option<[u8; 16]>,
    pub m_unsat_multiplier: f32,
    pub solve_time_ms: f64,
}

/// High-speed two-tier cache for SMT query resolutions.
pub struct TwoTierUnsatCache {
    l1: Arc<DashMap<[u8; 16], SmtResolution>>,
    l2_env: heed::Env,
    l2_db: heed::Database<heed::types::Bytes, heed::types::Bytes>,
    max_l1_capacity: usize,
}

impl TwoTierUnsatCache {
    /// Creates or opens a `TwoTierUnsatCache` storing L2 persistence at `lmdb_path`.
    pub fn new(lmdb_path: &Path) -> GfnResult<Self> {
        let mut options = heed::EnvOpenOptions::new();
        options.max_dbs(16);
        options.map_size(10 * 1024 * 1024 * 1024); // 10 GB

        let l2_env = options
            .open(lmdb_path)
            .map_err(|e| GfnError::SmtSolverError {
                details: format!(
                    "Failed to open heed LMDB environment at {}: {e}",
                    lmdb_path.display()
                ),
            })?;

        let mut wtxn = l2_env.write_txn().map_err(|e| GfnError::SmtSolverError {
            details: format!("Failed to create write txn for heed: {e}"),
        })?;

        let l2_db = l2_env
            .create_database(&mut wtxn, Some("smt_cache_l2"))
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to open or create heed database: {e}"),
            })?;

        wtxn.commit().map_err(|e| GfnError::SmtSolverError {
            details: format!("Failed to commit heed initial txn: {e}"),
        })?;

        Ok(Self {
            l1: Arc::new(DashMap::with_capacity(65_536)),
            l2_env,
            l2_db,
            max_l1_capacity: 65_536,
        })
    }

    /// Canonicalizes SMT-LIBv2 expressions to produce an invariant byte sequence.
    ///
    /// Strips comments, collapses extraneous whitespace, normalizes commutative operand
    /// orderings (`=`, `bvadd`, `bvmul`, `bvand`, `bvor`, `bvxor`), and alpha-renames bound
    /// symbolic variables (`x_0`, `x_1`, ...) so structurally identical branch constraints
    /// across parallel trajectories produce the identical `xxh3_128` digest.
    #[must_use]
    pub fn canonicalize_smt2(raw_smt2: &str) -> Vec<u8> {
        let mut cleaned_lines = Vec::new();
        for line in raw_smt2.lines() {
            let line_no_comment = match line.find(';') {
                Some(idx) => &line[..idx],
                None => line,
            };
            let trimmed = line_no_comment.trim();
            if !trimmed.is_empty() {
                cleaned_lines.push(trimmed);
            }
        }
        let joined = cleaned_lines.join(" ");

        // Tokenize s-expressions
        let mut tokens = Vec::new();
        let mut cur = String::new();
        for ch in joined.chars() {
            if ch == '(' || ch == ')' {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
                tokens.push(ch.to_string());
            } else if ch.is_whitespace() {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            tokens.push(cur);
        }

        // Alpha-rename variables (replace non-keyword symbols with $v0, $v1, ...)
        let mut var_map: BTreeMap<String, String> = BTreeMap::new();
        let mut canonical_tokens = Vec::with_capacity(tokens.len());

        for tok in tokens {
            let is_keyword = matches!(
                tok.as_str(),
                "(" | ")"
                    | "assert"
                    | "declare-const"
                    | "declare-fun"
                    | "define-fun"
                    | "set-logic"
                    | "check-sat"
                    | "get-model"
                    | "get-unsat-core"
                    | "QF_BV"
                    | "QF_UFBV"
                    | "QF_ABV"
                    | "_"
                    | "BitVec"
                    | "Array"
                    | "Bool"
                    | "select"
                    | "store"
                    | "evm_keccak256"
                    | "bvadd"
                    | "bvsub"
                    | "bvmul"
                    | "bvudiv"
                    | "bvsdiv"
                    | "bvurem"
                    | "bvsrem"
                    | "bvsmod"
                    | "bvshl"
                    | "bvlshr"
                    | "bvashr"
                    | "bvand"
                    | "bvor"
                    | "bvxor"
                    | "bvnot"
                    | "bvneg"
                    | "bvult"
                    | "bvugt"
                    | "bvslt"
                    | "bvsgt"
                    | "bvule"
                    | "bvuge"
                    | "bvsle"
                    | "bvsge"
                    | "="
                    | "distinct"
                    | "and"
                    | "or"
                    | "not"
                    | "ite"
                    | "true"
                    | "false"
            ) || tok.starts_with("#x")
                || tok.starts_with("#b")
                || tok.starts_with("bv")
                    && tok[2..].chars().all(|c| c.is_ascii_digit())
                    && tok.len() > 2
                || tok.chars().all(|c| c.is_ascii_digit());

            if is_keyword {
                canonical_tokens.push(tok);
            } else {
                let next_idx = var_map.len();
                let renamed = var_map
                    .entry(tok)
                    .or_insert_with(|| format!("$v{next_idx}"))
                    .clone();
                canonical_tokens.push(renamed);
            }
        }

        // Normalize simple binary commutative expressions: `( op A B )` where A, B are atomic tokens
        let mut idx = 0;
        while idx + 4 < canonical_tokens.len() {
            if canonical_tokens[idx] == "("
                && canonical_tokens[idx + 4] == ")"
                && matches!(
                    canonical_tokens[idx + 1].as_str(),
                    "=" | "distinct"
                        | "bvadd"
                        | "bvmul"
                        | "bvand"
                        | "bvor"
                        | "bvxor"
                        | "and"
                        | "or"
                )
                && canonical_tokens[idx + 2] != "("
                && canonical_tokens[idx + 2] != ")"
                && canonical_tokens[idx + 3] != "("
                && canonical_tokens[idx + 3] != ")"
                && canonical_tokens[idx + 2] > canonical_tokens[idx + 3]
            {
                canonical_tokens.swap(idx + 2, idx + 3);
            }
            idx += 1;
        }

        canonical_tokens.join(" ").into_bytes()
    }

    /// Computes the 128-bit `xxh3_128` digest for a canonicalized byte sequence.
    #[must_use]
    pub fn compute_hash(canonical_bytes: &[u8]) -> [u8; 16] {
        xxhash_rust::xxh3::xxh3_128(canonical_bytes).to_le_bytes()
    }

    /// Looks up a cached `SmtResolution` in L1 and L2.
    ///
    /// Automatically promotes L2 disk hits into L1 memory for subsequent sub-microsecond access.
    pub fn get(&self, key: &[u8; 16]) -> GfnResult<Option<SmtResolution>> {
        // 1. Check L1 in-memory DashMap (< 0.01 ms)
        if let Some(entry) = self.l1.get(key) {
            return Ok(Some(entry.value().clone()));
        }

        // 2. Check L2 persistent LMDB store via heed
        let rtxn = self
            .l2_env
            .read_txn()
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to acquire heed read txn: {e}"),
            })?;

        let l2_opt = self
            .l2_db
            .get(&rtxn, key)
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to read from heed database: {e}"),
            })?;

        if let Some(val_bytes) = l2_opt {
            let res: SmtResolution =
                serde_json::from_slice(val_bytes).map_err(|e| GfnError::SmtSolverError {
                    details: format!("Failed to deserialize SmtResolution from heed: {e}"),
                })?;

            // Promote to L1
            self.l1.insert(*key, res.clone());
            return Ok(Some(res));
        }

        Ok(None)
    }

    /// Inserts an `SmtResolution` into both L1 in-memory cache and L2 LMDB persistent store.
    pub fn insert(&self, key: [u8; 16], resolution: SmtResolution) -> GfnResult<()> {
        // Enforce L1 capacity limit
        if self.l1.len() >= self.max_l1_capacity {
            if let Some(first_key) = self.l1.iter().next().map(|entry| *entry.key()) {
                self.l1.remove(&first_key);
            }
        }

        self.l1.insert(key, resolution.clone());

        let val_bytes = serde_json::to_vec(&resolution).map_err(|e| GfnError::SmtSolverError {
            details: format!("Failed to serialize SmtResolution for heed: {e}"),
        })?;

        let mut wtxn = self
            .l2_env
            .write_txn()
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to acquire heed write txn: {e}"),
            })?;

        self.l2_db
            .put(&mut wtxn, &key, &val_bytes)
            .map_err(|e| GfnError::SmtSolverError {
                details: format!("Failed to write to heed database: {e}"),
            })?;

        wtxn.commit().map_err(|e| GfnError::SmtSolverError {
            details: format!("Failed to commit heed write txn: {e}"),
        })?;

        Ok(())
    }

    /// Returns the number of entries currently resident in L1.
    #[must_use]
    pub fn l1_len(&self) -> usize {
        self.l1.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonicalize_and_hash_consistency() {
        let query1 = r#"
            ; This is a comment
            (assert (= (bvmul x #x2710) #x26f2))
        "#;
        let query2 = "(assert (= (bvmul   y   #x2710) #x26f2)) ; different var name & spacing";

        let c1 = TwoTierUnsatCache::canonicalize_smt2(query1);
        let c2 = TwoTierUnsatCache::canonicalize_smt2(query2);

        let h1 = TwoTierUnsatCache::compute_hash(&c1);
        let h2 = TwoTierUnsatCache::compute_hash(&c2);

        assert_eq!(
            h1, h2,
            "Alpha-renaming and whitespace collapsing must yield identical hashes"
        );

        let comm1 = "(assert (= x #x2710))";
        let comm2 = "(assert (= #x2710 y))";
        let hc1 = TwoTierUnsatCache::compute_hash(&TwoTierUnsatCache::canonicalize_smt2(comm1));
        let hc2 = TwoTierUnsatCache::compute_hash(&TwoTierUnsatCache::canonicalize_smt2(comm2));
        assert_eq!(
            hc1, hc2,
            "Commutative binary operand normalization must yield identical hashes"
        );
    }

    #[test]
    fn test_two_tier_cache_put_get_and_persistence() -> GfnResult<()> {
        let temp_dir =
            std::env::temp_dir().join(format!("gfn_test_cache_{}", uuid::Uuid::new_v4()));
        let cache = TwoTierUnsatCache::new(&temp_dir)?;

        let key = [42u8; 16];
        let resolution = SmtResolution {
            status: SmtStatus::Sat,
            concrete_witness_calldata: Some(Bytes::from(vec![0x12, 0x34])),
            unsat_core_hash: None,
            m_unsat_multiplier: 1.0,
            solve_time_ms: 1.25,
        };

        cache.insert(key, resolution.clone())?;

        // 1. Direct L1 hit
        let l1_hit = cache.get(&key)?;
        assert_eq!(l1_hit, Some(resolution.clone()));

        // 2. Clear L1 and verify L2 promotion
        cache.l1.clear();
        assert_eq!(cache.l1_len(), 0);

        let l2_hit = cache.get(&key)?;
        assert_eq!(l2_hit, Some(resolution));
        assert_eq!(cache.l1_len(), 1, "Must promote L2 hit into L1");

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }
}
