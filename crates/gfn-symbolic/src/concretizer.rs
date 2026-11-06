#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Partial Symbolic Concretizer (`SymbolicConcretizer`).
//!
//! Linearizes non-linear EVM arithmetic by substituting concrete 256-bit storage slot values
//! from `RevmArena` snapshots and replaces bit-blasted `KECCAK256` hashes with an Uninterpreted
//! Function (`UF`) paired with concrete bijection equality axioms.

use alloy_primitives::U256;
use std::collections::BTreeMap;

/// Converts a 256-bit unsigned integer into a 64-character lowercase hex SMT-LIBv2 literal (`#x...`).
#[must_use]
pub fn u256_to_smt_hex(val: &U256) -> String {
    format!("#x{:064x}", val)
}

/// Concretizes SMT queries against concrete EVM state snapshots and observed hash tables.
pub struct SymbolicConcretizer;

impl SymbolicConcretizer {
    /// Concretizes storage references in `smt2_query` against `state_storage_snapshot`
    /// and injects `evm_keccak256` uninterpreted function declarations and observed bijection axioms.
    #[must_use]
    pub fn concretize_query(
        smt2_query: &str,
        state_storage_snapshot: &BTreeMap<u64, U256>,
        observed_hashes: &BTreeMap<(U256, U256), U256>,
    ) -> String {
        let mut result = smt2_query.to_string();

        // 1. Partial Storage Concretization: substitute concrete U256 hex literals.
        // Iterate in descending slot_id order (.rev()) so `slot_10` is replaced before `slot_1`,
        // preventing prefix-substring corruption (e.g. `slot_1` corrupting `slot_10`).
        for (&slot_id, val) in state_storage_snapshot.iter().rev() {
            let slot_prefix = format!("slot_{slot_id}");
            let slot_func = format!("slot({slot_id})");
            if !result.contains(&slot_prefix) && !result.contains(&slot_func) {
                continue;
            }

            let hex_val = u256_to_smt_hex(val);

            // Remove redundant declarations for concretized slots first
            let decl_patterns = [
                format!("(declare-const slot_{slot_id}_curr (_ BitVec 256))\n"),
                format!("(declare-const slot_{slot_id}_prev (_ BitVec 256))\n"),
                format!("(declare-const slot_{slot_id}_0 (_ BitVec 256))\n"),
                format!("(declare-const slot_{slot_id}_n (_ BitVec 256))\n"),
                format!("(declare-const slot_{slot_id} (_ BitVec 256))\n"),
            ];

            for decl in &decl_patterns {
                if result.contains(decl) {
                    result = result.replace(decl, "");
                }
            }

            // Replace standard slot variable patterns (longest suffix first)
            let patterns = [
                format!("slot_{slot_id}_curr"),
                format!("slot_{slot_id}_prev"),
                format!("slot_{slot_id}_0"),
                format!("slot_{slot_id}_n"),
                format!("slot({slot_id})"),
                format!("slot_{slot_id}"),
            ];

            for pat in &patterns {
                result = replace_smt_token(&result, pat, &hex_val);
            }
        }

        // 2. Uninterpreted Function (UF) + Concrete Hash Table Bijection for KECCAK256
        let mut uf_header = String::new();
        let mut uf_axioms = String::new();

        if result.contains("evm_keccak256")
            || result.contains("keccak256")
            || !observed_hashes.is_empty()
        {
            uf_header.push_str(
                "(declare-fun evm_keccak256 ((_ BitVec 256) (_ BitVec 256)) (_ BitVec 256))\n",
            );

            for (&(key, slot), &hash) in observed_hashes {
                let k_hex = u256_to_smt_hex(&key);
                let s_hex = u256_to_smt_hex(&slot);
                let h_hex = u256_to_smt_hex(&hash);

                // Inject bijection equality axiom
                uf_axioms.push_str(&format!(
                    "(assert (= (evm_keccak256 {k_hex} {s_hex}) {h_hex}))\n"
                ));

                // If exact concrete call appears in the query, substitute directly
                let call_pattern = format!("(evm_keccak256 {k_hex} {s_hex})");
                result = result.replace(&call_pattern, &h_hex);
            }
        }

        if !uf_header.is_empty() || !uf_axioms.is_empty() {
            format!("{uf_header}{uf_axioms}{result}")
        } else {
            result
        }
    }
}

/// Replaces `target` in `input` only when bounded by S-expression delimiters or whitespace,
/// preventing `slot_1` from corrupting `slot_10` or `my_slot_1`.
fn replace_smt_token(input: &str, target: &str, replacement: &str) -> String {
    if target.is_empty() || !input.contains(target) {
        return input.to_string();
    }
    let is_token_char = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    let mut prev_char: Option<char> = None;

    while let Some(pos) = rest.find(target) {
        let prefix = &rest[..pos];
        out.push_str(prefix);
        let before = prefix.chars().last().or(prev_char);
        let after_slice = &rest[pos + target.len()..];
        let after = after_slice.chars().next();

        let valid_before = before.is_none_or(|c| !is_token_char(c));
        let valid_after = after.is_none_or(|c| !is_token_char(c));

        if valid_before && valid_after {
            out.push_str(replacement);
        } else {
            out.push_str(target);
        }
        prev_char = target.chars().last();
        rest = after_slice;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_concretization() {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(1, U256::from(100)); // #x...64
        snapshot.insert(5, U256::from(500)); // #x...1f4

        let query = r#"
(declare-const slot_1_curr (_ BitVec 256))
(declare-const x (_ BitVec 256))
(assert (bvugt x slot_1_curr))
(assert (bvult x slot_5_curr))
"#;

        let empty_hashes = BTreeMap::new();
        let concretized = SymbolicConcretizer::concretize_query(query, &snapshot, &empty_hashes);

        assert!(!concretized.contains("slot_1_curr"));
        assert!(concretized
            .contains("#x0000000000000000000000000000000000000000000000000000000000000064"));
        assert!(concretized
            .contains("#x00000000000000000000000000000000000000000000000000000000000001f4"));
    }

    #[test]
    fn test_keccak256_uf_bijection() {
        let empty_snapshot = BTreeMap::new();
        let mut hashes = BTreeMap::new();

        let key = U256::from(0xbeef);
        let slot = U256::from(2);
        let hash = U256::from(0x1337c0de);
        hashes.insert((key, slot), hash);

        let query = "(assert (= (evm_keccak256 x #x0000000000000000000000000000000000000000000000000000000000000002) y))";
        let concretized = SymbolicConcretizer::concretize_query(query, &empty_snapshot, &hashes);

        assert!(concretized.contains("(declare-fun evm_keccak256"));
        assert!(concretized.contains("assert (= (evm_keccak256 #x000000000000000000000000000000000000000000000000000000000000beef"));
    }

    #[test]
    fn test_multi_digit_slot_no_prefix_collision() {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(1, U256::from(1));
        snapshot.insert(10, U256::from(10));

        let query = "(assert (bvugt slot_10 slot_1)) (assert (bvugt slot_10_curr slot_1_curr))";
        let concretized = SymbolicConcretizer::concretize_query(query, &snapshot, &BTreeMap::new());

        assert!(concretized
            .contains("#x000000000000000000000000000000000000000000000000000000000000000a"));
        assert!(concretized
            .contains("#x0000000000000000000000000000000000000000000000000000000000000001"));
        assert!(!concretized.contains("slot_"));
    }
}
