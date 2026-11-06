#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Concolic EVM opcode to SMT-LIBv2 QF_BV translator and witness encoder.

use alloy_primitives::{keccak256, Bytes, U256};
use gfn_common::error::{GfnError, GfnResult};

/// Translates 256-bit EVM opcodes into SMT-LIBv2 expressions and formats ABI calldata.
pub struct ConcolicEvmTranslator;

impl ConcolicEvmTranslator {
    /// Translates an EVM opcode into an SMT-LIBv2 QF_BV expression over 256-bit BitVectors.
    pub fn translate_opcode(opcode: u8, operands: &[&str]) -> GfnResult<String> {
        match opcode {
            // Arithmetic (0x01..0x09)
            0x01 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvadd {a} {b})"))
            }
            0x02 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvmul {a} {b})"))
            }
            0x03 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvsub {a} {b})"))
            }
            0x04 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvudiv {a} {b})"))
            }
            0x05 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvsdiv {a} {b})"))
            }
            0x06 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvurem {a} {b})"))
            }
            0x07 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvsrem {a} {b})"))
            }
            0x08 => {
                let (a, b, m) = Self::ternary_operands(operands)?;
                Ok(format!("(bvurem (bvadd {a} {b}) {m})"))
            }
            0x09 => {
                let (a, b, m) = Self::ternary_operands(operands)?;
                Ok(format!("(bvurem (bvmul {a} {b}) {m})"))
            }

            // Comparison & Bitwise (0x10..0x1D)
            0x10 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(ite (bvult {a} {b}) (_ bv1 256) (_ bv0 256))"))
            }
            0x11 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(ite (bvugt {a} {b}) (_ bv1 256) (_ bv0 256))"))
            }
            0x12 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(ite (bvslt {a} {b}) (_ bv1 256) (_ bv0 256))"))
            }
            0x13 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(ite (bvsgt {a} {b}) (_ bv1 256) (_ bv0 256))"))
            }
            0x14 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(ite (= {a} {b}) (_ bv1 256) (_ bv0 256))"))
            }
            0x15 => {
                let a = match operands.first() {
                    Some(op) => *op,
                    None => {
                        return Err(GfnError::SmtSolverError {
                            details: "ISZERO opcode requires 1 operand".to_string(),
                        })
                    }
                };
                Ok(format!("(ite (= {a} (_ bv0 256)) (_ bv1 256) (_ bv0 256))"))
            }
            0x16 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvand {a} {b})"))
            }
            0x17 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvor {a} {b})"))
            }
            0x18 => {
                let (a, b) = Self::binary_operands(operands)?;
                Ok(format!("(bvxor {a} {b})"))
            }
            0x19 => {
                let a = match operands.first() {
                    Some(op) => *op,
                    None => {
                        return Err(GfnError::SmtSolverError {
                            details: "NOT opcode requires 1 operand".to_string(),
                        })
                    }
                };
                Ok(format!("(bvnot {a})"))
            }
            0x1B => {
                let (shift, val) = Self::binary_operands(operands)?;
                Ok(format!("(bvshl {val} {shift})"))
            }
            0x1C => {
                let (shift, val) = Self::binary_operands(operands)?;
                Ok(format!("(bvlshr {val} {shift})"))
            }
            0x1D => {
                let (shift, val) = Self::binary_operands(operands)?;
                Ok(format!("(bvashr {val} {shift})"))
            }
            0x20 => {
                let (offset, size) = Self::binary_operands(operands)?;
                Ok(format!("(evm_keccak256 {offset} {size})"))
            }

            _ => Err(GfnError::SmtSolverError {
                details: format!("Unsupported EVM opcode 0x{opcode:02x} for SMT translation"),
            }),
        }
    }

    /// Composes a composite invariant violation query:
    ///
    /// $$\Phi_{\text{query}}(x_1, \dots, x_k) \equiv \Phi_\tau(x_1, \dots, x_k) \land \neg \mathcal{I}(s_n)$$
    #[must_use]
    pub fn compose_violation_query(
        path_constraints_smt2: &str,
        negated_invariant_smt2: &str,
    ) -> String {
        let mut composite =
            String::with_capacity(path_constraints_smt2.len() + negated_invariant_smt2.len() + 128);
        composite
            .push_str("; --- GFlowNet-Prover Composite Path & Invariant Violation Query ---\n");
        composite.push_str(path_constraints_smt2);
        if !path_constraints_smt2.ends_with('\n') {
            composite.push('\n');
        }
        composite.push_str(negated_invariant_smt2);
        if !negated_invariant_smt2.ends_with('\n') {
            composite.push('\n');
        }
        composite
    }

    /// Computes the 4-byte ABI selector from a function signature (e.g. `transfer(address,uint256)`).
    #[must_use]
    pub fn compute_function_selector(function_sig: &str) -> [u8; 4] {
        let sig_trimmed = function_sig.trim();
        // Check if already provided as 0x-prefixed 4-byte hex
        if (sig_trimmed.starts_with("0x") || sig_trimmed.starts_with("0X"))
            && sig_trimmed.len() == 10
        {
            if let Ok(bytes) = hex::decode(&sig_trimmed[2..]) {
                if bytes.len() == 4 {
                    return [bytes[0], bytes[1], bytes[2], bytes[3]];
                }
            }
        }

        let hash = keccak256(sig_trimmed.as_bytes());
        [hash[0], hash[1], hash[2], hash[3]]
    }

    /// Encodes a 4-byte selector and 256-bit model arguments into canonical ABI `Bytes`.
    #[must_use]
    pub fn encode_witness_calldata(target_function_sig: &str, model_args: &[U256]) -> Bytes {
        let selector = Self::compute_function_selector(target_function_sig);
        let mut calldata = Vec::with_capacity(4 + model_args.len() * 32);
        calldata.extend_from_slice(&selector);

        for arg in model_args {
            calldata.extend_from_slice(&arg.to_be_bytes::<32>());
        }

        Bytes::from(calldata)
    }

    fn binary_operands<'a>(operands: &'a [&'a str]) -> GfnResult<(&'a str, &'a str)> {
        if operands.len() < 2 {
            return Err(GfnError::SmtSolverError {
                details: format!("Expected at least 2 operands, got {}", operands.len()),
            });
        }
        let a = match operands.first() {
            Some(op) => *op,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing operand 0".to_string(),
                })
            }
        };
        let b = match operands.get(1) {
            Some(op) => *op,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing operand 1".to_string(),
                })
            }
        };
        Ok((a, b))
    }

    fn ternary_operands<'a>(operands: &'a [&'a str]) -> GfnResult<(&'a str, &'a str, &'a str)> {
        if operands.len() < 3 {
            return Err(GfnError::SmtSolverError {
                details: format!("Expected at least 3 operands, got {}", operands.len()),
            });
        }
        let a = match operands.first() {
            Some(op) => *op,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing operand 0".to_string(),
                })
            }
        };
        let b = match operands.get(1) {
            Some(op) => *op,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing operand 1".to_string(),
                })
            }
        };
        let c = match operands.get(2) {
            Some(op) => *op,
            None => {
                return Err(GfnError::SmtSolverError {
                    details: "Missing operand 2".to_string(),
                })
            }
        };
        Ok((a, b, c))
    }
}

// Simple hex module for selector hex decoding
mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        if s.len() % 2 != 0 {
            return Err(());
        }
        let mut bytes = Vec::with_capacity(s.len() / 2);
        for i in (0..s.len()).step_by(2) {
            let byte = u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ())?;
            bytes.push(byte);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_arithmetic_and_comparison_opcodes() -> GfnResult<()> {
        let add_smt = ConcolicEvmTranslator::translate_opcode(0x01, &["x", "y"])?;
        assert_eq!(add_smt, "(bvadd x y)");

        let mul_smt = ConcolicEvmTranslator::translate_opcode(0x02, &["x", "#x2710"])?;
        assert_eq!(mul_smt, "(bvmul x #x2710)");

        let lt_smt = ConcolicEvmTranslator::translate_opcode(0x10, &["a", "b"])?;
        assert_eq!(lt_smt, "(ite (bvult a b) (_ bv1 256) (_ bv0 256))");

        Ok(())
    }

    #[test]
    fn test_encode_witness_calldata() {
        let sig = "deposit(uint256)";
        let selector = ConcolicEvmTranslator::compute_function_selector(sig);
        assert_eq!(selector, [0xb6, 0xb5, 0x5f, 0x25]); // keccak256("deposit(uint256)")[0..4]

        let args = vec![U256::from(1000)];
        let calldata = ConcolicEvmTranslator::encode_witness_calldata(sig, &args);

        assert_eq!(calldata.len(), 36);
        assert_eq!(&calldata[0..4], &selector);
        assert_eq!(&calldata[4..36], &U256::from(1000).to_be_bytes::<32>());
    }
}
