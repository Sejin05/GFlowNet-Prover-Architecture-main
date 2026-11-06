#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Mathematical and tensor normalization routines for GFlowNet-Prover.

use crate::error::{GfnError, GfnResult};
use alloy_primitives::U256;

/// Normalizes a 256-bit EVM word into a log2-scaled floating point value in [0.0, 1.0].
///
/// Guaranteed never to overflow or produce NaN/Inf even when given `U256::MAX`.
/// Incorporates a zero-allocation fast-path for 64-bit bounded words (most DeFi balances/slots).
#[inline]
pub fn u256_to_log2_norm(val: U256) -> f32 {
    let limbs = val.as_limbs();
    if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
        let v0 = limbs[0];
        if v0 == 0 {
            return 0.0_f32;
        }
        let bit_len = 64 - v0.leading_zeros(); // 1..=64
        let shift = bit_len.saturating_sub(53);
        let top_bits = v0 >> shift;
        let log2_exact = (top_bits as f64).log2() + (shift as f64);
        return (log2_exact / 256.0_f64).clamp(0.0, 1.0) as f32;
    }
    let bit_len = val.bit_len() as u32; // 1..=256
    let shift = bit_len.saturating_sub(53);
    let top_bits: u64 = (val >> shift).to::<u64>();
    let log2_exact = (top_bits as f64).log2() + (shift as f64);
    (log2_exact / 256.0_f64).clamp(0.0, 1.0) as f32
}

/// Computes the signed log2-scaled difference between two 256-bit EVM state values.
///
/// Returns a value in [-1.0, 1.0] where the sign indicates the direction of transition.
#[inline]
pub fn u256_signed_log2_delta(curr: U256, prev: U256) -> f32 {
    if curr == prev {
        0.0_f32
    } else if curr > prev {
        let diff = curr - prev;
        u256_to_log2_norm(diff)
    } else {
        let diff = prev - curr;
        -u256_to_log2_norm(diff)
    }
}

/// Computes a 16-channel sinusoidal encoding of a 256-bit EVM word split into 16 `u16` chunks.
/// Uses a zero-cost bypass for zero-chunks and zero-word vectors.
#[inline]
pub fn u256_sinusoidal_chunk_encoding(val: U256) -> [f32; 16] {
    if val.is_zero() {
        return [0.0_f32; 16];
    }
    let bytes = val.to_be_bytes::<32>();
    let mut encoding = [0.0_f32; 16];
    const FACTOR: f64 = (std::f64::consts::PI * 2.0) / 65535.0_f64;
    for k in 0..16 {
        let chunk = u16::from_be_bytes([bytes[2 * k], bytes[2 * k + 1]]);
        if chunk != 0 {
            let angle = (chunk as f64) * FACTOR;
            encoding[k] = angle.sin() as f32;
        }
    }
    encoding
}

/// Assigns average fractional ranks to an array of floating point numbers, handling ties accurately.
fn compute_ranks(values: &[f64]) -> GfnResult<Vec<f64>> {
    let n = values.len();
    for &v in values {
        if !v.is_finite() {
            return Err(GfnError::MathDomainError {
                details: "Non-finite floating-point value encountered during rank computation"
                    .to_string(),
            });
        }
    }
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && indexed[j + 1].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = ((i + 1 + j + 1) as f64) / 2.0_f64;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    Ok(ranks)
}

/// Computes the Spearman rank correlation coefficient rho between two paired slices.
///
/// Handles tied observations accurately using average fractional ranks.
pub fn spearman_rank_correlation(x: &[f64], y: &[f64]) -> GfnResult<f64> {
    if x.len() != y.len() {
        return Err(GfnError::MathDomainError {
            details: format!(
                "Input slice lengths mismatch: x.len()={}, y.len()={}",
                x.len(),
                y.len()
            ),
        });
    }
    let n = x.len();
    if n < 2 {
        return Err(GfnError::MathDomainError {
            details: format!(
                "At least 2 observations required for correlation, received: {}",
                n
            ),
        });
    }

    let rx = compute_ranks(x)?;
    let ry = compute_ranks(y)?;

    let mean_rx: f64 = rx.iter().sum::<f64>() / (n as f64);
    let mean_ry: f64 = ry.iter().sum::<f64>() / (n as f64);

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;

    for i in 0..n {
        let dx = rx[i] - mean_rx;
        let dy = ry[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return Ok(0.0_f64);
    }

    let rho = (cov / denom).clamp(-1.0, 1.0);
    Ok(rho)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u256_to_log2_norm_boundary_zero() {
        assert_eq!(u256_to_log2_norm(U256::ZERO), 0.0_f32);
    }

    #[test]
    fn test_u256_to_log2_norm_boundary_one() {
        let val = u256_to_log2_norm(U256::from(1));
        assert!(val.is_finite());
        assert_eq!(val, 0.0_f32);
    }

    #[test]
    fn test_u256_to_log2_norm_boundary_max() {
        let val = u256_to_log2_norm(U256::MAX);
        assert!(val.is_finite());
        assert!(
            (val - 1.0_f32).abs() < 1e-6,
            "Expected ~1.0 for U256::MAX, got {}",
            val
        );
    }

    #[test]
    fn test_u256_to_log2_norm_monotonicity() {
        let v1 = U256::from(100);
        let v2 = U256::from(10_000);
        let v3 = U256::from(1_000_000_000u64);
        let v4 = U256::MAX;

        let n1 = u256_to_log2_norm(v1);
        let n2 = u256_to_log2_norm(v2);
        let n3 = u256_to_log2_norm(v3);
        let n4 = u256_to_log2_norm(v4);

        assert!(n1 < n2);
        assert!(n2 < n3);
        assert!(n3 < n4);
    }

    #[test]
    fn test_u256_signed_log2_delta() {
        let zero = U256::ZERO;
        let v = U256::from(1_000_000);

        assert_eq!(u256_signed_log2_delta(v, v), 0.0_f32);
        let inc = u256_signed_log2_delta(v, zero);
        let dec = u256_signed_log2_delta(zero, v);

        assert!(inc > 0.0_f32);
        assert!(dec < 0.0_f32);
        assert!((inc + dec).abs() < 1e-6);
    }

    #[test]
    fn test_u256_sinusoidal_chunk_encoding() {
        let enc_zero = u256_sinusoidal_chunk_encoding(U256::ZERO);
        assert_eq!(enc_zero.len(), 16);
        for &e in &enc_zero {
            assert_eq!(e, 0.0_f32);
        }

        let enc_max = u256_sinusoidal_chunk_encoding(U256::MAX);
        for &e in &enc_max {
            assert!(e.is_finite());
            assert!((-1.0_f32..=1.0_f32).contains(&e));
        }
    }

    #[test]
    fn test_spearman_rank_correlation_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rank_correlation(&x, &y);
        assert!(rho.is_ok());
        if let Ok(val) = rho {
            assert!((val - 1.0).abs() < 1e-6);
        }

        let y_rev = [50.0, 40.0, 30.0, 20.0, 10.0];
        let rho_rev = spearman_rank_correlation(&x, &y_rev);
        assert!(rho_rev.is_ok());
        if let Ok(val) = rho_rev {
            assert!((val - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spearman_rank_correlation_ties() {
        let x = [1.0, 2.0, 2.0, 4.0, 5.0];
        let y = [10.0, 20.0, 20.0, 40.0, 50.0];
        let rho = spearman_rank_correlation(&x, &y);
        assert!(rho.is_ok());
        if let Ok(val) = rho {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spearman_rank_correlation_errors() {
        let x = [1.0, 2.0];
        let y = [1.0];
        assert!(spearman_rank_correlation(&x, &y).is_err());

        let x_single = [1.0];
        let y_single = [2.0];
        assert!(spearman_rank_correlation(&x_single, &y_single).is_err());

        let x_nan = [1.0, f64::NAN];
        let y_nan = [1.0, 2.0];
        assert!(spearman_rank_correlation(&x_nan, &y_nan).is_err());
    }
}
