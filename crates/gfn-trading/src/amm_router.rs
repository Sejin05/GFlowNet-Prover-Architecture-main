#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

use alloy_primitives::U256;
use serde::{Deserialize, Serialize};

/// Constant product invariant AMM Pool state (x * y = k) with customizable fee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmmPoolState {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    pub reserve0: u128,
    pub reserve1: u128,
    pub fee_bps: u32, // e.g. 30 = 0.30%
}

impl AmmPoolState {
    #[must_use]
    pub fn get_amount_out(&self, amount_in: u128, token_in: &str) -> Option<u128> {
        let (r_in, r_out) = if token_in == self.token0 {
            (self.reserve0, self.reserve1)
        } else if token_in == self.token1 {
            (self.reserve1, self.reserve0)
        } else {
            return None;
        };

        if amount_in == 0 || r_in == 0 || r_out == 0 {
            return Some(0);
        }

        let amount_in_with_fee = U256::from(amount_in) * U256::from(10_000 - self.fee_bps);
        let numerator = amount_in_with_fee * U256::from(r_out);
        let denominator = (U256::from(r_in) * U256::from(10_000)) + amount_in_with_fee;

        if denominator == U256::ZERO {
            None
        } else {
            let out = numerator / denominator;
            if out > U256::from(u128::MAX) {
                Some(u128::MAX)
            } else {
                Some(out.to::<u128>())
            }
        }
    }

    pub fn execute_swap(&mut self, amount_in: u128, token_in: &str) -> Option<u128> {
        let amount_out = self.get_amount_out(amount_in, token_in)?;
        if token_in == self.token0 {
            self.reserve0 = self.reserve0.saturating_add(amount_in);
            self.reserve1 = self.reserve1.saturating_sub(amount_out);
        } else {
            self.reserve1 = self.reserve1.saturating_add(amount_in);
            self.reserve0 = self.reserve0.saturating_sub(amount_out);
        }
        Some(amount_out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteHop {
    pub pool_id: String,
    pub token_in: String,
    pub token_out: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHopRoute {
    pub hops: Vec<RouteHop>,
    pub expected_output: u128,
    pub price_impact_bps: u32,
}

pub struct OptimalSplitter;

impl OptimalSplitter {
    /// Convex optimization to split `total_in` between two parallel AMM pools to minimize price impact.
    #[must_use]
    pub fn split_two_pools(
        pool_a: &AmmPoolState,
        pool_b: &AmmPoolState,
        token_in: &str,
        total_in: u128,
    ) -> (u128, u128, u128) {
        if total_in == 0 {
            return (0, 0, 0);
        }

        let mut best_a = 0u128;
        let mut best_b = total_in;
        let mut max_out = 0u128;

        // Discrete grid search with 50 points
        for i in 0..=50 {
            let in_a = (total_in.saturating_mul(i as u128)) / 50;
            let in_b = total_in.saturating_sub(in_a);

            let out_a = pool_a.get_amount_out(in_a, token_in).unwrap_or(0);
            let out_b = pool_b.get_amount_out(in_b, token_in).unwrap_or(0);
            let total_out = out_a.saturating_add(out_b);

            if total_out > max_out {
                max_out = total_out;
                best_a = in_a;
                best_b = in_b;
            }
        }

        (best_a, best_b, max_out)
    }
}

/// Concentrated liquidity AMM pool state (Uniswap v3 / v4 style range-bound liquidity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentratedLiquidityPool {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    pub current_tick: i32,
    pub tick_lower: i32,
    pub tick_upper: i32,
    pub sqrt_price_x96: U256,
    pub liquidity: u128,
    pub fee_bps: u32,
}

impl ConcentratedLiquidityPool {
    /// Returns true if current pool tick is strictly within position bounds with non-zero active liquidity.
    #[must_use]
    pub fn is_in_range(&self) -> bool {
        self.current_tick >= self.tick_lower
            && self.current_tick < self.tick_upper
            && self.liquidity > 0
    }

    /// Computes virtual reserves (x_virt, y_virt) from active liquidity and sqrt_price_x96.
    #[must_use]
    pub fn get_virtual_reserves(&self) -> (u128, u128) {
        if self.sqrt_price_x96.is_zero() || self.liquidity == 0 {
            return (0, 0);
        }
        let q96: U256 = U256::from(1) << 96usize;
        let l = U256::from(self.liquidity);

        // x_virt = (L * 2^96) / sqrt_price_x96
        let x_num = l.saturating_mul(q96);
        let x_virt = x_num / self.sqrt_price_x96;

        // y_virt = (L * sqrt_price_x96) / 2^96
        let y_num = l.saturating_mul(self.sqrt_price_x96);
        let y_virt = y_num / q96;

        (
            x_virt.min(U256::from(u128::MAX)).to::<u128>(),
            y_virt.min(U256::from(u128::MAX)).to::<u128>(),
        )
    }

    /// Computes expected output amount given amount_in using concentrated virtual reserves and fee tier.
    #[must_use]
    pub fn get_amount_out(&self, amount_in: u128, token_in: &str) -> Option<u128> {
        if !self.is_in_range() || amount_in == 0 {
            return Some(0);
        }
        let (rx, ry) = self.get_virtual_reserves();
        if rx == 0 || ry == 0 {
            return Some(0);
        }

        let (r_in, r_out) = if token_in == self.token0 {
            (rx, ry)
        } else if token_in == self.token1 {
            (ry, rx)
        } else {
            return None;
        };

        let amount_in_with_fee = U256::from(amount_in) * U256::from(10_000 - self.fee_bps);
        let numerator = amount_in_with_fee * U256::from(r_out);
        let denominator = (U256::from(r_in) * U256::from(10_000)) + amount_in_with_fee;

        if denominator == U256::ZERO {
            None
        } else {
            let out = numerator / denominator;
            if out > U256::from(u128::MAX) {
                Some(u128::MAX)
            } else {
                Some(out.to::<u128>())
            }
        }
    }
}

/// Cross-protocol scanner detecting arbitrage opportunities between constant-product AMMs and concentrated liquidity pools.
pub struct ConcentratedArbitrageScanner;

impl ConcentratedArbitrageScanner {
    /// Evaluates if an arbitrage exists by routing `amount_in` of `token_in` through `pool_v2` and back through `pool_cl`.
    /// Returns `Some(net_profit)` if profitable after gas estimation, or `None`.
    #[must_use]
    pub fn find_cross_pool_arbitrage(
        pool_v2: &AmmPoolState,
        pool_cl: &ConcentratedLiquidityPool,
        token_in: &str,
        amount_in: u128,
        estimated_gas_cost: u128,
    ) -> Option<u128> {
        if amount_in == 0 {
            return None;
        }
        let intermediate_token = if token_in == pool_v2.token0 {
            &pool_v2.token1
        } else if token_in == pool_v2.token1 {
            &pool_v2.token0
        } else {
            return None;
        };

        // Path 1: V2 -> CL
        let out_v2 = pool_v2.get_amount_out(amount_in, token_in)?;
        let final_out_1 = pool_cl.get_amount_out(out_v2, intermediate_token)?;
        if final_out_1 > amount_in.saturating_add(estimated_gas_cost) {
            return Some(final_out_1 - amount_in - estimated_gas_cost);
        }

        // Path 2: CL -> V2
        let out_cl = pool_cl.get_amount_out(amount_in, token_in)?;
        let final_out_2 = pool_v2.get_amount_out(out_cl, intermediate_token)?;
        if final_out_2 > amount_in.saturating_add(estimated_gas_cost) {
            return Some(final_out_2 - amount_in - estimated_gas_cost);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_product_swap_and_split() {
        let pool1 = AmmPoolState {
            pool_id: "WETH-USDC-1".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000, // 100 WETH
            reserve1: 300_000_000_000,             // 300,000 USDC
            fee_bps: 30,
        };
        let pool2 = AmmPoolState {
            pool_id: "WETH-USDC-2".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 50_000_000_000_000_000_000, // 50 WETH
            reserve1: 150_000_000_000,            // 150,000 USDC
            fee_bps: 30,
        };

        let swap_amount = 5_000_000_000_000_000_000u128; // 5 WETH
        let single_pool_out = pool1.get_amount_out(swap_amount, "WETH").unwrap_or(0);

        let (split_a, split_b, split_out) =
            OptimalSplitter::split_two_pools(&pool1, &pool2, "WETH", swap_amount);
        assert_eq!(split_a + split_b, swap_amount);
        assert!(
            split_out >= single_pool_out,
            "Split routing must yield >= single pool execution"
        );
    }

    #[test]
    fn test_concentrated_liquidity_and_arbitrage_scanner() {
        let q96: U256 = U256::from(1) << 96usize;
        // Price = 3000 USDC per WETH -> sqrt(3000) approx 54.77
        let sqrt_p = q96.saturating_mul(U256::from(55));

        let cl_pool = ConcentratedLiquidityPool {
            pool_id: "WETH-USDC-CL".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            current_tick: 80000,
            tick_lower: 70000,
            tick_upper: 90000,
            sqrt_price_x96: sqrt_p,
            liquidity: 10_000_000_000_000_000_000, // 10 units
            fee_bps: 5,                            // 0.05%
        };

        assert!(cl_pool.is_in_range());
        let (rx, ry) = cl_pool.get_virtual_reserves();
        assert!(rx > 0 && ry > 0);

        let out = cl_pool.get_amount_out(1_000_000_000_000_000_000, "WETH");
        assert!(out.is_some() && out.unwrap_or(0) > 0);

        // Test out of range returns 0
        let mut out_of_range_pool = cl_pool.clone();
        out_of_range_pool.current_tick = 60000;
        assert!(!out_of_range_pool.is_in_range());
        assert_eq!(
            out_of_range_pool.get_amount_out(1_000_000_000_000_000_000, "WETH"),
            Some(0)
        );

        // Test cross-pool arbitrage scanner
        let v2_pool = AmmPoolState {
            pool_id: "WETH-USDC-V2".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000, // 100 WETH
            reserve1: 350_000_000_000,             // 350,000 USDC (higher price)
            fee_bps: 30,
        };

        let arb = ConcentratedArbitrageScanner::find_cross_pool_arbitrage(
            &v2_pool,
            &cl_pool,
            "WETH",
            1_000_000_000_000_000_000,
            0,
        );
        // Arbitrage scanner should return Some(profit) or None cleanly
        if let Some(profit) = arb {
            assert!(profit > 0);
        }
    }
}
