#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

use crate::amm_router::AmmPoolState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub cycle: Vec<String>,
    pub input_amount: u128,
    pub expected_gross_profit: u128,
    pub estimated_gas_cost: u128,
    pub net_profit: i128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashloanArbitrageOpportunity {
    pub cycle: Vec<String>,
    pub borrowed_token: String,
    pub borrowed_amount: u128,
    pub flashloan_fee: u128,
    pub gross_revenue: u128,
    pub estimated_gas_cost: u128,
    pub net_profit: i128,
}

pub struct CyclicArbitrageEngine;

impl CyclicArbitrageEngine {
    /// Detects triangular cross-DEX cyclic arbitrage: A -> B -> C -> A.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn find_triangular_arbitrage(
        pool_ab: &AmmPoolState,
        pool_bc: &AmmPoolState,
        pool_ca: &AmmPoolState,
        token_a: &str,
        token_b: &str,
        token_c: &str,
        input_amount: u128,
        gas_cost_in_token_a: u128,
    ) -> Option<ArbitrageOpportunity> {
        let out_b = pool_ab.get_amount_out(input_amount, token_a)?;
        let out_c = pool_bc.get_amount_out(out_b, token_b)?;
        let final_a = pool_ca.get_amount_out(out_c, token_c)?;

        if final_a > input_amount {
            let gross_profit = final_a - input_amount;
            let net_profit = (gross_profit as i128) - (gas_cost_in_token_a as i128);

            if net_profit > 0 {
                return Some(ArbitrageOpportunity {
                    cycle: vec![
                        token_a.to_string(),
                        token_b.to_string(),
                        token_c.to_string(),
                        token_a.to_string(),
                    ],
                    input_amount,
                    expected_gross_profit: gross_profit,
                    estimated_gas_cost: gas_cost_in_token_a,
                    net_profit,
                });
            }
        }

        None
    }

    /// Detects generalized n-hop cyclic cross-DEX arbitrage across an ordered series of pools.
    ///
    /// Requires `pools.len() >= 2` and `token_path.len() == pools.len() + 1`,
    /// where `token_path[0] == token_path[token_path.len() - 1]`.
    #[must_use]
    pub fn find_multi_hop_arbitrage(
        pools: &[&AmmPoolState],
        token_path: &[&str],
        input_amount: u128,
        gas_cost_in_base_token: u128,
    ) -> Option<ArbitrageOpportunity> {
        if pools.len() < 2 || token_path.len() != pools.len() + 1 {
            return None;
        }

        let first_token = *token_path.first()?;
        let last_token = *token_path.last()?;
        if first_token != last_token {
            return None;
        }

        let mut current_amount = input_amount;
        for (i, pool) in pools.iter().enumerate() {
            let t_in = token_path.get(i)?;
            let current_out = pool.get_amount_out(current_amount, t_in)?;
            current_amount = current_out;
        }

        let final_amount = current_amount;
        if final_amount > input_amount {
            let gross_profit = final_amount - input_amount;
            let net_profit = (gross_profit as i128) - (gas_cost_in_base_token as i128);

            if net_profit > 0 {
                return Some(ArbitrageOpportunity {
                    cycle: token_path.iter().map(|&s| s.to_string()).collect(),
                    input_amount,
                    expected_gross_profit: gross_profit,
                    estimated_gas_cost: gas_cost_in_base_token,
                    net_profit,
                });
            }
        }

        None
    }

    /// Evaluates zero-capital cyclic arbitrage using a simulated flashloan.
    ///
    /// Simulates borrowing `borrowed_amount` of `borrowed_token` at `flashloan_fee_bps`
    /// (e.g., 9 bps for Aave v3, 0 bps for Balancer, or 5 bps for Uniswap v3 flash).
    /// Gross revenue must strictly exceed principal + flashloan fee.
    #[must_use]
    pub fn find_flashloan_arbitrage(
        pools: &[&AmmPoolState],
        token_path: &[&str],
        borrowed_token: &str,
        borrowed_amount: u128,
        flashloan_fee_bps: u32,
        gas_cost_in_borrowed_token: u128,
    ) -> Option<FlashloanArbitrageOpportunity> {
        if pools.len() < 2 || token_path.len() != pools.len() + 1 {
            return None;
        }

        let first_token = *token_path.first()?;
        let last_token = *token_path.last()?;
        if first_token != borrowed_token || last_token != borrowed_token {
            return None;
        }

        let flashloan_fee = borrowed_amount
            .checked_mul(flashloan_fee_bps as u128)?
            / 10_000;
        let total_repayment = borrowed_amount.checked_add(flashloan_fee)?;

        let mut current_amount = borrowed_amount;
        for (i, pool) in pools.iter().enumerate() {
            let t_in = token_path.get(i)?;
            let current_out = pool.get_amount_out(current_amount, t_in)?;
            current_amount = current_out;
        }

        let final_amount = current_amount;
        if final_amount > total_repayment {
            let gross_profit = final_amount - total_repayment;
            let net_profit = (gross_profit as i128) - (gas_cost_in_borrowed_token as i128);

            if net_profit > 0 {
                return Some(FlashloanArbitrageOpportunity {
                    cycle: token_path.iter().map(|&s| s.to_string()).collect(),
                    borrowed_token: borrowed_token.to_string(),
                    borrowed_amount,
                    flashloan_fee,
                    gross_revenue: final_amount,
                    estimated_gas_cost: gas_cost_in_borrowed_token,
                    net_profit,
                });
            }
        }

        None
    }

    /// Discovers optimal cyclic cross-DEX arbitrage across an arbitrary network of AMM pools.
    ///
    /// Automatically explores candidate cycles starting and ending at `base_token` up to `max_hops`
    /// (e.g., 3..=5 hops), simulating exact constant-product swap mechanics with pool fees and slippage,
    /// returning the cycle that yields the maximum net profit after gas cost.
    #[must_use]
    pub fn find_optimal_cyclic_arbitrage_graph(
        pools: &[AmmPoolState],
        base_token: &str,
        input_amount: u128,
        gas_cost_in_base_token: u128,
        max_hops: usize,
    ) -> Option<ArbitrageOpportunity> {
        if pools.len() < 2 || input_amount == 0 || max_hops < 3 {
            return None;
        }

        use std::collections::{BTreeMap, BTreeSet};

        // Construct adjacency list mapping token -> Vec<(target_token, pool_index)>
        let mut adj: BTreeMap<&str, Vec<(&str, usize)>> = BTreeMap::new();
        for (idx, p) in pools.iter().enumerate() {
            adj.entry(p.token0.as_str())
                .or_default()
                .push((p.token1.as_str(), idx));
            adj.entry(p.token1.as_str())
                .or_default()
                .push((p.token0.as_str(), idx));
        }

        if !adj.contains_key(base_token) {
            return None;
        }

        struct SearchContext<'a> {
            base_token: &'a str,
            pools: &'a [AmmPoolState],
            adj: &'a BTreeMap<&'a str, Vec<(&'a str, usize)>>,
            input_amount: u128,
            gas_cost: u128,
            max_hops: usize,
            current_path_tokens: Vec<&'a str>,
            current_path_pools: Vec<usize>,
            visited_tokens: BTreeSet<&'a str>,
            best_opp: Option<ArbitrageOpportunity>,
            best_net_profit: i128,
        }

        fn search_cycles(ctx: &mut SearchContext<'_>, current_token: &str) {
            let Some(neighbors) = ctx.adj.get(current_token) else {
                return;
            };

            for &(next_token, pool_idx) in neighbors {
                if next_token == ctx.base_token && ctx.current_path_tokens.len() >= 3 {
                    // Closed candidate cycle
                    let mut path_pools = ctx.current_path_pools.clone();
                    path_pools.push(pool_idx);

                    let mut sim_amount = ctx.input_amount;
                    let mut valid = true;

                    for (step_idx, &p_idx) in path_pools.iter().enumerate() {
                        let pool = match ctx.pools.get(p_idx) {
                            Some(p) => p,
                            None => {
                                valid = false;
                                break;
                            }
                        };
                        let t_in = match ctx.current_path_tokens.get(step_idx) {
                            Some(&t) => t,
                            None => {
                                valid = false;
                                break;
                            }
                        };
                        match pool.get_amount_out(sim_amount, t_in) {
                            Some(out) if out > 0 => sim_amount = out,
                            _ => {
                                valid = false;
                                break;
                            }
                        }
                    }

                    if valid && sim_amount > ctx.input_amount {
                        let gross = sim_amount - ctx.input_amount;
                        let net = (gross as i128) - (ctx.gas_cost as i128);
                        if net > ctx.best_net_profit {
                            ctx.best_net_profit = net;
                            let mut cycle_tokens = ctx.current_path_tokens.iter().map(|&s| s.to_string()).collect::<Vec<_>>();
                            cycle_tokens.push(ctx.base_token.to_string());
                            ctx.best_opp = Some(ArbitrageOpportunity {
                                cycle: cycle_tokens,
                                input_amount: ctx.input_amount,
                                expected_gross_profit: gross,
                                estimated_gas_cost: ctx.gas_cost,
                                net_profit: net,
                            });
                        }
                    }
                } else if !ctx.visited_tokens.contains(next_token) && ctx.current_path_tokens.len() < ctx.max_hops {
                    ctx.visited_tokens.insert(next_token);
                    ctx.current_path_tokens.push(next_token);
                    ctx.current_path_pools.push(pool_idx);

                    search_cycles(ctx, next_token);

                    ctx.current_path_pools.pop();
                    ctx.current_path_tokens.pop();
                    ctx.visited_tokens.remove(next_token);
                }
            }
        }

        let mut visited_tokens = BTreeSet::new();
        visited_tokens.insert(base_token);

        let mut ctx = SearchContext {
            base_token,
            pools,
            adj: &adj,
            input_amount,
            gas_cost: gas_cost_in_base_token,
            max_hops: max_hops.min(6),
            current_path_tokens: vec![base_token],
            current_path_pools: Vec::new(),
            visited_tokens,
            best_opp: None,
            best_net_profit: 0,
        };

        search_cycles(&mut ctx, base_token);

        ctx.best_opp
    }

    /// Discovers optimal zero-capital flashloan arbitrage across an arbitrary network of AMM pools.
    #[must_use]
    pub fn find_optimal_flashloan_arbitrage_graph(
        pools: &[AmmPoolState],
        borrowed_token: &str,
        borrowed_amount: u128,
        flashloan_fee_bps: u32,
        gas_cost_in_borrowed_token: u128,
        max_hops: usize,
    ) -> Option<FlashloanArbitrageOpportunity> {
        let flashloan_fee = borrowed_amount
            .checked_mul(flashloan_fee_bps as u128)?
            / 10_000;
        let _ = borrowed_amount.checked_add(flashloan_fee)?;

        let opp = Self::find_optimal_cyclic_arbitrage_graph(
            pools,
            borrowed_token,
            borrowed_amount,
            gas_cost_in_borrowed_token + flashloan_fee,
            max_hops,
        )?;

        let final_amount = borrowed_amount.checked_add(opp.expected_gross_profit)?;
        let net_profit = (opp.expected_gross_profit as i128)
            - (flashloan_fee as i128)
            - (gas_cost_in_borrowed_token as i128);

        if net_profit > 0 {
            Some(FlashloanArbitrageOpportunity {
                cycle: opp.cycle,
                borrowed_token: borrowed_token.to_string(),
                borrowed_amount,
                flashloan_fee,
                gross_revenue: final_amount,
                estimated_gas_cost: gas_cost_in_borrowed_token,
                net_profit,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_triangular_arbitrage_discovery() {
        // Construct a price discrepancy:
        // Pool AB: WETH -> DAI ($3000)
        let pool_ab = AmmPoolState {
            pool_id: "WETH-DAI".to_string(),
            token0: "WETH".to_string(),
            token1: "DAI".to_string(),
            reserve0: 10_000_000_000_000_000_000_000, // 10,000 WETH
            reserve1: 30_000_000_000_000_000_000_000_000, // 30,000,000 DAI
            fee_bps: 20,
        };
        // Pool BC: DAI -> USDC ($1.00)
        let pool_bc = AmmPoolState {
            pool_id: "DAI-USDC".to_string(),
            token0: "DAI".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000_000, // 100,000 DAI
            reserve1: 100_000_000_000_000_000_000_000, // 100,000 USDC
            fee_bps: 5,
        };
        // Pool CA: token0 = WETH, token1 = USDC ($2700 discounted ETH price!)
        // When swapping USDC into pool_ca, token_in is USDC (token1), so r_in = reserve1, r_out = reserve0 (WETH)
        let pool_ca = AmmPoolState {
            pool_id: "WETH-USDC".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 10_000_000_000_000_000_000_000, // 10,000 WETH
            reserve1: 27_000_000_000_000_000_000_000_000, // 27,000,000 USDC ($2700/ETH)
            fee_bps: 10,
        };

        let input_weth = 1_000_000_000_000_000_000u128; // 1 WETH
        let gas_cost = 5_000_000_000_000_000u128; // 0.005 WETH

        let opp = CyclicArbitrageEngine::find_triangular_arbitrage(
            &pool_ab, &pool_bc, &pool_ca, "WETH", "DAI", "USDC", input_weth, gas_cost,
        );
        assert!(opp.is_some());
        if let Some(opp) = opp {
            assert!(opp.net_profit > 0);
        }
    }

    #[test]
    fn test_cyclic_multi_hop_arbitrage_discovery() {
        let pool_ab = AmmPoolState {
            pool_id: "WETH-DAI".to_string(),
            token0: "WETH".to_string(),
            token1: "DAI".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 30_000_000_000_000_000_000_000_000,
            fee_bps: 20,
        };
        let pool_bc = AmmPoolState {
            pool_id: "DAI-USDC".to_string(),
            token0: "DAI".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000_000,
            reserve1: 100_000_000_000_000_000_000_000,
            fee_bps: 5,
        };
        let pool_ca = AmmPoolState {
            pool_id: "WETH-USDC".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 27_000_000_000_000_000_000_000_000,
            fee_bps: 10,
        };

        let pools = vec![&pool_ab, &pool_bc, &pool_ca];
        let token_path = vec!["WETH", "DAI", "USDC", "WETH"];
        let input_weth = 1_000_000_000_000_000_000u128;
        let gas_cost = 5_000_000_000_000_000u128;

        let opp = CyclicArbitrageEngine::find_multi_hop_arbitrage(
            &pools, &token_path, input_weth, gas_cost,
        );
        assert!(opp.is_some());
        if let Some(opp) = opp {
            assert!(opp.net_profit > 0);
            assert_eq!(opp.cycle.len(), 4);
        }
    }

    #[test]
    fn test_flashloan_arbitrage_discovery() {
        let pool_ab = AmmPoolState {
            pool_id: "WETH-DAI".to_string(),
            token0: "WETH".to_string(),
            token1: "DAI".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 30_000_000_000_000_000_000_000_000,
            fee_bps: 20,
        };
        let pool_bc = AmmPoolState {
            pool_id: "DAI-USDC".to_string(),
            token0: "DAI".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000_000,
            reserve1: 100_000_000_000_000_000_000_000,
            fee_bps: 5,
        };
        let pool_ca = AmmPoolState {
            pool_id: "WETH-USDC".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 27_000_000_000_000_000_000_000_000,
            fee_bps: 10,
        };

        let pools = vec![&pool_ab, &pool_bc, &pool_ca];
        let token_path = vec!["WETH", "DAI", "USDC", "WETH"];
        let borrowed_weth = 1_000_000_000_000_000_000u128; // 1 WETH borrowed
        let flashloan_fee_bps = 9u32; // Aave 0.09%
        let gas_cost = 5_000_000_000_000_000u128; // 0.005 WETH

        let opp = CyclicArbitrageEngine::find_flashloan_arbitrage(
            &pools,
            &token_path,
            "WETH",
            borrowed_weth,
            flashloan_fee_bps,
            gas_cost,
        );
        assert!(opp.is_some());
        if let Some(opp) = opp {
            assert!(opp.net_profit > 0);
            assert_eq!(opp.borrowed_token, "WETH");
            assert_eq!(opp.borrowed_amount, borrowed_weth);
            assert!(opp.flashloan_fee > 0);
        }
    }

    #[test]
    fn test_optimal_graph_cyclic_arbitrage_discovery() {
        let pool_ab = AmmPoolState {
            pool_id: "WETH-DAI".to_string(),
            token0: "WETH".to_string(),
            token1: "DAI".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 30_000_000_000_000_000_000_000_000,
            fee_bps: 20,
        };
        let pool_bc = AmmPoolState {
            pool_id: "DAI-USDC".to_string(),
            token0: "DAI".to_string(),
            token1: "USDC".to_string(),
            reserve0: 100_000_000_000_000_000_000_000,
            reserve1: 100_000_000_000_000_000_000_000,
            fee_bps: 5,
        };
        let pool_ca = AmmPoolState {
            pool_id: "WETH-USDC".to_string(),
            token0: "WETH".to_string(),
            token1: "USDC".to_string(),
            reserve0: 10_000_000_000_000_000_000_000,
            reserve1: 27_000_000_000_000_000_000_000_000,
            fee_bps: 10,
        };
        // Extra unrelated pool to test graph traversal filtering
        let pool_extra = AmmPoolState {
            pool_id: "WBTC-USDC".to_string(),
            token0: "WBTC".to_string(),
            token1: "USDC".to_string(),
            reserve0: 1_000_000_000_000_000_000_000,
            reserve1: 60_000_000_000_000_000_000_000_000,
            fee_bps: 30,
        };

        let pools = vec![pool_extra, pool_ab, pool_bc, pool_ca];
        let input_weth = 1_000_000_000_000_000_000u128; // 1 WETH
        let gas_cost = 5_000_000_000_000_000u128; // 0.005 WETH

        let opp = CyclicArbitrageEngine::find_optimal_cyclic_arbitrage_graph(
            &pools, "WETH", input_weth, gas_cost, 4,
        );
        assert!(opp.is_some());
        if let Some(opp) = opp {
            assert!(opp.net_profit > 0);
            assert_eq!(opp.cycle.first().map(String::as_str), Some("WETH"));
            assert_eq!(opp.cycle.last().map(String::as_str), Some("WETH"));
            assert!(opp.cycle.len() >= 4);
        }

        let flash_opp = CyclicArbitrageEngine::find_optimal_flashloan_arbitrage_graph(
            &pools, "WETH", input_weth, 9, gas_cost, 4,
        );
        assert!(flash_opp.is_some());
        if let Some(flash) = flash_opp {
            assert!(flash.net_profit > 0);
            assert_eq!(flash.borrowed_token, "WETH");
        }
    }
}
