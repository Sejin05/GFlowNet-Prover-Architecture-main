#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingEnvironmentState {
    pub mid_price: f64,
    pub spread_bps: f64,
    pub orderbook_imbalance: f64, // (bids - asks) / (bids + asks)
    pub rolling_volatility_5m: f64,
    pub gas_price_gwei: f64,
    pub current_inventory_ratio: f64, // -1.0 to +1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeAction {
    Hold,
    MarketBuy,
    MarketSell,
    LimitBid(u32), // offset bps
    LimitAsk(u32), // offset bps
}

/// Generative Flow Network trading policy mapping market microstructure state to action logits.
#[derive(Debug, Clone)]
pub struct GfnTradingPolicy {
    pub weights: Vec<f64>, // [6 state features * 5 actions]
    pub biases: Vec<f64>,  // [5 actions]
    pub log_z: f64,
}

impl Default for GfnTradingPolicy {
    fn default() -> Self {
        Self {
            weights: vec![
                // Hold, Buy, Sell, LimitBid, LimitAsk
                0.0, 0.5, -0.5, 0.2, -0.2, // mid_price drift
                0.1, -0.2, -0.2, 0.8, 0.8, // spread_bps (wide spread favors market making)
                0.0, 0.9, -0.9, 0.3, -0.3, // orderbook_imbalance
                0.2, -0.4, -0.4, -0.5,
                -0.5, // rolling_volatility (high vol favors hold/caution)
                0.3, -0.6, -0.6, -0.2, -0.2, // gas_price (high gas favors hold)
                0.0, -0.8, 0.8, -0.5, 0.5, // inventory (rebalance towards 0)
            ],
            biases: vec![0.5, 0.0, 0.0, 0.2, 0.2],
            log_z: 1.0,
        }
    }
}

impl GfnTradingPolicy {
    #[must_use]
    pub fn state_to_features(state: &TradingEnvironmentState) -> [f64; 6] {
        [
            (state.mid_price - 3000.0) / 3000.0,
            state.spread_bps,
            state.orderbook_imbalance,
            state.rolling_volatility_5m,
            state.gas_price_gwei,
            state.current_inventory_ratio,
        ]
    }

    #[must_use]
    pub fn compute_action_logits(&self, state: &TradingEnvironmentState) -> [f64; 5] {
        let feats = Self::state_to_features(state);
        let mut logits = [0.0; 5];

        for (action_idx, logit_slot) in logits.iter_mut().enumerate() {
            let mut acc = self.biases.get(action_idx).copied().unwrap_or(0.0);
            for (feat_idx, &val) in feats.iter().enumerate() {
                let w_idx = feat_idx * 5 + action_idx;
                acc += self.weights.get(w_idx).copied().unwrap_or(0.0) * val;
            }
            *logit_slot = acc.clamp(-20.0, 20.0);
        }

        logits
    }

    #[must_use]
    pub fn select_best_action(&self, state: &TradingEnvironmentState) -> TradeAction {
        let logits = self.compute_action_logits(state);
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;

        for (idx, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }

        match best_idx {
            1 => TradeAction::MarketBuy,
            2 => TradeAction::MarketSell,
            3 => TradeAction::LimitBid(5),
            4 => TradeAction::LimitAsk(5),
            _ => TradeAction::Hold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gfn_trading_policy_inventory_rebalancing() {
        let policy = GfnTradingPolicy::default();

        // Strong positive inventory imbalance (+0.9) must favor selling/rebalancing
        let long_state = TradingEnvironmentState {
            mid_price: 3000.0,
            spread_bps: 2.0,
            orderbook_imbalance: 0.0,
            rolling_volatility_5m: 0.01,
            gas_price_gwei: 15.0,
            current_inventory_ratio: 0.9,
        };

        let logits = policy.compute_action_logits(&long_state);
        assert!(
            logits[2] > logits[1],
            "Sell logit must exceed Buy logit when heavily long"
        );
    }
}
