#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_drawdown_pct: f64, // e.g. 0.05 = 5% max drawdown circuit breaker
    pub max_position_size_usd: f64, // Maximum exposure per single asset
    pub max_portfolio_leverage: f64, // Maximum debt / equity ratio
    pub value_at_risk_95_pct: f64, // VaR threshold
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_drawdown_pct: 0.08,
            max_position_size_usd: 100_000.0,
            max_portfolio_leverage: 2.0,
            value_at_risk_95_pct: 0.03,
        }
    }
}

pub struct PortfolioRiskManager {
    pub peak_portfolio_value: f64,
    pub current_portfolio_value: f64,
    pub limits: RiskLimits,
    pub positions: HashMap<String, Position>,
    pub circuit_breaker_tripped: bool,
}

impl PortfolioRiskManager {
    #[must_use]
    pub fn new(initial_capital: f64, limits: RiskLimits) -> Self {
        Self {
            peak_portfolio_value: initial_capital,
            current_portfolio_value: initial_capital,
            limits,
            positions: HashMap::new(),
            circuit_breaker_tripped: false,
        }
    }

    pub fn update_portfolio_value(&mut self, new_value: f64) {
        self.current_portfolio_value = new_value;
        if new_value > self.peak_portfolio_value {
            self.peak_portfolio_value = new_value;
        }

        if self.peak_portfolio_value > 0.0 {
            let drawdown = (self.peak_portfolio_value - new_value) / self.peak_portfolio_value;
            if drawdown >= self.limits.max_drawdown_pct {
                self.circuit_breaker_tripped = true;
            }
        }
    }

    #[must_use]
    pub fn can_open_position(&self, symbol: &str, size_usd: f64) -> bool {
        if self.circuit_breaker_tripped {
            return false;
        }

        if size_usd > self.limits.max_position_size_usd {
            return false;
        }

        let existing = self
            .positions
            .get(symbol)
            .map_or(0.0, |p| p.quantity * p.current_price);
        if existing + size_usd > self.limits.max_position_size_usd {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_circuit_breaker() {
        let mut risk = PortfolioRiskManager::new(100_000.0, RiskLimits::default());
        assert!(!risk.circuit_breaker_tripped);
        assert!(risk.can_open_position("WETH", 50_000.0));

        // Drawdown of 10% trips 8% circuit breaker
        risk.update_portfolio_value(89_000.0);
        assert!(risk.circuit_breaker_tripped);
        assert!(!risk.can_open_position("WETH", 10_000.0));
    }
}
