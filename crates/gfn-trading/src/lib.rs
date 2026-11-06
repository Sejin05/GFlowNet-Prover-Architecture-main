#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Autonomous High-Frequency & Cross-DEX Trading Model Engine (`gfn-trading`).
//!
//! Synthesizes generative flow network policies for optimal order routing,
//! cyclical cross-DEX arbitrage discovery, atomic MEV bundling, and
//! real-time slippage / fee minimization under Cancun EVM gas constraints.

pub mod amm_router;
pub mod arbitrage;
pub mod orderbook;
pub mod risk;
pub mod strategy;

pub use amm_router::{
    AmmPoolState, ConcentratedArbitrageScanner, ConcentratedLiquidityPool, MultiHopRoute,
    OptimalSplitter, RouteHop,
};
pub use arbitrage::{ArbitrageOpportunity, CyclicArbitrageEngine, FlashloanArbitrageOpportunity};
pub use orderbook::{LimitOrder, OrderBook, Side};
pub use risk::{PortfolioRiskManager, Position, RiskLimits};
pub use strategy::{GfnTradingPolicy, TradeAction, TradingEnvironmentState};
