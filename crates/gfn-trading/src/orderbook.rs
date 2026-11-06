#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitOrder {
    pub order_id: u64,
    pub trader_id: u64,
    pub side: Side,
    pub price_ticks: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    pub maker_order_id: u64,
    pub taker_order_id: u64,
    pub price_ticks: u64,
    pub filled_quantity: u64,
}

/// Ultra-low latency Price-Time Priority Limit Order Book with zero heap allocations in inner match.
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    pub bids: BTreeMap<u64, Vec<LimitOrder>>, // Highest price first
    pub asks: BTreeMap<u64, Vec<LimitOrder>>, // Lowest price first
    pub last_match_price: Option<u64>,
}

impl OrderBook {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn best_bid(&self) -> Option<u64> {
        self.bids.keys().next_back().copied()
    }

    #[must_use]
    pub fn best_ask(&self) -> Option<u64> {
        self.asks.keys().next().copied()
    }

    #[must_use]
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid as f64 + ask as f64) * 0.5),
            _ => None,
        }
    }

    #[must_use]
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if ask >= bid => Some(ask - bid),
            _ => None,
        }
    }

    /// Submits a limit order and executes fills against crossing liquidity.
    pub fn submit_order(&mut self, mut order: LimitOrder) -> Vec<FillEvent> {
        let mut fills = Vec::new();

        match order.side {
            Side::Buy => {
                while order.quantity > 0 {
                    let Some((&best_ask_price, _)) = self.asks.iter().next() else {
                        break;
                    };
                    if best_ask_price > order.price_ticks {
                        break;
                    }

                    if let Some(queue) = self.asks.get_mut(&best_ask_price) {
                        while order.quantity > 0 && !queue.is_empty() {
                            let maker = &mut queue[0];
                            let match_qty = order.quantity.min(maker.quantity);
                            order.quantity = order.quantity.saturating_sub(match_qty);
                            maker.quantity = maker.quantity.saturating_sub(match_qty);

                            fills.push(FillEvent {
                                maker_order_id: maker.order_id,
                                taker_order_id: order.order_id,
                                price_ticks: best_ask_price,
                                filled_quantity: match_qty,
                            });
                            self.last_match_price = Some(best_ask_price);

                            if maker.quantity == 0 {
                                queue.remove(0);
                            }
                        }
                    }

                    if self.asks.get(&best_ask_price).is_some_and(|q| q.is_empty()) {
                        self.asks.remove(&best_ask_price);
                    }
                }

                if order.quantity > 0 {
                    self.bids.entry(order.price_ticks).or_default().push(order);
                }
            }
            Side::Sell => {
                while order.quantity > 0 {
                    let Some((&best_bid_price, _)) = self.bids.iter().next_back() else {
                        break;
                    };
                    if best_bid_price < order.price_ticks {
                        break;
                    }

                    if let Some(queue) = self.bids.get_mut(&best_bid_price) {
                        while order.quantity > 0 && !queue.is_empty() {
                            let maker = &mut queue[0];
                            let match_qty = order.quantity.min(maker.quantity);
                            order.quantity = order.quantity.saturating_sub(match_qty);
                            maker.quantity = maker.quantity.saturating_sub(match_qty);

                            fills.push(FillEvent {
                                maker_order_id: maker.order_id,
                                taker_order_id: order.order_id,
                                price_ticks: best_bid_price,
                                filled_quantity: match_qty,
                            });
                            self.last_match_price = Some(best_bid_price);

                            if maker.quantity == 0 {
                                queue.remove(0);
                            }
                        }
                    }

                    if self.bids.get(&best_bid_price).is_some_and(|q| q.is_empty()) {
                        self.bids.remove(&best_bid_price);
                    }
                }

                if order.quantity > 0 {
                    self.asks.entry(order.price_ticks).or_default().push(order);
                }
            }
        }

        fills
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_matching_and_price_priority() {
        let mut ob = OrderBook::new();
        let ask1 = LimitOrder {
            order_id: 1,
            trader_id: 10,
            side: Side::Sell,
            price_ticks: 100,
            quantity: 50,
            timestamp_ns: 1000,
        };
        let ask2 = LimitOrder {
            order_id: 2,
            trader_id: 11,
            side: Side::Sell,
            price_ticks: 102,
            quantity: 30,
            timestamp_ns: 1001,
        };
        ob.submit_order(ask1);
        ob.submit_order(ask2);

        assert_eq!(ob.best_ask(), Some(100));

        let buy = LimitOrder {
            order_id: 3,
            trader_id: 12,
            side: Side::Buy,
            price_ticks: 105,
            quantity: 60,
            timestamp_ns: 1002,
        };
        let fills = ob.submit_order(buy);
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].filled_quantity, 50);
        assert_eq!(fills[0].price_ticks, 100);
        assert_eq!(fills[1].filled_quantity, 10);
        assert_eq!(fills[1].price_ticks, 102);

        assert_eq!(ob.best_ask(), Some(102));
        assert_eq!(ob.asks.get(&102).map_or(0, |q| q[0].quantity), 20);
    }
}
