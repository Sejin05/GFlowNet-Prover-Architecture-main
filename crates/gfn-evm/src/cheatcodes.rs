#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! Foundry cheatcode runtime for GFlowNet-Prover.
//!
//! Intercepts calls to the canonical Foundry cheatcode address
//! `0x7109709ECfa91a80626fF3989D68f67F5b1DD12D`, implementing `vm.prank`,
//! `vm.startPrank`, `vm.stopPrank`, `vm.warp`, `vm.roll`, `deal`,
//! `vm.mockCall`, `vm.clearMockedCalls`, `vm.expectEmit`, and `vm.recordLogs`.

use alloy_primitives::{address, Address, Bytes, U256};
use alloy_sol_types::{sol, SolInterface};
use gfn_common::error::{GfnError, GfnResult};
use revm::{primitives::db::Database, EvmContext};
use std::collections::HashMap;

/// Canonical Foundry cheatcode contract address:
/// `0x7109709ECfa91a80626fF3989D68f67F5b1DD12D`
pub const CHEATCODE_ADDRESS: Address = address!("7109709ECfa91a80626fF3989D68f67F5b1DD12D");

/// The 5 canonical actor personas defined for multi-agent adversarial exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ActorPersona {
    /// 0: Universal Dynamic Callback Router Contract (capable of re-entering on flash-loan/ERC-777/Uniswap hooks).
    Attacker0 = 0,
    /// 1: Sybil secondary attacker address.
    Attacker1Sybil = 1,
    /// 2: Benign protocol user.
    NormalUser = 2,
    /// 3: Liquidity provider providing AMM/vault liquidity.
    LiquidityProvider = 3,
    /// 4: Liquidator / arbitrage bot / keeper.
    LiquidatorKeeper = 4,
}

impl ActorPersona {
    /// Array of all 5 canonical personas.
    pub const ALL: [Self; 5] = [
        Self::Attacker0,
        Self::Attacker1Sybil,
        Self::NormalUser,
        Self::LiquidityProvider,
        Self::LiquidatorKeeper,
    ];

    /// Attempts to convert an actor index `0..=4` into an `ActorPersona`.
    #[inline]
    pub fn try_from_u8(val: u8) -> GfnResult<Self> {
        match val {
            0 => Ok(Self::Attacker0),
            1 => Ok(Self::Attacker1Sybil),
            2 => Ok(Self::NormalUser),
            3 => Ok(Self::LiquidityProvider),
            4 => Ok(Self::LiquidatorKeeper),
            other => Err(GfnError::RevmExecution {
                env_idx: 0,
                message: format!("Invalid actor persona index: {other}"),
            }),
        }
    }

    /// Canonical Ethereum address assigned to this persona.
    #[inline]
    pub const fn address(&self) -> Address {
        match self {
            Self::Attacker0 => address!("00000000000000000000000000000000000000A0"),
            Self::Attacker1Sybil => address!("00000000000000000000000000000000000000A1"),
            Self::NormalUser => address!("00000000000000000000000000000000000000B2"),
            Self::LiquidityProvider => address!("00000000000000000000000000000000000000C3"),
            Self::LiquidatorKeeper => address!("00000000000000000000000000000000000000D4"),
        }
    }

    /// Returns `true` if the given address is one of the 5 canonical actor personas.
    #[inline]
    pub fn is_actor_address(addr: &Address) -> bool {
        Self::ALL.iter().any(|p| p.address() == *addr)
    }

    /// Precomputed 256-bit unsigned integer representations of the 5 canonical actor addresses.
    pub const ALL_U256: [U256; 5] = [
        U256::from_limbs([0xa0, 0, 0, 0]),
        U256::from_limbs([0xa1, 0, 0, 0]),
        U256::from_limbs([0xb2, 0, 0, 0]),
        U256::from_limbs([0xc3, 0, 0, 0]),
        U256::from_limbs([0xd4, 0, 0, 0]),
    ];

    /// Returns `true` if the given U256 word corresponds to one of the 5 canonical actor addresses.
    #[inline]
    pub fn is_actor_u256(val: U256) -> bool {
        Self::ALL_U256.contains(&val)
    }
}

// Sol interface declaration for Foundry cheatcodes
sol! {
    interface VmCheatcodes {
        function prank(address msgSender) external;
        function prank(address msgSender, address txOrigin) external;
        function startPrank(address msgSender) external;
        function startPrank(address msgSender, address txOrigin) external;
        function stopPrank() external;
        function warp(uint256 newTimestamp) external;
        function roll(uint256 newBlockNumber) external;
        function deal(address who, uint256 newBalance) external;
        function mockCall(address target, bytes calldata data, bytes calldata returndata) external;
        function clearMockedCalls() external;
        function expectEmit(bool checkTopic1, bool checkTopic2, bool checkTopic3, bool checkData) external;
        function recordLogs() external;
    }
}

/// Recorded prank state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrankRecord {
    pub caller: Address,
    pub origin: Option<Address>,
    pub single_call: bool,
}

/// State maintained across cheatcode invocations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheatcodeState {
    pub active_prank: Option<PrankRecord>,
    pub timestamp_override: Option<U256>,
    pub block_number_override: Option<U256>,
    pub deals: Vec<(Address, U256)>,
    pub mocked_calls: HashMap<(Address, Bytes), Bytes>,
    pub recording_logs: bool,
    pub expect_emit: Option<(bool, bool, bool, bool)>,
}

impl CheatcodeState {
    /// Creates a new empty `CheatcodeState`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets all overrides.
    #[inline]
    pub fn reset(&mut self) {
        self.active_prank = None;
        self.timestamp_override = None;
        self.block_number_override = None;
        self.deals.clear();
        self.mocked_calls.clear();
        self.recording_logs = false;
        self.expect_emit = None;
    }

    /// Checks if a call has been mocked via `vm.mockCall`. Returns the mock return data if found.
    #[inline]
    pub fn find_mocked_call(&self, target: &Address, calldata: &[u8]) -> Option<&Bytes> {
        if self.mocked_calls.is_empty() {
            return None;
        }
        for ((mock_addr, mock_data), ret) in &self.mocked_calls {
            if mock_addr == target && calldata.starts_with(mock_data) {
                return Some(ret);
            }
        }
        None
    }

    /// Checks if a call targets the Foundry cheatcode address.
    #[inline]
    pub fn is_cheatcode_call(target: &Address) -> bool {
        *target == CHEATCODE_ADDRESS
    }

    /// Returns the active prank (caller, optional origin) and consumes it if single-call.
    pub fn consume_prank_for_next_tx(&mut self) -> Option<(Address, Option<Address>)> {
        match self.active_prank.take() {
            Some(prank) => {
                let res = (prank.caller, prank.origin);
                if !prank.single_call {
                    self.active_prank = Some(prank);
                }
                Some(res)
            }
            None => None,
        }
    }

    /// Dispatches and executes an intercepted cheatcode call.
    pub fn dispatch_call<DB: Database>(
        &mut self,
        calldata: &[u8],
        context: &mut EvmContext<DB>,
    ) -> GfnResult<Bytes> {
        let decoded = VmCheatcodes::VmCheatcodesCalls::abi_decode(calldata, true).map_err(|e| {
            GfnError::RevmExecution {
                env_idx: 0,
                message: format!("Failed to decode cheatcode ABI: {e}"),
            }
        })?;

        match decoded {
            VmCheatcodes::VmCheatcodesCalls::prank_0(call) => {
                self.active_prank = Some(PrankRecord {
                    caller: call.msgSender,
                    origin: None,
                    single_call: true,
                });
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::prank_1(call) => {
                self.active_prank = Some(PrankRecord {
                    caller: call.msgSender,
                    origin: Some(call.txOrigin),
                    single_call: true,
                });
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::startPrank_0(call) => {
                self.active_prank = Some(PrankRecord {
                    caller: call.msgSender,
                    origin: None,
                    single_call: false,
                });
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::startPrank_1(call) => {
                self.active_prank = Some(PrankRecord {
                    caller: call.msgSender,
                    origin: Some(call.txOrigin),
                    single_call: false,
                });
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::stopPrank(_) => {
                self.active_prank = None;
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::warp(call) => {
                self.timestamp_override = Some(call.newTimestamp);
                context.env.block.timestamp = call.newTimestamp;
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::roll(call) => {
                self.block_number_override = Some(call.newBlockNumber);
                context.env.block.number = call.newBlockNumber;
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::deal(call) => {
                self.deals.push((call.who, call.newBalance));
                let state_load =
                    context
                        .load_account(call.who)
                        .map_err(|_e| GfnError::RevmExecution {
                            env_idx: 0,
                            message: "Failed to load account for vm.deal".to_string(),
                        })?;
                let account = state_load.data;
                account.info.balance = call.newBalance;
                account.mark_touch();
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::mockCall(call) => {
                self.mocked_calls.insert((call.target, call.data), call.returndata);
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::clearMockedCalls(_) => {
                self.mocked_calls.clear();
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::expectEmit(call) => {
                self.expect_emit = Some((call.checkTopic1, call.checkTopic2, call.checkTopic3, call.checkData));
                Ok(Bytes::new())
            }
            VmCheatcodes::VmCheatcodesCalls::recordLogs(_) => {
                self.recording_logs = true;
                Ok(Bytes::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_sol_types::SolCall;
    use revm::{
        db::{CacheDB, EmptyDB},
        Context,
    };

    #[test]
    fn test_actor_personas() {
        assert_eq!(ActorPersona::ALL.len(), 5);
        for (idx, persona) in ActorPersona::ALL.iter().enumerate() {
            let parsed = ActorPersona::try_from_u8(idx as u8);
            assert!(parsed.is_ok());
            if let Ok(p) = parsed {
                assert_eq!(p, *persona);
                assert!(ActorPersona::is_actor_address(&p.address()));
            }
        }
        assert!(ActorPersona::try_from_u8(5).is_err());
    }

    #[test]
    fn test_prank_lifecycle() {
        let mut state = CheatcodeState::new();
        let attacker = ActorPersona::Attacker0.address();

        // Single prank call
        let prank_call = VmCheatcodes::prank_0Call {
            msgSender: attacker,
        };
        let encoded = prank_call.abi_encode();

        let db = CacheDB::new(EmptyDB::default());
        let mut ctx = Context::new_with_db(db);

        let res = state.dispatch_call(&encoded, &mut ctx.evm);
        assert!(res.is_ok());

        // First consume succeeds
        let consumed = state.consume_prank_for_next_tx();
        assert_eq!(consumed, Some((attacker, None)));

        // Second consume is None because it was single-call
        assert_eq!(state.consume_prank_for_next_tx(), None);

        // Persistent startPrank
        let start_prank = VmCheatcodes::startPrank_0Call {
            msgSender: attacker,
        };
        let encoded_start = start_prank.abi_encode();
        let res = state.dispatch_call(&encoded_start, &mut ctx.evm);
        assert!(res.is_ok());

        assert_eq!(state.consume_prank_for_next_tx(), Some((attacker, None)));
        assert_eq!(state.consume_prank_for_next_tx(), Some((attacker, None)));

        // stopPrank
        let stop_prank = VmCheatcodes::stopPrankCall {};
        let res = state.dispatch_call(&stop_prank.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(state.consume_prank_for_next_tx(), None);
    }

    #[test]
    fn test_warp_and_roll() {
        let mut state = CheatcodeState::new();
        let db = CacheDB::new(EmptyDB::default());
        let mut ctx = Context::new_with_db(db);

        let target_ts = U256::from(1_700_000_100);
        let warp_call = VmCheatcodes::warpCall {
            newTimestamp: target_ts,
        };
        let res = state.dispatch_call(&warp_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(ctx.evm.env.block.timestamp, target_ts);
        assert_eq!(state.timestamp_override, Some(target_ts));

        let target_block = U256::from(18_000_500);
        let roll_call = VmCheatcodes::rollCall {
            newBlockNumber: target_block,
        };
        let res = state.dispatch_call(&roll_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(ctx.evm.env.block.number, target_block);
        assert_eq!(state.block_number_override, Some(target_block));
    }

    #[test]
    fn test_deal() {
        let mut state = CheatcodeState::new();
        let db = CacheDB::new(EmptyDB::default());
        let mut ctx = Context::new_with_db(db);

        let user = ActorPersona::NormalUser.address();
        let target_bal = U256::from(1_000_000_000_000_000_000_u128);

        let deal_call = VmCheatcodes::dealCall {
            who: user,
            newBalance: target_bal,
        };
        let res = state.dispatch_call(&deal_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(state.deals.len(), 1);
        assert_eq!(state.deals[0], (user, target_bal));

        let account = ctx.evm.load_account(user);
        assert!(account.is_ok());
        if let Ok(acc) = account {
            assert_eq!(acc.data.info.balance, target_bal);
        }
    }

    #[test]
    fn test_actor_persona_u256_correspondence() {
        for persona in &ActorPersona::ALL {
            let u256_from_slice = U256::from_be_slice(persona.address().as_slice());
            assert!(
                ActorPersona::is_actor_u256(u256_from_slice),
                "Persona {:?} U256 {:?} must match is_actor_u256",
                persona,
                u256_from_slice
            );
        }
        assert!(!ActorPersona::is_actor_u256(U256::ZERO));
        assert!(!ActorPersona::is_actor_u256(U256::from(0x99)));
        assert!(!ActorPersona::is_actor_u256(U256::MAX));
    }

    #[test]
    fn test_mock_calls_and_expect_emit() {
        let mut state = CheatcodeState::new();
        let db = CacheDB::new(EmptyDB::default());
        let mut ctx = Context::new_with_db(db);

        let target_contract = address!("1111111111111111111111111111111111111111");
        let mock_data = Bytes::from(vec![0x12, 0x34, 0x56, 0x78]);
        let mock_ret = Bytes::from(vec![0xaa, 0xbb, 0xcc, 0xdd]);

        let mock_call = VmCheatcodes::mockCallCall {
            target: target_contract,
            data: mock_data.clone(),
            returndata: mock_ret.clone(),
        };
        let res = state.dispatch_call(&mock_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());

        assert_eq!(state.find_mocked_call(&target_contract, &[0x12, 0x34, 0x56, 0x78]), Some(&mock_ret));
        assert_eq!(state.find_mocked_call(&target_contract, &[0x12, 0x34, 0x56, 0x78, 0x99]), Some(&mock_ret));
        assert_eq!(state.find_mocked_call(&target_contract, &[0x00, 0x00]), None);

        let emit_call = VmCheatcodes::expectEmitCall {
            checkTopic1: true,
            checkTopic2: true,
            checkTopic3: false,
            checkData: true,
        };
        let res = state.dispatch_call(&emit_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(state.expect_emit, Some((true, true, false, true)));

        let rec_logs_call = VmCheatcodes::recordLogsCall {};
        let res = state.dispatch_call(&rec_logs_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert!(state.recording_logs);

        let clear_call = VmCheatcodes::clearMockedCallsCall {};
        let res = state.dispatch_call(&clear_call.abi_encode(), &mut ctx.evm);
        assert!(res.is_ok());
        assert_eq!(state.find_mocked_call(&target_contract, &[0x12, 0x34, 0x56, 0x78]), None);
    }
}
