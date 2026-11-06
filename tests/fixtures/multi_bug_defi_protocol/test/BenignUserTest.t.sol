// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {VaultErc4626} from "../src/VaultErc4626.sol";
import {AmmPool} from "../src/AmmPool.sol";
import {LendingMarket} from "../src/LendingMarket.sol";
import {GovTimelock} from "../src/GovTimelock.sol";

contract MockTarget {
    uint256 public value;

    function setValue(uint256 _value) external {
        value = _value;
    }
}

abstract contract ProtocolFixtureBase is Test {
    VaultErc4626 public vault;
    AmmPool public ammPool;
    LendingMarket public lendingMarket;
    GovTimelock public govTimelock;
    MockTarget public target;

    address public userA = address(0x1111);
    address public userB = address(0x2222);

    function setUp() public virtual {
        vault = new VaultErc4626();
        ammPool = new AmmPool();
        lendingMarket = new LendingMarket(address(ammPool));
        govTimelock = new GovTimelock(address(this));
        target = new MockTarget();

        // Seed initial liquidity in AMM pool (10,000 of token0 and token1, initial price = 1.0)
        ammPool.addLiquidity(10000 ether, 10000 ether, address(this));
    }
}

contract BenignUserTest is ProtocolFixtureBase {

    /// @notice 1. Standard deposit of 1000 tokens yields 1000 shares in fresh vault.
    function test_BenignVaultDeposit() public {
        vm.prank(userA);
        uint256 shares = vault.deposit(1000 ether, userA);
        assertEq(shares, 1000 ether, "Initial deposit 1:1 ratio mismatch");
        assertEq(vault.balanceOf(userA), 1000 ether);
        assertEq(vault.totalAssets(), 1000 ether);
    }

    /// @notice 2. Withdraw deposited tokens and burn shares.
    function test_BenignVaultWithdraw() public {
        vm.prank(userA);
        vault.deposit(1000 ether, userA);

        vm.prank(userA);
        uint256 sharesBurned = vault.withdraw(500 ether, userA, userA);
        assertEq(sharesBurned, 500 ether);
        assertEq(vault.balanceOf(userA), 500 ether);
        assertEq(vault.totalAssets(), 500 ether);
    }

    /// @notice 3. Multi-user deposit maintains fair share distribution under benign conditions.
    function test_BenignVaultMultiUser() public {
        vm.prank(userA);
        vault.deposit(1000 ether, userA);

        vm.prank(userB);
        uint256 sharesB = vault.deposit(2000 ether, userB);
        assertEq(sharesB, 2000 ether);
        assertEq(vault.totalAssets(), 3000 ether);
    }

    /// @notice 4. Swap token0 for token1 in AMM pool updates reserves correctly.
    function test_BenignAmmSwap() public {
        (uint256 r0Before, uint256 r1Before) = ammPool.getReserves();
        assertEq(r0Before, 10000 ether);
        assertEq(r1Before, 10000 ether);

        ammPool.swap(100 ether, 0, userA, "");
        (uint256 r0After, uint256 r1After) = ammPool.getReserves();
        assertEq(r0After, 9900 ether);
        assertGe(r1After, 10000 ether);
    }

    /// @notice 5. Provide additional liquidity and receive LP shares.
    function test_BenignAmmAddLiquidity() public {
        uint256 lpShares = ammPool.addLiquidity(1000 ether, 1000 ether, userB);
        assertGe(lpShares, 1000 ether);
        assertEq(ammPool.liquidityOf(userB), lpShares);
    }

    /// @notice 6. Deposit collateral into LendingMarket.
    function test_BenignCollateralDeposit() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);
        assertEq(lendingMarket.collateralOf(userA), 100 ether);
        assertEq(lendingMarket.totalCollateral(), 100 ether);
    }

    /// @notice 7. Borrow within safe collateral limits (50% LTV).
    function test_BenignBorrowHealthy() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);

        // Max borrow at 75% LTV with price 1.0 is 75 ether. Borrowing 50 ether is safe.
        vm.prank(userA);
        lendingMarket.borrow(50 ether);
        assertEq(lendingMarket.debtOf(userA), 50 ether);
    }

    /// @notice 8. Repay borrowed debt and restore collateral capacity.
    function test_BenignRepayLoan() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);

        vm.prank(userA);
        lendingMarket.borrow(50 ether);

        vm.prank(userA);
        lendingMarket.repay(userA, 25 ether);
        assertEq(lendingMarket.debtOf(userA), 25 ether);
    }

    /// @notice 9. Queue governance proposal with valid delay.
    function test_BenignGovQueueProposal() public {
        bytes memory data = abi.encodeWithSelector(MockTarget.setValue.selector, 42);
        govTimelock.queueProposal(1, address(target), data, 2 days);

        assertTrue(govTimelock.proposalQueued(1));
        assertEq(govTimelock.proposalEta(1), block.timestamp + 2 days);
    }

    /// @notice 10. Warp time past delay and execute proposal successfully.
    function test_BenignGovExecuteAfterDelay() public {
        bytes memory data = abi.encodeWithSelector(MockTarget.setValue.selector, 1337);
        govTimelock.queueProposal(2, address(target), data, 2 days);

        vm.warp(block.timestamp + 2 days + 1);
        govTimelock.executeProposal(2);

        assertTrue(govTimelock.proposalExecuted(2));
        assertEq(target.value(), 1337);
    }

    /// @notice 11. Fee accrual calculation across transactions.
    function test_BenignFeeAccrual() public {
        uint256 initialPrice = ammPool.getPrice();
        assertEq(initialPrice, 1 ether);

        // Execute small swap
        ammPool.swap(50 ether, 0, userA, "");
        uint256 newPrice = ammPool.getPrice();
        assertTrue(newPrice != initialPrice, "Price should adjust dynamically with invariant");
    }

    /// @notice 12. View functions return consistent and valid values.
    function test_BenignViewAccounting() public {
        assertEq(vault.totalAssets(), 0);
        assertEq(vault.totalSupply(), 0);
        assertEq(ammPool.totalLiquidity(), 10000 ether);
        assertEq(lendingMarket.totalCollateral(), 0);
        assertEq(lendingMarket.totalDebt(), 0);
    }
}
