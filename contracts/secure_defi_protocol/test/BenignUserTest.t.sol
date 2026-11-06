// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {VaultErc4626} from "../src/VaultErc4626.sol";
import {AmmPool, IAmmCallback} from "../src/AmmPool.sol";
import {LendingMarket} from "../src/LendingMarket.sol";
import {GovTimelock} from "../src/GovTimelock.sol";

contract MockTarget {
    uint256 public value;

    function setValue(uint256 _value) external {
        value = _value;
    }
}

/// @notice Malicious callback contract attempting AMM read-only reentrancy against LendingMarket.
contract ReentrantOracleAttacker is IAmmCallback {
    AmmPool public immutable pool;
    LendingMarket public immutable market;

    constructor(AmmPool _pool, LendingMarket _market) {
        pool = _pool;
        market = _market;
    }

    function executeAttack() external {
        market.depositCollateral(address(this), 100 ether);
        pool.swap(1000 ether, 0, address(this), hex"01");
    }

    function ammCallback(uint256, uint256, bytes calldata) external override {
        // Attempt undercollateralized borrow while AMM swap is mid-flight
        market.borrow(80 ether);
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
    address public attacker = address(0xA0A0);

    function setUp() public virtual {
        vault = new VaultErc4626();
        ammPool = new AmmPool();
        lendingMarket = new LendingMarket(address(ammPool));
        govTimelock = new GovTimelock(address(this));
        target = new MockTarget();

        ammPool.addLiquidity(10000 ether, 10000 ether, address(this));
    }
}

contract BenignUserTest is ProtocolFixtureBase {
    // =========================================================================
    // Part A: 12 Benign Protocol Functional Parity Tests
    // =========================================================================

    function test_BenignVaultDeposit() public {
        vm.prank(userA);
        uint256 shares = vault.deposit(1000 ether, userA);
        assertGt(shares, 0);
        assertEq(vault.balanceOf(userA), shares);
        assertEq(vault.totalAssets(), 1000 ether);
        assertEq(vault.convertToAssets(shares), 1000 ether);
    }

    function test_BenignVaultWithdraw() public {
        vm.prank(userA);
        vault.deposit(1000 ether, userA);

        vm.prank(userA);
        uint256 sharesBurned = vault.withdraw(500 ether, userA, userA);
        assertGt(sharesBurned, 0);
        assertEq(vault.totalAssets(), 500 ether);
        uint256 remAssets = vault.convertToAssets(vault.balanceOf(userA));
        assertGe(remAssets, 500 ether - 1);
        assertLe(remAssets, 500 ether);
    }

    function test_BenignVaultMultiUser() public {
        vm.prank(userA);
        uint256 sharesA = vault.deposit(1000 ether, userA);

        vm.prank(userB);
        uint256 sharesB = vault.deposit(2000 ether, userB);
        assertGe(sharesB, sharesA * 2 - 2e6);
        assertLe(sharesB, sharesA * 2 + 2e6);
        assertEq(vault.totalAssets(), 3000 ether);
    }

    function test_BenignAmmSwap() public {
        (uint256 r0Before, uint256 r1Before) = ammPool.getReserves();
        assertEq(r0Before, 10000 ether);
        assertEq(r1Before, 10000 ether);

        ammPool.swap(100 ether, 0, userA, "");
        (uint256 r0After, uint256 r1After) = ammPool.getReserves();
        assertEq(r0After, 9900 ether);
        assertGt(r1After, 10000 ether);
        assertGe(r0After * r1After, r0Before * r1Before);
    }

    function test_BenignAmmAddLiquidity() public {
        uint256 lpShares = ammPool.addLiquidity(1000 ether, 1000 ether, userB);
        assertGe(lpShares, 999 ether);
        assertEq(ammPool.liquidityOf(userB), lpShares);
    }

    function test_BenignCollateralDeposit() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);
        assertEq(lendingMarket.collateralOf(userA), 100 ether);
        assertEq(lendingMarket.totalCollateral(), 100 ether);
    }

    function test_BenignBorrowHealthy() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);

        vm.prank(userA);
        lendingMarket.borrow(50 ether);
        assertEq(lendingMarket.debtOf(userA), 50 ether);
    }

    function test_BenignRepayLoan() public {
        vm.prank(userA);
        lendingMarket.depositCollateral(userA, 100 ether);

        vm.prank(userA);
        lendingMarket.borrow(50 ether);

        vm.prank(userA);
        lendingMarket.repay(userA, 25 ether);
        assertEq(lendingMarket.debtOf(userA), 25 ether);
    }

    function test_BenignGovQueueProposal() public {
        bytes memory data = abi.encodeWithSelector(MockTarget.setValue.selector, 42);
        govTimelock.queueProposal(1, address(target), data, 2 days);

        assertTrue(govTimelock.proposalQueued(1));
        assertEq(govTimelock.proposalEta(1), block.timestamp + 2 days);
    }

    function test_BenignGovExecuteAfterDelay() public {
        bytes memory data = abi.encodeWithSelector(MockTarget.setValue.selector, 1337);
        govTimelock.queueProposal(2, address(target), data, 2 days);

        vm.warp(block.timestamp + 2 days + 1);
        govTimelock.executeProposal(2);

        assertTrue(govTimelock.proposalExecuted(2));
        assertFalse(govTimelock.proposalQueued(2));
        assertEq(target.value(), 1337);
    }

    function test_BenignTwoStepAdminTransfer() public {
        govTimelock.proposeAdmin(userA);
        assertEq(govTimelock.pendingAdmin(), userA);

        vm.prank(userA);
        govTimelock.acceptAdmin();
        assertEq(govTimelock.admin(), userA);
        assertEq(govTimelock.pendingAdmin(), address(0));
    }

    function test_BenignViewAccounting() public {
        assertEq(vault.totalAssets(), 0);
        assertEq(vault.totalSupply(), 0);
        assertEq(ammPool.totalLiquidity(), 10000 ether);
        assertEq(ammPool.getTwapPrice(), 1 ether);
        assertEq(lendingMarket.totalCollateral(), 0);
        assertEq(lendingMarket.totalDebt(), 0);
    }

    // =========================================================================
    // Part B: 6 Adversarial Security Regression Tests (Exploits Must Revert)
    // =========================================================================

    function test_DefenseBlocksVaultInflationAndDustDeposit() public {
        vm.startPrank(attacker);
        vm.expectRevert(abi.encodeWithSelector(VaultErc4626.InvalidAssetAmount.selector));
        vault.deposit(1, attacker);

        // Even with MIN_DEPOSIT, donation before 100 ether active liquidity reverts
        vault.deposit(1e6, attacker);
        vm.expectRevert(abi.encodeWithSelector(VaultErc4626.DonationRequiresActivePool.selector));
        vault.donateAssets(1000 ether);
        vm.stopPrank();
    }

    function test_DefenseBlocksUnbackedPayableDeposit() public {
        vm.deal(attacker, 1 ether);
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSelector(VaultErc4626.UnbackedPayableDeposit.selector, 0, 500 ether));
        vault.deposit(500 ether);
    }

    function test_DefenseBlocksAmmReadOnlyReentrancy() public {
        ReentrantOracleAttacker mal = new ReentrantOracleAttacker(ammPool, lendingMarket);
        vm.expectRevert(abi.encodeWithSelector(LendingMarket.OracleReentrantState.selector));
        mal.executeAttack();
    }

    function test_DefenseBlocksFlashSwapOracleManipulationBorrow() public {
        vm.startPrank(attacker);
        lendingMarket.depositCollateral(attacker, 100 ether);

        // 50% reserve drain swap is blocked by single-swap impact circuit breaker
        vm.expectRevert(abi.encodeWithSelector(AmmPool.ExcessiveSwapImpact.selector));
        ammPool.swap(5000 ether, 0, attacker, "");

        // Even after a 25% swap distorting spot price, TWAP deviation guard blocks borrow
        ammPool.swap(2500 ether, 0, attacker, "");
        vm.expectRevert();
        lendingMarket.borrow(100 ether);
        vm.stopPrank();
    }

    function test_DefenseBlocksUnauthenticatedSetAdmin() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSelector(GovTimelock.Unauthorized.selector));
        govTimelock.setAdmin(attacker);
    }

    function test_DefenseBlocksPrematureAndExpiredProposalExecution() public {
        bytes memory data = abi.encodeWithSelector(MockTarget.setValue.selector, 999);
        govTimelock.queueProposal(99, address(target), data, 2 days);

        // Premature execution before 2-day ETA must revert
        vm.prank(attacker);
        vm.expectRevert(
            abi.encodeWithSelector(
                GovTimelock.TimelockNotExpired.selector,
                block.timestamp,
                block.timestamp + 2 days
            )
        );
        govTimelock.executeProposal(99);

        // Execution after GRACE_PERIOD (2 days + 14 days + 1s) must also revert
        uint256 eta = govTimelock.proposalEta(99);
        vm.warp(eta + govTimelock.GRACE_PERIOD() + 1);
        vm.expectRevert(
            abi.encodeWithSelector(
                GovTimelock.ProposalExpired.selector,
                block.timestamp,
                eta + govTimelock.GRACE_PERIOD()
            )
        );
        govTimelock.executeProposal(99);
    }
}
