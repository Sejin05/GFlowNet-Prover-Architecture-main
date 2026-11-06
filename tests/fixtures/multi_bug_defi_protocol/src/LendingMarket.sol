// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AmmPool} from "./AmmPool.sol";

/// @title LendingMarket
/// @notice Synthetic collateralized lending market vulnerable to price distortion via AMM pool read-only reentrancy.
contract LendingMarket {
    AmmPool public immutable ammPool;

    uint256 public constant COLLATERAL_FACTOR_BPS = 7500; // 75% max LTV
    uint256 public constant BPS_DIVISOR = 10000;

    mapping(address => uint256) public collateralOf;
    mapping(address => uint256) public debtOf;

    uint256 public totalCollateral;
    uint256 public totalDebt;

    event CollateralDeposited(address indexed user, uint256 amount);
    event CollateralWithdrawn(address indexed user, uint256 amount);
    event Borrowed(address indexed user, uint256 amount);
    event Repaid(address indexed user, uint256 amount);

    constructor(address _ammPool) {
        ammPool = AmmPool(_ammPool);
    }

    /// @notice Convenience deposit function for collateral.
    function deposit(uint256 assets) external {
        collateralOf[msg.sender] += assets;
        totalCollateral += assets;
        emit CollateralDeposited(msg.sender, assets);
    }

    /// @notice Deposits collateral assets into the lending market.
    function depositCollateral(address onBehalfOf, uint256 amount) external {
        require(amount > 0, "INVALID_AMOUNT");
        collateralOf[onBehalfOf] += amount;
        totalCollateral += amount;
        emit CollateralDeposited(onBehalfOf, amount);
    }

    /// @notice Withdraws collateral if health condition remains satisfied.
    function withdrawCollateral(uint256 amount) external {
        require(amount > 0 && collateralOf[msg.sender] >= amount, "INVALID_AMOUNT");

        collateralOf[msg.sender] -= amount;
        totalCollateral -= amount;

        // Health check on remaining collateral
        uint256 price = ammPool.getPrice();
        uint256 collateralValue = (collateralOf[msg.sender] * price) / 1e18;
        uint256 maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        require(debtOf[msg.sender] <= maxBorrow, "INSUFFICIENT_COLLATERAL");

        emit CollateralWithdrawn(msg.sender, amount);
    }

    /// @notice Borrows debt assets against deposited collateral evaluated via AMM spot price.
    /// @dev Vulnerable to read-only reentrancy price distortion when invoked during AmmPool callback.
    function borrow(uint256 borrowAmount) external {
        require(borrowAmount > 0, "INVALID_BORROW_AMOUNT");

        uint256 price = ammPool.getPrice();
        uint256 collateralValue = (collateralOf[msg.sender] * price) / 1e18;
        uint256 maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;

        require(debtOf[msg.sender] + borrowAmount <= maxBorrow, "UNDERCOLLATERALIZED");

        debtOf[msg.sender] += borrowAmount;
        totalDebt += borrowAmount;

        emit Borrowed(msg.sender, borrowAmount);
    }

    /// @notice Repays outstanding borrow debt.
    function repay(address onBehalfOf, uint256 amount) external {
        require(amount > 0 && debtOf[onBehalfOf] >= amount, "INVALID_REPAY_AMOUNT");
        debtOf[onBehalfOf] -= amount;
        totalDebt -= amount;
        emit Repaid(onBehalfOf, amount);
    }

    /// @notice Returns user account health metrics.
    function getAccountHealth(address user) external view returns (uint256 collateralValue, uint256 maxBorrow, uint256 currentDebt) {
        uint256 price = ammPool.getPrice();
        collateralValue = (collateralOf[user] * price) / 1e18;
        maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        currentDebt = debtOf[user];
    }
}
