// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AmmPool} from "./AmmPool.sol";

/// @title LendingMarket (Hardened v2)
/// @notice Security-hardened collateralized lending market immune to AMM read-only reentrancy
///         and single-block flash-swap price manipulation via dual-oracle bounds and solvency caps.
contract LendingMarket {
    AmmPool public immutable ammPool;

    uint256 public constant COLLATERAL_FACTOR_BPS = 7500; // 75% max LTV
    uint256 public constant MAX_ORACLE_DEVIATION_BPS = 1000; // Max 10% spot-vs-TWAP divergence
    uint256 public constant BPS_DIVISOR = 10000;

    uint256 private constant REENTRANCY_SLOT = 0x2f81d98c0711906a14f7e49e8a9c5a2b3c4d5e6f708192a3b4c5d6e7f8091a02;

    mapping(address => uint256) public collateralOf;
    mapping(address => uint256) public debtOf;

    uint256 public totalCollateral;
    uint256 public totalDebt;

    error ReentrancyLocked();
    error OracleReentrantState();
    error OracleDeviationExceeded(uint256 spotPrice, uint256 twapPrice);
    error ZeroAddress();
    error InvalidAmount();
    error Undercollateralized();
    error GlobalSolvencyCapExceeded();

    event CollateralDeposited(address indexed user, uint256 amount);
    event CollateralWithdrawn(address indexed user, uint256 amount);
    event Borrowed(address indexed user, uint256 amount);
    event Repaid(address indexed user, uint256 amount);

    modifier nonReentrant() {
        bool locked;
        assembly {
            locked := tload(REENTRANCY_SLOT)
        }
        if (locked) revert ReentrancyLocked();
        assembly {
            tstore(REENTRANCY_SLOT, 1)
        }
        _;
        assembly {
            tstore(REENTRANCY_SLOT, 0)
        }
    }

    constructor(address _ammPool) {
        if (_ammPool == address(0)) revert ZeroAddress();
        ammPool = AmmPool(_ammPool);
    }

    /// @notice Reads manipulation-resistant collateral price: rejects active AMM lock, checks spot/TWAP divergence,
    ///         and returns `min(spotPrice, twapPrice)` so transient spikes never expand borrowing power.
    function _validatedConservativePrice() internal view returns (uint256) {
        if (ammPool.isLocked()) revert OracleReentrantState();
        uint256 spot = ammPool.getPrice();
        uint256 twap = ammPool.getTwapPrice();

        uint256 diff = spot > twap ? spot - twap : twap - spot;
        if (diff * BPS_DIVISOR > twap * MAX_ORACLE_DEVIATION_BPS) {
            revert OracleDeviationExceeded(spot, twap);
        }
        return spot < twap ? spot : twap;
    }

    /// @notice Convenience deposit function for caller's own collateral.
    function deposit(uint256 assets) external nonReentrant {
        if (assets == 0) revert InvalidAmount();
        collateralOf[msg.sender] += assets;
        totalCollateral += assets;
        emit CollateralDeposited(msg.sender, assets);
    }

    /// @notice Deposits collateral assets on behalf of `onBehalfOf`.
    function depositCollateral(address onBehalfOf, uint256 amount) external nonReentrant {
        if (onBehalfOf == address(0)) revert ZeroAddress();
        if (amount == 0) revert InvalidAmount();
        collateralOf[onBehalfOf] += amount;
        totalCollateral += amount;
        emit CollateralDeposited(onBehalfOf, amount);
    }

    /// @notice Withdraws collateral only if post-withdrawal account and global solvency remain healthy.
    function withdrawCollateral(uint256 amount) external nonReentrant {
        if (amount == 0 || collateralOf[msg.sender] < amount) revert InvalidAmount();

        collateralOf[msg.sender] -= amount;
        totalCollateral -= amount;

        uint256 price = _validatedConservativePrice();
        uint256 collateralValue = (collateralOf[msg.sender] * price) / 1e18;
        uint256 maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        if (debtOf[msg.sender] > maxBorrow) revert Undercollateralized();

        uint256 globalMaxDebt = (((totalCollateral * price) / 1e18) * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        if (totalDebt > globalMaxDebt) revert GlobalSolvencyCapExceeded();

        emit CollateralWithdrawn(msg.sender, amount);
    }

    /// @notice Borrows debt assets against conservatively priced collateral and enforces global solvency.
    function borrow(uint256 borrowAmount) external nonReentrant {
        if (borrowAmount == 0) revert InvalidAmount();

        uint256 price = _validatedConservativePrice();
        uint256 collateralValue = (collateralOf[msg.sender] * price) / 1e18;
        uint256 maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;

        if (debtOf[msg.sender] + borrowAmount > maxBorrow) revert Undercollateralized();

        uint256 globalMaxDebt = (((totalCollateral * price) / 1e18) * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        if (totalDebt + borrowAmount > globalMaxDebt) revert GlobalSolvencyCapExceeded();

        debtOf[msg.sender] += borrowAmount;
        totalDebt += borrowAmount;

        emit Borrowed(msg.sender, borrowAmount);
    }

    /// @notice Repays outstanding borrow debt.
    function repay(address onBehalfOf, uint256 amount) external nonReentrant {
        if (onBehalfOf == address(0)) revert ZeroAddress();
        if (amount == 0 || debtOf[onBehalfOf] < amount) revert InvalidAmount();
        debtOf[onBehalfOf] -= amount;
        totalDebt -= amount;
        emit Repaid(onBehalfOf, amount);
    }

    /// @notice Returns user account health metrics using the conservative validated oracle price.
    function getAccountHealth(address user)
        external
        view
        returns (uint256 collateralValue, uint256 maxBorrow, uint256 currentDebt)
    {
        uint256 price = _validatedConservativePrice();
        collateralValue = (collateralOf[user] * price) / 1e18;
        maxBorrow = (collateralValue * COLLATERAL_FACTOR_BPS) / BPS_DIVISOR;
        currentDebt = debtOf[user];
    }
}
