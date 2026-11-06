// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IAmmCallback {
    function ammCallback(uint256 amount0Out, uint256 amount1Out, bytes calldata data) external;
}

/// @title AmmPool (Hardened v2)
/// @notice Security-hardened constant-product AMM enforcing strict Checks-Effects-Interactions (CEI),
///         EIP-1153 transient reentrancy locks (including read-only reentrancy view guards),
///         minimum locked liquidity, single-swap impact caps, and a bounded TWAP price oracle.
contract AmmPool {
    uint256 public constant MINIMUM_LIQUIDITY = 1000;
    uint256 public constant MAX_SWAP_IMPACT_BPS = 3000; // Max 30% of reserve per swap
    uint256 public constant MAX_TWAP_STEP_BPS = 500;    // Max 5% TWAP adjustment per block
    uint256 public constant BPS_DIVISOR = 10000;

    /// @dev Cancun EIP-1153 transient storage slot for swap/LP reentrancy lock
    uint256 private constant REENTRANCY_SLOT = 0x5c36a4e9a62091c5128a1d329c5f2134901f2d9b8f3a11e6809822a40a99c101;

    uint256 public reserve0;
    uint256 public reserve1;
    uint256 public totalLiquidity;
    uint256 public twapPriceE18;
    uint256 public lastTwapBlock;

    mapping(address => uint256) public liquidityOf;

    error ReentrancyLocked();
    error InvalidAmounts();
    error InvalidRecipient();
    error InsufficientLiquidityMinted();
    error InsufficientLiquidity();
    error ExcessiveSwapImpact();
    error InvariantViolationK();
    error ZeroReserve();

    event Mint(address indexed sender, uint256 amount0, uint256 amount1, uint256 liquidity);
    event Burn(address indexed sender, uint256 amount0, uint256 amount1, uint256 liquidity);
    event Swap(address indexed sender, address indexed to, uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out);

    modifier nonReentrant() {
        if (isLocked()) revert ReentrancyLocked();
        assembly {
            tstore(REENTRANCY_SLOT, 1)
        }
        _;
        assembly {
            tstore(REENTRANCY_SLOT, 0)
        }
    }

    /// @notice Exposes transient lock status so external protocols (e.g. LendingMarket) reject read-only reentrancy.
    function isLocked() public view returns (bool locked) {
        assembly {
            locked := tload(REENTRANCY_SLOT)
        }
    }

    /// @notice Adds liquidity to the pool and initializes TWAP on bootstrap.
    function addLiquidity(uint256 amount0, uint256 amount1, address to) external nonReentrant returns (uint256 liquidity) {
        if (to == address(0)) revert InvalidRecipient();
        if (amount0 == 0 || amount1 == 0) revert InvalidAmounts();

        if (totalLiquidity == 0) {
            uint256 gross = (amount0 + amount1) / 2;
            if (gross <= MINIMUM_LIQUIDITY) revert InsufficientLiquidityMinted();
            totalLiquidity = MINIMUM_LIQUIDITY;
            liquidityOf[address(0)] = MINIMUM_LIQUIDITY;
            liquidity = gross - MINIMUM_LIQUIDITY;
            reserve0 = amount0;
            reserve1 = amount1;
            twapPriceE18 = (amount1 * 1e18) / amount0;
            lastTwapBlock = block.number;
        } else {
            uint256 l0 = (amount0 * totalLiquidity) / reserve0;
            uint256 l1 = (amount1 * totalLiquidity) / reserve1;
            liquidity = l0 < l1 ? l0 : l1;
            if (liquidity == 0) revert InsufficientLiquidityMinted();
            reserve0 += amount0;
            reserve1 += amount1;
        }

        totalLiquidity += liquidity;
        liquidityOf[to] += liquidity;

        emit Mint(msg.sender, amount0, amount1, liquidity);
    }

    /// @notice Removes liquidity from the pool under reentrancy protection.
    function removeLiquidity(uint256 liquidity, address to) external nonReentrant returns (uint256 amount0, uint256 amount1) {
        if (to == address(0)) revert InvalidRecipient();
        if (liquidity == 0 || liquidityOf[msg.sender] < liquidity) revert InsufficientLiquidity();

        amount0 = (liquidity * reserve0) / totalLiquidity;
        amount1 = (liquidity * reserve1) / totalLiquidity;
        if (amount0 == 0 || amount1 == 0) revert InsufficientLiquidity();

        liquidityOf[msg.sender] -= liquidity;
        totalLiquidity -= liquidity;
        reserve0 -= amount0;
        reserve1 -= amount1;

        emit Burn(msg.sender, amount0, amount1, liquidity);
    }

    /// @notice Swaps tokens enforcing strict CEI: all storage updates and k-invariant checks execute BEFORE callback.
    function swap(
        uint256 amount0Out,
        uint256 amount1Out,
        address to,
        bytes calldata data
    ) external nonReentrant {
        if (to == address(0) || to == address(this)) revert InvalidRecipient();
        if (amount0Out == 0 && amount1Out == 0) revert InvalidAmounts();
        if (amount0Out >= reserve0 || amount1Out >= reserve1) revert InsufficientLiquidity();

        // Single-swap price impact circuit breaker (blocks 50%+ flash-loan pool drains)
        if (
            amount0Out * BPS_DIVISOR > reserve0 * MAX_SWAP_IMPACT_BPS
                || amount1Out * BPS_DIVISOR > reserve1 * MAX_SWAP_IMPACT_BPS
        ) {
            revert ExcessiveSwapImpact();
        }

        uint256 kBefore = reserve0 * reserve1;
        uint256 newReserve0 = reserve0;
        uint256 newReserve1 = reserve1;
        uint256 amount0In = 0;
        uint256 amount1In = 0;

        if (amount0Out > 0) {
            newReserve0 = reserve0 - amount0Out;
            // Ceiling division guarantees kAfter >= kBefore
            amount1In = (kBefore + newReserve0 - 1) / newReserve0 - reserve1;
            newReserve1 = reserve1 + amount1In;
        }
        if (amount1Out > 0) {
            newReserve1 = newReserve1 - amount1Out;
            amount0In = (kBefore + newReserve1 - 1) / newReserve1 - newReserve0;
            newReserve0 = newReserve0 + amount0In;
        }

        if (newReserve0 * newReserve1 < kBefore) revert InvariantViolationK();

        // EFFECTS BEFORE INTERACTIONS: Persist reserves and bounded TWAP prior to any external call
        reserve0 = newReserve0;
        reserve1 = newReserve1;
        _updateBoundedTwap(newReserve0, newReserve1);

        // INTERACTIONS: Transient lock remains active (`isLocked() == true`), zero SSTORE follows this CALL
        if (data.length > 0) {
            IAmmCallback(to).ammCallback(amount0Out, amount1Out, data);
        }

        emit Swap(msg.sender, to, amount0In, amount1In, amount0Out, amount1Out);
    }

    function _updateBoundedTwap(uint256 r0, uint256 r1) internal {
        if (block.number <= lastTwapBlock || r0 == 0) {
            return;
        }
        uint256 spot = (r1 * 1e18) / r0;
        uint256 maxUp = (twapPriceE18 * (BPS_DIVISOR + MAX_TWAP_STEP_BPS)) / BPS_DIVISOR;
        uint256 maxDown = (twapPriceE18 * (BPS_DIVISOR - MAX_TWAP_STEP_BPS)) / BPS_DIVISOR;
        if (spot > maxUp) {
            twapPriceE18 = maxUp;
        } else if (spot < maxDown) {
            twapPriceE18 = maxDown;
        } else {
            twapPriceE18 = spot;
        }
        lastTwapBlock = block.number;
    }

    /// @notice Returns current reserves (reverts if queried mid-swap to prevent read-only reentrancy).
    function getReserves() external view returns (uint256, uint256) {
        if (isLocked()) revert ReentrancyLocked();
        return (reserve0, reserve1);
    }

    /// @notice Returns spot price of token0 in token1 (reverts if AMM is mid-swap).
    function getPrice() external view returns (uint256) {
        if (isLocked()) revert ReentrancyLocked();
        if (reserve0 == 0) revert ZeroReserve();
        return (reserve1 * 1e18) / reserve0;
    }

    /// @notice Returns manipulation-resistant bounded TWAP price (reverts if AMM is mid-swap).
    function getTwapPrice() external view returns (uint256) {
        if (isLocked()) revert ReentrancyLocked();
        if (twapPriceE18 == 0) revert ZeroReserve();
        return twapPriceE18;
    }
}
