// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IAmmCallback {
    function ammCallback(uint256 amount0Out, uint256 amount1Out, bytes calldata data) external;
}

/// @title AmmPool
/// @notice Synthetic constant-product automated market maker exhibiting read-only reentrancy via deferred reserve updates.
contract AmmPool {
    uint256 public reserve0;
    uint256 public reserve1;
    uint256 public totalLiquidity;

    mapping(address => uint256) public liquidityOf;

    event Mint(address indexed sender, uint256 amount0, uint256 amount1, uint256 liquidity);
    event Burn(address indexed sender, uint256 amount0, uint256 amount1, uint256 liquidity);
    event Swap(address indexed sender, address indexed to, uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out);

    /// @notice Adds liquidity to the pool.
    function addLiquidity(uint256 amount0, uint256 amount1, address to) external returns (uint256 liquidity) {
        require(amount0 > 0 && amount1 > 0, "INVALID_AMOUNTS");

        if (totalLiquidity == 0) {
            liquidity = (amount0 + amount1) / 2;
        } else {
            uint256 l0 = (amount0 * totalLiquidity) / reserve0;
            uint256 l1 = (amount1 * totalLiquidity) / reserve1;
            liquidity = l0 < l1 ? l0 : l1;
        }

        require(liquidity > 0, "INSUFFICIENT_LIQUIDITY_MINTED");

        reserve0 += amount0;
        reserve1 += amount1;
        totalLiquidity += liquidity;
        liquidityOf[to] += liquidity;

        emit Mint(msg.sender, amount0, amount1, liquidity);
    }

    /// @notice Removes liquidity from the pool.
    function removeLiquidity(uint256 liquidity, address /* to */) external returns (uint256 amount0, uint256 amount1) {
        require(liquidity > 0 && liquidityOf[msg.sender] >= liquidity, "INVALID_LIQUIDITY");

        amount0 = (liquidity * reserve0) / totalLiquidity;
        amount1 = (liquidity * reserve1) / totalLiquidity;

        liquidityOf[msg.sender] -= liquidity;
        totalLiquidity -= liquidity;
        reserve0 -= amount0;
        reserve1 -= amount1;

        emit Burn(msg.sender, amount0, amount1, liquidity);
    }

    /// @notice Swaps tokens in the pool.
    /// @dev Vulnerable: invokes the external callback before updating reserve0 / reserve1 state variables.
    function swap(
        uint256 amount0Out,
        uint256 amount1Out,
        address to,
        bytes calldata data
    ) external {
        require(amount0Out > 0 || amount1Out > 0, "INSUFFICIENT_OUTPUT");
        require(amount0Out < reserve0 && amount1Out < reserve1, "INSUFFICIENT_LIQUIDITY");

        // External callback before updating state variables (Read-Only Reentrancy vulnerability)
        if (data.length > 0) {
            IAmmCallback(to).ammCallback(amount0Out, amount1Out, data);
        }

        // Deferred state update (Checks-Effects-Interactions violation)
        if (amount0Out > 0) {
            reserve0 -= amount0Out;
            reserve1 += (amount0Out * reserve1) / (reserve0 + amount0Out);
        } else {
            reserve1 -= amount1Out;
            reserve0 += (amount1Out * reserve0) / (reserve1 + amount1Out);
        }

        emit Swap(msg.sender, to, 0, 0, amount0Out, amount1Out);
    }

    /// @notice Returns current reserves.
    function getReserves() external view returns (uint256, uint256) {
        return (reserve0, reserve1);
    }

    /// @notice Calculates spot price of token0 relative to token1 (scaled by 1e18).
    function getPrice() external view returns (uint256) {
        require(reserve0 > 0, "ZERO_RESERVE");
        return (reserve1 * 1e18) / reserve0;
    }
}
