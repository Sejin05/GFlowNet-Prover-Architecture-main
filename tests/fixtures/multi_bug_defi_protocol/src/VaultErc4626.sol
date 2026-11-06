// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title VaultErc4626
/// @notice Synthetic ERC-4626 vault exhibiting a zero-supply first-depositor share inflation vulnerability.
contract VaultErc4626 {
    string public name = "Synthetic 4626 Vault";
    string public symbol = "sVAULT";
    uint8 public constant decimals = 18;

    uint256 public totalSupply;
    uint256 public totalAssets;

    mapping(address => uint256) public balanceOf;

    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);
    event Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares);
    event Donated(address indexed donor, uint256 amount);

    /// @notice Converts assets to shares using raw spot ratio without virtual share/asset offset.
    /// @dev Vulnerable to inflation rounding exploit when totalSupply is small and totalAssets is artificially inflated.
    function convertToShares(uint256 assets) public view returns (uint256) {
        uint256 supply = totalSupply;
        if (supply == 0) {
            return assets;
        }
        return (assets * supply) / totalAssets;
    }

    /// @notice Converts shares to assets.
    function convertToAssets(uint256 shares) public view returns (uint256) {
        uint256 supply = totalSupply;
        if (supply == 0) {
            return shares;
        }
        return (shares * totalAssets) / supply;
    }

    /// @notice Overloaded deposit for single-parameter convenience.
    function deposit(uint256 assets) public payable returns (uint256 shares) {
        if (totalAssets == 0) {
            shares = assets;
        } else {
            shares = (assets * totalSupply) / totalAssets;
        }
        balanceOf[msg.sender] += shares;
        totalSupply += shares;
        totalAssets += assets;
        emit Deposit(msg.sender, msg.sender, assets, shares);
    }

    /// @notice Deposit assets to receive vault shares.
    function deposit(uint256 assets, address receiver) public returns (uint256 shares) {
        require(assets > 0, "INVALID_ASSETS");
        shares = convertToShares(assets);
        require(shares > 0, "ZERO_SHARES");

        balanceOf[receiver] += shares;
        totalSupply += shares;
        totalAssets += assets;

        emit Deposit(msg.sender, receiver, assets, shares);
    }

    /// @notice Withdraw assets by burning shares.
    function withdraw(uint256 assets, address receiver, address owner) public returns (uint256 shares) {
        require(assets > 0, "INVALID_ASSETS");
        shares = convertToShares(assets);
        if (shares == 0 && totalSupply > 0) {
            shares = 1;
        }
        require(balanceOf[owner] >= shares, "INSUFFICIENT_SHARES");

        balanceOf[owner] -= shares;
        totalSupply -= shares;
        totalAssets -= assets;

        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }

    /// @notice Direct donation to inflate vault assets (simulating first-depositor inflation attack).
    function donateAssets(uint256 amount) public {
        require(amount > 0, "INVALID_DONATION");
        totalAssets += amount;
        emit Donated(msg.sender, amount);
    }
}
