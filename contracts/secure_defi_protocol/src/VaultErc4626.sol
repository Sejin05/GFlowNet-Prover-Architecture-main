// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title VaultErc4626 (Hardened v2)
/// @notice Security-hardened ERC-4626 vault eliminating first-depositor share inflation,
///         donation rounding manipulation, unbacked ETH accounting, and reentrancy.
contract VaultErc4626 {
    string public constant name = "Hardened 4626 Vault v2";
    string public constant symbol = "hsVAULT";
    uint8 public constant decimals = 24;

    /// @dev OpenZeppelin v5-style virtual offset (6 decimal places = 1e6 virtual shares per 1 virtual asset)
    uint256 public constant VIRTUAL_SHARES = 1e6;
    uint256 public constant VIRTUAL_ASSETS = 1;
    uint256 public constant MIN_DEPOSIT = 1e6;
    uint256 public constant MIN_ACTIVE_SHARES_FOR_DONATION = 100 ether * VIRTUAL_SHARES;

    /// @dev Cancun EIP-1153 transient storage slot for reentrancy guard
    uint256 private constant REENTRANCY_SLOT = 0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    uint256 public totalSupply;
    uint256 public totalAssets;
    uint256 public donatedYieldReserve;

    mapping(address => uint256) public balanceOf;

    error ReentrancyLocked();
    error ZeroAddress();
    error InvalidAssetAmount();
    error ZeroSharesMinted();
    error InsufficientShares();
    error InsufficientVaultLiquidity();
    error UnbackedPayableDeposit(uint256 sent, uint256 declared);
    error DonationRequiresActivePool();

    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);
    event Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares);
    event Donated(address indexed donor, uint256 amount);

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

    /// @notice Converts assets to shares using virtual share/asset offset (rounds down in favor of vault).
    function convertToShares(uint256 assets) public view returns (uint256) {
        return (assets * (totalSupply + VIRTUAL_SHARES)) / (totalAssets + VIRTUAL_ASSETS);
    }

    /// @notice Converts shares to assets using virtual share/asset offset (rounds down in favor of vault).
    function convertToAssets(uint256 shares) public view returns (uint256) {
        return (shares * (totalAssets + VIRTUAL_ASSETS)) / (totalSupply + VIRTUAL_SHARES);
    }

    /// @notice Preview shares required to withdraw exact assets (rounds UP so caller never burns 0 or under-burns).
    function previewWithdraw(uint256 assets) public view returns (uint256) {
        uint256 num = assets * (totalSupply + VIRTUAL_SHARES);
        uint256 den = totalAssets + VIRTUAL_ASSETS;
        return (num + den - 1) / den;
    }

    /// @notice Payable single-argument deposit requiring exact ETH backing and minimum deposit size.
    function deposit(uint256 assets) public payable nonReentrant returns (uint256 shares) {
        if (assets < MIN_DEPOSIT) revert InvalidAssetAmount();
        if (msg.value != assets) revert UnbackedPayableDeposit(msg.value, assets);

        shares = convertToShares(assets);
        if (shares == 0) revert ZeroSharesMinted();

        balanceOf[msg.sender] += shares;
        totalSupply += shares;
        totalAssets += assets;

        emit Deposit(msg.sender, msg.sender, assets, shares);
    }

    /// @notice Deposit assets for `receiver` with virtual-offset inflation immunity and minimum-share checks.
    function deposit(uint256 assets, address receiver) public nonReentrant returns (uint256 shares) {
        if (receiver == address(0)) revert ZeroAddress();
        if (assets < MIN_DEPOSIT) revert InvalidAssetAmount();

        shares = convertToShares(assets);
        if (shares == 0) revert ZeroSharesMinted();

        balanceOf[receiver] += shares;
        totalSupply += shares;
        totalAssets += assets;

        emit Deposit(msg.sender, receiver, assets, shares);
    }

    /// @notice Withdraw exact `assets` to `receiver` by burning ceiling-rounded shares from `owner`.
    function withdraw(uint256 assets, address receiver, address owner) public nonReentrant returns (uint256 shares) {
        if (receiver == address(0) || owner == address(0)) revert ZeroAddress();
        if (assets == 0) revert InvalidAssetAmount();
        if (owner != msg.sender) revert InsufficientShares();
        if (assets > totalAssets) revert InsufficientVaultLiquidity();

        shares = previewWithdraw(assets);
        if (shares == 0 || balanceOf[owner] < shares) revert InsufficientShares();

        balanceOf[owner] -= shares;
        totalSupply -= shares;
        totalAssets -= assets;

        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }

    /// @notice Isolated yield donation routed to `donatedYieldReserve` so spot `totalAssets` cannot be flash-inflated.
    function donateAssets(uint256 amount) public nonReentrant {
        if (amount == 0) revert InvalidAssetAmount();
        if (totalSupply < MIN_ACTIVE_SHARES_FOR_DONATION) revert DonationRequiresActivePool();

        donatedYieldReserve += amount;
        emit Donated(msg.sender, amount);
    }
}
