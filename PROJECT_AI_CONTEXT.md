# Project AI Context Bundle

- **Target Project**: `.`
- **Format**: `markdown`
- **Generated**: `2026-11-06T02:51:45Z`
- **Solidity Units Scanned**: `10`

## Discovered Contracts & Entrypoints

### `AmmPool` (`contracts/secure_defi_protocol/src/AmmPool.sol`, 195 LOC)
- External low-level call sites: `0`
- `function ammCallback(uint256 amount0Out, uint256 amount1Out, bytes calldata data)`
- `function isLocked()`
- `function addLiquidity(uint256 amount0, uint256 amount1, address to)`
- `function removeLiquidity(uint256 liquidity, address to)`
- `function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data)`
- `function _updateBoundedTwap(uint256 r0, uint256 r1)`
- `function getReserves()`
- `function getPrice()`
- `function getTwapPrice()`

### `GovTimelock` (`contracts/secure_defi_protocol/src/GovTimelock.sol`, 147 LOC)
- External low-level call sites: `1`
- `function setAdmin(address newAdmin)`
- `function proposeAdmin(address candidateAdmin)`
- `function acceptAdmin()`
- `function queueProposal(uint256 proposalId, address target, bytes calldata data, uint256 delay)`
- `function executeProposal(uint256 proposalId)`
- `function cancelProposal(uint256 proposalId)`

### `LendingMarket` (`contracts/secure_defi_protocol/src/LendingMarket.sol`, 145 LOC)
- External low-level call sites: `0`
- `function _validatedConservativePrice()`
- `function deposit(uint256 assets)`
- `function depositCollateral(address onBehalfOf, uint256 amount)`
- `function withdrawCollateral(uint256 amount)`
- `function borrow(uint256 borrowAmount)`
- `function repay(address onBehalfOf, uint256 amount)`
- `function getAccountHealth(address user)`

### `VaultErc4626` (`contracts/secure_defi_protocol/src/VaultErc4626.sol`, 127 LOC)
- External low-level call sites: `0`
- `function convertToShares(uint256 assets)`
- `function convertToAssets(uint256 shares)`
- `function previewWithdraw(uint256 assets)`
- `function deposit(uint256 assets)`
- `function deposit(uint256 assets, address receiver)`
- `function withdraw(uint256 assets, address receiver, address owner)`
- `function donateAssets(uint256 amount)`

### `MockTarget, attempting, ReentrantOracleAttacker, ProtocolFixtureBase, BenignUserTest` (`contracts/secure_defi_protocol/test/BenignUserTest.t.sol`, 261 LOC)
- External low-level call sites: `0`
- `function setValue(uint256 _value)`
- `function executeAttack()`
- `function ammCallback(uint256, uint256, bytes calldata)`
- `function setUp()`
- `function test_BenignVaultDeposit()`
- `function test_BenignVaultWithdraw()`
- `function test_BenignVaultMultiUser()`
- `function test_BenignAmmSwap()`
- `function test_BenignAmmAddLiquidity()`
- `function test_BenignCollateralDeposit()`
- `function test_BenignBorrowHealthy()`
- `function test_BenignRepayLoan()`
- `function test_BenignGovQueueProposal()`
- `function test_BenignGovExecuteAfterDelay()`
- `function test_BenignTwoStepAdminTransfer()`
- `function test_BenignViewAccounting()`
- `function test_DefenseBlocksVaultInflationAndDustDeposit()`
- `function test_DefenseBlocksUnbackedPayableDeposit()`
- `function test_DefenseBlocksAmmReadOnlyReentrancy()`
- `function test_DefenseBlocksFlashSwapOracleManipulationBorrow()`
- `function test_DefenseBlocksUnauthenticatedSetAdmin()`
- `function test_DefenseBlocksPrematureAndExpiredProposalExecution()`

### `AmmPool` (`tests/fixtures/multi_bug_defi_protocol/src/AmmPool.sol`, 96 LOC)
- External low-level call sites: `0`
- `function ammCallback(uint256 amount0Out, uint256 amount1Out, bytes calldata data)`
- `function addLiquidity(uint256 amount0, uint256 amount1, address to)`
- `function removeLiquidity(uint256 liquidity, address /* to */)`
- `function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data)`
- `function getReserves()`
- `function getPrice()`

### `GovTimelock` (`tests/fixtures/multi_bug_defi_protocol/src/GovTimelock.sol`, 80 LOC)
- External low-level call sites: `1`
- `function setAdmin(address newAdmin)`
- `function queueProposal(uint256 proposalId, address target, bytes calldata data, uint256 delay)`
- `function executeProposal(uint256 proposalId)`
- `function cancelProposal(uint256 proposalId)`

### `LendingMarket` (`tests/fixtures/multi_bug_defi_protocol/src/LendingMarket.sol`, 92 LOC)
- External low-level call sites: `0`
- `function deposit(uint256 assets)`
- `function depositCollateral(address onBehalfOf, uint256 amount)`
- `function withdrawCollateral(uint256 amount)`
- `function borrow(uint256 borrowAmount)`
- `function repay(address onBehalfOf, uint256 amount)`
- `function getAccountHealth(address user)`

### `VaultErc4626` (`tests/fixtures/multi_bug_defi_protocol/src/VaultErc4626.sol`, 87 LOC)
- External low-level call sites: `0`
- `function convertToShares(uint256 assets)`
- `function convertToAssets(uint256 shares)`
- `function deposit(uint256 assets)`
- `function deposit(uint256 assets, address receiver)`
- `function withdraw(uint256 assets, address receiver, address owner)`
- `function donateAssets(uint256 amount)`

### `MockTarget, ProtocolFixtureBase, BenignUserTest` (`tests/fixtures/multi_bug_defi_protocol/test/BenignUserTest.t.sol`, 165 LOC)
- External low-level call sites: `0`
- `function setValue(uint256 _value)`
- `function setUp()`
- `function test_BenignVaultDeposit()`
- `function test_BenignVaultWithdraw()`
- `function test_BenignVaultMultiUser()`
- `function test_BenignAmmSwap()`
- `function test_BenignAmmAddLiquidity()`
- `function test_BenignCollateralDeposit()`
- `function test_BenignBorrowHealthy()`
- `function test_BenignRepayLoan()`
- `function test_BenignGovQueueProposal()`
- `function test_BenignGovExecuteAfterDelay()`
- `function test_BenignFeeAccrual()`
- `function test_BenignViewAccounting()`
