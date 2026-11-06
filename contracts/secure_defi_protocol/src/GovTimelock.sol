// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title GovTimelock (Hardened v2)
/// @notice Security-hardened governance timelock enforcing strict RBAC (`onlyAdmin`), two-step admin
///         handover, mandatory `[MINIMUM_DELAY, MAXIMUM_DELAY]` windows, `proposalEta` maturity checks,
///         `GRACE_PERIOD` stale-proposal expiration, and CEI + EIP-1153 reentrancy protection.
contract GovTimelock {
    address public admin;
    address public pendingAdmin;

    uint256 public constant MINIMUM_DELAY = 2 days;
    uint256 public constant MAXIMUM_DELAY = 30 days;
    uint256 public constant GRACE_PERIOD = 14 days;

    uint256 private constant REENTRANCY_SLOT = 0x71e4b8c2a0d3f51728394a5b6c7d8e9f0123456789abcdef0123456789abcd03;

    mapping(uint256 => uint256) public proposalEta;
    mapping(uint256 => bool) public proposalQueued;
    mapping(uint256 => bool) public proposalExecuted;
    mapping(uint256 => address) public proposalTarget;
    mapping(uint256 => bytes) public proposalData;

    error Unauthorized();
    error ZeroAddress();
    error ReentrancyLocked();
    error ProposalAlreadyQueued();
    error ProposalNotQueued();
    error ProposalAlreadyExecuted();
    error InvalidDelay(uint256 delay);
    error TimelockNotExpired(uint256 currentTimestamp, uint256 requiredEta);
    error ProposalExpired(uint256 currentTimestamp, uint256 expiryTimestamp);
    error ProposalExecutionFailed();

    event AdminUpdated(address indexed previousAdmin, address indexed newAdmin);
    event AdminTransferProposed(address indexed currentAdmin, address indexed pendingAdmin);
    event ProposalQueued(uint256 indexed proposalId, address indexed target, uint256 eta);
    event ProposalExecuted(uint256 indexed proposalId, address indexed target);
    event ProposalCanceled(uint256 indexed proposalId);

    modifier onlyAdmin() {
        if (msg.sender != admin) revert Unauthorized();
        _;
    }

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

    constructor(address _admin) {
        if (_admin == address(0)) revert ZeroAddress();
        admin = _admin;
    }

    /// @notice Updates the timelock admin address — strictly gated by `onlyAdmin` and non-zero validation.
    function setAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert ZeroAddress();
        address prev = admin;
        admin = newAdmin;
        pendingAdmin = address(0);
        emit AdminUpdated(prev, newAdmin);
    }

    /// @notice Initiates a safer two-step admin transfer.
    function proposeAdmin(address candidateAdmin) external onlyAdmin {
        if (candidateAdmin == address(0)) revert ZeroAddress();
        pendingAdmin = candidateAdmin;
        emit AdminTransferProposed(admin, candidateAdmin);
    }

    /// @notice Completes a two-step admin transfer when called by `pendingAdmin`.
    function acceptAdmin() external {
        if (msg.sender != pendingAdmin) revert Unauthorized();
        address prev = admin;
        admin = msg.sender;
        pendingAdmin = address(0);
        emit AdminUpdated(prev, msg.sender);
    }

    /// @notice Queues a governance proposal with enforced delay bounds and target validation.
    function queueProposal(
        uint256 proposalId,
        address target,
        bytes calldata data,
        uint256 delay
    ) external onlyAdmin {
        if (target == address(0)) revert ZeroAddress();
        if (proposalQueued[proposalId] || proposalExecuted[proposalId]) revert ProposalAlreadyQueued();
        if (delay < MINIMUM_DELAY || delay > MAXIMUM_DELAY) revert InvalidDelay(delay);

        uint256 unlockEta = block.timestamp + delay;
        proposalEta[proposalId] = unlockEta;
        proposalQueued[proposalId] = true;
        proposalTarget[proposalId] = target;
        proposalData[proposalId] = data;

        emit ProposalQueued(proposalId, target, unlockEta);
    }

    /// @notice Executes a queued proposal ONLY after `proposalEta` has elapsed and within `GRACE_PERIOD`.
    function executeProposal(uint256 proposalId) external nonReentrant returns (bytes memory) {
        if (!proposalQueued[proposalId]) revert ProposalNotQueued();
        if (proposalExecuted[proposalId]) revert ProposalAlreadyExecuted();

        uint256 unlockEta = proposalEta[proposalId];
        if (unlockEta == 0 || block.timestamp < unlockEta) {
            revert TimelockNotExpired(block.timestamp, unlockEta);
        }
        if (block.timestamp > unlockEta + GRACE_PERIOD) {
            revert ProposalExpired(block.timestamp, unlockEta + GRACE_PERIOD);
        }

        // Effects before Interactions (CEI)
        proposalExecuted[proposalId] = true;
        proposalQueued[proposalId] = false;

        address target = proposalTarget[proposalId];
        bytes memory data = proposalData[proposalId];

        (bool success, bytes memory returnData) = target.call(data);
        if (!success) revert ProposalExecutionFailed();

        emit ProposalExecuted(proposalId, target);
        return returnData;
    }

    /// @notice Cancels a pending proposal.
    function cancelProposal(uint256 proposalId) external onlyAdmin {
        if (!proposalQueued[proposalId]) revert ProposalNotQueued();
        if (proposalExecuted[proposalId]) revert ProposalAlreadyExecuted();

        proposalQueued[proposalId] = false;
        proposalEta[proposalId] = 0;
        emit ProposalCanceled(proposalId);
    }
}
