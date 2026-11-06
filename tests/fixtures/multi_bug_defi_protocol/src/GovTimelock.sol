// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title GovTimelock
/// @notice Synthetic governance timelock controller exhibiting an unprotected timelock execution bypass.
contract GovTimelock {
    address public admin;
    uint256 public constant MINIMUM_DELAY = 2 days;
    uint256 internal eta;

    mapping(uint256 => uint256) public proposalEta;
    mapping(uint256 => bool) public proposalQueued;
    mapping(uint256 => bool) public proposalExecuted;
    mapping(uint256 => address) public proposalTarget;
    mapping(uint256 => bytes) public proposalData;

    event ProposalQueued(uint256 indexed proposalId, address indexed target, uint256 eta);
    event ProposalExecuted(uint256 indexed proposalId, address indexed target);
    event ProposalCanceled(uint256 indexed proposalId);

    constructor(address _admin) {
        admin = _admin;
    }

    /// @notice Updates the timelock admin address.
    /// @dev Vulnerable: missing authorization check require(msg.sender == admin, "Unauthorized");
    function setAdmin(address newAdmin) external {
        admin = newAdmin;
    }

    /// @notice Queues a governance proposal for delayed execution.
    function queueProposal(
        uint256 proposalId,
        address target,
        bytes calldata data,
        uint256 delay
    ) external {
        require(msg.sender == admin, "UNAUTHORIZED");
        require(!proposalQueued[proposalId], "PROPOSAL_ALREADY_QUEUED");
        require(delay >= MINIMUM_DELAY, "DELAY_TOO_SHORT");

        eta = block.timestamp + delay;
        proposalEta[proposalId] = eta;
        proposalQueued[proposalId] = true;
        proposalTarget[proposalId] = target;
        proposalData[proposalId] = data;

        emit ProposalQueued(proposalId, target, proposalEta[proposalId]);
    }

    /// @notice Executes a queued proposal.
    /// @dev Vulnerability: omits require(block.timestamp >= proposalEta[proposalId], "TIMELOCK_NOT_EXPIRED");
    function executeProposal(uint256 proposalId) external returns (bytes memory) {
        require(proposalQueued[proposalId], "PROPOSAL_NOT_QUEUED");
        require(!proposalExecuted[proposalId], "PROPOSAL_ALREADY_EXECUTED");

        // Missing timelock delay check allows immediate bypass of governance delay:
        // require(block.timestamp >= proposalEta[proposalId], "TIMELOCK_NOT_EXPIRED");

        proposalExecuted[proposalId] = true;
        address target = proposalTarget[proposalId];
        bytes memory data = proposalData[proposalId];

        (bool success, bytes memory returnData) = target.call(data);
        require(success, "PROPOSAL_EXECUTION_FAILED");

        emit ProposalExecuted(proposalId, target);
        return returnData;
    }

    /// @notice Cancels a pending proposal.
    function cancelProposal(uint256 proposalId) external {
        require(msg.sender == admin, "UNAUTHORIZED");
        require(proposalQueued[proposalId], "PROPOSAL_NOT_QUEUED");
        require(!proposalExecuted[proposalId], "PROPOSAL_ALREADY_EXECUTED");

        proposalQueued[proposalId] = false;
        emit ProposalCanceled(proposalId);
    }
}
