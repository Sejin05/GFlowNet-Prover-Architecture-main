// SPDX-License-Identifier: MIT
pragma solidity >=0.6.2 <0.9.0;

interface Vm {
    function warp(uint256 newTimestamp) external;
    function roll(uint256 newNumber) external;
    function prank(address sender) external;
    function startPrank(address sender) external;
    function stopPrank() external;
    function deal(address who, uint256 newBalance) external;
    function expectRevert() external;
    function expectRevert(bytes calldata revertData) external;
    function load(address target, bytes32 slot) external view returns (bytes32);
    function store(address target, bytes32 slot, bytes32 value) external;
}
