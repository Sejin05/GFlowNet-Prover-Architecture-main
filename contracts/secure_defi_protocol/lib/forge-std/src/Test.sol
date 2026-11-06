// SPDX-License-Identifier: MIT
pragma solidity >=0.6.2 <0.9.0;

import {Vm} from "./Vm.sol";

abstract contract Test {
    address private constant VM_ADDRESS = address(uint160(uint256(keccak256("hevm cheat code"))));
    Vm internal constant vm = Vm(VM_ADDRESS);

    function assertTrue(bool condition) internal pure {
        require(condition, "Assertion failed: condition is false");
    }

    function assertTrue(bool condition, string memory message) internal pure {
        require(condition, message);
    }

    function assertFalse(bool condition) internal pure {
        require(!condition, "Assertion failed: condition is true");
    }

    function assertFalse(bool condition, string memory message) internal pure {
        require(!condition, message);
    }

    function assertEq(uint256 a, uint256 b) internal pure {
        require(a == b, "Assertion failed: uint256 equality");
    }

    function assertEq(uint256 a, uint256 b, string memory message) internal pure {
        require(a == b, message);
    }

    function assertEq(address a, address b) internal pure {
        require(a == b, "Assertion failed: address equality");
    }

    function assertEq(address a, address b, string memory message) internal pure {
        require(a == b, message);
    }

    function assertNotEq(uint256 a, uint256 b) internal pure {
        require(a != b, "Assertion failed: uint256 equality");
    }

    function assertNotEq(uint256 a, uint256 b, string memory message) internal pure {
        require(a != b, message);
    }

    function assertNotEq(address a, address b) internal pure {
        require(a != b, "Assertion failed: address equality");
    }

    function assertNotEq(address a, address b, string memory message) internal pure {
        require(a != b, message);
    }

    function assertGe(uint256 a, uint256 b) internal pure {
        require(a >= b, "Assertion failed: uint256 not greater or equal");
    }

    function assertGe(uint256 a, uint256 b, string memory message) internal pure {
        require(a >= b, message);
    }

    function assertLe(uint256 a, uint256 b) internal pure {
        require(a <= b, "Assertion failed: uint256 not less or equal");
    }

    function assertLe(uint256 a, uint256 b, string memory message) internal pure {
        require(a <= b, message);
    }

    function assertGt(uint256 a, uint256 b) internal pure {
        require(a > b, "Assertion failed: uint256 not greater than");
    }

    function assertGt(uint256 a, uint256 b, string memory message) internal pure {
        require(a > b, message);
    }

    function assertLt(uint256 a, uint256 b) internal pure {
        require(a < b, "Assertion failed: uint256 not less than");
    }

    function assertLt(uint256 a, uint256 b, string memory message) internal pure {
        require(a < b, message);
    }
}
