#!/usr/bin/env python3
"""
Very simple test script that doesn't require external dependencies
"""

import sys
import os

print("=" * 50)
print("SIMPLE TEST SCRIPT")
print("=" * 50)

# Test 1: Basic Python functionality
print("\n[Test 1] Python version:")
print(f"  Version: {sys.version}")
print(f"  Executable: {sys.executable}")

# Test 2: Math operations
print("\n[Test 2] Basic math operations:")
result_add = 10 + 5
result_mult = 10 * 5
result_div = 10 / 2
print(f"  10 + 5 = {result_add}")
print(f"  10 * 5 = {result_mult}")
print(f"  10 / 2 = {result_div}")

# Test 3: String operations
print("\n[Test 3] String operations:")
test_string = "Hello, World!"
print(f"  Original: {test_string}")
print(f"  Upper: {test_string.upper()}")
print(f"  Lower: {test_string.lower()}")
print(f"  Length: {len(test_string)}")

# Test 4: List operations
print("\n[Test 4] List operations:")
test_list = [1, 2, 3, 4, 5]
print(f"  List: {test_list}")
print(f"  Sum: {sum(test_list)}")
print(f"  Max: {max(test_list)}")
print(f"  Min: {min(test_list)}")
print(f"  Length: {len(test_list)}")

# Test 5: File system check
print("\n[Test 5] File system check:")
print(f"  Current directory: {os.getcwd()}")
print(f"  Directory exists: {os.path.isdir(os.getcwd())}")
print(f"  Files in current dir: {len(os.listdir(os.getcwd()))}")

# Test 6: Function definition and execution
print("\n[Test 6] Function test:")

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

test_number = 5
result = factorial(test_number)
print(f"  Factorial of {test_number} = {result}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
