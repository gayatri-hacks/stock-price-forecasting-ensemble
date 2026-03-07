#!/usr/bin/env python
"""Quick test of directional_accuracy function."""

import numpy as np
from utils.helpers import directional_accuracy

print("Testing directional_accuracy function...")

# Test 1: Normal 1D arrays
y_true = np.array([1.0, 2.0, 3.0, 2.5, 4.0])
y_pred = np.array([1.1, 2.1, 2.9, 2.6, 3.9])
acc = directional_accuracy(y_true, y_pred)
print(f"✓ Test 1 (1D arrays): {acc:.2f}")

# Test 2: 2D arrays (should be flattened)
y_true_2d = np.array([[1.0], [2.0], [3.0], [2.5], [4.0]])
y_pred_2d = np.array([[1.1], [2.1], [2.9], [2.6], [3.9]])
acc = directional_accuracy(y_true_2d, y_pred_2d)
print(f"✓ Test 2 (2D arrays): {acc:.2f}")

# Test 3: Lists (should be converted)
y_true_list = [1.0, 2.0, 3.0, 2.5, 4.0]
y_pred_list = [1.1, 2.1, 2.9, 2.6, 3.9]
acc = directional_accuracy(y_true_list, y_pred_list)
print(f"✓ Test 3 (Lists): {acc:.2f}")

print("\n✓ All tests passed!")
