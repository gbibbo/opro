#!/usr/bin/env python3
"""
Simple test script to verify environment setup
"""

import sys
print(f"Python version: {sys.version}")

# Test basic imports
try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA version: {torch.version.cuda}")
        print(f"     GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"[FAIL] PyTorch import failed: {e}")

try:
    import transformers
    print(f"[OK] Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"[FAIL] Transformers import failed: {e}")

try:
    import pandas as pd
    print(f"[OK] Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"[FAIL] Pandas import failed: {e}")

try:
    import numpy as np
    print(f"[OK] NumPy version: {np.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy import failed: {e}")

# Test simple calculation
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    result = arr.mean()
    print(f"\n[OK] Simple NumPy test passed: mean of [1,2,3,4,5] = {result}")
except Exception as e:
    print(f"\n[FAIL] Simple NumPy test failed: {e}")

print("\n=== Basic environment check completed ===")
