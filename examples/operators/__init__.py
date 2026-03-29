# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Operator examples — single-kernel programs demonstrating individual ops.

Suggested reading order:
  1. elementwise.py   — add, mul (simplest tile ops)
  2. fused_ops.py     — add+scale, add+relu, matmul+bias, linear+relu
  3. matmul.py        — cube unit matmul and matmul_acc
  4. concat.py        — tile concatenation
  5. activation.py    — SiLU, GELU, SwiGLU, GeGLU
  6. softmax.py       — row-wise numerically stable softmax
  7. normalization.py — RMSNorm, LayerNorm
  8. assemble.py      — tile assembly patterns (Acc->Mat, Vec->Vec)
"""
