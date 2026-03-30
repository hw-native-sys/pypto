# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Kernel examples — single-kernel programs, ordered by complexity.

  01_elementwise.py    — add, mul (start here)
  02_fused_ops.py      — add+scale, add+relu, matmul+bias, linear+relu
  03_matmul.py         — cube unit matmul and matmul_acc
  04_concat.py         — tile concatenation
  05_activation.py     — SiLU, GELU, SwiGLU, GeGLU
  06_softmax.py        — numerically stable row-wise softmax
  07_normalization.py  — RMSNorm, LayerNorm
  08_assemble.py       — tile assembly patterns (Acc->Mat, Vec->Vec)
"""

import importlib
import sys

_ALIASES = {
    "elementwise": "01_elementwise",
    "fused_ops": "02_fused_ops",
    "matmul": "03_matmul",
    "concat": "04_concat",
    "activation": "05_activation",
    "softmax": "06_softmax",
    "normalization": "07_normalization",
    "assemble": "08_assemble",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
