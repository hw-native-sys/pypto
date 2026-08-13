# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Intermediate examples — real-kernel patterns, ordered by complexity.

  01_fused_linear.py    — matmul+bias, linear+relu (cube + vector orchestration)
  02_softmax.py         — numerically stable row-wise softmax
  03_normalization.py   — RMSNorm, LayerNorm
  04_matmul_acc.py      — cube unit matmul with K-dimension tiling/accumulation
  05_assemble.py        — tile assembly patterns (Acc->Mat, Vec->Vec)
  06_dyn_valid_shape.py — dynamic valid_shape via scalar, if/else and loop patterns
  07_task_graph.py      — an inferred dependency edge, and the same edge declared
"""

import importlib
import sys

_ALIASES = {
    "fused_linear": "01_fused_linear",
    "softmax": "02_softmax",
    "normalization": "03_normalization",
    "matmul_acc": "04_matmul_acc",
    "assemble": "05_assemble",
    "dyn_valid_shape": "06_dyn_valid_shape",
    "task_graph": "07_task_graph",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
