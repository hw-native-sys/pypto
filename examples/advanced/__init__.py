# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Advanced examples — performance and low-level techniques.

  01_split_k.py          — split-K matmul (parallel K reduction, atomic-add)
  02_auto_tile_matmul.py — compiler-driven L0 matmul tiling (DDR/Mat-scratch x full-K/split-K)
  03_mixed_kernel.py     — cube and vector in one scope via pl.split (vs. two scopes)
  04_task_granularity.py — bigger tiles, loop-inside-pl.at, merged producer/consumer
  05_runtime_overhead.py — SPMD fan-out, allow_early_resolve, in-kernel soft syncall
  06_dependencies.py     — no_dep_args / manual_dep / manual_scope / output slicing
  07_double_buffer.py    — pl.pipeline(stage=) vs. explicit pl.MemRef slots
  08_scope_placement.py  — scope depth picks the runtime ring; auto_scope=False
"""

import importlib
import sys

_ALIASES = {
    "split_k": "01_split_k",
    "auto_tile_matmul": "02_auto_tile_matmul",
    "mixed_kernel": "03_mixed_kernel",
    "task_granularity": "04_task_granularity",
    "runtime_overhead": "05_runtime_overhead",
    "dependencies": "06_dependencies",
    "double_buffer": "07_double_buffer",
    "scope_placement": "08_scope_placement",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
