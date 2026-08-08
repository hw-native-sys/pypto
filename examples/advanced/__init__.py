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
  03_auto_tile_vector.py — compiler-driven vector tiling for softmax, norms, and SiLU
"""

import importlib
import sys

_ALIASES = {
    "split_k": "01_split_k",
    "auto_tile_matmul": "02_auto_tile_matmul",
    "auto_tile_vector": "03_auto_tile_vector",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
