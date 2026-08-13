# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Beginner examples — language basics, one concept per file, ordered by complexity.

  01_hello_world.py — simplest program: 128x128 add (start here)
  02_elementwise.py — add, mul
  03_scalar_ops.py  — scaling a tile by a constant
  04_activation.py  — relu, SiLU, GELU, SwiGLU, GeGLU
  05_matmul.py      — cube unit matmul (cube basics)
  06_concat.py      — tile concatenation
"""

import importlib
import sys

_ALIASES = {
    "hello_world": "01_hello_world",
    "elementwise": "02_elementwise",
    "scalar_ops": "03_scalar_ops",
    "activation": "04_activation",
    "matmul": "05_matmul",
    "concat": "06_concat",
}

for _alias, _numbered in _ALIASES.items():
    _mod = importlib.import_module(f".{_numbered}", __package__)
    globals()[_alias] = _mod
    sys.modules[f"{__package__}.{_alias}"] = _mod
