# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO arithmetic simplification module.

Provides constant folding and (future) expression analysis utilities.
"""

from pypto.pypto_core.arith import (
    ConstIntBound,
    ConstIntBoundAnalyzer,
    ModularSet,
    ModularSetAnalyzer,
    extended_euclidean,
    floordiv,
    floormod,
    fold_const,
    gcd,
    lcm,
)

__all__ = [
    "ConstIntBound",
    "ConstIntBoundAnalyzer",
    "ModularSet",
    "ModularSetAnalyzer",
    "extended_euclidean",
    "floordiv",
    "floormod",
    "fold_const",
    "gcd",
    "lcm",
]
