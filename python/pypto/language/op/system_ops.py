# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO Language DSL.

Re-exports from pypto.ir.op.system_ops. System ops have no language-level
type wrappers (they take/return no Tensor/Tile/Scalar), so this is a
straight pass-through.
"""

from pypto.ir.op.system_ops import bar_all, bar_m, bar_v, sync_dst, sync_src

__all__ = [
    "sync_src",
    "sync_dst",
    "bar_v",
    "bar_m",
    "bar_all",
]
