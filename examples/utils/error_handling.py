# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Diagnose rebinding an Out parameter in a JIT kernel.

The last assignment gives ``result`` a new tensor value. The compiler emits
``OutParamWriteDropped`` because the caller's output buffer is never written.
Use ``result[:] = ...`` for a write to that buffer.
"""

import pypto.language as pl


@pl.jit
def rebind_out_parameter(x: pl.Tensor, result: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.mul(x, 2.0)
        result = pl.add(x, 1.0)
    return result


if __name__ == "__main__":
    import torch
    from pypto.runtime import RunConfig

    x = torch.arange(64, dtype=torch.float32)
    result = torch.zeros_like(x)
    rebind_out_parameter.compile(x, result, config=RunConfig(codegen_only=True))
    print("Check for OutParamWriteDropped in the compiler diagnostics.")
