# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Demonstrates that the @pl.jit pipeline rejects an invalid kernel at compile time.

The body rebinds ``result`` to ``pl.add(x, 1.0)``, discarding the prior write
of ``pl.mul(x, 2.0)``. Each rebinding assigns a value that does not write into
the ``result`` Out parameter, so the external buffer would never be written and
the kernel would silently produce all-zero. The ``OutParamNotShadowed`` verifier
catches this up front (at pipeline input, before SSA) and fails compilation with
a clear diagnostic instead — see issue #1525.
"""

import pypto.language as pl


@pl.jit
def test_ssa_violation(x: pl.Tensor, result: pl.Out[pl.Tensor]):
    with pl.incore():
        result = pl.mul(x, 2.0)
        result = pl.add(x, 1.0)  # rebinding -- discards the prior write to result
    return result


if __name__ == "__main__":
    import sys

    import torch
    from pypto.runtime import RunConfig

    x = torch.randn(64, dtype=torch.float32)
    result = torch.zeros_like(x)
    try:
        test_ssa_violation(x, result, config=RunConfig())
        print("ERROR: expected the invalid kernel to be rejected")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 -- demo: any compile-time rejection is the success path
        msg = str(e)
        if "OutParamNotShadowed" in msg or "issue #1525" in msg:
            print("OK -- rejected at compile time by the OutParamNotShadowed verifier")
        else:
            print(f"OK -- rejected at compile time: {type(e).__name__}")
