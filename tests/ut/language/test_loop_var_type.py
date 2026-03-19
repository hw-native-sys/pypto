# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# pyright: reportCallIssue=true
# pyright: reportArgumentType=true

"""Pyright regression test: loop variables from pl.range/parallel/unroll must be Scalar.

If the loop variable type regresses to ``int``, pyright will report:
  "Argument of type 'int' cannot be assigned to parameter of type 'Scalar'"
on every ``accept_scalar()`` call below.
"""

import pypto.language as pl
import pytest


def accept_scalar(x: pl.Scalar) -> None: ...


class TestLoopVarType:
    """Verify loop variables from pl.range/parallel/unroll are typed as Scalar."""

    def test_range_loop_var(self):
        for i in pl.range(10):
            accept_scalar(i)

    def test_range_loop_var_with_init_values(self):
        for j, (t,) in pl.range(0, 8, 1, init_values=(pl.Scalar[pl.INDEX],)):
            accept_scalar(j)

    def test_parallel_loop_var(self):
        for k in pl.parallel(4):
            accept_scalar(k)

    def test_parallel_loop_var_with_init_values(self):
        for m, (t,) in pl.parallel(0, 4, 1, init_values=(pl.Scalar[pl.INDEX],)):
            accept_scalar(m)

    def test_unroll_loop_var(self):
        for n in pl.unroll(3):
            accept_scalar(n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
