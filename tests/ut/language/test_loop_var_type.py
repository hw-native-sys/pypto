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

"""Pyright regression tests for the DSL loop signatures.

Two properties are pinned here. Both are checked by pyright rather than by an
assertion, so a regression surfaces as a type error on this file, not a failing
run.

1. Loop variables from pl.range/parallel/unroll are ``Scalar``. If one regresses
   to ``int``, pyright reports "Argument of type 'int' cannot be assigned to
   parameter of type 'Scalar'" on every ``accept_scalar()`` call below.
2. A loop-carried value keeps its concrete tensor subclass. The carry TypeVars
   in ``dsl_api`` are *bound*, so they solve to the argument's own type. If one
   regresses to a constrained ``TypeVar(..., "Tensor", ...)``, the solve
   collapses the subclass to plain ``Tensor`` and every ``accept_distributed()``
   call below reports "Argument of type 'Tensor' cannot be assigned to parameter
   of type 'DistributedTensor'".
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest


def accept_scalar(x: pl.Scalar) -> None: ...


def accept_distributed(x: pld.DistributedTensor) -> None: ...


def accept_tensor(x: pl.Tensor) -> None: ...


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


class TestLoopCarrySubclass:
    """Verify a loop-carried DistributedTensor is not widened to plain Tensor.

    ``pld.system.notify`` / ``wait`` / ``defer_wait`` annotate their signal
    operand as ``DistributedTensor``, matching the C++ deducer. A carried value
    that lost its subclass could not be passed to them at all, which is the
    regression these cases pin.
    """

    def test_range_carry_keeps_distributed_tensor(self):
        signal = pld.DistributedTensor[[16, 16], pl.INT32]
        for _i, (carried,) in pl.range(1, init_values=(signal,)):
            accept_distributed(carried)
            accept_distributed(pl.yield_(carried))

    def test_while_carry_keeps_distributed_tensor(self):
        signal = pld.DistributedTensor[[16, 16], pl.INT32]
        for (carried,) in pl.while_(init_values=(signal,)):
            accept_distributed(carried)
            accept_distributed(pl.yield_(carried))

    def test_multi_carry_keeps_each_position(self):
        """Each carry slot has its own TypeVar, so the subclass survives per slot."""
        signal = pld.DistributedTensor[[16, 16], pl.INT32]
        plain = pl.Tensor[[16, 16], pl.INT32]
        for _i, (dist, tensor) in pl.range(1, init_values=(signal, plain)):
            accept_distributed(dist)
            accept_tensor(tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
