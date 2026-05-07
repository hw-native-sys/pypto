# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for control flow code generation using PyPTO frontend.

This module validates code generation and execution for control flow patterns:
- scf.for loops (with and without yield/iter_args)
- scf.if (if-else with yield)
- scf.while (natural while loop with PTO codegen)
- break/continue (via CtrlFlowTransform pass, which converts to structured control flow)

Limitations:
    scf.if (if-else) inside InCore functions with tile operations is not yet
    fully supported at runtime. The PTO codegen's ForStmt/IfStmt type inference
    only handles ScalarType for iter_args/return_vars (TileType/TensorType fall
    through to "index"). These limitations also affect break/continue tests
    since the CtrlFlowTransform pass converts them into if-else + while
    constructs.
"""

from typing import Any

import pypto.language as pl
import pytest
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


class TestForLoopAdd(PTOTestCase):
    """Test tile add inside a for loop (1 iteration).

    Validates scf.for code generation. The loop runs once,
    performing a simple tile add. Expected result: c = a + b.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_add_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_loop(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_add_loop(a, b, c)
                return c

        return ForLoopAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestForLoopMul(PTOTestCase):
    """Test tile mul inside a for loop (1 iteration).

    Validates scf.for code generation with a different operation.
    Expected result: c = a * b.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_mul_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopMulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_mul_loop(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_mul_loop(a, b, c)
                return c

        return ForLoopMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestForLoopYieldAdd(PTOTestCase):
    """Test for loop with yield carrying a tensor across iterations.

    The loop iterates 4 times, each time loading a 64-row chunk from both
    inputs, adding them, storing to the output, and yielding the output
    tensor for the next iteration. Expected result: c = a + b.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_yield_add_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopYieldAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_yield(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.range(4, init_values=(c,)):
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], acc)
                    result = pl.yield_(out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_add_yield(a, b, c)
                return c

        return ForLoopYieldAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestForLoopYieldTileAccum(PTOTestCase):
    """Test for loop with yield carrying a tile accumulator across iterations.

    Loads 4 chunks of 64x64 from input (all 2.0), accumulates into a single
    tile via yield. Stores the final accumulated tile to output.
    Expected result: c[:] = 4 * 2.0 = 8.0.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_yield_tile_accum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopYieldTileAccumProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_tile_accum(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_init: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                for i, (acc,) in pl.range(1, 4, init_values=(tile_init,)):
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    new_acc: pl.Tile[[64, 64], pl.FP32] = pl.add(acc, tile_a)
                    result = pl.yield_(new_acc)
                out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                c = self.kernel_tile_accum(a, c)
                return c

        return ForLoopYieldTileAccumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 4 * 2.0


class TestIfYieldTensor(PTOTestCase):
    """Test if-else with yield carrying tensors.

    Conditionally selects between two operations based on a scalar condition.
    If condition is true (1), multiplies input by 2. Otherwise, multiplies by 3.
    Expected result: c = a * 2 (since condition is 1).
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "if_yield_tensor"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("condition", [1], DataType.INT32, init_value=1),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class IfYieldTensorProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_if_yield(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                cond: pl.Scalar[pl.INT32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                if cond == 1:
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_a, 2.0)
                else:
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_a, 3.0)
                out: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                cond_tensor: pl.Tensor[[1], pl.INT32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                cond: pl.Scalar[pl.INT32] = pl.tensor.read(cond_tensor, [0])
                c = self.kernel_if_yield(a, cond, c)
                return c

        return IfYieldTensorProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * 2.0


class TestForIfElseNested(PTOTestCase):
    """Test if-else nested inside a for loop.

    Iterates 4 chunks of 64 rows. In each iteration, if condition is 1,
    performs add(a, b); otherwise performs mul(a, b). With condition=1,
    expected result: c = a + b = 5.0.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_if_else_nested"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("condition", [1], DataType.INT32, init_value=1),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForIfElseNestedProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_for_if_else(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                cond: pl.Scalar[pl.INT32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.create_tile([64, 64], dtype=pl.FP32)

                    if cond == 1:
                        tile_c = pl.add(tile_a, tile_b)
                    else:
                        tile_c = pl.mul(tile_a, tile_b)

                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                cond_tensor: pl.Tensor[[1], pl.INT32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                cond: pl.Scalar[pl.INT32] = pl.tensor.read(cond_tensor, [0])
                c = self.kernel_for_if_else(a, b, cond, c)
                return c

        return ForIfElseNestedProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestWhileLoopAdd(PTOTestCase):
    """Test while loop performing tile add over 4 chunks.

    Uses a natural while loop to iterate over 4 chunks of 64 rows,
    adding corresponding tiles from a and b, storing to c.
    Validates that PTO codegen emits scf.while correctly.
    Expected result: c = a + b = 5.0.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "while_loop_add_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class WhileLoopAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_while_add(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = 0
                while i < 4:
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                    i = i + 1
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_while_add(a, b, c)
                return c

        return WhileLoopAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestForLoopBreak(PTOTestCase):
    """Test for loop with break — only first 2 chunks are processed.

    Loops over 4 chunks of 64 rows, but breaks when i >= 2.
    Only the first 128 rows get c = a + b = 5.0, the rest remain 0.
    Validates that CtrlFlowTransform correctly eliminates break.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_break_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopBreakProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_break(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    if i >= 2:
                        break
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_break(a, b, c)
                return c

        return ForLoopBreakProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:128] = tensors["a"][:128] + tensors["b"][:128]
        tensors["c"][128:] = 0.0


class TestForLoopContinue(PTOTestCase):
    """Test for loop with continue — odd iterations are skipped.

    Loops over 4 chunks of 64 rows, but skips odd iterations (i=1, i=3)
    via continue. Only even chunks (i=0, i=2) get c = a + b = 5.0.
    Validates that CtrlFlowTransform correctly eliminates continue.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_continue_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopContinueProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_continue(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    if i % 2 != 0:
                        continue
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_continue(a, b, c)
                return c

        return ForLoopContinueProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:64] = tensors["a"][:64] + tensors["b"][:64]
        tensors["c"][64:128] = 0.0
        tensors["c"][128:192] = tensors["a"][128:192] + tensors["b"][128:192]
        tensors["c"][192:] = 0.0


class TestForLoopBreakContinue(PTOTestCase):
    """Test for loop with both break and continue.

    Loops over 4 chunks. Breaks when i >= 3 (so only i=0,1,2 run).
    Skips odd iterations via continue (so i=1 is skipped).
    Only chunks i=0 and i=2 get c = a + b = 5.0.
    Validates the two-phase algorithm: eliminate continue first, then break.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "for_loop_break_continue_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [256, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [256, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [256, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ForLoopBreakContinueProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_break_continue(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    if i >= 3:
                        break
                    if i % 2 != 0:
                        continue
                    offset_i = i * 64
                    tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [offset_i, 0], [64, 64])
                    tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [offset_i, 0], [64, 64])
                    tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                    out: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile_c, [offset_i, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c = self.kernel_break_continue(a, b, c)
                return c

        return ForLoopBreakContinueProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:64] = tensors["a"][:64] + tensors["b"][:64]
        tensors["c"][64:128] = 0.0
        tensors["c"][128:192] = tensors["a"][128:192] + tensors["b"][128:192]
        tensors["c"][192:] = 0.0


class TestOrchRangeCarryRebind(PTOTestCase):
    """Reproducer for issue #1286: orchestration ``pl.range`` with carry-rebind to
    a freshly-allocated GM tensor.

    The orchestrator runs 4 iterations. Each iteration allocates a fresh
    ``next_hidden`` tensor via ``pl.create_tensor``, calls an InCore kernel
    that reads from ``current_hidden`` and writes ``+1`` into ``next_hidden``
    (no inout/output_existing on ``current_hidden``), then rebinds
    ``current_hidden = next_hidden``. After 4 iterations the result is
    copied to ``c``.

    Expected: c == a + 4 (each iter adds 1).
    Buggy behaviour (current code): each iter sees the loop-entry value of
    ``current_hidden`` (i.e. the original input ``a``), so c == a + 1 (only
    the last iteration's +1 is reflected, because ``next_hidden`` is what the
    final copy reads — actually since we copy ``current_hidden`` after the
    loop, c == a (no rebind ever observed)).
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "orch_range_carry_rebind"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=10.0),
            TensorSpec("c", [16, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class OrchRangeCarryRebindProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_add_one(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                dst: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                tile_in: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                tile_out: pl.Tile[[16, 64], pl.FP32] = pl.add(tile_in, 1.0)
                out: pl.Tensor[[16, 64], pl.FP32] = pl.store(tile_out, [0, 0], dst)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_copy(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                dst: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                tile_in: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                out: pl.Tensor[[16, 64], pl.FP32] = pl.store(tile_in, [0, 0], dst)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                # Initialise carry from the input (one copy so we don't write
                # back into `a` itself).
                current_hidden: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                current_hidden = self.kernel_copy(a, current_hidden)
                for _ in pl.range(4):
                    next_hidden: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                    next_hidden = self.kernel_add_one(current_hidden, next_hidden)
                    current_hidden = next_hidden  # carry rebind across iterations
                c = self.kernel_copy(current_hidden, c)
                return c

        return OrchRangeCarryRebindProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + 4.0


class TestCtrlFlowOperations:
    """Test suite for control flow operations."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_add(self, test_runner, platform):
        """Test for loop wrapping tile add."""
        result = test_runner.run(TestForLoopAdd(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_mul(self, test_runner, platform):
        """Test for loop wrapping tile mul."""
        result = test_runner.run(TestForLoopMul(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_yield_add(self, test_runner, platform):
        """Test for loop with yield carrying tensor across iterations."""
        result = test_runner.run(TestForLoopYieldAdd(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_yield_tile_accum(self, test_runner, platform):
        """Test for loop with yield carrying tile accumulator across iterations."""
        result = test_runner.run(TestForLoopYieldTileAccum(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_if_yield_tensor(self, test_runner, platform):
        """Test if-else with yield carrying tensors."""
        result = test_runner.run(TestIfYieldTensor(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_if_else_nested(self, test_runner, platform):
        """Test if-else nested inside a for loop."""
        result = test_runner.run(TestForIfElseNested(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(reason="PTOAS BUG")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_while_loop_add(self, test_runner, platform):
        """Test while loop add (scf.while codegen)."""
        result = test_runner.run(TestWhileLoopAdd(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(reason="PTOAS BUG")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_break(self, test_runner, platform):
        """Test for loop with break."""
        result = test_runner.run(TestForLoopBreak(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(reason="PTOAS BUG")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_continue(self, test_runner, platform):
        """Test for loop with continue."""
        result = test_runner.run(TestForLoopContinue(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(reason="PTOAS BUG")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_for_loop_break_continue(self, test_runner, platform):
        """Test for loop with break and continue."""
        result = test_runner.run(TestForLoopBreakContinue(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_orch_range_carry_rebind(self, test_runner, platform):
        """Reproducer for issue #1286: orchestration pl.range carry rebind to fresh tensor."""
        result = test_runner.run(TestOrchRangeCarryRebind(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
