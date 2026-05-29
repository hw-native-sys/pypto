# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ruff: noqa: F841
# pyright: reportAttributeAccessIssue=false

"""Before/Expected tests for the ExpandManualPhaseFence pass.

The pass runs after DeriveCallDirections, so these tests write post-derive call
sites directly with ``attrs["manual_dep_edges"]`` and compare the transformed
program to an explicit Expected program via ``ir.assert_structural_equal``.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes


def _expand(program: ir.Program) -> ir.Program:
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.expand_manual_phase_fence()(program)


def _assert_expands(before: ir.Program, expected: ir.Program) -> None:
    after = _expand(before)
    ir.assert_structural_equal(after, expected)


def test_profitable_parallel_array_dep_inserts_dummy_and_rewrites_consumers():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.parallel(4):
                    a = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
                    b = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_parallel_iter_arg_dep_falls_back_even_with_visible_init_value():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_iter,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_iter,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_parallel_iter_arg_without_visible_init_is_not_hoisted():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                for p, (tids_iter,) in pl.parallel(4, init_values=(pl.array.create(4, pl.TASK_ID),)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                for p, (tids_iter,) in pl.parallel(4, init_values=(pl.array.create(4, pl.TASK_ID),)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_iter_arg_dep_falls_back_even_with_visible_init_value():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_iter,) in pl.range(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_iter,) in pl.range(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids_iter]})
                    tids_out = pl.yield_(tids_iter)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_parallel_loop_local_dep_array_is_not_moved_before_definition():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    local_tids = tids
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    local_tids = tids
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_loop_local_dep_array_is_not_moved_before_definition():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    local_tids = tids
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    local_tids = tids
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_same_carrier_dep_array_update_in_body_falls_back():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_after_a = pl.array.update_element(tids, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_after_a = pl.array.update_element(tids, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_parallel_iter_arg_alias_update_of_dep_array_falls_back():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_branch,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_branch_after_a = pl.array.update_element(tids_branch, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_out = pl.yield_(tids_branch_after_a)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_branch,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_branch_after_a = pl.array.update_element(tids_branch, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_out = pl.yield_(tids_branch_after_a)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_iter_arg_alias_update_of_dep_array_falls_back():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_branch,) in pl.range(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_branch_after_a = pl.array.update_element(tids_branch, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_out = pl.yield_(tids_branch_after_a)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (tids_branch,) in pl.range(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_branch_after_a = pl.array.update_element(tids_branch, 0, a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_out = pl.yield_(tids_branch_after_a)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_nested_iter_arg_alias_update_of_dep_array_falls_back():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (outer_tids,) in pl.parallel(4, init_values=(tids,)):
                    for q, (inner_tids,) in pl.parallel(4, init_values=(outer_tids,)):
                        a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                        inner_tids_after_a = pl.array.update_element(inner_tids, 0, a)
                        b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                        inner_out = pl.yield_(inner_tids_after_a)
                    outer_out = pl.yield_(inner_out)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (outer_tids,) in pl.parallel(4, init_values=(tids,)):
                    for q, (inner_tids,) in pl.parallel(4, init_values=(outer_tids,)):
                        a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                        inner_tids_after_a = pl.array.update_element(inner_tids, 0, a)
                        b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                        inner_out = pl.yield_(inner_tids_after_a)
                    outer_out = pl.yield_(inner_out)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_outer_dep_array_consumers_fall_back_when_nested_loop_updates_alias():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (outer_tids,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    for q, (inner_tids,) in pl.parallel(4, init_values=(outer_tids,)):
                        inner_tids_after_a = pl.array.update_element(inner_tids, 0, a)
                        inner_out = pl.yield_(inner_tids_after_a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    outer_out = pl.yield_(inner_out)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p, (outer_tids,) in pl.parallel(4, init_values=(tids,)):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    for q, (inner_tids,) in pl.parallel(4, init_values=(outer_tids,)):
                        inner_tids_after_a = pl.array.update_element(inner_tids, 0, a)
                        inner_out = pl.yield_(inner_tids_after_a)
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    outer_out = pl.yield_(inner_out)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_double_buffered_dep_array_update_remains_compressible():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    tids_new = tids
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    tids_new_after_a = pl.array.update_element(tids_new, 0, a)
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.parallel(4):
                    tids_new = tids
                    a = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
                    b = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
                    tids_new_after_a = pl.array.update_element(tids_new, 0, a)
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_nested_if_return_dep_array_is_not_moved_before_definition():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    if True:
                        local_tids = pl.yield_(tids)
                    else:
                        local_tids = pl.yield_(tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    if True:
                        local_tids = pl.yield_(tids)
                    else:
                        local_tids = pl.yield_(tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_parallel_nested_for_return_dep_array_is_not_moved_before_definition():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    for q, (iter_tids,) in pl.range(1, init_values=(tids,)):
                        local_tids = pl.yield_(iter_tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    for q, (iter_tids,) in pl.range(1, init_values=(tids,)):
                        local_tids = pl.yield_(iter_tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_nested_for_return_dep_array_is_not_moved_before_definition():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    for q, (iter_tids,) in pl.range(1, init_values=(tids,)):
                        local_tids = pl.yield_(iter_tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    for q, (iter_tids,) in pl.range(1, init_values=(tids,)):
                        local_tids = pl.yield_(iter_tids)
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [local_tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_non_orchestration_function_is_ignored():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.AIV)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.AIV)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_scalar_dep_does_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tid = pl.system.task_invalid()
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tid]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tid]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tid = pl.system.task_invalid()
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tid]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tid]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_mixed_array_scalar_deps_do_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                tid = pl.system.task_invalid()
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids, tid]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids, tid]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                tid = pl.system.task_invalid()
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids, tid]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids, tid]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_partial_slot_dep_does_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                slot = pl.array.get_element(tids, 0)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [slot]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [slot]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                slot = pl.array.get_element(tids, 0)
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [slot]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [slot]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_two_by_two_low_benefit_does_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(2, pl.TASK_ID)
                for p in pl.parallel(2):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(2, pl.TASK_ID)
                for p in pl.parallel(2):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_two_by_two_low_benefit_does_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(2, pl.TASK_ID)
                for p in pl.range(2):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(2, pl.TASK_ID)
                for p in pl.range(2):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_known_zero_sequential_loop_does_not_insert_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                a = pl.system.task_invalid()
                for p in pl.range(0, 0, 1):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                a = pl.system.task_invalid()
                for p in pl.range(0, 0, 1):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_three_by_three_min_profitable_inserts_dummy():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(3, pl.TASK_ID)
                for p in pl.parallel(3):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
                    c = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(3, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.parallel(3):
                    a = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
                    b = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
                    c = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_user_dummy_and_auto_phase_fence_can_mix_on_same_array():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                user_barrier = pl.system.task_dummy(deps=[tids])
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [user_barrier]})
                    b = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                user_barrier = pl.system.task_dummy(deps=[tids])
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.parallel(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [user_barrier]})
                    b = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_sequential_stable_outer_dep_hoists_barrier_once():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.range(4):
                    a = self.kernel(
                        attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                    )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_nested_sequential_stable_outer_dep_hoists_barrier_once():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    for q in pl.range(2):
                        a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.range(4):
                    for q in pl.range(2):
                        a = self.kernel(
                            attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                        )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


def test_nested_parallel_fanout_in_sequential_loop_hoists_barrier_once():
    @pl.program
    class Before:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                for p in pl.range(4):
                    for q in pl.parallel(4):
                        a = self.kernel(attrs={"arg_directions": [], "manual_dep_edges": [tids]})
            return pl.system.task_invalid()

    @pl.program
    class Expected:
        @pl.function
        def kernel(self) -> pl.Scalar[pl.TASK_ID]:
            return pl.system.task_invalid()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                tids = pl.array.create(4, pl.TASK_ID)
                phase_fence_barrier_0_tid = pl.system.task_dummy(deps=[tids])
                for p in pl.range(4):
                    for q in pl.parallel(4):
                        a = self.kernel(
                            attrs={"arg_directions": [], "manual_dep_edges": [phase_fence_barrier_0_tid]}
                        )
            return pl.system.task_invalid()

    _assert_expands(Before, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
