# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for parsing @pl.function(level=..., role=...) (Step 05)."""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir

# ─── Standalone @pl.function ──────────────────────────────────────────────


def test_function_with_level_and_role():
    """@pl.function(level=HOST, role=Worker) sets both."""

    @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
    def worker(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert worker.level == ir.Level.HOST
    assert worker.role == ir.Role.Worker


def test_function_with_level_only():
    """@pl.function(level=GLOBAL) sets level, role defaults to None."""

    @pl.function(level=pl.Level.GLOBAL)
    def orch(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert orch.level == ir.Level.GLOBAL
    assert orch.role is None


def test_function_with_role_only():
    """@pl.function(role=Worker) sets role without level."""

    @pl.function(role=pl.Role.Worker)
    def worker(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert worker.level is None
    assert worker.role == ir.Role.Worker


def test_function_with_type_and_level():
    """@pl.function(type=Orchestration, level=CHIP) sets both."""

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP)
    def orch(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert orch.func_type == ir.FunctionType.Orchestration
    assert orch.level == ir.Level.CHIP


# ─── Backward compatibility ───────────────────────────────────────────────


def test_function_backward_compat_no_level():
    """Existing @pl.function without level/role still works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert f.level is None
    assert f.role is None
    assert f.func_type == ir.FunctionType.Opaque


def test_function_backward_compat_type_only():
    """Existing @pl.function(type=InCore) still works."""

    @pl.function(type=pl.FunctionType.InCore)
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    assert f.func_type == ir.FunctionType.InCore
    assert f.level is None
    assert f.role is None


# ─── @pl.program integration ──────────────────────────────────────────────


def test_program_with_level_role():
    """@pl.function(level=..., role=...) works inside @pl.program."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
        def worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
        def orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

    worker_fn = P.get_function("worker")
    assert worker_fn is not None
    assert worker_fn.level == ir.Level.HOST
    assert worker_fn.role == ir.Role.Worker

    orch_fn = P.get_function("orch")
    assert orch_fn is not None
    assert orch_fn.level is not None
    # POD is alias for CLUSTER_0
    assert ir.level_to_linqu_level(orch_fn.level) == ir.level_to_linqu_level(ir.Level.CLUSTER_0)
    assert orch_fn.role == ir.Role.Orchestrator


def test_program_mixed_with_and_without_level():
    """Mix of @pl.function with and without level in @pl.program."""

    @pl.program
    class P:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
        def worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

    main_fn = P.get_function("main")
    assert main_fn is not None
    assert main_fn.level is None
    assert main_fn.role is None

    worker_fn = P.get_function("worker")
    assert worker_fn is not None
    assert worker_fn.level == ir.Level.HOST
    assert worker_fn.role == ir.Role.Worker


# ─── Printer output ──────────────────────────────────────────────────────


def test_function_level_role_in_printer():
    """Function with level/role prints correctly."""

    @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
    def worker(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    printed = str(worker)
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
