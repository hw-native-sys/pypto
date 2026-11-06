# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile public SIMT authoring through the complete JIT pipeline."""

import json
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig, backend, ir
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def scope_copy(
    src: pl.Tensor[[1, 32], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(32, 1, 1)):
        i = pl.simt.thread_idx(pl.simt.Axis.X)
        if i < n:
            value = pl.simt.load(src, (0, i))
            pl.simt.store(dst, (0, i), value)
    return dst


@pl.jit
def scope_entry(
    src: pl.Tensor[[1, 32], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    return scope_copy(src, dst, n)


def test_public_jit_scope_full_compile(tmp_path: Path):
    backend.reset_for_testing()
    try:
        compiled = scope_entry.compile(
            config=RunConfig(
                backend_type=backend.BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(tmp_path),
                dump_passes=True,
            ),
        )
        assert isinstance(compiled, CompiledProgram)
        helpers = list((tmp_path / "vpto").rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        assert "pto.simt_entry" in text
        assert "pto.simt_launch" in text
        assert "pto.ldg" in text and "pto.stg" in text
        manifest = json.loads(helpers[0].with_name("manifest.json").read_text())
        assert manifest["helpers"][0]["kind"] == "simt"
    finally:
        backend.reset_for_testing()


def test_jit_aiv_specialize_preserves_transient_scope():
    lowered = scope_entry.specialize()
    kernel = lowered.get_function("scope_copy")
    assert kernel is not None and kernel.func_type == ir.FunctionType.AIV
    scopes: list[ir.SimtScopeStmt] = []

    class Collector(ir.IRVisitor):
        def visit_simt_scope_stmt(self, op: ir.SimtScopeStmt) -> None:
            scopes.append(op)

    Collector().visit_program(lowered)
    assert len(scopes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
