# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Linx physical topology and target-scoped IR lowering."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes

_LINX_PIPELINE = [
    "InlineFunctions",
    "ConvertToSSA",
    "Simplify",
    "NormalizeStmtStructure",
    "FlattenCallExpr",
    "OutlineIncoreScopes",
    "ConvertTensorToTileOps",
    "LowerLinxTiles",
    "IRVerifier",
]


def test_physical_topology():
    target = backend.get_backend_instance(backend.BackendType.Linx)
    assert target is backend.BackendLinx.instance()
    soc = target.soc
    assert soc.total_die_count() == 4
    assert soc.total_cluster_count() == 0
    assert soc.total_core_count() == target.get_core_count(ir.CoreType.SuperScalar) == 144
    assert soc.total_pe_count() == 576
    assert target.get_core_count(ir.CoreType.CUBE) == target.get_core_count(ir.CoreType.VECTOR) == 0
    ((die, die_count),) = soc.die_counts.items()
    ((core, cores_per_die),) = die.direct_core_counts.items()
    assert (die_count, cores_per_die, len(core.pes)) == (4, 36, 4)
    assert die.total_core_count() == 36
    assert core.core_type == ir.CoreType.SuperScalar
    assert core.shared_capacity_bytes == 256 * 1024
    for pe in core.pes:
        assert isinstance(pe, backend.PE)
        assert pe.cell_capacity_bytes == 256 * 1024
        assert pe.execution_units == ["scalar", "cube", "fixpipe", "v-int", "v-fma", "v-lnexp"]


def test_export_topology(tmp_path):
    msgpack = pytest.importorskip("msgpack")
    path = tmp_path / "linx.msgpack"
    backend.get_backend_instance(backend.BackendType.Linx).export_to_file(str(path))
    snapshot = msgpack.unpackb(path.read_bytes(), raw=False)
    assert snapshot["type"] == "Linx"
    (die,) = snapshot["soc"]["dies"]
    (entry,) = die["die"]["cores"]
    assert (die["count"], entry["count"]) == (4, 36)
    assert entry["core"]["shared_capacity_bytes"] == 262144
    assert len(entry["core"]["pes"]) == 4
    assert entry["core"]["pes"][0]["cell_capacity_bytes"] == 262144


@pytest.mark.parametrize("configured", [None, backend.BackendType.Ascend910, backend.BackendType.Ascend950])
def test_lower_scalar_ir(configured):
    @pl.program
    class ScalarProgram:
        @pl.function
        def main(self, x: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            return x + 0

    if configured is not None:
        backend.set_backend_type(configured)
    seen = []

    def before_pass(current_pass, _program):
        ctx = passes.PassContext.current()
        assert ctx is not None
        assert ctx.get_backend().get_type_name() == "Linx"
        seen.append(current_pass.get_name())

    with ir.PassContext([passes.CallbackInstrument(before_pass=before_pass)]):
        lowered = ir.lower(ScalarProgram, backend_type=backend.BackendType.Linx)

    assert seen == _LINX_PIPELINE
    assert " + " not in ir.python_print(lowered)
    assert backend.is_backend_configured() == (configured is not None)
    if configured is not None:
        assert backend.get_backend_type() == configured


def test_nested_pass_context_inherits_linx_and_restores_ascend():
    backend.set_backend_type(backend.BackendType.Ascend910)
    with ir.PassContext([], backend_type=backend.BackendType.Linx):
        with ir.PassContext([]) as inner:
            assert inner.get_backend() is backend.BackendLinx.instance()
            assert ir.PassManager.get_strategy().pass_names == _LINX_PIPELINE
    assert backend.get_backend_type() == backend.BackendType.Ascend910


def test_conflicting_pto_context_does_not_replace_active_linx():
    backend.set_backend_type(backend.BackendType.Ascend910)
    with ir.PassContext([], backend_type=backend.BackendType.Linx) as linx:
        with pytest.raises(ValueError, match="already set"):
            with ir.PassContext([], backend_type=backend.BackendType.Ascend950):
                pass
        assert ir.PassContext.current() is linx


def test_ir_only_target_rejects_pto_codegen(tmp_path):
    @pl.program
    class ScalarProgram:
        @pl.function
        def main(self, x: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            return x

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Linx supports IR lowering only"):
        ir.compile(ScalarProgram, backend_type=backend.BackendType.Linx, output_dir=str(output))
    assert not output.exists()
    assert not backend.is_backend_configured()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
