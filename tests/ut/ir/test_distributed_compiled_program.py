# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``DistributedCompiledProgram.__call__`` argument acceptance.

These tests compile a small L3 program (no device needed, ``skip_ptoas=True``)
and mock ``execute_distributed`` so the calling convention can be exercised
without a Worker. The focus is the G1 widening: tensor parameters now accept a
worker-resident :class:`DeviceTensor` in addition to a host ``torch.Tensor``.
"""

import json
from unittest.mock import patch

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import (
    _DISTRIBUTED_META_FILENAME,
    DistributedCompiledProgram,
    DistributedConfig,
)
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import DeviceTensor


@pl.program
class _L3AddProgram:
    """L3: HOST orch → CHIP worker (a + b → f)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.add(tile_a, tile_b)
        return pl.store(tile_f, [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.tile_add(a, b, f)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.chip_orch(a, b, f)


@pytest.fixture
def compiled(tmp_path) -> DistributedCompiledProgram:
    prog = ir.compile(
        _L3AddProgram,
        output_dir=str(tmp_path),
        platform="a2a3sim",
        skip_ptoas=True,
        dump_passes=False,
    )
    assert isinstance(prog, DistributedCompiledProgram)
    return prog


def test_call_accepts_device_tensor(compiled):
    """A DeviceTensor input is accepted and passed through to execute_distributed."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    weight = DeviceTensor(0xABCD0000, (128, 128), torch.float32)  # worker-resident
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed") as mock_exec:
        compiled(a, weight, out)

    mock_exec.assert_called_once()
    coerced = mock_exec.call_args.args[1]
    assert coerced[1] is weight  # DeviceTensor reached the runner unchanged


def test_call_rejects_non_tensor(compiled):
    """Non-tensor / non-DeviceTensor args still raise TypeError with guidance."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed"):
        with pytest.raises(TypeError, match="DeviceTensor"):
            compiled(a, "not a tensor", out)  # type: ignore[arg-type]


def test_call_validates_device_tensor_shape(compiled):
    """A DeviceTensor with the wrong shape is rejected by _validate_device_tensor."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    bad = DeviceTensor(0xABCD0000, (64, 64), torch.float32)  # wrong shape
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed"):
        with pytest.raises(TypeError, match="shape"):
            compiled(a, bad, out)


# ---------------------------------------------------------------------------
# from_dir / distributed_meta.json (replay an already-compiled L3 build, #1689)
# ---------------------------------------------------------------------------


def test_compile_persists_distributed_meta(compiled, tmp_path):
    """ir.compile() of an L3 program writes a distributed_meta.json sidecar."""
    meta_path = tmp_path / _DISTRIBUTED_META_FILENAME
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())

    # Param metadata mirrors the HOST orchestrator (post-SSA names that match
    # the generated host_orch.py): a, b are In; f is Out; all 128x128 fp32.
    directions = [p["direction"] for p in meta["params"]]
    dtypes = {p["dtype"] for p in meta["params"]}
    shapes = [p["shape"] for p in meta["params"]]
    assert directions == ["In", "In", "Out"]
    assert dtypes == {"fp32"}
    assert shapes == [[128, 128], [128, 128], [128, 128]]
    assert meta["num_return_types"] == 1
    assert meta["platform"] == "a2a3sim"
    assert meta["backend_type"] == "Ascend910B"
    assert meta["distributed_config"]["runtime"] == "tensormap_and_ringbuffer"


def test_from_dir_round_trips_param_metadata(compiled, tmp_path):
    """from_dir reconstructs the same param metadata as the live compile."""
    reloaded = DistributedCompiledProgram.from_dir(tmp_path)

    def _key(prog):
        infos, _, _ = prog._get_metadata()
        return [(p.name, p.direction, p.shape, str(p.dtype)) for p in infos]

    assert _key(reloaded) == _key(compiled)
    assert reloaded.program is None  # reconstructed from disk, no live IR
    assert reloaded.platform == "a2a3sim"


def test_from_dir_dispatches_via_runner(compiled, tmp_path):
    """A reconstructed program is callable and reaches execute_distributed."""
    reloaded = DistributedCompiledProgram.from_dir(tmp_path)
    a = torch.zeros(128, 128, dtype=torch.float32)
    b = torch.zeros(128, 128, dtype=torch.float32)
    f = torch.zeros(128, 128, dtype=torch.float32)
    with patch("pypto.runtime.distributed_runner.execute_distributed") as mock_exec:
        reloaded(a, b, f)
    mock_exec.assert_called_once()
    # arg 0 is the compiled program; arg 1 the coerced args in param order.
    assert mock_exec.call_args.args[0] is reloaded
    assert list(mock_exec.call_args.args[1]) == [a, b, f]


def test_from_dir_does_not_clobber_debug_runner(compiled, tmp_path):
    """Reloading must preserve a hand-edited debug/run.py (the replay workflow)."""
    run_py = tmp_path / "debug" / "run.py"
    if not run_py.exists():
        pytest.skip("debug/run.py not emitted for this program")
    sentinel = "# hand-edited by the user — must survive from_dir\n"
    run_py.write_text(sentinel)
    DistributedCompiledProgram.from_dir(tmp_path)
    assert run_py.read_text() == sentinel


def test_from_dir_missing_meta_raises(tmp_path):
    """A directory without distributed_meta.json raises with a recompile hint."""
    with pytest.raises(FileNotFoundError, match=r"distributed_meta\.json"):
        DistributedCompiledProgram.from_dir(tmp_path)


def test_from_dir_incompatible_schema_raises(compiled, tmp_path):
    """A distributed_meta.json written under a different schema version is rejected."""
    meta_path = tmp_path / _DISTRIBUTED_META_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["schema"] = meta["schema"] + 1  # simulate an incompatible future format
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema"):
        DistributedCompiledProgram.from_dir(tmp_path)


def test_from_dir_overrides_platform_and_config(compiled, tmp_path):
    """Explicit platform / distributed_config override the persisted defaults."""
    dc = DistributedConfig(device_ids=[0, 1], block_dim=8)
    reloaded = DistributedCompiledProgram.from_dir(tmp_path, platform="a2a3", distributed_config=dc)
    assert reloaded.platform == "a2a3"
    assert reloaded._distributed_config.device_ids == [0, 1]
    assert reloaded._distributed_config.block_dim == 8


def test_from_dir_output_indices_match_out_params(compiled, tmp_path):
    """output_indices are rederived from persisted directions (f is the lone Out)."""
    reloaded = DistributedCompiledProgram.from_dir(tmp_path)
    param_infos, output_indices, _ = reloaded._get_metadata()
    assert output_indices == [2]
    assert param_infos[2].direction == ParamDirection.Out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
