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
from pypto import DataType, ir
from pypto.backend import BackendType
from pypto.ir.distributed_compiled_program import (
    _DISTRIBUTED_META_FILENAME,
    DistributedCompiledProgram,
    DistributedConfig,
)
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import DeviceTensor, StackedDeviceTensor


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


def _stacked(full_shape):
    """Build a StackedDeviceTensor of ``full_shape`` (B shards resident per rank)."""
    b = full_shape[0]
    tail = tuple(full_shape[1:])
    shards = [DeviceTensor(0x10000 + i * 0x1000, tail, torch.float32) for i in range(b)]
    return StackedDeviceTensor(shards, full_shape, tuple(range(b)))


def test_call_accepts_stacked_device_tensor(compiled):
    """A StackedDeviceTensor is accepted at the one-shot entry (parity with
    DeviceTensor) and passed through to execute_distributed unchanged."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    stacked = _stacked((128, 128))  # per-card resident shards for the [128,128] param
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed") as mock_exec:
        compiled(a, stacked, out)

    mock_exec.assert_called_once()
    coerced = mock_exec.call_args.args[1]
    assert coerced[1] is stacked  # StackedDeviceTensor reached the runner unchanged


def test_call_validates_stacked_device_tensor_shape(compiled):
    """A StackedDeviceTensor whose full_shape mismatches the param is rejected."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    bad = _stacked((64, 64))  # wrong full_shape for the [128,128] param
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
    assert meta["distributed_config"]["aicpu_thread_num"] == 0


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


def test_from_dir_rejects_non_member_backend_type(compiled, tmp_path):
    """A 'backend_type' naming a class attribute is not a backend.

    ``getattr(BackendType, "mro")`` resolves to a bound method, so a lax lookup
    would accept it and only fail much later inside backend dispatch.
    """
    meta_path = tmp_path / _DISTRIBUTED_META_FILENAME
    meta = json.loads(meta_path.read_text())
    for bogus in ("mro", "AscendNope", 7):
        meta_path.write_text(json.dumps({**meta, "backend_type": bogus}))
        with pytest.raises(ValueError, match=r"backend_type") as excinfo:
            DistributedCompiledProgram.from_dir(tmp_path)
        assert "ir.compile()" in str(excinfo.value), f"{bogus!r}: message lacks a recompile hint"

    # An explicit override does not excuse the payload: the sidecar is loaded as
    # a whole, and every other field (the params!) is still being trusted.
    with pytest.raises(ValueError, match=r"backend_type"):
        DistributedCompiledProgram.from_dir(tmp_path, backend_type=BackendType.Ascend950)


def test_malformed_meta_raises_value_error(compiled, tmp_path):
    """Every malformed payload surfaces as ValueError naming the file, not a raw KeyError.

    Same contract as the L2 ``compiled_meta.json`` (both sidecars share
    ``_load_meta``): a build directory is user-supplied input, so a hand-edited
    or truncated sidecar must fail like bad input rather than leaking whichever
    ``JSONDecodeError`` / ``KeyError`` / ``TypeError`` the offending field
    happened to raise.
    """
    meta_path = tmp_path / _DISTRIBUTED_META_FILENAME
    good = json.loads(meta_path.read_text())
    serialised = json.dumps(good)

    def _param(**overrides):
        """Serialise ``good`` with its first param entry field-patched."""
        return json.dumps({**good, "params": [{**good["params"][0], **overrides}]})

    def _dist_config(value):
        return json.dumps({**good, "distributed_config": value})

    broken = {
        "not JSON": "{ this is not json",
        # A non-atomic writer used to be able to leave exactly this behind.
        "truncated mid-write": serialised[: len(serialised) // 2],
        "top-level list": json.dumps([good]),
        "params not a list": json.dumps({**good, "params": "abc"}),
        "param entry not an object": json.dumps({**good, "params": ["abc"]}),
        "param missing a key": json.dumps({**good, "params": [{"name": "a"}]}),
        # ``getattr(ParamDirection, "mro")`` resolves to a bound method.
        "direction names an attribute": _param(direction="mro"),
        "shape holds a bool": _param(shape=[128, True]),
        "bad dtype": _param(dtype="fp99"),
        "negative return count": json.dumps({**good, "num_return_types": -1}),
        "backend names an attribute": json.dumps({**good, "backend_type": "mro"}),
        "non-string platform": json.dumps({**good, "platform": 7}),
        "distributed_config not an object": _dist_config("abc"),
        "distributed_config unknown key": _dist_config({"device_id": 0}),
        "device_ids not a list": _dist_config({"device_ids": "0"}),
        "device_ids holds a bool": _dist_config({"device_ids": [True]}),
        "non-int aicpu_thread_num": _dist_config({"aicpu_thread_num": "two"}),
        "non-string runtime": _dist_config({"runtime": 7}),
        # Correctly typed but violating the constraints DistributedConfig
        # documents — the reload is held to the same bar as a live config.
        "empty device_ids": _dist_config({"device_ids": []}),
        "duplicate device_ids": _dist_config({"device_ids": [0, 0]}),
        "negative device id": _dist_config({"device_ids": [-1]}),
        "negative num_sub_workers": _dist_config({"num_sub_workers": -1}),
        "aicpu_thread_num below the floor": _dist_config({"aicpu_thread_num": 1}),
    }
    for label, payload in broken.items():
        meta_path.write_text(payload)
        with pytest.raises(ValueError, match=_DISTRIBUTED_META_FILENAME) as excinfo:
            DistributedCompiledProgram.from_dir(tmp_path)
        assert "ir.compile()" in str(excinfo.value), f"{label}: message lacks a recompile hint"


def test_distributed_config_rejects_values_it_documents_as_invalid():
    """The constraints in the class docstring are enforced at construction.

    Validated on the dataclass rather than at either call site so a reloaded
    ``distributed_meta.json`` cannot express a config the live API rejects.
    """
    with pytest.raises(ValueError, match="device_ids must not be empty"):
        DistributedConfig(device_ids=[])  # nothing to run on
    with pytest.raises(ValueError, match="device_ids must be distinct"):
        DistributedConfig(device_ids=[0, 0])  # two ranks would drive one card
    with pytest.raises(ValueError, match="device_ids must be non-negative"):
        DistributedConfig(device_ids=[1, -1])
    with pytest.raises(ValueError, match="num_sub_workers must be non-negative"):
        DistributedConfig(num_sub_workers=-1)
    # 0 means "architecture default"; 1 is below the runtime's floor.
    with pytest.raises(ValueError, match="aicpu_thread_num"):
        DistributedConfig(aicpu_thread_num=1)

    # The documented defaults and the shapes the suite actually uses stay valid.
    DistributedConfig()
    DistributedConfig(device_ids=[0, 1], num_sub_workers=1, aicpu_thread_num=4)
    DistributedConfig(aicpu_thread_num=0)


def test_unextractable_recompile_drops_stale_meta(tmp_path):
    """Recompiling into a reused dir must not leave the previous program's ABI behind.

    ``ir.compile`` never clears ``output_dir``, so a sidecar the new compile
    declines to write would otherwise survive and drive the new artifacts with
    the old parameter ABI — a mismatch ``from_dir`` cannot detect.
    """
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([128, 128], DataType.FP32)
    body = ir.SeqStmts([], span)
    orch = ir.Function(
        "orch",
        [
            (ir.Var("a", tensor_type, span), ParamDirection.In),
            (ir.Var("b", tensor_type, span), ParamDirection.In),
            (ir.Var("c", tensor_type, span), ParamDirection.Out),
        ],
        [],
        body,
        span,
        ir.FunctionType.Orchestration,
    )
    DistributedCompiledProgram(ir.Program([orch], "WithSignature", span), str(tmp_path))
    assert (tmp_path / _DISTRIBUTED_META_FILENAME).exists()  # sanity: written by the first compile

    # No Orchestration and several InCore functions — no resolvable signature,
    # so this compile writes nothing and must remove what the previous one left.
    unextractable = ir.Program(
        [
            ir.Function("k1", [], [], body, span, ir.FunctionType.InCore),
            ir.Function("k2", [], [], body, span, ir.FunctionType.InCore),
        ],
        "NoSignature",
        span,
    )
    DistributedCompiledProgram(unextractable, str(tmp_path))

    assert not (tmp_path / _DISTRIBUTED_META_FILENAME).exists()
    with pytest.raises(FileNotFoundError, match=r"distributed_meta\.json"):
        DistributedCompiledProgram.from_dir(tmp_path)


def test_no_temp_files_left_behind(compiled, tmp_path):
    """The atomic write leaves no ``.tmp`` residue next to the sidecar."""
    assert (tmp_path / _DISTRIBUTED_META_FILENAME).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_from_dir_overrides_platform_and_config(compiled, tmp_path):
    """Explicit platform / distributed_config override the persisted defaults."""
    dc = DistributedConfig(device_ids=[0, 1], aicpu_thread_num=3)
    reloaded = DistributedCompiledProgram.from_dir(tmp_path, platform="a2a3", distributed_config=dc)
    assert reloaded.platform == "a2a3"
    assert reloaded._distributed_config.device_ids == [0, 1]
    assert reloaded._distributed_config.aicpu_thread_num == 3


def test_from_dir_output_indices_match_out_params(compiled, tmp_path):
    """output_indices are rederived from persisted directions (f is the lone Out)."""
    reloaded = DistributedCompiledProgram.from_dir(tmp_path)
    param_infos, output_indices, _ = reloaded._get_metadata()
    assert output_indices == [2]
    assert param_infos[2].direction == ParamDirection.Out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
