# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821
"""Tests for the AOT comm-group manifest pipeline (v2 schema, N1.4).

Compile-time:  Program → ``lift_comm_manifest`` → JSON-safe dict
On disk:       ``output_dir/orchestration/comm_manifest.json``
Runtime:       dict → ``_build_chip_bootstrap_configs_from_manifest`` → list[ChipBootstrapConfig]

The CollectCommGroups pass that *infers* CommGroups from
``pld.alloc_window_buffer`` ops is added in N4. These tests pre-stage the
CommGroups directly on a hand-built ``ir.Program`` so the manifest pipeline
can be exercised independently of the inference logic.
"""

import json

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    CommGroup,
    ConstInt,
    Program,
    Span,
    Var,
    WindowBuffer,
)


def _make_dc(device_ids):
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    return DistributedConfig(device_ids=list(device_ids))


def _lift(program):
    from pypto.ir.comm_manifest import lift_comm_manifest  # noqa: PLC0415

    return lift_comm_manifest(program)


def _build(manifest, device_ids, rootinfo_path="/tmp/_test_rootinfo.bin"):
    # Skip in environments without simpler installed (e.g. unit-tests CI).
    pytest.importorskip("simpler.task_interface")
    from pypto.runtime.distributed_runner import (  # noqa: PLC0415
        _build_chip_bootstrap_configs_from_manifest,
    )

    return _build_chip_bootstrap_configs_from_manifest(manifest, _make_dc(device_ids), rootinfo_path)


def _const(value: int) -> ConstInt:
    return ConstInt(value, DataType.INT64, Span.unknown())


def _trivial_program(groups: list[CommGroup] | None = None) -> Program:
    """Build a minimal Program (optionally with CommGroups) via @pl.program +
    immediate Program(...) reconstruction. Until the CollectCommGroups pass
    lands (N4), tests pre-stage the groups directly through the constructor.
    """

    @pl.program
    class P:
        @pl.function
        def f(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

    if not groups:
        return P
    funcs = list(P.functions.values())
    return Program(funcs, list(groups), P.name, P.span)


# ---------------------------------------------------------------------------
# Compile-time: lift_comm_manifest
# ---------------------------------------------------------------------------


def test_lift_no_comm_group_returns_none():
    """A program without any CommGroup must skip the manifest entirely."""
    p = _trivial_program()
    assert _lift(p) is None


def test_lift_const_size_emits_json_safe_manifest():
    """All-devices group with literal sizes lifts to a JSON-safe v2 manifest."""
    slots = [
        WindowBuffer("data", _const(256), DataType.FP32),
        WindowBuffer("signal", _const(2), DataType.INT32),
    ]
    p = _trivial_program([CommGroup([], slots)])  # empty devices = all

    manifest = _lift(p)
    assert manifest is not None
    # Manifest must round-trip through JSON without losing fidelity.
    assert json.loads(json.dumps(manifest)) == manifest

    assert manifest["version"] == 2
    assert len(manifest["comm_groups"]) == 1
    g = manifest["comm_groups"][0]
    assert g["devices"] == []  # empty = all devices
    assert g["slots"] == [
        {
            "name": "data",
            "dtype": "float32",
            "size": 256,
            "bits_per_element": 32,
            "load_from_host": False,
            "store_to_host": False,
        },
        {
            "name": "signal",
            "dtype": "int32",
            "size": 2,
            "bits_per_element": 32,
            "load_from_host": False,
            "store_to_host": False,
        },
    ]


def test_lift_explicit_device_list():
    """A group with explicit devices serializes the literal list."""
    p = _trivial_program([CommGroup([0, 1], [WindowBuffer("data", _const(64), DataType.FP32)])])

    manifest = _lift(p)
    assert manifest is not None
    assert manifest["comm_groups"][0]["devices"] == [0, 1]


def test_lift_dynamic_size_unsupported_raises():
    """Symbolic ``size`` is rejected at compile time so authors get a clear error."""
    sym = Var("N", _const(0).type, Span.unknown())
    p = _trivial_program([CommGroup([], [WindowBuffer("signal", sym, DataType.INT32)])])
    with pytest.raises(RuntimeError, match="dynamic WindowBuffer size is not supported"):
        _lift(p)


def test_lift_two_comm_groups_raises():
    """Multi-group programs are not yet supported by the runner."""
    p = _trivial_program(
        [
            CommGroup([], [WindowBuffer("a", _const(64), DataType.FP32)]),
            CommGroup([], [WindowBuffer("b", _const(64), DataType.FP32)]),
        ]
    )
    with pytest.raises(RuntimeError, match="at most one CommGroup"):
        _lift(p)


def test_lift_load_store_to_host_flags_propagate():
    """``load_from_host`` / ``store_to_host`` bool flags pass through 1:1."""
    slots = [
        WindowBuffer("lut", _const(64), DataType.FP32, load_from_host=True),
        WindowBuffer("met", _const(64), DataType.INT32, store_to_host=True),
    ]
    p = _trivial_program([CommGroup([], slots)])

    manifest = _lift(p)
    assert manifest is not None
    slot_data = manifest["comm_groups"][0]["slots"]
    assert slot_data[0]["load_from_host"] is True
    assert slot_data[0]["store_to_host"] is False
    assert slot_data[1]["load_from_host"] is False
    assert slot_data[1]["store_to_host"] is True


# ---------------------------------------------------------------------------
# Runtime: _build_chip_bootstrap_configs_from_manifest
# ---------------------------------------------------------------------------


def test_build_none_manifest_returns_none():
    pytest.importorskip("simpler.task_interface")
    assert _build(None, [0, 1]) is None


def _make_manifest(devices: list[int], slots: list[dict]) -> dict:
    return {"version": 2, "comm_groups": [{"devices": devices, "slots": slots}]}


def _slot(
    name: str,
    size: int,
    dtype: str = "float32",
    bits: int = 32,
    *,
    load_from_host: bool = False,
    store_to_host: bool = False,
) -> dict:
    return {
        "name": name,
        "dtype": dtype,
        "size": size,
        "bits_per_element": bits,
        "load_from_host": load_from_host,
        "store_to_host": store_to_host,
    }


def test_build_full_coverage_via_empty_devices():
    """Empty ``devices`` list ⇒ every device in the dc gets a comm config."""
    pytest.importorskip("simpler.task_interface")
    manifest = _make_manifest([], [_slot("data", 256), _slot("signal", 2, "int32")])
    cfgs = _build(manifest, [0, 1])
    assert cfgs is not None
    assert len(cfgs) == 2
    assert all(c.comm is not None for c in cfgs)
    assert cfgs[0].comm.rank == 0 and cfgs[1].comm.rank == 1
    assert all(c.comm.nranks == 2 for c in cfgs)
    assert all(c.comm.window_size == 256 * 4 + 2 * 4 for c in cfgs)
    assert [b.name for b in cfgs[0].buffers] == ["data", "signal"]


def test_build_explicit_subset_keeps_extras_commless():
    """Devices not in the list get bare ``ChipBootstrapConfig()`` (comm=None)."""
    pytest.importorskip("simpler.task_interface")
    manifest = _make_manifest([0, 1], [_slot("data", 64)])
    cfgs = _build(manifest, [0, 1, 2, 3])
    assert cfgs is not None
    assert len(cfgs) == 4
    assert cfgs[0].comm is not None and cfgs[0].comm.rank == 0
    assert cfgs[1].comm is not None and cfgs[1].comm.rank == 1
    assert cfgs[2].comm is None and cfgs[3].comm is None


def test_build_subset_out_of_range_raises():
    pytest.importorskip("simpler.task_interface")
    manifest = _make_manifest([3, 4], [_slot("data", 1)])
    with pytest.raises(RuntimeError, match="outside DistributedConfig.device_ids range"):
        _build(manifest, [0, 1])


def test_build_unknown_version_raises():
    pytest.importorskip("simpler.task_interface")
    manifest = {"version": 999, "comm_groups": [{"devices": [], "slots": []}]}
    with pytest.raises(RuntimeError, match="version mismatch"):
        _build(manifest, [0, 1])


def test_build_rejects_two_groups():
    pytest.importorskip("simpler.task_interface")
    manifest = {
        "version": 2,
        "comm_groups": [
            {"devices": [], "slots": []},
            {"devices": [], "slots": []},
        ],
    }
    with pytest.raises(RuntimeError, match="exactly one CommGroup"):
        _build(manifest, [0, 1])


def test_build_subbyte_dtype_byte_calculation():
    """nbytes rounds up for sub-byte dtypes (e.g. INT4)."""
    pytest.importorskip("simpler.task_interface")
    # INT4: 4 bits per element, 7 elements → ceil(7*4/8) = 4 bytes.
    manifest = _make_manifest([], [_slot("x", 7, "int4", bits=4)])
    cfgs = _build(manifest, [0, 1])
    assert cfgs is not None
    assert cfgs[0].buffers[0].nbytes == 4
    assert cfgs[0].comm.window_size == 4


def test_build_load_store_host_flags():
    """Slot bool flags propagate to ChipBufferSpec without modification."""
    pytest.importorskip("simpler.task_interface")
    manifest = _make_manifest(
        [],
        [
            _slot("lut", 4, load_from_host=True),
            _slot("out", 4, store_to_host=True),
        ],
    )
    cfgs = _build(manifest, [0, 1])
    assert cfgs is not None
    assert cfgs[0].buffers[0].load_from_host is True
    assert cfgs[0].buffers[0].store_to_host is False
    assert cfgs[0].buffers[1].load_from_host is False
    assert cfgs[0].buffers[1].store_to_host is True


# ---------------------------------------------------------------------------
# AOT roundtrip: writing manifest to disk and re-reading it
# ---------------------------------------------------------------------------


def test_aot_roundtrip_writes_and_loads_manifest(tmp_path):
    """``emit_comm_manifest`` writes the file and ``json.load`` recovers an
    identical dict to the in-memory ``lift_comm_manifest`` output.
    """
    from pypto.ir.comm_manifest import (  # noqa: PLC0415
        COMM_MANIFEST_FILENAME,
        emit_comm_manifest,
    )

    p = _trivial_program([CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32)])])

    expected = _lift(p)
    out_path = emit_comm_manifest(p, tmp_path)
    assert out_path is not None
    assert out_path == tmp_path / "orchestration" / COMM_MANIFEST_FILENAME

    with out_path.open("r", encoding="utf-8") as fh:
        actual = json.load(fh)
    assert actual == expected


def test_aot_roundtrip_no_group_writes_no_file(tmp_path):
    from pypto.ir.comm_manifest import emit_comm_manifest  # noqa: PLC0415

    p = _trivial_program()
    assert emit_comm_manifest(p, tmp_path) is None
    assert not (tmp_path / "orchestration").exists()


# ---------------------------------------------------------------------------
# Sanity: ensure the new pld.* surface area exposes only DistributedTensor
# ---------------------------------------------------------------------------


def test_pld_does_not_export_legacy_dataclasses():
    """N1 removes the user-declared ``pld.CommGroup`` / ``pld.WindowBuffer``."""
    assert not hasattr(pld, "CommGroup")
    assert not hasattr(pld, "WindowBuffer")
    assert hasattr(pld, "DistributedTensor")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
