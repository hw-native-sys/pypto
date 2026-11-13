# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CANNBotDSL public-export preparation, without optional-compiler collection imports."""

import hashlib
import importlib
import inspect
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pypto.backend.external_bridge import RuntimeTarget
from pypto.backend.tmr_bridge import TMRBridge
from pypto.jit.external.contracts import (
    ABIValue,
    Access,
    CompileTarget,
    Core,
    Coverage,
    KernelRequest,
    ParameterSpec,
    ScalarSpec,
    TensorSpec,
)
from pypto.jit.external.identity import preparation_key, prepared_digest
from pypto.jit.external.normalization import plan_call
from pypto.jit.external.providers import cannbotdsl as adapter
from pypto.jit.external.validation import ContractError
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout


def _kernel(alpha, x, out):
    pass


def _request():
    tensor = TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1))
    return KernelRequest(
        (
            ParameterSpec("alpha", ScalarSpec(DataType.FP32)),
            ParameterSpec("x", tensor),
            ParameterSpec("out", tensor),
        ),
        CompileTarget("a5", "ccec", "test"),
    )


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    include = tmp_path / "include"
    (include / "asc_ext").mkdir(parents=True)
    header = b"// provider header\n"
    (include / "asc_ext/b64.h").write_bytes(header)
    source = "// device source\n"
    manifest = {
        "protocol": adapter.ABI_NAME,
        "schema_version": 1,
        "arch": "dav-3510",
        "calling_convention": "device_function",
        "symbol": "kernel_scaled",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "producer": {"name": "cannbot-dsl", "version": "0.8.1"},
        "parameters": [
            {
                "logical_index": 1,
                "name": "x",
                "kind": "tensor",
                "dtype": "f32",
                "access": "read",
                "layout_map": 0,
                "shape": [-1, 32],
                "stride": [-1, 1],
                "size": 24,
                "alignment": 8,
            },
            {
                "logical_index": 2,
                "name": "out",
                "kind": "tensor",
                "dtype": "f32",
                "access": "write",
                "layout_map": 0,
                "shape": [-1, 32],
                "stride": [-1, 1],
                "size": 24,
                "alignment": 8,
            },
            {"logical_index": 0, "name": "alpha", "kind": "scalar", "size": 4, "alignment": 4},
        ],
        "core_type": "aiv",
        "participants": {"aic": 0, "aiv": 1},
        "completion": {
            "local_pipelines_drained": True,
            "scalar_gm_flushed": True,
            "ddr_barrier": True,
            "publication": "consumer_after_all_participants_return",
        },
        "contract": {"tensor_views": {"pointer": "first_valid_element", "offset_applied_by": "consumer"}},
        "resources": {
            "gm_write_isolation": {"granularity_bytes": 64, "rule": "serialize_conflicting_cache_lines"}
        },
        "state": {"clobbered": ["DMA", "local_memory"]},
        "loading": {"text_only": "requires_binary_validation"},
        "build_requirements": {"dsl_headers": {"asc_ext/b64.h": hashlib.sha256(header).hexdigest()}},
    }
    calls = []
    export = SimpleNamespace(source=source, manifest=manifest)

    def export_device(kernel, *args, **kwargs):
        calls.append((kernel, args, kwargs))
        return export

    cb = SimpleNamespace(
        dtypes=SimpleNamespace(float32=object()),
        inspect_kernel=lambda kernel: SimpleNamespace(configured=False, signature=inspect.signature(kernel)),
        Dim=lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        TensorSpec=lambda shape, dtype, **kwargs: SimpleNamespace(shape=shape, dtype=dtype, **kwargs),
        export_device=export_device,
        get_include=lambda: str(include),
    )
    monkeypatch.setitem(sys.modules, "cannbotdsl", cb)
    return SimpleNamespace(cb=cb, manifest=manifest, export=export, calls=calls, include=include)


def test_preparation_preserves_physical_order_and_hidden_partial_writes(compiler):
    request = _request()
    prepared = adapter.CANNBotDSLProvider().prepare(_kernel, request)
    assert prepared.request is request
    assert prepared.native_abi.name == adapter.ABI_NAME
    assert [p.parameter for p in prepared.native_abi.parameters] == [1, 2, 0]
    assert [p.value for p in prepared.native_abi.parameters] == [
        ABIValue.DESCRIPTOR,
        ABIValue.DESCRIPTOR,
        ABIValue.SCALAR,
    ]
    assert prepared.contract.public_returns == ()
    assert [(e.parameter, e.access, e.coverage) for e in prepared.contract.effects] == [
        (1, Access.READ, Coverage.NONE),
        (2, Access.WRITE, Coverage.UNKNOWN),
    ]
    plan = plan_call(prepared.contract)
    assert plan.public_slots == ()
    assert plan.writeback_slots == ((2, 0),)
    assert preparation_key(request, prepared.identity) is None
    assert len(prepared_digest(prepared)) == 64
    adapter.CANNBotDSLProvider().prepare(_kernel, request)
    assert len(compiler.calls) == 2  # No hidden provider compilation cache.


def test_dynamic_symbols_and_scalar_types_are_shared_without_addresses(compiler):
    adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    kernel, args, kwargs = compiler.calls[0]
    assert kernel is _kernel and kwargs == {"arch": "dav-3510"}
    assert args[0] is compiler.cb.dtypes.float32
    assert args[1].shape[0] is args[2].shape[0]
    assert args[1].stride[0] is args[2].stride[0]
    assert args[1].shape[0].min == 0
    assert args[1].stride[0].min == 1
    assert args[1].stride[-1] == 1


def test_symbol_shared_by_extent_and_stride_is_positive(compiler):
    request = _request()
    tensor = TensorSpec(("rows", 32), DataType.FP32, strides=("rows", 1))
    request = replace(
        request,
        parameters=(request.parameters[0], ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
    )
    adapter.CANNBotDSLProvider().prepare(_kernel, request)
    _, args, _ = compiler.calls[0]
    assert args[1].shape[0] is args[1].stride[0]
    assert args[1].shape[0] is args[2].shape[0]
    assert args[1].shape[0].min == 1


@pytest.mark.parametrize("unreadable", [False, True])
def test_declared_header_read_failure_is_a_contract_error(compiler, monkeypatch, unreadable):
    if unreadable:

        def deny_read(path):
            raise PermissionError("header is not readable")

        monkeypatch.setattr(adapter.Path, "read_bytes", deny_read)
    else:
        (compiler.include / "asc_ext/b64.h").unlink()
    with pytest.raises(
        ContractError, match="CANNBotDSL: cannot read declared DSL header 'asc_ext/b64.h'"
    ) as error:
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    assert isinstance(error.value.__cause__, PermissionError if unreadable else FileNotFoundError)


def test_artifact_owns_headers_and_retains_execution_requirements(compiler):
    prepared = adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    before = prepared_digest(prepared)
    (compiler.include / "asc_ext/b64.h").unlink()
    compiler.manifest["resources"].clear()
    owned = {f.path: f.content for f in prepared.artifact.files}
    manifest = json.loads(owned["device_manifest.json"])
    assert manifest["resources"]["gm_write_isolation"]["granularity_bytes"] == 64
    assert manifest["contract"]["tensor_views"]["offset_applied_by"] == "consumer"
    assert owned["include/asc_ext/b64.h"] == b"// provider header\n"
    assert prepared_digest(prepared) == before


def test_named_descriptor_abi_cannot_be_executed_by_c_pointer_bridge(compiler):
    prepared = adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    target = RuntimeTarget("tmr", "1", prepared.request.target, frozenset({("c_pointer", 1)}), frozenset())
    with pytest.raises(ContractError, match="unsupported native ABI"):
        TMRBridge().lower(prepared, target)


@pytest.mark.parametrize(
    "core,participants,expected",
    [
        ("aic", {"aic": 1, "aiv": 0}, (Core.AIC,)),
        ("aiv", {"aic": 0, "aiv": 1}, (Core.AIV,)),
        ("mixed", {"aic": 1, "aiv": 2}, (Core.AIC, Core.AIV)),
    ],
)
def test_execution_topology_comes_from_export(compiler, core, participants, expected):
    compiler.manifest.update(core_type=core, participants=participants)
    prepared = adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    assert prepared.execution.cores == expected
    assert prepared.execution.aiv_lanes == participants["aiv"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"schema_version": True}, "protocol version"),
        ({"arch": "dav-2201"}, "architecture"),
        ({"calling_convention": "host"}, "device function"),
        ({"source_sha256": "0" * 64}, "source digest"),
        ({"participants": {"aic": 0, "aiv": True}}, "topology"),
        ({"completion": {}}, "local completion"),
        ({"resources": None}, "resources metadata"),
    ],
)
def test_incompatible_exports_fail_before_handoff(compiler, change, match):
    compiler.manifest.update(change)
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("logical_index", 0, "every logical parameter"),
        ("logical_index", True, "every logical parameter"),
        ("name", "wrong", "different name"),
        ("kind", "scalar", "kind mismatch"),
        ("dtype", "i32", "dtype mismatch"),
        ("shape", [8, 32], "shape differs"),
        ("stride", [32, 1], "stride differs"),
        ("layout_map", 1, "ND descriptor"),
        ("layout_map", False, "ND descriptor"),
        ("access", "unknown", "invalid device export"),
        ("size", True, "immutable"),
    ],
)
def test_invalid_parameter_metadata_is_rejected(compiler, field, value, match):
    compiler.manifest["parameters"][0][field] = value
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


def test_header_snapshot_rejects_changes_and_path_escape(compiler):
    path = compiler.include / "asc_ext/b64.h"
    path.write_bytes(b"changed since export")
    with pytest.raises(ContractError, match="header digest mismatch"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    compiler.manifest["build_requirements"]["dsl_headers"] = {"../outside.h": "0" * 64}
    with pytest.raises(ContractError, match="invalid DSL header path"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "spec,match",
    [
        (TensorSpec((32,), DataType.FP32), "explicit strides"),
        (TensorSpec((32,), DataType.FP32, strides=(2,)), "innermost stride 1"),
        (TensorSpec((2, 32), DataType.FP32, strides=(0, 1)), "positive strides"),
        (TensorSpec((32,), DataType.FP32, strides=(1,), layout=TensorLayout.NZ), "ND layout"),
    ],
)
def test_request_does_not_silently_assume_contiguity(compiler, spec, match):
    request = _request()
    request = replace(
        request, parameters=(request.parameters[0], ParameterSpec("x", spec), request.parameters[2])
    )
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, request)
    assert not compiler.calls


def test_constexpr_values_follow_declared_order(compiler):
    def kernel(alpha, scale, x, out):
        pass

    request = replace(_request(), constants=(("scale", 2.5),))
    prepared = adapter.CANNBotDSLProvider().prepare(kernel, request)
    assert compiler.calls[0][1][1] == 2.5
    assert prepared.request.constants == (("scale", 2.5),)
    assert len(prepared.native_abi.parameters) == 3


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"options": (("fast_math", True),)}, "compiler options"),
        ({"target": CompileTarget("a3", "ccec", "test")}, "a5/ccec"),
        ({"constants": (("alpha", 1),)}, "names overlap"),
        ({"constants": (("missing", 1),)}, "kernel signature"),
    ],
)
def test_invalid_requests_do_not_export(compiler, changes, match):
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, replace(_request(), **changes))
    assert not compiler.calls


def test_public_inspection_rejects_configured_or_unknown_kernel(compiler):
    compiler.cb.inspect_kernel = lambda _: None
    with pytest.raises(ContractError, match="recognized by inspect_kernel"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    compiler.cb.inspect_kernel = lambda _: SimpleNamespace(configured=True)
    with pytest.raises(ContractError, match="unconfigured"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    assert not compiler.calls


def test_no_optional_import_until_preparation(monkeypatch):
    monkeypatch.setitem(sys.modules, "cannbotdsl", None)
    importlib.reload(adapter)
    with pytest.raises(ContractError, match="install cannbot-dsl"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


def test_real_public_export_uses_shared_contract_and_owned_source():
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def scaled(alpha, x, out):
        for row in cb.range(x.shape[0]):
            for col in cb.range(32):
                out[row, col] = x[row, col] * alpha

    prepared = adapter.CANNBotDSLProvider().prepare(scaled, _request())
    assert prepared.contract.public_returns == ()
    assert prepared.contract.writebacks[0].parameter == 2
    assert prepared.execution.cores == (Core.AIV,)
    assert [p.parameter for p in prepared.native_abi.parameters] == [1, 2, 0]
    assert preparation_key(prepared.request, prepared.identity) is None
    artifact = {f.path: f.content for f in prepared.artifact.files}
    manifest = json.loads(artifact["device_manifest.json"])
    assert manifest["protocol"] == adapter.ABI_NAME
    assert manifest["parameters"][0]["dynamic_fields"]
    assert b"kernel_scaled" in artifact["device_callable.hpp"]
    assert any(name.startswith("include/asc_ext/") for name in artifact)


def test_real_constexpr_and_e8m0_export():
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    def copy_scales(source, out, width):
        # Copy scale bits without arithmetic: the ABI remains E8M0, rather
        # than being silently changed to uint8 by the provider adapter.
        staging = cb.Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, width)))
        cb.mem_copy(staging, source.view(dtype=cb.dtypes.uint8))
        cb.mem_copy(out.view(dtype=cb.dtypes.uint8), staging)

    # Set the optional compiler annotation at runtime so collecting/type-checking
    # this test does not require importing the compiler package.
    copy_scales.__annotations__["width"] = cb.Constexpr[int]
    kernel = cb.kernel(copy_scales)

    tensor = TensorSpec((1, 32), DataType.FP8E8M0, strides=(32, 1))
    request = KernelRequest(
        (ParameterSpec("source", tensor), ParameterSpec("out", tensor)),
        CompileTarget("a5", "ccec", "test"),
        constants=(("width", 32),),
    )
    prepared = adapter.CANNBotDSLProvider().prepare(kernel, request)
    assert prepared.request.parameters[0].type.dtype == DataType.FP8E8M0
    manifest = json.loads(
        next(f.content for f in prepared.artifact.files if f.path == "device_manifest.json")
    )
    assert [p["dtype"] for p in manifest["parameters"]] == ["f8e8m0", "f8e8m0"]
    assert len(prepared.native_abi.parameters) == 2
    assert prepared.contract.writebacks[0].parameter == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
