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
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto.backend.cannbotdsl_abi import build_call_adapter
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
                "cpp_type": "NativeTensor",
                "element_cpp_type": "float",
                "access": "read",
                "layout_map": 0,
                "shape": [-1, 32],
                "stride": [-1, 1],
                "size": 24,
                "alignment": 8,
                "pointer_offset": 0,
                "dynamic_fields": [
                    {"field": "shape", "dimension": 0, "slot": 0, "offset": 8},
                    {"field": "stride", "dimension": 0, "slot": 0, "offset": 16},
                ],
            },
            {
                "logical_index": 2,
                "name": "out",
                "kind": "tensor",
                "dtype": "f32",
                "cpp_type": "NativeTensor",
                "element_cpp_type": "float",
                "access": "write",
                "layout_map": 0,
                "shape": [-1, 32],
                "stride": [-1, 1],
                "size": 24,
                "alignment": 8,
                "pointer_offset": 0,
                "dynamic_fields": [
                    {"field": "shape", "dimension": 0, "slot": 0, "offset": 8},
                    {"field": "stride", "dimension": 0, "slot": 0, "offset": 16},
                ],
            },
            {
                "logical_index": 0,
                "name": "alpha",
                "kind": "scalar",
                "size": 4,
                "alignment": 4,
                "scalar_kind": "float",
                "cpp_type": "float",
                "bits": 32,
            },
        ],
        "core_type": "aiv",
        "participants": {"aic": 0, "aiv": 1},
        "completion": {
            "local_pipelines_drained": True,
            "scalar_gm_flushed": True,
            "ddr_barrier": True,
            "publication": "consumer_after_all_participants_return",
        },
        "contract": {
            "schema_version": 6,
            "dimensions": [
                {"name": "rows", "min": 0, "max": None, "multiple_of": 1},
                {"name": "pitch", "min": 1, "max": None, "multiple_of": 1},
            ],
            "parameters": [
                {"name": "alpha", "role": "runtime", "argument": {"kind": "scalar", "element": "f32"}},
                *[
                    {
                        "name": name,
                        "role": "runtime",
                        "argument": {
                            "kind": "tensor",
                            "element": "f32",
                            "address_space": "GM",
                            "shape": [
                                {"kind": "dimension", "name": "rows"},
                                {"kind": "constant", "value": 32},
                            ],
                            "stride": [
                                {"kind": "dimension", "name": "pitch"},
                                {"kind": "constant", "value": 1},
                            ],
                        },
                    }
                    for name in ("x", "out")
                ],
            ],
            "tensor_views": {
                "layout": "ND",
                "pointer": "first_valid_element",
                "offset_applied_by": "consumer",
                "positive_strides": True,
                "innermost_stride": 1,
                "non_overlapping": True,
                "validation": "consumer_must_prove_or_check_shapes_strides_and_dimension_relations",
            },
            "aliasing": {
                "single_parameter_inplace": True,
                "readonly_overlap": True,
                "cross_parameter_write_overlap": False,
            },
        },
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
    contract = compiler.manifest["contract"]
    contract["dimensions"] = [{"name": "rows", "min": 1, "max": None, "multiple_of": 1}]
    for parameter in contract["parameters"][1:]:
        parameter["argument"]["stride"][0]["name"] = "rows"
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


@pytest.mark.parametrize("attribute", ["manifest", "source"])
def test_missing_export_attributes_raise_contract_error(compiler, attribute):
    delattr(compiler.export, attribute)
    with pytest.raises(ContractError, match=f"invalid device export:.*{attribute}") as error:
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    assert isinstance(error.value.__cause__, AttributeError)


@pytest.mark.parametrize(
    "argument,match",
    [
        ({"kind": "scalar", "element": "i32"}, "scalar dtype mismatch: expected f32, got 'i32'"),
        ({"kind": "tensor", "element": "f32"}, "scalar contract kind mismatch"),
        ({"kind": "scalar"}, "invalid device export"),
        (None, "invalid device export"),
    ],
)
def test_scalar_contract_must_match_request_even_at_same_byte_width(compiler, argument, match):
    compiler.manifest["contract"]["parameters"][0]["argument"] = argument
    scalar = compiler.manifest["parameters"][2]
    assert scalar["size"] == scalar["alignment"] == 4
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize("names", [[], ["x", "alpha", "out"], ["alpha", "alpha", "out"]])
def test_logical_runtime_parameters_must_match_request(compiler, names):
    compiler.manifest["contract"]["parameters"] = [{"name": name, "role": "runtime"} for name in names]
    with pytest.raises(ContractError, match="runtime contract parameter names/order"):
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
        ("size", True, "descriptor size"),
    ],
)
def test_invalid_parameter_metadata_is_rejected(compiler, field, value, match):
    compiler.manifest["parameters"][0][field] = value
    with pytest.raises(ContractError, match=match):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "field,value",
    [
        ("pointer_offset", 8),
        ("pointer_offset", False),
        ("alignment", 4),
        ("alignment", 8.0),
        ("size", 16),
        ("size", 32),
        ("dynamic_fields", []),
        ("dynamic_fields", None),
    ],
)
def test_descriptor_layout_is_checked_before_handoff(compiler, field, value):
    compiler.manifest["parameters"][0][field] = value
    with pytest.raises(ContractError, match="descriptor"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "field,value",
    [
        ("offset", 0),
        ("offset", 16),
        ("offset", 9),
        ("offset", 8.0),
        ("dimension", 1),
        ("dimension", False),
        ("slot", 1),
        ("slot", False),
        ("field", "offset"),
    ],
)
def test_dynamic_fields_cannot_overlap_or_address_another_axis(compiler, field, value):
    compiler.manifest["parameters"][0]["dynamic_fields"][0][field] = value
    with pytest.raises(ContractError, match="descriptor dynamic fields"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


def test_dynamic_field_records_may_be_reordered_but_not_duplicated(compiler):
    fields = compiler.manifest["parameters"][0]["dynamic_fields"]
    fields.reverse()
    adapter.CANNBotDSLProvider().prepare(_kernel, _request())
    fields.append(dict(fields[0]))
    with pytest.raises(ContractError, match="descriptor dynamic fields"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize("dynamic", [False, True])
def test_descriptor_slots_cover_static_and_multiple_dynamic_axes(compiler, dynamic):
    if dynamic:
        spec = TensorSpec(("rows", "cols", 32), DataType.FP32, strides=("plane", "pitch", 1), offset=16)
        physical_shape, physical_stride = [-1, -1, 32], [-1, -1, 1]
        shape = [{"kind": "dimension", "name": name} for name in ("rows", "cols")]
        stride = [{"kind": "dimension", "name": name} for name in ("plane", "pitch")]
        fields = [
            {"field": "shape", "dimension": 0, "slot": 0, "offset": 8},
            {"field": "shape", "dimension": 1, "slot": 1, "offset": 16},
            {"field": "stride", "dimension": 0, "slot": 0, "offset": 24},
            {"field": "stride", "dimension": 1, "slot": 1, "offset": 32},
        ]
        dimensions = [
            {"name": name, "min": minimum, "max": None, "multiple_of": 1}
            for name, minimum in (("rows", 0), ("cols", 0), ("plane", 1), ("pitch", 1))
        ]
    else:
        spec = TensorSpec((2, 32), DataType.FP32, strides=(32, 1), offset=16)
        physical_shape, physical_stride = [2, 32], [32, 1]
        shape, stride = [{"kind": "constant", "value": 2}], [{"kind": "constant", "value": 32}]
        fields, dimensions = [], []
    contract = compiler.manifest["contract"]
    contract["dimensions"] = dimensions
    for parameter in contract["parameters"][1:]:
        parameter["argument"].update(
            shape=[*shape, {"kind": "constant", "value": 32}],
            stride=[*stride, {"kind": "constant", "value": 1}],
        )
    for parameter in compiler.manifest["parameters"][:2]:
        parameter.update(
            shape=physical_shape, stride=physical_stride, dynamic_fields=fields, size=40 if dynamic else 8
        )
    request = replace(
        _request(),
        parameters=(_request().parameters[0], ParameterSpec("x", spec), ParameterSpec("out", spec)),
    )
    prepared = adapter.CANNBotDSLProvider().prepare(_kernel, request)
    assert prepared.native_abi.parameters[0].size_bytes == (40 if dynamic else 8)
    # The view offset is not a field offset in the descriptor. It stays in the
    # request for a future bridge to apply exactly once to the data pointer.
    assert prepared.request.parameters[1].type == spec


@pytest.mark.parametrize(
    "field,value",
    [("size", 8), ("alignment", 8), ("bits", 64), ("bits", 32.0), ("scalar_kind", "int")],
)
def test_scalar_physical_abi_must_match_logical_type(compiler, field, value):
    compiler.manifest["parameters"][2][field] = value
    with pytest.raises(ContractError, match=f"scalar {field}"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "scalar"),
        ("element", "i32"),
        ("address_space", "UB"),
        ("shape", [{"kind": "dimension", "name": "pitch"}, {"kind": "constant", "value": 32}]),
        ("stride", [{"kind": "dimension", "name": "rows"}, {"kind": "constant", "value": 1}]),
    ],
)
def test_logical_tensor_contract_cannot_disagree_with_physical_markers(compiler, field, value):
    # Both symbols become -1 in the physical ABI. Checking that ABI alone
    # misses a changed equality relationship between shape and stride.
    compiler.manifest["contract"]["parameters"][1]["argument"][field] = value
    with pytest.raises(ContractError, match="logical tensor contract"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "field,value", [("min", 1), ("min", False), ("max", 1024), ("multiple_of", 16), ("name", "other")]
)
def test_export_cannot_silently_strengthen_dimension_constraints(compiler, field, value):
    compiler.manifest["contract"]["dimensions"][0][field] = value
    with pytest.raises(ContractError, match="logical dimension constraints"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unused"])
def test_dimension_declarations_must_cover_exactly_the_request(compiler, mutation):
    dimensions = compiler.manifest["contract"]["dimensions"]
    if mutation == "missing":
        dimensions.pop()
    else:
        dimensions.append({**dimensions[0], "name": "unused" if mutation == "unused" else "rows"})
    with pytest.raises(ContractError, match="logical dimension constraints"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize("version", [5, 6.0, True])
def test_unknown_or_mistyped_logical_schema_is_rejected(compiler, version):
    compiler.manifest["contract"]["schema_version"] = version
    with pytest.raises(ContractError, match="logical contract schema"):
        adapter.CANNBotDSLProvider().prepare(_kernel, _request())


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("tensor_views", "pointer", "allocation_base"),
        ("tensor_views", "offset_applied_by", "producer"),
        ("tensor_views", "non_overlapping", False),
        ("tensor_views", "innermost_stride", True),
        ("aliasing", "cross_parameter_write_overlap", True),
    ],
)
def test_view_and_aliasing_conventions_are_explicit(compiler, section, field, value):
    compiler.manifest["contract"][section][field] = value
    with pytest.raises(ContractError, match="convention"):
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
    def kernel(scale, alpha, x, out):
        pass

    compiler.manifest["contract"]["parameters"].insert(
        0, {"name": "scale", "role": "constexpr", "argument": None}
    )
    request = replace(_request(), constants=(("scale", 2.5),))
    prepared = adapter.CANNBotDSLProvider().prepare(kernel, request)
    assert compiler.calls[0][1][0] == 2.5
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


@pytest.mark.parametrize("dtype", [DataType.BOOL, DataType.UINT32, DataType.INT64, DataType.INDEX])
def test_real_scalar_storage_width_and_kind(dtype):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def store_value(value, out):
        # Bool scalars are supported even though sub-byte tensor exports are
        # not. Use the scalar in control flow and keep the output byte-addressable.
        if value:
            out[0] = 1
        else:
            out[0] = 0

    request = KernelRequest(
        (
            ParameterSpec("value", ScalarSpec(dtype)),
            ParameterSpec("out", TensorSpec((1,), DataType.INT32, strides=(1,))),
        ),
        CompileTarget("a5", "ccec", "test"),
    )
    prepared = adapter.CANNBotDSLProvider().prepare(store_value, request)
    scalar = next(p for p in prepared.native_abi.parameters if p.value == ABIValue.SCALAR)
    assert scalar.size_bytes == dtype.get_byte()


_CALL_SOURCE = """
struct NativeTensor { __gm__ float* ptr; int64_t shape[1]; int64_t stride[1]; };
__aicore__ inline void kernel_scaled(NativeTensor x, NativeTensor out, float alpha) {
    for (int64_t row = 0; row < x.shape[0]; ++row)
        for (int64_t col = 0; col < 32; ++col)
            out.ptr[row * out.stride[0] + col] = x.ptr[row * x.stride[0] + col];
    out.ptr[0] = alpha;
}
"""


def _prepare_call(compiler, source=_CALL_SOURCE, request=None):
    compiler.export.source = source
    compiler.manifest["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return adapter.CANNBotDSLProvider().prepare(_kernel, _request() if request is None else request)


def _compile_call(artifact, tmp_path, main, *, device=False):
    compiler = shutil.which("ccec") if device else shutil.which("g++-15") or shutil.which("g++")
    if compiler is None:
        pytest.skip("CANN device compiler unavailable" if device else "C++ compiler unavailable")
    for file in artifact.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    root = Path(__file__).resolve().parents[3]
    includes = [root / "runtime/src/common", root / "runtime/src/common/task_interface"]
    includes.extend(tmp_path / path for path in artifact.include_dirs)
    source = tmp_path / "caller.cpp"
    source.write_text(
        ("" if device else "#define __aicore__\n#define __gm__\n") + f'#include "{artifact.entry}"\n' + main
    )
    flags = ["-std=c++17", "-O2"]
    output = tmp_path / ("caller.o" if device else "caller")
    if device:
        # Derive SDK include roots from the selected compiler installation,
        # rather than taking headers from another CANN version on PATH.
        sdk = Path(compiler).resolve().parents[3]
        headers = list(sdk.glob("*/asc/include/c_api/asc_simd.h"))
        if not headers:
            pytest.skip("CANN C API headers unavailable")
        includes.extend(header.parents[1] for header in headers)
        includes.extend(header.parents[2] for header in headers)
        flags += ["-c", "-x", "cce", "--cce-aicore-only", "--cce-aicore-arch=dav-c310-vec"]
    result = subprocess.run(
        [compiler, *flags, *(f"-I{path}" for path in includes), str(source), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


@pytest.mark.parametrize("bits", [0x80000000, 0x7FC12345, 0x3F800000])
@pytest.mark.parametrize("reorder", [False, True])
def test_call_adapter_packs_runtime_views_and_preserves_scalar_bits(compiler, tmp_path, bits, reorder):
    source = _CALL_SOURCE
    if reorder:
        compiler.manifest["parameters"][:2] = reversed(compiler.manifest["parameters"][:2])
        source = source.replace("NativeTensor x, NativeTensor out", "NativeTensor out, NativeTensor x")
    prepared = _prepare_call(compiler, source)
    artifact = build_call_adapter(prepared)
    # Generated adapters are callable headers, not loadable runtime entries.
    assert artifact.entry == "call_adapter.hpp" and artifact.symbol.endswith("::invoke")
    assert b"kernel_entry" not in artifact.files[-1].content
    main = f"""
#include <cstring>
int main() {{
    float input[100]{{}}, output[120]{{}};
    for (int i = 0; i < 100; ++i) input[i] = float(i + 1);
    simpler::tmr::Tensor x{{}}, out{{}};
    x.dtype = out.dtype = DataType::FLOAT32;
    x.buffer = {{uint64_t(input), sizeof(input)}}; x.ndims = 2;
    x.start_offset = 3; x.shapes[0] = 2; x.shapes[1] = 32;
    x.strides[0] = 40; x.strides[1] = 1;
    out.buffer = {{uint64_t(output), sizeof(output)}}; out.ndims = 2;
    out.start_offset = 7; out.shapes[0] = 2; out.shapes[1] = 32;
    out.strides[0] = 40; out.strides[1] = 1;
    int64_t args[] = {{int64_t(&x), int64_t(&out), {bits}LL}};
    {artifact.symbol}(args);
    uint32_t actual; std::memcpy(&actual, &output[7], 4);
    if (actual != {bits}U) return 1;
    if (output[8] != input[4] || output[47] != input[43] || output[78] != input[74]) return 2;
    if (output[0] != 0 || output[6] != 0 || output[39] != 0 || output[79] != 0) return 3;
    // Reuse the same generated source with a different invocation's pointers
    // and offsets. No call-specific address is captured in the artifact.
    x.start_offset = 4; out.start_offset = 8;
    {artifact.symbol}(args);
    if (output[9] != input[5] || output[48] != input[44]) return 4;
}}
"""
    result, binary = _compile_call(artifact, tmp_path, main)
    assert result.returncode == 0, result.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert executed.returncode == 0, executed.stderr


@pytest.mark.parametrize(
    "change,message",
    [
        ("fields", "dynamic field offset mismatch"),
        ("pointer", "descriptor pointer type mismatch"),
        ("dtype", "tensor element dtype mismatch"),
        ("scalar", "native type mismatch"),
    ],
)
def test_compiler_checks_manifest_against_actual_native_layout(compiler, tmp_path, change, message):
    source = _CALL_SOURCE
    if change == "fields":
        source = source.replace("shape[1]; int64_t stride[1]", "stride[1]; int64_t shape[1]")
    elif change in ("pointer", "dtype"):
        source = source.replace("float* ptr", "int32_t* ptr")
        if change == "dtype":
            # Equal widths and internally consistent native metadata must not
            # authorize an int32 descriptor for a logical FP32 request.
            for parameter in compiler.manifest["parameters"][:2]:
                parameter["element_cpp_type"] = "int32_t"
    else:
        source = source.replace("float alpha", "int32_t alpha")
    artifact = build_call_adapter(_prepare_call(compiler, source))
    result, _ = _compile_call(artifact, tmp_path, "int main() {}")
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("change", ["source", "manifest", "metadata", "missing_header", "cpp_type"])
def test_adapter_revalidates_owned_artifacts_without_the_compiler(compiler, monkeypatch, change):
    prepared = _prepare_call(compiler)
    shutil.rmtree(compiler.include)
    monkeypatch.setitem(sys.modules, "cannbotdsl", None)
    assert build_call_adapter(prepared).symbol
    if change == "metadata":
        prepared = replace(prepared, execution=replace(prepared.execution, cores=(Core.AIC,), aiv_lanes=0))
    else:
        files = list(prepared.artifact.files)
        if change == "source":
            files[0] = replace(
                files[0], content=b"// changed", sha256=hashlib.sha256(b"// changed").hexdigest()
            )
        elif change == "missing_header":
            files.pop()
        else:
            manifest = json.loads(files[1].content)
            if change == "manifest":
                manifest["parameters"][0]["dynamic_fields"][0]["offset"] = 16
            else:
                manifest["parameters"][0]["cpp_type"] = "NativeTensor; unsafe()"
            content = json.dumps(manifest).encode()
            files[1] = replace(files[1], content=content, sha256=hashlib.sha256(content).hexdigest())
        prepared = replace(prepared, artifact=replace(prepared.artifact, files=tuple(files)))
    with pytest.raises(ContractError):
        build_call_adapter(prepared)


@pytest.mark.parametrize(
    "dtype,dsl_type,element,ctype,kind,bits,expected",
    [
        (DataType.UINT32, "uint32", "u32", "int32_t", "integer", 0xFFFFFFFF, 0xFFFFFFFF),
        (DataType.INT64, "int64", "i64", "int64_t", "integer", 0xFEDCBA9876543210, 0xFEDCBA9876543210),
        (DataType.BOOL, "bool", "bool", "bool", "bool", 0, 0),
        (DataType.BOOL, "bool", "bool", "bool", "bool", 1, 1),
        (DataType.BOOL, "bool", "bool", "bool", "bool", 2, 1),
    ],
)
def test_call_adapter_integer_carriers_and_bool_normalization(
    compiler, tmp_path, dtype, dsl_type, element, ctype, kind, bits, expected
):
    setattr(compiler.cb.dtypes, dsl_type, object())
    compiler.manifest["contract"]["parameters"][0]["argument"]["element"] = element
    compiler.manifest["parameters"][2].update(
        cpp_type=ctype,
        scalar_kind=kind,
        signed=kind == "integer",
        size=dtype.get_byte(),
        alignment=dtype.get_byte(),
        bits=1 if kind == "bool" else dtype.get_byte() * 8,
    )
    request = _request()
    request = replace(
        request, parameters=(ParameterSpec("alpha", ScalarSpec(dtype)), *request.parameters[1:])
    )
    # Unsigned logical u32 travels through the export's signed C++ carrier;
    # the generated adapter must copy bits, not convert numerically through float.
    conversion = "uint64_t(uint32_t(alpha))" if dtype == DataType.UINT32 else "uint64_t(alpha)"
    source = f"""
struct NativeTensor {{ __gm__ float* ptr; int64_t shape[1]; int64_t stride[1]; }};
uint64_t received = 0;
inline void kernel_scaled(NativeTensor, NativeTensor, {ctype} alpha) {{ received = {conversion}; }}
"""
    artifact = build_call_adapter(_prepare_call(compiler, source, request))
    main = f"""int main() {{
    simpler::tmr::Tensor x{{}}, out{{}};
    x.dtype = out.dtype = DataType::FLOAT32;
    x.ndims = out.ndims = 2; x.shapes[1] = out.shapes[1] = 32;
    x.strides[0] = out.strides[0] = 32; x.strides[1] = out.strides[1] = 1;
    int64_t args[] = {{int64_t(&x), int64_t(&out), int64_t({bits}ULL)}};
    {artifact.symbol}(args);
    return received == {expected}ULL ? 0 : 1;
}}"""
    result, binary = _compile_call(artifact, tmp_path, main)
    assert result.returncode == 0, result.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert executed.returncode == 0, executed.stderr


@pytest.mark.parametrize("dynamic", [False, True])
def test_real_export_call_adapter_compiles_with_ccec(tmp_path, dynamic):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def scaled(alpha, x, out):
        for row in cb.range(x.shape[0]):
            for col in cb.range(32):
                out[row, col] = x[row, col] * alpha

    request = _request()
    if not dynamic:
        tensor = TensorSpec((2, 32), DataType.FP32, strides=(32, 1))
        request = replace(
            request,
            parameters=(request.parameters[0], ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
        )
    artifact = build_call_adapter(adapter.CANNBotDSLProvider().prepare(scaled, request))
    # Only the test supplies a global entry. Production output intentionally
    # stops at a device-callable function until runtime guarantees are implemented.
    main = f'extern "C" __aicore__ void test_entry(__gm__ int64_t* args) {{ {artifact.symbol}(args); }}'
    result, binary = _compile_call(artifact, tmp_path, main, device=True)
    assert result.returncode == 0, result.stderr
    assert binary.stat().st_size > 0


def test_call_adapter_populates_multiple_dynamic_slots(compiler, tmp_path):
    tensor = TensorSpec(("rows", "cols", 32), DataType.FP32, strides=("plane", "pitch", 1))
    request = replace(
        _request(),
        parameters=(_request().parameters[0], ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
    )
    contract = compiler.manifest["contract"]
    contract["dimensions"] = [
        {"name": name, "min": minimum, "max": None, "multiple_of": 1}
        for name, minimum in (("rows", 0), ("cols", 0), ("plane", 1), ("pitch", 1))
    ]
    for parameter in contract["parameters"][1:]:
        parameter["argument"]["shape"] = [
            {"kind": "dimension", "name": "rows"},
            {"kind": "dimension", "name": "cols"},
            {"kind": "constant", "value": 32},
        ]
        parameter["argument"]["stride"] = [
            {"kind": "dimension", "name": "plane"},
            {"kind": "dimension", "name": "pitch"},
            {"kind": "constant", "value": 1},
        ]
    for parameter in compiler.manifest["parameters"][:2]:
        parameter.update(
            shape=[-1, -1, 32],
            stride=[-1, -1, 1],
            size=40,
            dynamic_fields=[
                {"field": "shape", "dimension": 0, "slot": 0, "offset": 8},
                {"field": "shape", "dimension": 1, "slot": 1, "offset": 16},
                {"field": "stride", "dimension": 0, "slot": 0, "offset": 24},
                {"field": "stride", "dimension": 1, "slot": 1, "offset": 32},
            ],
        )
    source = """
struct NativeTensor { __gm__ float* ptr; int64_t shape[2]; int64_t stride[2]; };
inline void kernel_scaled(NativeTensor x, NativeTensor out, float) {
    out.ptr[0] = float(x.shape[0] + 10*x.shape[1] + 100*x.stride[0] + 10000*x.stride[1]);
    out.ptr[1] = x.ptr[0];
}
"""
    artifact = build_call_adapter(_prepare_call(compiler, source, request))
    main = f"""int main() {{
    float input[300]{{}}, output[300]{{}}; input[3] = 17;
    simpler::tmr::Tensor x{{}}, out{{}};
    x.dtype = out.dtype = DataType::FLOAT32;
    x.buffer = {{uint64_t(input), sizeof(input)}}; x.start_offset = 3;
    x.ndims = 3; x.shapes[0] = 2; x.shapes[1] = 3; x.shapes[2] = 32;
    x.strides[0] = 128; x.strides[1] = 40; x.strides[2] = 1;
    out = x; out.buffer = {{uint64_t(output), sizeof(output)}}; out.start_offset = 7;
    int64_t args[] = {{int64_t(&x), int64_t(&out), 0}};
    {artifact.symbol}(args);
    return output[7] == 412832 && output[8] == 17 && output[0] == 0 ? 0 : 1;
}}"""
    result, binary = _compile_call(artifact, tmp_path, main)
    assert result.returncode == 0, result.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert executed.returncode == 0, executed.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
