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
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pypto.backend.cannbotdsl_abi import build_binding_checker, build_call_adapter, build_conflict_checker
from pypto.backend.cannbotdsl_submit import build_ordered_submitter
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

if TYPE_CHECKING:
    from pypto.jit.specializer import SpecializeContext
    from pypto.runtime.kernel_compiler import KernelCompiler


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


def _compile_call(artifact, tmp_path, main, *, device=False, orchestration=False, extra_includes=()):
    compiler = shutil.which("ccec") if device else shutil.which("g++-15") or shutil.which("g++")
    if compiler is None:
        pytest.skip("CANN device compiler unavailable" if device else "C++ compiler unavailable")
    for file in artifact.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    root = Path(__file__).resolve().parents[3]
    includes = [root / "runtime/src/common", root / "runtime/src/common/task_interface"]
    if orchestration:
        # The runtime's declared header order selects TMR's TensorRef/Arg
        # implementation instead of a similarly named header from another ABI.
        src = root / "runtime/src"
        runtime = src / "a5/runtime/tensormap_and_ringbuffer"
        config = runpy.run_path(str(runtime / "build_config.py"))["BUILD_CONFIG"]["orchestration"]
        includes = [runtime / p for p in config["include_dirs"]] + includes
        includes += [
            src / "a5/platform/include",
            src / "common/platform/include",
            src / "common/log/include",
            src,
        ]
    includes.extend(extra_includes)
    includes.extend(tmp_path / path for path in artifact.include_dirs)
    source = tmp_path / "caller.cpp"
    source.write_text(
        ("" if device else "#define __aicore__\n#define __gm__\n") + f'#include "{artifact.entry}"\n' + main
    )
    flags = ["-std=c++17", "-O2", "-DNDEBUG"]
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
    checker = build_binding_checker(prepared)
    artifact = replace(artifact, files=(*checker.files, *artifact.files))
    main = f"""
#include <cstring>
#include "{checker.entry}"
[[noreturn]] void assert_impl(const char* condition, const char*, int) {{ throw condition; }}
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
    CoreTaskArgs bindings; bindings.add_input(x); bindings.add_output(out); bindings.add_scalar({bits}ULL);
    int64_t args[] = {{int64_t(&x), int64_t(&out), {bits}LL}};
    {checker.symbol}(bindings);
    {artifact.symbol}(args);
    uint32_t actual; std::memcpy(&actual, &output[7], 4);
    if (actual != {bits}U) return 1;
    if (output[8] != input[4] || output[47] != input[43] || output[78] != input[74]) return 2;
    if (output[0] != 0 || output[6] != 0 || output[39] != 0 || output[79] != 0) return 3;
    // Reuse the same generated source with a different invocation's pointers
    // and offsets. No call-specific address is captured in the artifact.
    x.start_offset = 4; out.start_offset = 8;
    {checker.symbol}(bindings);
    {artifact.symbol}(args);
    if (output[9] != input[5] || output[48] != input[44]) return 4;
}}
"""
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True)
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
@pytest.mark.parametrize("kind", ["call", "binding", "conflict_first", "conflict_second"])
def test_adapter_revalidates_owned_artifacts_without_the_compiler(compiler, monkeypatch, change, kind):
    prepared = _prepare_call(compiler)
    original = prepared
    builders = {
        "call": build_call_adapter,
        "binding": build_binding_checker,
        "conflict_first": lambda p: build_conflict_checker(p, original),
        "conflict_second": lambda p: build_conflict_checker(original, p),
    }
    builder = builders[kind]
    shutil.rmtree(compiler.include)
    monkeypatch.setitem(sys.modules, "cannbotdsl", None)
    assert builder(prepared).symbol
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
        builder(prepared)


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
    # The standalone adapter remains a callable header. Its caller supplies
    # an entry; the production AIV bridge adds its resource/completion policy.
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


def _prepare_binding(compiler, tensor, *, element="f32", cpp_type="float"):
    """Declare a fake export; runtime bindings are independently supplied in C++."""
    contract = compiler.manifest["contract"]
    dimensions = {d for d in (*tensor.shape, *tensor.strides) if isinstance(d, str)}
    contract["dimensions"] = [
        {"name": d, "min": int(d in tensor.strides), "max": None, "multiple_of": 1}
        for d in sorted(dimensions)
    ]
    fields = []
    members = [f"__gm__ {cpp_type}* ptr;"]
    for field, values in (("shape", tensor.shape), ("stride", tensor.strides)):
        dynamic = [axis for axis, value in enumerate(values) if isinstance(value, str)]
        for slot, axis in enumerate(dynamic):
            fields.append({"field": field, "dimension": axis, "slot": slot, "offset": 8 * (len(fields) + 1)})
        if dynamic:
            members.append(f"int64_t {field}[{len(dynamic)}];")
        expressions = [
            {"kind": "dimension", "name": v} if isinstance(v, str) else {"kind": "constant", "value": v}
            for v in values
        ]
        for parameter in contract["parameters"][1:]:
            parameter["argument"].update(element=element, **{field: expressions})
        for parameter in compiler.manifest["parameters"][:2]:
            parameter[field] = [-1 if isinstance(v, str) else v for v in values]
    for parameter in compiler.manifest["parameters"][:2]:
        parameter.update(
            dtype=element, element_cpp_type=cpp_type, dynamic_fields=fields, size=8 * (len(fields) + 1)
        )
    request = _request()
    request = replace(
        request, parameters=(request.parameters[0], ParameterSpec("x", tensor), ParameterSpec("out", tensor))
    )
    source = (
        f"struct NativeTensor {{ {' '.join(members)} }};\n"
        "inline void kernel_scaled(NativeTensor, NativeTensor, float) {}\n"
    )
    return _prepare_call(compiler, source, request)


def _run_binding_checker(artifact, tmp_path, body):
    main = (
        """
#include <cstring>
[[noreturn]] void assert_impl(const char* condition, const char*, int) { throw condition; }
int main() {
    simpler::tmr::Tensor x{}, out{};
    x.buffer = {0x1000, 4096}; x.ndims = 2; x.dtype = DataType::FLOAT32;
    x.shapes[0] = 2; x.shapes[1] = 32; x.strides[0] = 40; x.strides[1] = 1;
    x.start_offset = 3;
    out = x; out.buffer.addr = 0x4000; out.start_offset = 7;
    CoreTaskArgs args; args.add_input(x); args.add_output(out); args.add_scalar(2.0f);
"""
        + f"""
    auto check = [&]() {{ {artifact.symbol}(args); }};
    auto rejects = [&](const char* message) {{
        try {{ check(); }} catch (const char* why) {{ return std::strstr(why, message) != nullptr; }}
        return false;
    }};
{body}
}}
"""
    )
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"checker exited {result.returncode}: {result.stderr}"


def test_binding_checker_rechecks_actual_arguments(compiler, tmp_path):
    artifact = build_binding_checker(_prepare_call(compiler))
    # One generated checker observes mutations between invocations. Assertions
    # must stay enabled under NDEBUG; no storage is dereferenced by the checker.
    _run_binding_checker(
        artifact,
        tmp_path,
        f"""
    check();
    out.ndims = 1; if (!rejects("rank")) return 1; out.ndims = 2;
    out.dtype = DataType::INT32; if (!rejects("dtype")) return 2; out.dtype = DataType::FLOAT32;
    out.shapes[0] = 3; if (!rejects("shared symbol rows")) return 3; out.shapes[0] = 2;
    out.shapes[1] = 31; if (!rejects("dimension 1")) return 4; out.shapes[1] = 32;
    out.strides[0] = 41; if (!rejects("shared symbol pitch")) return 5; out.strides[0] = 40;
    out.strides[1] = 2; if (!rejects("stride 1")) return 6; out.strides[1] = 1;
    out.strides[0] = 0; if (!rejects("positive stride")) return 7; out.strides[0] = 40;
    out.start_offset = 1024; if (!rejects("storage bounds")) return 8; out.start_offset = 7;
    out.start_offset = UINT64_MAX; if (!rejects("overflow")) return 9; out.start_offset = 7;
    out.start_offset = (1ULL << 62); if (!rejects("byte extent overflow")) return 10; out.start_offset = 7;
    out.buffer.addr = UINT64_MAX - 4; if (!rejects("address overflow")) return 11;
    out.buffer.addr = 0; if (!rejects("null storage")) return 12; out.buffer.addr = 0x4000;
    out.buffer.addr++; if (!rejects("alignment")) return 13; out.buffer.addr--;
    x.strides[0] = out.strides[0] = 31;
    if (!rejects("non-overlapping elements")) return 14;
    x.strides[0] = out.strides[0] = 40;
    out.buffer.addr = x.buffer.addr; if (!rejects("overlap needs alias permission")) return 15;
    out.start_offset = 100; check();  // Disjoint sibling views of one allocation.
    out.buffer.addr = 0x4000; out.start_offset = 7;
    args.tag(1) = TensorArgType::OUTPUT;
    if (!rejects("runtime-allocated outputs")) return 16;
    args.tag(1) = TensorArgType::OUTPUT_EXISTING;
    // Empty views still validate offsets, positive strides and pointer arithmetic.
    x.shapes[0] = out.shapes[0] = 0;
    x.buffer.addr = out.buffer.addr = 0; check();
    out.start_offset = 1025; if (!rejects("storage bounds")) return 17; out.start_offset = 7;
    out.strides[0] = 0; if (!rejects("positive stride")) return 18; out.strides[0] = 40;
    CoreTaskArgs missing;
    try {{ {artifact.symbol}(missing); return 19; }} catch (const char* why) {{
        if (!std::strstr(why, "argument counts")) return 20;
    }}
    args.add_scalar(1.0f); if (!rejects("argument counts")) return 21;
""",
    )


@pytest.mark.parametrize(
    "address,offset,accepted", [(0x1004, 3, True), (0x1000, 1, False), (0x1000, 4, True), (0x1001, 0, False)]
)
def test_binding_alignment_uses_first_element_pointer(compiler, tmp_path, address, offset, accepted):
    tensor = TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1), alignment=16)
    artifact = build_binding_checker(_prepare_binding(compiler, tensor))
    _run_binding_checker(
        artifact,
        tmp_path,
        f"""
    x.buffer.addr = {address}; x.start_offset = {offset}; out.start_offset = 4;
    return rejects("alignment") == {str(accepted).lower()} ? 1 : 0;
""",
    )


@pytest.mark.parametrize("access", ["read", "read_write", "write"])
def test_binding_alias_rules_follow_effects(compiler, tmp_path, access):
    compiler.manifest["parameters"][1]["access"] = access
    artifact = build_binding_checker(_prepare_call(compiler))
    _run_binding_checker(
        artifact,
        tmp_path,
        f"""
    out.buffer = x.buffer; out.start_offset = x.start_offset;
    return rejects("overlap needs alias permission") == {str(access != "read").lower()} ? 0 : 1;
""",
    )


def test_binding_nonoverlap_proof_handles_permuted_and_empty_views(compiler, tmp_path):
    tensor = TensorSpec(("rows", "cols", 32), DataType.FP32, strides=("plane", "pitch", 1))
    artifact = build_binding_checker(_prepare_binding(compiler, tensor))
    _run_binding_checker(
        artifact,
        tmp_path,
        """
    struct Case { uint32_t rows, cols, plane, pitch; bool accepted; } cases[] = {
        {2, 3, 128, 40, true},   // Row-major with padding.
        {2, 3, 32, 64, true},    // Permuted outer axes, innermost stride still one.
        {1, 3, 1, 40, true},    // Singleton axis contributes no addresses.
        {2, 3, 64, 32, false},   // Two coordinates refer to the same element.
        {3, 2, 64, 96, false},   // Disjoint interleaving is conservatively rejected.
        {0, 3, 1, 1, true},     // Empty views have no overlapping elements.
    };
    for (auto c : cases) {
        x.ndims = out.ndims = 3;
        x.shapes[0] = out.shapes[0] = c.rows; x.shapes[1] = out.shapes[1] = c.cols;
        x.shapes[2] = out.shapes[2] = 32;
        x.strides[0] = out.strides[0] = c.plane; x.strides[1] = out.strides[1] = c.pitch;
        x.strides[2] = out.strides[2] = 1;
        if (rejects("non-overlapping elements") == c.accepted) return 1;
    }
    // Every field fits TMR's uint32, but the total reachable span can overflow.
    x.shapes[0] = x.shapes[1] = UINT32_MAX;
    x.strides[0] = x.strides[1] = UINT32_MAX;
    x.buffer.size = UINT64_MAX;
    if (!rejects("extent overflow")) return 2;
""",
    )


@pytest.mark.parametrize("offset", [3, "pitch"])
def test_binding_checker_constrains_offset(compiler, tmp_path, offset):
    tensor = TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1), offset=offset)
    artifact = build_binding_checker(_prepare_binding(compiler, tensor))
    _run_binding_checker(
        artifact,
        tmp_path,
        f"""
    x.start_offset = out.start_offset = {3 if offset == 3 else 40}; check();
    out.start_offset++; return rejects("offset") ? 0 : 1;
""",
    )


@pytest.mark.parametrize(
    "dtype,dsl_type,element,ctype,tag",
    [
        (DataType.INDEX, "int64", "i64", "int64_t", "INT64"),
        (DataType.FP8E4M3FN, "float8_e4m3fn", "f8e4m3", "fp8_e4m3fn_t", "FP8E4M3FN"),
        (DataType.FP8E8M0, "float8_e8m0", "f8e8m0", "fp8_e8m0_t", "FP8E8M0"),
        (DataType.FP8E5M2, "float8_e5m2", "f8e5m2", "fp8_e5m2_t", None),
    ],
)
def test_binding_checker_runtime_dtype_support(compiler, tmp_path, dtype, dsl_type, element, ctype, tag):
    setattr(compiler.cb.dtypes, dsl_type, object())
    tensor = TensorSpec(("rows", 32), dtype, strides=("pitch", 1))
    prepared = _prepare_binding(compiler, tensor, element=element, cpp_type=ctype)
    if tag is None:
        with pytest.raises(ContractError, match="unsupported dtype"):
            build_binding_checker(prepared)
        return
    artifact = build_binding_checker(prepared)
    _run_binding_checker(
        artifact,
        tmp_path,
        f"""
    x.dtype = out.dtype = DataType::{tag}; check();
    out.dtype = DataType::FLOAT32; return rejects("dtype") ? 0 : 1;
""",
    )


@pytest.mark.parametrize("field,value", [("shape", 1 << 32), ("stride", 1 << 32), ("offset", 1 << 64)])
def test_binding_checker_rejects_unrepresentable_constraints(compiler, field, value):
    tensor = TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1))
    if field == "shape":
        tensor = replace(tensor, shape=(value, 32))
    elif field == "stride":
        tensor = replace(tensor, strides=(value, 1))
    else:
        tensor = replace(tensor, offset=value)
    prepared = _prepare_binding(compiler, tensor)
    with pytest.raises(ContractError, match="exceeds uint"):
        build_binding_checker(prepared)


def _run_conflict_checker(artifact, tmp_path, body):
    main = f"""
#include <cstring>
[[noreturn]] void assert_impl(const char* condition, const char*, int) {{ throw condition; }}
int main() {{
    simpler::tmr::Tensor x{{}}, out{{}};
    x.buffer = {{0x1000, 4096}}; x.ndims = 2; x.dtype = DataType::FLOAT32;
    x.shapes[0] = x.shapes[1] = x.strides[0] = x.strides[1] = 1;
    out = x; out.buffer.addr = 0x2000;
    auto x2 = x, out2 = out; x2.buffer.addr = 0x3000; out2.buffer.addr = 0x2004;
    CoreTaskArgs first, second;
    first.add_input(x); first.add_output(out); first.add_scalar(1.0f);
    second.add_input(x2); second.add_output(out2); second.add_scalar(1.0f);
    auto conflicts = [&]() {{ return {artifact.symbol}(first, second); }};
{body}
}}
"""
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"conflict checker exited {result.returncode}: {result.stderr}"


@pytest.mark.parametrize(
    "first_access,second_access,conflict",
    [
        ("write", "write", True),
        ("read", "write", True),
        ("write", "read", True),
        ("read_write", "read", True),
        ("read", "read", False),
        ("none", "write", False),
        ("read_write", "none", False),
    ],
)
def test_cache_line_conflicts_follow_both_access_modes(
    compiler, tmp_path, first_access, second_access, conflict
):
    tensor = TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1))
    compiler.manifest["parameters"][1]["access"] = first_access
    first = _prepare_binding(compiler, tensor)
    compiler.manifest["parameters"][1]["access"] = second_access
    second = _prepare_binding(compiler, tensor)
    # Each invocation touches disjoint bytes, but both touch line 0x2000.
    # NONE is an unused parameter, not a read or write dependency.
    _run_conflict_checker(
        build_conflict_checker(first, second),
        tmp_path,
        f"return conflicts() == {str(conflict).lower()} ? 0 : 1;",
    )


def test_cache_line_conflicts_recheck_views_and_validate_before_return(compiler, tmp_path):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_conflict_checker(prepared, prepared)
    _run_conflict_checker(
        artifact,
        tmp_path,
        f"""
    if (!conflicts()) return 1;  // Adjacent float elements share a cache line.
    out2.buffer.addr = 0x2040;
    if (conflicts()) return 2;   // Exactly the next line is independent.
    out2.buffer.addr = 0x2000; out2.start_offset = 16;
    if (conflicts()) return 3;   // Apply the view offset before line rounding.
    out2.start_offset = 15;
    if (!conflicts()) return 4;
    out2.buffer.addr = 0x2004; out2.start_offset = 15;
    if (conflicts()) return 5;   // A misaligned allocation can have an aligned view.
    out2.buffer.addr = 0x2000; out2.start_offset = 0;
    x2.shapes[0] = out2.shapes[0] = 0;
    if (conflicts()) return 6;   // Empty writes have no cache-line footprint.
    x2.shapes[0] = out2.shapes[0] = 1;
    out2.buffer.addr = 0x4000; x2.buffer.addr = 0x2004;
    if (!conflicts()) return 7;  // First output conflicts with second input.
    x2.buffer.addr = 0x1000;
    if (conflicts()) return 8;   // Shared read-only input needs no serialization.
    out2.buffer.addr = 0x1004;
    if (!conflicts()) return 9;  // First input conflicts with second output.
    out2.buffer.addr = 0x2004; x2.buffer.addr = 0x3000;
    out2.dtype = DataType::INT32;
    try {{ conflicts(); return 10; }} catch (const char* why) {{
        if (!std::strstr(why, "dtype")) return 11;
    }}
    out2.dtype = DataType::FLOAT32; out.buffer.size = 0;
    try {{ conflicts(); return 12; }} catch (const char* why) {{
        if (!std::strstr(why, "storage bounds")) return 13;
    }}
    out.buffer.size = 4096;
    if (!{artifact.symbol}(first, first)) return 14; // Reusing the same bindings writes the same line.
    // The byte envelope includes holes; uncertainty requires serialization.
    x.shapes[0] = out.shapes[0] = 2; x.strides[0] = out.strides[0] = 32;
    out2.buffer.addr = 0x2040;
    if (!conflicts()) return 15;
    x.shapes[0] = out.shapes[0] = 1; x.strides[0] = out.strides[0] = 1;
    // Inclusive last-line comparison never rounds the top address up to zero.
    out.buffer.addr = UINT64_MAX - 7; out.buffer.size = 4;
    out2.buffer.addr = UINT64_MAX - 11; out2.buffer.size = 4;
    if (!conflicts()) return 16;
    out2.buffer.addr = UINT64_MAX - 67;
    if (conflicts()) return 17;
    return 0;
""",
    )


def test_cache_line_conflicts_support_different_dtypes_and_header_composition(compiler, tmp_path):
    first = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    compiler.cb.dtypes.uint8 = object()
    second = _prepare_binding(
        compiler,
        TensorSpec(("rows", 1), DataType.UINT8, strides=("pitch", 1)),
        element="u8",
        cpp_type="uint8_t",
    )
    artifact = build_conflict_checker(first, second)
    reverse = build_conflict_checker(second, first)
    standalone = build_binding_checker(first)
    # Standalone, forward and reverse checkers share binding definitions. This
    # exercises deduplication across different physical include paths as well.
    files = {f.path: f for f in (*artifact.files, *reverse.files, *standalone.files)}
    forward_file = next(f for f in artifact.files if f.path == artifact.entry)
    content = forward_file.content + b'\n#include "conflict_check.hpp"\n#include "binding_check.hpp"\n'
    files["forward.hpp"] = replace(
        forward_file, path="forward.hpp", content=content, sha256=hashlib.sha256(content).hexdigest()
    )
    artifact = replace(artifact, files=tuple(files.values()), entry="forward.hpp")
    _run_conflict_checker(
        artifact,
        tmp_path,
        f"""
    x2.dtype = out2.dtype = DataType::UINT8;
    out2.buffer.addr = 0x203f;  // Last byte of the float output's cache line.
    if (!conflicts() || !{reverse.symbol}(second, first)) return 1;
    out2.buffer.addr = 0x2040;
    if (conflicts() || {reverse.symbol}(second, first)) return 2;
    return 0;
""",
    )


@pytest.mark.parametrize(
    "isolation",
    [
        None,
        {},
        {"granularity_bytes": 64.0, "rule": "serialize_conflicting_cache_lines"},
        {"granularity_bytes": 32, "rule": "serialize_conflicting_cache_lines"},
        {"granularity_bytes": 64, "rule": "assume_disjoint"},
    ],
)
@pytest.mark.parametrize("side", ["first", "second"])
def test_conflict_checker_rejects_unknown_cache_line_policy(compiler, isolation, side):
    first = _prepare_call(compiler)
    compiler.manifest["resources"]["gm_write_isolation"] = isolation
    second = _prepare_call(compiler)
    pair = (second, first) if side == "first" else (first, second)
    with pytest.raises(ContractError, match="64-byte GM cache-line isolation"):
        build_conflict_checker(*pair)


def _run_ordered_submitter(artifact, tmp_path, body, *, prefix="", direct=False):
    invoke = f"{artifact.symbol}(bindings" + (")" if direct else ", submit)")
    main = f"""
{prefix}
#include <cstring>
[[noreturn]] void assert_impl(const char* condition, const char*, int) {{ throw condition; }}
int main() {{
    simpler::tmr::Tensor x[3]{{}}, out[3]{{}};
    CoreTaskArgs args[3];
    std::array<CoreTaskArgs*, 3> bindings{{&args[0], &args[1], &args[2]}};
    for (unsigned i = 0; i < 3; ++i) {{
        x[i].buffer = {{0x1000, 4096}}; x[i].ndims = 2; x[i].dtype = DataType::FLOAT32;
        x[i].shapes[0] = x[i].shapes[1] = x[i].strides[0] = x[i].strides[1] = 1;
        out[i] = x[i]; out[i].buffer.addr = 0x2000 + i * 4096;
        args[i].add_input(x[i]); args[i].add_output(out[i]); args[i].add_scalar(1.0f);
        args[i].set_allow_early_resolve(true);
    }}
    out[2].buffer.addr = 0x2004; // Nonadjacent writes share a line, not bytes.
    unsigned calls = 0;
    std::vector<TaskId> observed[3];
    std::vector<DepFlags> flags[3];
    auto submit = [&](size_t i, CoreTaskArgs& arg) {{
        if (i != calls++ || &arg != &args[i] || arg.allow_early_resolve()) throw "submit contract";
        for (uint32_t d = 0; d < arg.explicit_dep_count(); ++d) {{
            observed[i].push_back(arg.explicit_dep(d));
            flags[i].push_back(arg.explicit_dep_kind(d));
        }}
        return TaskId::make(0, 100 + int(i));
    }};
    auto run = [&]() {{ return {invoke}; }};
{body}
}}
"""
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"ordered submitter exited {result.returncode}: {result.stderr}"


@pytest.mark.parametrize("scenario", ["nonadjacent", "chain", "all", "read_only", "empty", "offset"])
def test_ordered_submitter_builds_complete_dependency_graph(compiler, tmp_path, scenario):
    if scenario == "read_only":
        compiler.manifest["parameters"][1]["access"] = "read"
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,) * 3)
    setup, edges = {
        "nonadjacent": ("", [(0, 2)]),
        "chain": (
            "x[1].buffer.addr = 0x2004; x[2].buffer.addr = 0x3004; out[2].buffer.addr = 0x4000;",
            [(0, 1), (1, 2)],
        ),
        "all": ("out[1].buffer.addr = 0x2008;", [(0, 1), (0, 2), (1, 2)]),
        "read_only": ("", []),
        "empty": ("x[0].shapes[0] = out[0].shapes[0] = 0;", []),
        "offset": ("out[2].start_offset = 15;", []),
    }[scenario]
    expected = []
    for i in range(3):
        preds = [j for j, target in edges if target == i]
        expected.append(f"if (observed[{i}].size() != {len(preds)}) return 10;")
        for d, j in enumerate(preds):
            expected.append(f"if (observed[{i}][{d}] != ids[{j}] || flags[{i}][{d}] != DEP_WAIT) return 11;")
    _run_ordered_submitter(
        artifact,
        tmp_path,
        setup
        + "\nauto ids = run();\n"
        + "\n".join(expected)
        + """
    if (calls != 3) return 12;
    for (auto& arg : args)
        if (arg.explicit_dep_count() != 0 || !arg.allow_early_resolve()) return 13;
    // No address/dependency cache: rebinding a later batch removes its edge.
    out[0].buffer.addr = 0x5000; out[1].buffer.addr = 0x6000; out[2].buffer.addr = 0x7000;
    calls = 0;
    for (unsigned i = 0; i < 3; ++i) { observed[i].clear(); flags[i].clear(); }
    run();
    for (auto& deps : observed) if (!deps.empty()) return 14;
    return 0;
""",
    )


@pytest.mark.parametrize("explicit_kinds", [False, True])
def test_ordered_submitter_preserves_and_merges_existing_dependencies(compiler, tmp_path, explicit_kinds):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,) * 3)
    binding = (
        "args[2].set_dependencies_with_kinds(original, original_kinds, 2);"
        if explicit_kinds
        else "args[2].set_dependencies(original, 2);"
    )
    # A retain-only edge to the first task must gain WAIT rather than losing
    # RETAIN or consuming a second slot. Plain dependencies default to both.
    _run_ordered_submitter(
        artifact,
        tmp_path,
        f"""
    TaskId original[] = {{TaskId::make(0, 9), TaskId::make(0, 100)}};
    DepFlags original_kinds[] = {{DEP_WAIT, DEP_RETAIN}};
    {binding}
    auto ids = run();
    if (observed[2].size() != 2 || observed[2][0] != original[0] || observed[2][1] != ids[0]) return 1;
    if (flags[2][0] != {"DEP_WAIT" if explicit_kinds else "(DEP_WAIT | DEP_RETAIN)"}) return 2;
    if (flags[2][1] != (DEP_WAIT | DEP_RETAIN)) return 3;
    if (args[2].explicit_deps_data() != original || args[2].explicit_dep_count() != 2) return 4;
    if (args[2].explicit_dep_kinds_data() != {"original_kinds" if explicit_kinds else "nullptr"}) return 5;
    if (!args[2].allow_early_resolve()) return 6;
    return 0;
""",
    )


@pytest.mark.parametrize(
    "invalid,diagnostic",
    [
        ("out[2].dtype = DataType::INT32;", "dtype"),
        ("bindings[2] = nullptr;", "null arguments"),
        ('args[2].set_error("invalid test binding");', "invalid arguments"),
        ("args[2].set_dependencies(bad, 1);", "invalid dependency"),
        ("args[2].set_dependencies_with_kinds(good, unknown, 1);", "unknown dependency flags"),
    ],
)
def test_ordered_submitter_validates_entire_batch_before_submit(compiler, tmp_path, invalid, diagnostic):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,) * 3)
    _run_ordered_submitter(
        artifact,
        tmp_path,
        f"""
    TaskId bad[] = {{TaskId::invalid()}}, good[] = {{TaskId::make(0, 9)}};
    DepFlags unknown[] = {{static_cast<DepFlags>(4)}};
    {invalid}
    try {{ run(); return 1; }} catch (const char* why) {{
        if (!std::strstr(why, "{diagnostic}")) return 2;
    }}
    if (calls != 0 || args[0].explicit_dep_count() != 0 || !args[0].allow_early_resolve()) return 3;
    return 0;
""",
    )


@pytest.mark.parametrize("failure", ["invalid_id", "throw", "reused_id"])
def test_ordered_submitter_stops_and_restores_after_submit_failure(compiler, tmp_path, failure):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,) * 3)
    failed = {
        "invalid_id": "return TaskId::invalid();",
        "throw": 'throw "callback failed";',
        "reused_id": "return TaskId::make(0, 100);",
    }[failure]
    _run_ordered_submitter(
        artifact,
        tmp_path,
        f"""
    TaskId original[] = {{TaskId::make(0, 9)}};
    args[1].set_dependencies(original, 1);
    auto fail = [&](size_t i, CoreTaskArgs& arg) -> TaskId {{
        auto id = submit(i, arg);
        if (i == 1) {{ {failed} }}
        return id;
    }};
    try {{ {artifact.symbol}(bindings, fail); return 1; }} catch (const char*) {{}}
    if (calls != 2 || args[1].explicit_deps_data() != original || args[1].explicit_dep_count() != 1) return 2;
    if (!args[1].allow_early_resolve() || args[2].explicit_dep_count() != 0) return 3;
    return 0;
""",
    )


def test_real_export_ordered_submitter_accepts_group_isolation(tmp_path):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def scaled(alpha, x, out):
        for row in cb.range(x.shape[0]):
            out[row, 0] = x[row, 0] * alpha

    tensor = TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1))
    request = replace(
        _request(),
        parameters=(_request().parameters[0], ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
    )
    prepared = adapter.CANNBotDSLProvider().prepare(scaled, request)
    artifact = build_ordered_submitter((prepared,) * 3)
    _run_ordered_submitter(
        artifact,
        tmp_path,
        """
    auto ids = run();
    return calls == 3 && observed[0].empty() && observed[1].empty() &&
        observed[2].size() == 1 && observed[2][0] == ids[0] && flags[2][0] == DEP_WAIT ? 0 : 1;
""",
    )


@pytest.mark.parametrize("policy", ["consumer_must_ensure", None, True])
def test_conflict_checker_rejects_unknown_within_group_policy(compiler, policy):
    compiler.manifest["resources"]["gm_write_isolation"]["within_group"] = policy
    prepared = _prepare_call(compiler)
    with pytest.raises(ContractError, match="64-byte GM cache-line isolation"):
        build_ordered_submitter((prepared,))


@pytest.mark.parametrize("prepared", [(), [], None])
def test_ordered_submitter_requires_nonempty_kernel_tuple(prepared):
    with pytest.raises(ContractError, match="nonempty tuple"):
        build_ordered_submitter(prepared)


def test_ordered_submitter_composes_owned_headers_for_different_kernels(compiler, tmp_path, monkeypatch):
    tensor = TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1))
    writer = _prepare_binding(compiler, tensor)
    compiler.manifest["parameters"][1]["access"] = "read"
    reader = _prepare_binding(compiler, tensor)
    # Generation consumes the owned manifest, not the exporter's current state
    # or installation. A batch can combine distinct logical access contracts.
    shutil.rmtree(compiler.include)
    monkeypatch.setitem(sys.modules, "cannbotdsl", None)
    artifact = build_ordered_submitter((writer, reader, writer))
    standalone = build_conflict_checker(writer, reader)
    files = {f.path: f for f in (*artifact.files, *standalone.files)}
    entry = files[artifact.entry]
    content = entry.content + f'\n#include "{standalone.entry}"\n'.encode()
    files[entry.path] = replace(entry, content=content, sha256=hashlib.sha256(content).hexdigest())
    artifact = replace(artifact, files=tuple(files.values()))
    _run_ordered_submitter(
        artifact,
        tmp_path,
        """
    out[1].buffer.addr = 0x2008;
    auto ids = run();
    if (!observed[0].empty() || observed[1].size() != 1 || observed[2].size() != 2) return 1;
    if (observed[1][0] != ids[0] || observed[2][0] != ids[0] || observed[2][1] != ids[1]) return 2;
    return 0;
""",
    )


def test_ordered_submitter_singleton_still_validates_bindings(compiler, tmp_path):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,))
    main = f"""
[[noreturn]] void assert_impl(const char* condition, const char*, int) {{ throw condition; }}
int main() {{
    simpler::tmr::Tensor x{{}}, out{{}};
    x.buffer = {{0x1000, 4096}}; x.ndims = 2; x.dtype = DataType::FLOAT32;
    x.shapes[0] = x.shapes[1] = x.strides[0] = x.strides[1] = 1;
    out = x; out.buffer.addr = 0x2000;
    CoreTaskArgs args;
    args.add_input(x); args.add_output(out); args.add_scalar(1.0f);
    std::array<CoreTaskArgs*, 1> bindings{{&args}};
    unsigned calls = 0;
    auto submit = [&](size_t, CoreTaskArgs& arg) {{
        ++calls;
        if (arg.explicit_dep_count() != 0) throw "unexpected dependency";
        return TaskId::make(0, 7);
    }};
    auto ids = {artifact.symbol}(bindings, submit);
    if (calls != 1 || ids[0] != TaskId::make(0, 7)) return 1;
    out.buffer.size = 0;
    try {{ {artifact.symbol}(bindings, submit); return 2; }} catch (const char*) {{}}
    return calls == 1 ? 0 : 3;
}}
"""
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_ordered_submitter_uses_public_tmr_submit_api(compiler, tmp_path):
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    artifact = build_ordered_submitter((prepared,) * 3)
    # Exercise the real public submit wrapper with an instrumented ops table.
    # The test consumes actual TMR arguments without starting a device runtime.
    _run_ordered_submitter(
        artifact,
        tmp_path,
        f"""
    static unsigned dispatches = 0;
    RuntimeOps ops{{}};
    ops.is_fatal = [](RuntimeContext*) {{ return false; }};
    ops.submit_task = [](RuntimeContext*, const MixedKernels& kernels, const CoreTaskArgs& arg) {{
        unsigned i = dispatches++;
        if (kernels.aiv0_kernel_id != int(10 + i) || arg.allow_early_resolve()) throw "dispatch contract";
        if (arg.explicit_dep_count() != (i == 2 ? 1U : 0U)) throw "dependency count";
        if (i == 2 && (arg.explicit_dep(0) != TaskId::make(0, 100) ||
                      arg.explicit_dep_kind(0) != DEP_WAIT)) throw "dependency kind";
        TaskOutputTensors result;
        result.set_task_id(TaskId::make(0, 100 + int(i)));
        return result;
    }};
    RuntimeContext runtime{{&ops, ScopeMode::AUTO}};
    test_runtime = &runtime;
    auto ids = {artifact.symbol}(bindings, [](size_t i, CoreTaskArgs& arg) {{
        return rt_submit_aiv_task(10 + int(i), arg).task_id();
    }});
    return dispatches == 3 && ids[2] == TaskId::make(0, 102) ? 0 : 1;
""",
        prefix="""
#include "orchestration_api.h"
RuntimeContext* test_runtime = nullptr;
RuntimeContext* framework_current_runtime() { return test_runtime; }
""",
    )


def _aiv_runtime_manifest(compiler, core_type="aiv"):
    compiler.manifest["core_type"] = core_type
    compiler.manifest["participants"] = {
        "aic": int(core_type != "aiv"),
        "aiv": 2 if core_type == "mixed" else int(core_type == "aiv"),
    }
    compiler.manifest["build_requirements"].update(
        dsl_headers_sha256="0" * 64,
        cann_headers="consumer_resolved_sdk",
        device_function_mode=True,
        language="AscendC",
        separate_translation_unit=True,
        source_includes=["c_api/asc_simd.h"],
        standard="c++17",
        targets=["aic", "aiv"] if core_type == "mixed" else [core_type],
    )
    compiler.manifest["resources"].update(
        allocations=[],
        entry_requires_previous_task_complete=True,
        exclusive_local_memory=True,
        exclusive_sync_resources=True,
        implicit_workspace_bytes=0,
        local_memory_bytes=[],
        reentrant=False,
        stack="consumer_must_validate_compiler_requirement",
        subblock_id="physical_aiv_lane",
        topology="hardware_paired_aic_and_two_aiv" if core_type == "mixed" else "single_core",
    )
    compiler.manifest["state"] = {
        "clobbered": ["DMA", "padding", "quantization", "mask", "local_memory", "sync_resources"],
        "entry": "asc_init_and_default_dma_state_atomic_none",
        "exit_atomic": "none",
        "preserved": ["CTRL"],
    }
    compiler.manifest["completion"].update(
        pipe_barrier_is_cross_core=False, wait_for_all_participants=core_type == "mixed"
    )
    compiler.manifest["loading"]["host_runtime"] = False


def _aiv_target(prepared):
    return RuntimeTarget(
        "tmr",
        "1",
        prepared.request.target,
        frozenset({(adapter.ABI_NAME, 1)}),
        frozenset(
            {
                *(f"core:{core.value}" for core in prepared.execution.cores),
                f"aiv_lanes:{prepared.execution.aiv_lanes}",
                "completion:entry_return_visible",
                "tensor_writeback",
                "partial_write",
            }
        ),
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("resources", "exclusive_local_memory", False),
        ("resources", "exclusive_sync_resources", False),
        ("resources", "entry_requires_previous_task_complete", False),
        ("resources", "implicit_workspace_bytes", True),
        ("resources", "implicit_workspace_bytes", 64),
        ("resources", "local_memory_bytes", [64]),
        ("resources", "allocations", [{"space": "UB", "bytes": 64}]),
        ("resources", "reentrant", True),
        ("resources", "topology", "cluster"),
        ("resources", "unknown_requirement", True),
        ("state", "preserved", []),
        ("state", "clobbered", ["CTRL"]),
        ("state", "exit_atomic", "add"),
        ("completion", "pipe_barrier_is_cross_core", True),
        ("completion", "wait_for_all_participants", True),
        ("loading", "host_runtime", True),
        ("build_requirements", "standard", "c++20"),
        ("build_requirements", "targets", ["aic", "aiv"]),
        ("build_requirements", "unknown_link_flag", True),
    ],
)
def test_aiv_bridge_rejects_unsupported_resource_contracts(compiler, section, key, value):
    _aiv_runtime_manifest(compiler)
    compiler.manifest[section][key] = value
    prepared = _prepare_call(compiler)
    with pytest.raises(ContractError, match="runtime"):
        TMRBridge().lower(prepared, _aiv_target(prepared))


def _local_allocation(*, name="UB", space=1, core="aiv", offset=0, size=512):
    return {
        "memory_space_name": name,
        "memory_space": space,
        "core": core,
        "byte_offset": offset,
        "bytes": size,
        "max_bytes": size,
    }


def test_runtime_admits_static_local_memory(compiler):
    _aiv_runtime_manifest(compiler)
    compiler.manifest["resources"].update(
        allocations=[_local_allocation(), _local_allocation(offset=1024)],
        local_memory_bytes=[{"core": "aiv", "memory_space_name": "UB", "bytes": 1536}],
    )
    prepared = _prepare_call(compiler)
    assert TMRBridge().lower(prepared, _aiv_target(prepared)).binary_policy == "a5-text-only"


@pytest.mark.parametrize(
    "change",
    [
        {"byte_offset": -32},
        {"byte_offset": 1},
        {"byte_offset": True},
        {"bytes": 0},
        {"bytes": 512.0},
        {"max_bytes": 1024},
        {"max_bytes": True},
        {"memory_space": True},
        {"memory_space": 2},
        {"core": "aic"},
        {"memory_space_name": "SSBUF"},
        {"bytes": 256 * 1024, "max_bytes": 256 * 1024},
        {"unknown": 0},
    ],
)
def test_runtime_rejects_invalid_local_allocation(compiler, change):
    _aiv_runtime_manifest(compiler)
    compiler.manifest["resources"].update(
        allocations=[{**_local_allocation(), **change}],
        local_memory_bytes=[{"core": "aiv", "memory_space_name": "UB", "bytes": 512}],
    )
    prepared = _prepare_call(compiler)
    with pytest.raises(ContractError, match="runtime"):
        TMRBridge().lower(prepared, _aiv_target(prepared))


@pytest.mark.parametrize("overlap", [False, True])
def test_runtime_rejects_inconsistent_local_ranges(compiler, overlap):
    _aiv_runtime_manifest(compiler)
    compiler.manifest["resources"].update(
        allocations=[_local_allocation(), _local_allocation(offset=256 if overlap else 1024)],
        local_memory_bytes=[{"core": "aiv", "memory_space_name": "UB", "bytes": 1024}],
    )
    prepared = _prepare_call(compiler)
    with pytest.raises(ContractError, match="runtime"):
        TMRBridge().lower(prepared, _aiv_target(prepared))


def test_aiv_bridge_owns_checked_source_and_requires_strict_compilation(compiler):
    from pypto.backend.tmr_bridge import materialize  # noqa: PLC0415

    _aiv_runtime_manifest(compiler)
    prepared = _prepare_call(compiler)
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    assert task.binary_policy == "a5-text-only"
    assert task.source.symbol == "kernel_entry"
    with materialize(task) as source:
        assert Path(source.source).is_file()
        with pytest.raises(ContractError, match="descriptor JIT integration"):
            source.configure_context(cast("SpecializeContext", SimpleNamespace()))
    assert not Path(source.source).exists()
    with pytest.raises(ContractError, match="unmodified bridge output"):
        with materialize(replace(task, binary_policy="ordinary")):
            pass


@pytest.mark.parametrize("core_type", ["aiv", "mixed"])
def test_compile_enforces_policy_and_cleans_source_on_failure(compiler, tmp_path, monkeypatch, core_type):
    from pypto.backend.cannbotdsl_runtime import compile_task  # noqa: PLC0415

    _aiv_runtime_manifest(compiler, core_type)
    prepared = _prepare_call(compiler)
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    header = tmp_path / "asc/include/c_api/asc_simd.h"
    header.parent.mkdir(parents=True)
    header.touch()
    observed = []
    cores = []
    monkeypatch.setattr("pypto.runtime.elf_parser.extract_text_section", lambda image: image)

    def fail_compile(source, **kwargs):
        observed.append(Path(source))
        assert Path(source).is_file()
        assert kwargs["text_only"] is True
        cores.append(kwargs["core_type"])
        assert kwargs["runtime_name"] == "tensormap_and_ringbuffer"
        assert str(header.parents[1]) in kwargs["extra_include_dirs"]
        if kwargs["core_type"] == "aic":
            return b"validated aic image"
        raise RuntimeError("compiler rejected stack usage")

    native = SimpleNamespace(
        platform="a5",
        sdk=SimpleNamespace(ccec=SimpleNamespace(ascend_home_path=str(tmp_path))),
        compile_incore=fail_compile,
    )
    with pytest.raises(ContractError, match="unmodified bridge output"):
        compile_task(
            replace(task, binary_policy="ordinary"),
            cast("KernelCompiler", native),
            pto_isa_root=str(tmp_path),
        )
    assert not observed
    with pytest.raises(RuntimeError, match="stack usage"):
        compile_task(task, cast("KernelCompiler", native), pto_isa_root=str(tmp_path))
    assert cores == (["aic", "aiv"] if core_type == "mixed" else ["aiv"])
    assert all(not path.exists() for path in observed)


@pytest.mark.parametrize("cube", [False, True])
def test_wrapper_drains_and_restores_before_return(compiler, tmp_path, cube):
    _aiv_runtime_manifest(compiler, "aic" if cube else "aiv")
    native = _CALL_SOURCE.replace(
        "    for (int64_t row",
        f"    if (ctrl != {0x1008 if cube else 0x400000001008}ULL || !defaults) throw 1;\n"
        "    events.push_back(9); ctrl = 0xffff; defaults = false;\n    for (int64_t row",
    )
    prepared = _prepare_call(compiler, source=native)
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    header = tmp_path / "c_api/asc_simd.h"
    header.parent.mkdir()
    header.write_text(
        ("#define __DAV_CUBE__ 1\n" if cube else "")
        + """
#pragma once
#include <cstdint>
#include <vector>
static std::vector<int> events;
static uint64_t ctrl = 0x400000001008;
static bool defaults = true;
constexpr int PIPE_ALL = 0, DSB_DDR = 0;
inline void pipe_barrier(int) { events.push_back(1); }
inline void asc_dcci_entire_out() { events.push_back(2); }
inline void dsb(int) { events.push_back(3); }
inline void set_atomic_none() { events.push_back(4); ctrl &= ~0x1c0ULL; }
inline uint64_t asc_get_ctrl() { events.push_back(5); return ctrl; }
inline void asc_set_ctrl(uint64_t value) { events.push_back(8); ctrl = value; }
inline void asc_init() { events.push_back(6); defaults = true; ctrl = 8; }
"""
        + "\n".join(
            f"template<class... T> inline void {name}(T...) {{}}"
            for name in (
                "asc_set_gm2l1_loop_size",
                "asc_set_gm2ub_loop_size",
                "asc_set_ub2gm_loop_size",
                "asc_set_l0c2gm_config",
                "asc_set_l0c2gm_nz2nd",
                "asc_set_l0c_copy_prequant",
                "asc_set_l0c2gm_quant_post",
                "asc_set_l0c2gm_relu_alpha",
                "asc_set_l0c2gm_lrelu_alpha",
                "asc_set_l0c2gm_channel_para",
                "asc_set_copy_pad_val",
                "asc_set_ndim_pad_value",
                "asc_set_gm2l1_pad",
                "asc_set_l13d_padding",
                "asc_set_l12l0_padding_val",
            )
        )
    )
    main = """
int main() {
    float x[32]{}, out[32]{};
    simpler::tmr::Tensor tx{}, ty{};
    tx.buffer.addr = reinterpret_cast<uint64_t>(x); tx.shapes[0] = 1; tx.strides[0] = 32;
    ty = tx; ty.buffer.addr = reinterpret_cast<uint64_t>(out);
    int64_t args[] = {reinterpret_cast<int64_t>(&tx), reinterpret_cast<int64_t>(&ty), 0x3f800000};
    for (int i = 0; i < 2; ++i) {
        kernel_entry(args);
        // The executor is allowed to publish FIN only after entry return.
        if (ctrl != 0x400000001008 || !defaults || out[0] != 1.0f) return 1;
        events.push_back(10);
    }
    const std::vector<int> one{1,2,3,4,5,9,1,2,3,6,4,8,10};
    std::vector<int> expected(one); expected.insert(expected.end(), one.begin(), one.end());
    return events == expected ? 0 : 2;
}
"""
    if cube:
        main = main.replace("{1,2,3,4,5,9", "{1,2,3,4,5,8,9")
    result, binary = _compile_call(task.source, tmp_path, main)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("bad_launch", ["none", "blocks", "sync", "null"])
@pytest.mark.parametrize("reuse_ids", [False, True])
def test_submitter_calls_tmr_with_checked_single_block_bindings(
    compiler, tmp_path, bad_launch, core_type, reuse_ids
):
    from pypto.backend.cannbotdsl_runtime import build_submitter  # noqa: PLC0415

    _aiv_runtime_manifest(compiler, core_type)
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    ids = tuple((10 + i, 20 + i) if core_type == "mixed" else (10 + i,) for i in range(3))
    if reuse_ids:
        ids = (ids[0],) * 3
    # Equal task records may reuse an image even when they are distinct Python objects.
    artifact = build_submitter((task, replace(task), replace(task)), ids)
    offset = "0" if reuse_ids else "i"
    expected_aic = f"int(10 + {offset})" if core_type != "aiv" else "-1"
    expected_aiv = (
        f"int(20 + {offset})"
        if core_type == "mixed"
        else f"int(10 + {offset})"
        if core_type == "aiv"
        else "-1"
    )
    expected_aiv1 = expected_aiv if core_type == "mixed" else "-1"
    invalid_launch = {
        "none": "",
        "blocks": "bindings[2]->launch_spec.set_block_num(2);",
        "sync": "bindings[2]->launch_spec.set_require_sync_start(true);",
        "null": "bindings[2] = nullptr;",
    }[bad_launch]
    body = f"""
    static unsigned dispatches = 0;
    RuntimeOps ops{{}};
    ops.is_fatal = [](RuntimeContext*) {{ return false; }};
    ops.submit_task = [](RuntimeContext*, const MixedKernels& kernels, const CoreTaskArgs& arg) {{
        unsigned i = dispatches++;
        if (kernels.aic_kernel_id != {expected_aic} || kernels.aiv0_kernel_id != {expected_aiv} ||
            kernels.aiv1_kernel_id != {expected_aiv1} || arg.allow_early_resolve() ||
            arg.launch_spec.block_num() != 1) throw "dispatch contract";
        if (arg.explicit_dep_count() != (i == 2 ? 1U : 0U)) throw "dependency count";
        if (i == 2 && arg.explicit_dep_kind(0) != DEP_WAIT) throw "completion dependency";
        TaskOutputTensors result;
        result.set_task_id(TaskId::make(0, 100 + int(i)));
        return result;
    }};
    RuntimeContext runtime{{&ops, ScopeMode::AUTO}};
    test_runtime = &runtime;
    if ({str(bad_launch != "none").lower()}) {{
        {invalid_launch}
        try {{ {artifact.symbol}(bindings); return 1; }} catch (const char*) {{}}
        return dispatches == 0 ? 0 : 2;
    }}
    auto ids = {artifact.symbol}(bindings);
    return dispatches == 3 && ids[2] == TaskId::make(0, 102) ? 0 : 3;
"""
    _run_ordered_submitter(
        artifact,
        tmp_path,
        body,
        direct=True,
        prefix="""
#include "orchestration_api.h"
RuntimeContext* test_runtime = nullptr;
RuntimeContext* framework_current_runtime() { return test_runtime; }
""",
    )


@pytest.mark.parametrize("core_type", ["aic", "mixed"])
def test_real_export_cube_and_mixed_topology(tmp_path, core_type):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def cube_copy(x, out):
        for i in cb.range(x.shape[0]):
            cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    @cb.kernel
    def mixed_copy(x, out):
        for i in cb.range(x.shape[0]):
            if (i // 16) % 3 == 0:
                cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)
            if (i // 16) % 3 == cb.get_subblock_id() + 1:
                cb.scalar.vec_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    tensor = TensorSpec((96,), DataType.INT32, strides=(1,))
    request = KernelRequest(
        (ParameterSpec("x", tensor), ParameterSpec("out", tensor)), CompileTarget("a5", "ccec", "test")
    )
    prepared = adapter.CANNBotDSLProvider().prepare(cube_copy if core_type == "aic" else mixed_copy, request)
    manifest = json.loads(
        next(f.content for f in prepared.artifact.files if f.path == "device_manifest.json")
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert manifest["core_type"] == core_type
    assert prepared.execution.cores == ((Core.AIC,) if core_type == "aic" else (Core.AIC, Core.AIV))


@pytest.mark.parametrize("ids", [((1,),), ((1, 1),), ((-1, 2),), ((True, 2),), ((1, 2, 3),)])
def test_mixed_submitter_rejects_incomplete_or_ambiguous_registration(compiler, ids):
    from pypto.backend.cannbotdsl_runtime import build_submitter  # noqa: PLC0415

    _aiv_runtime_manifest(compiler, "mixed")
    prepared = _prepare_call(compiler)
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    with pytest.raises(ContractError, match="function ID"):
        build_submitter((task,), ids)


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
def test_submitter_rejects_reused_id_for_different_task_images(compiler, core_type):
    from pypto.backend.cannbotdsl_runtime import build_submitter  # noqa: PLC0415

    _aiv_runtime_manifest(compiler, core_type)
    first = _prepare_call(compiler)
    second = _prepare_call(compiler, _CALL_SOURCE.replace("= alpha;", "= alpha + 1.0f;"))
    tasks = tuple(TMRBridge().lower(p, _aiv_target(p)) for p in (first, second))
    ids = ((10, 20), (10, 21)) if core_type == "mixed" else ((10,), (10,))
    with pytest.raises(ContractError, match="function ID 10.*different task/core images"):
        build_submitter(tasks, ids)


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("scenario", ["registration", "batch"])
def test_runtime_submitter_bundles_compose_in_one_translation_unit(compiler, tmp_path, core_type, scenario):
    from pypto.backend.cannbotdsl_runtime import build_submitter  # noqa: PLC0415

    _aiv_runtime_manifest(compiler, core_type)
    prepared = _prepare_binding(compiler, TensorSpec(("rows", 1), DataType.FP32, strides=("pitch", 1)))
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    ids = (10, 11) if core_type == "mixed" else (10,)
    first = build_submitter((task,) * 3, (ids,) * 3)
    # Exercise both different registrations of the same batch and a different
    # batch reusing the same images. Each used to overwrite a fixed header path.
    count = 3 if scenario == "registration" else 2
    second_ids = tuple(i + 10 for i in ids) if scenario == "registration" else ids
    second = build_submitter((task,) * count, (second_ids,) * count)
    files = {f.path: f for f in first.files}
    for file in second.files:
        if file.path in files:
            assert files[file.path] == file, f"conflicting bundle path: {file.path}"
        files[file.path] = file
    artifact = replace(first, files=tuple(files.values()))
    expected = (
        f"{{{base}, {base + 1}, {base + 1}}}"
        if core_type == "mixed"
        else f"{{{base}, -1, -1}}"
        if core_type == "aic"
        else f"{{-1, {base}, -1}}"
        for base in (10, second_ids[0])
    )
    body = f"""
    static unsigned dispatches = 0;
    RuntimeOps ops{{}};
    ops.is_fatal = [](RuntimeContext*) {{ return false; }};
    ops.submit_task = [](RuntimeContext*, const MixedKernels& kernels, const CoreTaskArgs&) {{
        const MixedKernels expected[] = {{{", ".join(expected)}}};
        const auto& want = expected[dispatches < 3 ? 0 : 1];
        if (kernels.aic_kernel_id != want.aic_kernel_id ||
            kernels.aiv0_kernel_id != want.aiv0_kernel_id ||
            kernels.aiv1_kernel_id != want.aiv1_kernel_id) throw "wrong registration";
        TaskOutputTensors result;
        result.set_task_id(TaskId::make(0, 100 + int(dispatches++)));
        return result;
    }};
    RuntimeContext runtime{{&ops, ScopeMode::AUTO}};
    test_runtime = &runtime;
    auto first = {first.symbol}(bindings);
    std::array<CoreTaskArgs*, {count}> next{{{", ".join(f"bindings[{i}]" for i in range(count))}}};
    auto second = {second.symbol}(next);
    return dispatches == {3 + count} && first[2] == TaskId::make(0, 102) &&
        second.back() == TaskId::make(0, {102 + count}) ? 0 : 1;
"""
    _run_ordered_submitter(
        artifact,
        tmp_path,
        body,
        direct=True,
        prefix=f"""
#include "{first.entry}"
#include "{second.entry}"
RuntimeContext* test_runtime = nullptr;
RuntimeContext* framework_current_runtime() {{ return test_runtime; }}
""",
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("resources", "topology", "single_core"),
        ("completion", "wait_for_all_participants", False),
        ("build_requirements", "targets", ["aiv"]),
    ],
)
def test_mixed_bridge_requires_whole_group_contract(compiler, section, key, value):
    _aiv_runtime_manifest(compiler, "mixed")
    compiler.manifest[section][key] = value
    prepared = _prepare_call(compiler)
    with pytest.raises(ContractError, match="runtime"):
        TMRBridge().lower(prepared, _aiv_target(prepared))


def test_mixed_submission_uses_whole_group_completion(compiler, tmp_path):
    from pypto.backend.cannbotdsl_runtime import build_submitter  # noqa: PLC0415

    root = Path(__file__).resolve().parents[3]
    isa = root / "runtime/build/pto-isa/include"
    if not (isa / "pto/comm/async/urma/urma_types.hpp").is_file():
        pytest.skip("pinned scheduler test needs runtime's managed PTO-ISA headers")
    _aiv_runtime_manifest(compiler, "mixed")
    prepared = _prepare_call(compiler)
    task = TMRBridge().lower(prepared, _aiv_target(prepared))
    artifact = build_submitter((task,), ((10, 11),))
    main = """
#include "scheduler/scheduler.h"
int main() {
    static SchedulerState scheduler{};
    ChipTaskSlotState slot{};
    slot.total_required_subtasks = 3;
    // Reuse the same slot to detect stale completion counters as well.
    for (int batch = 0; batch < 3; ++batch) {
        slot.completed_subtasks.store(0);
        if (scheduler.on_subtask_complete(slot)) return 1; // first lane FIN
        if (scheduler.on_subtask_complete(slot)) return 2; // second lane FIN
        if (!scheduler.on_subtask_complete(slot)) return 3; // whole group FIN
    }
    return 0;
}
"""
    result, binary = _compile_call(artifact, tmp_path, main, orchestration=True, extra_includes=(isa,))
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
