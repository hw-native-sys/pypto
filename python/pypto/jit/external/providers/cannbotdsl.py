# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepare owned CANNBotDSL exports through its public device-callable API.

This is the direct provider protocol, not a capturing registry implementation.
Python closures cannot yet be snapshotted completely, so every preparation exports
again. The named descriptor ABI deliberately requires a dedicated runtime bridge;
it must never be relabeled as c_pointer or executed without manifest checks.
"""

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, cast

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

from ..contracts import (
    ABIParameter,
    ABIValue,
    Access,
    Completion,
    Core,
    Coverage,
    ExecutionRequirements,
    KernelContract,
    KernelRequest,
    NativeABI,
    PreparationIdentity,
    PreparedKernel,
    ScalarSpec,
    SourceArtifact,
    SourceFile,
    TensorEffect,
    TensorSpec,
    WritebackBinding,
)
from ..validation import ContractError, validate_prepared, validate_request

ABI_NAME = "cannbotdsl.device_callable"
# Names and element tags belong to the public DSL API/export protocol. Keep
# INDEX distinct in the request even though its physical representation is i64.
_DTYPES = {
    DataType.FP16: ("float16", "f16"),
    DataType.BF16: ("bfloat16", "bf16"),
    DataType.FP32: ("float32", "f32"),
    DataType.FP8E4M3FN: ("float8_e4m3fn", "f8e4m3"),
    DataType.FP8E5M2: ("float8_e5m2", "f8e5m2"),
    DataType.FP8E8M0: ("float8_e8m0", "f8e8m0"),
    DataType.INT8: ("int8", "i8"),
    DataType.INT16: ("int16", "i16"),
    DataType.INT32: ("int32", "i32"),
    DataType.INT64: ("int64", "i64"),
    DataType.INDEX: ("int64", "i64"),
    DataType.UINT8: ("uint8", "u8"),
    DataType.UINT16: ("uint16", "u16"),
    DataType.UINT32: ("uint32", "u32"),
    DataType.UINT64: ("uint64", "u64"),
    DataType.BOOL: ("bool", "bool"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(f"CANNBotDSL: {message}")


def _arguments(cb: Any, kernel: object, request: KernelRequest) -> list[Any]:
    info = cb.inspect_kernel(kernel)
    _require(info is not None, "expected a kernel recognized by inspect_kernel()")
    _require(not info.configured, "expected an unconfigured kernel without host launch dimensions")
    parameters = tuple(info.signature.parameters.values())
    _require(
        all(
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
            for p in parameters
        ),
        "preparation requires explicit positional parameters without defaults",
    )
    constants = dict(request.constants)
    names = [p.name for p in request.parameters]
    _require(not set(names) & constants.keys(), "runtime and constexpr parameter names overlap")
    _require(
        [p.name for p in parameters if p.name not in constants] == names
        and constants.keys() <= info.signature.parameters.keys(),
        "request parameter names/order do not match the kernel signature",
    )
    values: dict[str, Any] = dict(constants)
    symbols: dict[str, Any] = {}
    stride_symbols = {
        stride
        for parameter in request.parameters
        if isinstance(parameter.type, TensorSpec)
        for stride in parameter.type.strides
        if isinstance(stride, str)
    }

    def dimension(value: int | str) -> Any:
        if isinstance(value, int):
            return value
        if value not in symbols:
            # Extent-only symbols admit empty views. A symbol used anywhere as
            # a stride must be positive, including when shared with an extent.
            symbols[value] = cb.Dim(value, min=1 if value in stride_symbols else 0)
        return symbols[value]

    for p in request.parameters:
        spec = p.type
        name, _ = _DTYPES[spec.dtype]
        dtype = getattr(cb.dtypes, name, None)
        _require(dtype is not None, f"installed compiler does not support dtype {spec.dtype}")
        if isinstance(spec, ScalarSpec):
            values[p.name] = dtype
        else:
            values[p.name] = cb.TensorSpec(
                tuple(dimension(d) for d in spec.shape),
                dtype,
                stride=tuple(dimension(d) for d in spec.strides),
            )
    return [values[p.name] for p in parameters]


def _validate_inputs(request: KernelRequest) -> None:
    validate_request(request)
    _require(
        (request.target.architecture, request.target.compiler) == ("a5", "ccec"),
        "device export requires the a5/ccec target",
    )
    _require(not request.options, "compiler options are not yet supported by this adapter")
    for p in request.parameters:
        spec = p.type
        _require(spec.dtype in _DTYPES, f"unsupported dtype {spec.dtype} for parameter {p.name!r}")
        if isinstance(spec, TensorSpec):
            _require(spec.layout == TensorLayout.ND, f"parameter {p.name!r} requires ND layout")
            # Do not infer contiguity from shape. The public exporter requires a
            # unit inner stride; callers must express it as an actual constraint.
            _require(
                bool(spec.shape) and bool(spec.strides) and spec.strides[-1] == 1,
                f"parameter {p.name!r} requires explicit strides with innermost stride 1",
            )
            _require(0 not in spec.strides, f"parameter {p.name!r} requires positive strides")


def _file(path: str, content: bytes) -> SourceFile:
    return SourceFile(path, hashlib.sha256(content).hexdigest(), content)


def _artifact(cb: Any, source: str, manifest: dict[str, Any]) -> SourceArtifact:
    _require(type(source) is str, "device source must be text")
    payload = source.encode("utf-8")
    _require(bool(payload), "device export returned empty source")
    _require(hashlib.sha256(payload).hexdigest() == manifest["source_sha256"], "source digest mismatch")
    # Preserve all provider-specific requirements for the named ABI's future
    # bridge: POD fields, views, state, resources, completion and binary loading.
    # Copying the JSON prevents subsequent mutations of DeviceExport.manifest.
    files = [
        _file("device_callable.hpp", payload),
        _file("device_manifest.json", json.dumps(manifest, sort_keys=True, allow_nan=False).encode()),
    ]
    root = Path(cb.get_include()).resolve()
    for name, digest in sorted(manifest["build_requirements"]["dsl_headers"].items()):
        relative = Path(name)
        _require(
            bool(name) and not relative.is_absolute() and ".." not in relative.parts,
            f"invalid DSL header path {name!r}",
        )
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), f"DSL header escapes include directory: {name!r}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ContractError(f"CANNBotDSL: cannot read declared DSL header {name!r}: {error}") from error
        _require(hashlib.sha256(content).hexdigest() == digest, f"DSL header digest mismatch: {name!r}")
        files.append(_file(f"include/{name}", content))
    return SourceArtifact(tuple(files), "device_callable.hpp", manifest["symbol"], ("include",))


def _prepare_export(cb: Any, export: Any, request: KernelRequest) -> PreparedKernel:
    manifest = export.manifest
    _require(isinstance(manifest, dict), "device manifest must be an object")
    _require(
        manifest.get("protocol") == ABI_NAME
        and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1,
        "expected device_callable export protocol version 1",
    )
    _require(manifest["arch"] == "dav-3510", "export architecture differs from a5")
    _require(manifest["calling_convention"] == "device_function", "expected a device function")
    for section in (
        "contract",
        "resources",
        "state",
        "loading",
        "build_requirements",
        "completion",
        "producer",
        "participants",
    ):
        _require(isinstance(manifest.get(section), dict), f"missing or invalid {section} metadata")
    physical = manifest["parameters"]
    indices = [p["logical_index"] for p in physical]
    _require(
        all(type(i) is int for i in indices) and sorted(indices) == list(range(len(request.parameters))),
        "native ABI must represent every logical parameter exactly once",
    )
    abi: list[ABIParameter] = []
    effects: dict[int, TensorEffect] = {}
    for item in physical:
        index = item["logical_index"]
        parameter = request.parameters[cast(int, index)]
        _require(item["name"] == parameter.name, f"native parameter {index} has a different name")
        tensor = isinstance(parameter.type, TensorSpec)
        _require(item["kind"] == ("tensor" if tensor else "scalar"), f"parameter {index} kind mismatch")
        abi.append(
            ABIParameter(
                index, ABIValue.DESCRIPTOR if tensor else ABIValue.SCALAR, item["size"], item["alignment"]
            )
        )
        if isinstance(parameter.type, TensorSpec):
            spec = parameter.type
            _require(item["dtype"] == _DTYPES[spec.dtype][1], f"parameter {index} dtype mismatch")
            _require(
                type(item["layout_map"]) is int and item["layout_map"] == 0,
                f"parameter {index} requires an ND descriptor",
            )
            for field, dimensions in (("shape", spec.shape), ("stride", spec.strides)):
                # The export protocol encodes dynamic extents/strides as -1;
                # their equality relationships remain in the retained contract.
                expected_dimensions = [d if isinstance(d, int) else -1 for d in dimensions]
                _require(
                    item[field] == expected_dimensions and all(type(d) is int for d in item[field]),
                    f"parameter {index} {field} differs from the request",
                )
            access = Access(item["access"])
            coverage = Coverage.UNKNOWN if access in (Access.WRITE, Access.READ_WRITE) else Coverage.NONE
            # An access mask proves a write exists, never that every element of
            # the argument is overwritten. Retain incoming dependencies.
            effects[index] = TensorEffect(index, access, coverage)
    cores = {"aic": (Core.AIC,), "aiv": (Core.AIV,), "mixed": (Core.AIC, Core.AIV)}
    core_type = manifest["core_type"]
    _require(core_type in cores, f"unsupported core type {core_type!r}")
    expected = {"aic": int(core_type != "aiv"), "aiv": 2 if core_type == "mixed" else int(core_type == "aiv")}
    _require(
        manifest["participants"] == expected
        and all(type(value) is int for value in manifest["participants"].values()),
        "unexpected AIC/AIV participant topology",
    )
    completion = manifest["completion"]
    _require(
        all(
            completion.get(key) is True
            for key in ("local_pipelines_drained", "scalar_gm_flushed", "ddr_barrier")
        )
        and completion.get("publication") == "consumer_after_all_participants_return",
        "export does not guarantee local completion before participant return",
    )
    artifact = _artifact(cb, export.source, manifest)
    prepared = PreparedKernel(
        request,
        KernelContract(
            request.parameters,
            tuple(effects[i] for i in sorted(effects)),
            writebacks=tuple(
                WritebackBinding(i) for i in sorted(effects) if effects[i].coverage != Coverage.NONE
            ),
        ),
        artifact,
        NativeABI(ABI_NAME, 1, tuple(abi)),
        ExecutionRequirements(cores[core_type], expected["aiv"], Completion.ENTRY_RETURN),
        # The output digest is useful provenance but cannot prove a complete
        # Python input snapshot. Never authorize preparation-cache reuse with it.
        PreparationIdentity(
            "cannbotdsl",
            manifest["producer"]["version"],
            "1",
            hashlib.sha256(artifact.files[0].content + artifact.files[1].content).hexdigest(),
            False,
        ),
    )
    validate_prepared(prepared)
    return prepared


class CANNBotDSLProvider:
    """Direct, uncached preparation; no JIT discovery or runtime dispatch yet.

    The descriptor ABI is intentionally unsupported by TMRBridge. A future
    bridge must consume device_manifest.json, validate all declared resources
    and views, and publish completion only after all participants return.
    """

    def prepare(self, kernel: object, request: KernelRequest) -> PreparedKernel:
        _validate_inputs(request)
        try:
            cb = importlib.import_module("cannbotdsl")
        except ModuleNotFoundError as error:
            if error.name != "cannbotdsl":
                raise
            raise ContractError(
                "CANNBotDSL: install cannbot-dsl with the public export_device API"
            ) from error
        _require(
            all(
                callable(getattr(cb, name, None))
                for name in ("export_device", "inspect_kernel", "get_include")
            ),
            "installed package needs export_device(), inspect_kernel() and get_include()",
        )
        export = cb.export_device(kernel, *_arguments(cb, kernel, request), arch="dav-3510")
        try:
            return _prepare_export(cb, export, request)
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ContractError):
                raise
            raise ContractError(f"CANNBotDSL: invalid device export: {error}") from error
