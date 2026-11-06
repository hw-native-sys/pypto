# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Pure validation of the internal v1 contract, independent of provider imports."""

import math
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import cast

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

from .contracts import (
    SCHEMA_VERSION,
    ABIValue,
    Access,
    Completion,
    Core,
    Coverage,
    KernelContract,
    KernelRequest,
    ParameterSpec,
    PreparationIdentity,
    PreparedKernel,
    ScalarSpec,
    TensorSpec,
    Unit,
)

# Capabilities are semantic commitments, not names of supported providers.
SUPPORTED_CAPABILITIES = frozenset({"parameter_returns", "tensor_writeback", "partial_write", "aliasing"})


class ContractError(ValueError):
    """A provider supplied an invalid or unsupported external-kernel contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _immutable(value: object, path: str) -> None:
    """Reject mutable nested values; frozen dataclasses alone are insufficient."""
    if value is None or isinstance(value, (str, bool, int, Enum, DataType, TensorLayout)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{path}: non-finite values are unsupported")
        return
    if isinstance(value, (tuple, frozenset)):
        for i, item in enumerate(cast(tuple[object, ...] | frozenset[object], value)):
            _immutable(item, f"{path}[{i}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        frozen = getattr(getattr(type(value), "__dataclass_params__"), "frozen")
        _require(frozen is True, f"{path}: expected immutable frozen record")
        for field in fields(value):
            _immutable(getattr(value, field.name), f"{path}.{field.name}")
        return
    raise ContractError(f"{path}: expected immutable contract data, got {type(value).__name__}")


def _version(value: int, context: str) -> None:
    _require(type(value) is int and value == SCHEMA_VERSION, f"{context}: unsupported schema version {value}")


def _alignment(value: int, context: str) -> None:
    _require(
        type(value) is int and value > 0 and value & (value - 1) == 0,
        f"{context}: alignment must be a positive power of two in bytes, got {value}",
    )


def _dimension(value: object, context: str) -> None:
    _require(
        (type(value) is int and value >= 0) or (isinstance(value, str) and value.isidentifier()),
        f"{context}: expected nonnegative extent/stride/offset or dimension symbol, got {value!r}",
    )


def _parameters(parameters: tuple[ParameterSpec, ...]) -> set[int]:
    names: set[str] = set()
    tensors: set[int] = set()
    for i, parameter in enumerate(parameters):
        _require(
            parameter.name.isidentifier() and parameter.name not in names,
            f"parameter {i}: invalid or duplicate name {parameter.name!r}",
        )
        names.add(parameter.name)
        spec = parameter.type
        _require(type(spec) in (TensorSpec, ScalarSpec), f"parameter {i}: unsupported type {spec!r}")
        _require(
            type(spec.dtype) is DataType and spec.dtype != DataType.TASK_ID,
            f"parameter {i}: unsupported dtype {spec.dtype!r}",
        )
        if isinstance(spec, TensorSpec):
            tensors.add(i)
            _alignment(spec.alignment, f"parameter {i}")
            _require(type(spec.layout) is TensorLayout, f"parameter {i}: invalid layout {spec.layout!r}")
            _require(
                not spec.strides or len(spec.strides) == len(spec.shape),
                f"parameter {i}: stride rank does not match shape",
            )
            for dim in (*spec.shape, *spec.strides):
                _dimension(dim, f"parameter {i}")
            if spec.offset is not None:
                _dimension(spec.offset, f"parameter {i} offset")
    return tensors


def validate_request(request: KernelRequest) -> None:
    """Validate normalized signature and compile inputs before provider preparation."""
    _immutable(request, "request")
    _version(request.schema_version, "request")
    _parameters(request.parameters)
    _require(
        all((request.target.architecture, request.target.compiler, request.target.compiler_version)),
        "request: architecture, compiler and compiler_version must be specified",
    )
    for label, entries in (("constants", request.constants), ("options", request.options)):
        names: set[str] = set()
        for name, value in entries:
            _require(bool(name) and name not in names, f"request {label}: duplicate/empty key {name!r}")
            names.add(name)
            _require(type(value) in (bool, int, float, str), f"request {label}.{name}: unsupported value")


def validate_contract(contract: KernelContract) -> None:
    """Reject missing effects, ambiguous writebacks and unsupported result kinds."""
    _immutable(contract, "contract")
    _version(contract.schema_version, "contract")
    tensors = _parameters(contract.parameters)
    unknown = contract.required_capabilities - SUPPORTED_CAPABILITIES
    _require(not unknown, f"contract: unsupported required capabilities {sorted(unknown)}")
    seen: set[int] = set()
    written: set[int] = set()
    for effect in contract.effects:
        i = effect.parameter
        _require(
            type(i) is int and i in tensors and i not in seen,
            f"effect: invalid/duplicate tensor parameter {i}",
        )
        seen.add(i)
        _require(type(effect.access) is Access, f"parameter {i}: unknown access {effect.access!r}")
        _require(type(effect.coverage) is Coverage, f"parameter {i}: unknown coverage {effect.coverage!r}")
        _require(type(effect.preserves_unwritten) is bool, f"parameter {i}: invalid preservation guarantee")
        if effect.access in (Access.WRITE, Access.READ_WRITE):
            written.add(i)
            _require(effect.coverage != Coverage.NONE, f"parameter {i}: write needs coverage")
            _require(
                effect.coverage == Coverage.FULL or effect.preserves_unwritten,
                f"parameter {i}: partial/unknown write must preserve untouched bytes",
            )
        else:
            _require(
                effect.coverage == Coverage.NONE, f"parameter {i}: non-writing effect has write coverage"
            )
    _require(seen == tensors, f"contract: missing tensor effects for parameters {sorted(tensors - seen)}")
    writebacks = [binding.parameter for binding in contract.writebacks]
    _require(all(type(i) is int for i in writebacks), "contract: writeback indices must be integers")
    _require(
        len(set(writebacks)) == len(writebacks) and set(writebacks) == written,
        "contract: writebacks must name each written tensor exactly once",
    )
    for binding in contract.public_returns:
        _require(
            type(binding.parameter) is int and binding.parameter in tensors,
            f"return: parameter {binding.parameter} is not a tensor parameter; "
            "fresh/scalar results unsupported",
        )
    pairs: set[tuple[int, int]] = set()
    for permission in contract.alias_permissions:
        a, b = permission.first, permission.second
        _require(
            type(a) is int and type(b) is int and a in tensors and b in tensors and a < b,
            f"alias permission: expected distinct ordered tensor indices, got {(a, b)}",
        )
        _require((a, b) not in pairs, f"alias permission: duplicate pair {(a, b)}")
        pairs.add((a, b))


def _validate_abi(prepared: PreparedKernel) -> None:
    abi = prepared.native_abi
    _require(bool(abi.name) and type(abi.version) is int and abi.version > 0, "ABI: missing name/version")
    _require(abi.byte_order == "little", f"ABI: unsupported byte order {abi.byte_order!r}")
    represented: set[int] = set()
    fields_seen: set[tuple[int, ABIValue, int | None]] = set()
    for position, physical in enumerate(abi.parameters):
        i = physical.parameter
        _require(
            type(i) is int and 0 <= i < len(prepared.contract.parameters),
            f"ABI argument {position}: invalid logical parameter {i}",
        )
        _require(
            type(physical.value) is ABIValue and type(physical.unit) is Unit,
            f"ABI argument {position}: unknown value kind or unit",
        )
        _alignment(physical.alignment, f"ABI argument {position}")
        _require(
            type(physical.size_bytes) is int
            and physical.size_bytes > 0
            and physical.size_bytes % physical.alignment == 0,
            f"ABI argument {position}: invalid byte size/alignment",
        )
        key = (i, physical.value, physical.axis)
        _require(key not in fields_seen, f"ABI argument {position}: duplicate field {key}")
        fields_seen.add(key)
        spec = prepared.contract.parameters[i].type
        if isinstance(spec, ScalarSpec):
            _require(
                physical.value == ABIValue.SCALAR and physical.size_bytes == spec.dtype.get_byte(),
                f"ABI argument {position}: scalar bit representation does not match {spec.dtype}",
            )
        else:
            _require(
                physical.value != ABIValue.SCALAR, f"ABI argument {position}: tensor cannot use scalar bits"
            )
        if physical.value in (ABIValue.DIMENSION, ABIValue.STRIDE):
            _require(
                isinstance(spec, TensorSpec)
                and type(physical.axis) is int
                and 0 <= physical.axis < len(spec.shape),
                f"ABI argument {position}: invalid tensor axis",
            )
        else:
            _require(physical.axis is None, f"ABI argument {position}: unexpected axis")
        if physical.value in (ABIValue.SCALAR, ABIValue.DATA_POINTER, ABIValue.DESCRIPTOR):
            _require(i not in represented, f"ABI: multiple primary representations for parameter {i}")
            represented.add(i)
    _require(
        represented == set(range(len(prepared.contract.parameters))),
        "ABI: every logical parameter needs one scalar, data-pointer or descriptor representation",
    )


def _bundle_path(path: str, *, directory: bool = False) -> None:
    value = PurePosixPath(path)
    _require(
        bool(path)
        and str(value) == path
        and not value.is_absolute()
        and ".." not in value.parts
        and "\\" not in path
        and "\x00" not in path
        and (directory or path != "."),
        f"artifact: expected bundle-relative path, got {path!r}",
    )


def _digest(value: str, context: str) -> None:
    _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{context}: expected lowercase SHA-256")


def validate_prepared(prepared: PreparedKernel) -> None:
    """Validate a provider result; bridge-specific ABI support is checked separately."""
    _immutable(prepared, "prepared")
    validate_request(prepared.request)
    validate_contract(prepared.contract)
    _require(
        prepared.contract.parameters == prepared.request.parameters,
        "prepared: provider changed the normalized request signature",
    )
    _validate_abi(prepared)
    execution = prepared.execution
    _require(
        bool(execution.cores)
        and len(set(execution.cores)) == len(execution.cores)
        and all(type(core) is Core for core in execution.cores),
        "execution: invalid core members",
    )
    for name in ("full_residency", "sync_start", "logical_launch_context"):
        _require(type(getattr(execution, name)) is bool, f"execution: {name} must be boolean")
    lanes = execution.aiv_lanes
    allowed = (0,) if Core.AIV not in execution.cores else (1,) if Core.AIC not in execution.cores else (1, 2)
    _require(
        type(lanes) is int and lanes in allowed, f"execution: invalid AIV lanes {lanes} for core members"
    )
    _require(execution.completion == Completion.ENTRY_RETURN, "execution: unsupported completion guarantee")
    _require(
        not execution.full_residency or execution.sync_start, "execution: full residency needs sync_start"
    )
    if execution.full_residency and len(execution.cores) == 2:
        _require(lanes == 2, "execution: mixed full residency requires both AIV lanes")
    artifact = prepared.artifact
    paths: set[str] = set()
    for file in artifact.files:
        _bundle_path(file.path)
        _require(file.path not in paths, f"artifact: duplicate file {file.path!r}")
        paths.add(file.path)
        _digest(file.sha256, f"artifact {file.path}")
    _require(artifact.entry in paths, "artifact: entry must name a bundled file")
    _require(bool(artifact.symbol), "artifact: missing entry symbol")
    for directory in artifact.include_dirs:
        _bundle_path(directory, directory=True)
    validate_identity(prepared.identity)


def validate_identity(identity: PreparationIdentity) -> None:
    """Validate identity metadata without promising that dependencies are complete."""
    _immutable(identity, "identity")
    _require(
        all((identity.provider, identity.provider_version, identity.adapter_version)),
        "identity: provider and adapter versions are required",
    )
    _require(type(identity.complete) is bool, "identity: complete must be an explicit boolean")
    _digest(identity.kernel_digest, "identity kernel_digest")
