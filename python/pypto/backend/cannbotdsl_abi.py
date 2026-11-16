# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-only CANNBotDSL call adapters for TMR argument storage.

This module constructs native arguments, not runtime tasks. Generated adapters
have no kernel_entry and cannot be passed to tmr_bridge.materialize(). A future
bridge must validate each binding and establish the manifest's resource,
cache-line isolation, stack, loading and completion guarantees before calling it.
"""

import hashlib
import json
import re
from typing import Any, cast

from pypto.jit.external.contracts import (
    ABIParameter,
    ABIValue,
    Access,
    PreparedKernel,
    SourceArtifact,
    TensorSpec,
)
from pypto.jit.external.identity import prepared_digest
from pypto.jit.external.providers.cannbotdsl import ABI_NAME, _contracts, _validate_inputs
from pypto.jit.external.validation import ContractError, validate_prepared
from pypto.pypto_core import DataType

from .tmr_bridge import _DTYPES, _file, _preflight, _slots, _validate_parameters

# CANN C API spellings differ from PTO's float8_* names. Other supported
# element types use the shared IR C++ spelling, including INDEX as int64_t.
_TENSOR_CPP_TYPES = {
    DataType.FP8E4M3FN: "fp8_e4m3fn_t",
    DataType.FP8E5M2: "fp8_e5m2_t",
    DataType.FP8E8M0: "fp8_e8m0_t",
}

# TMR has no E5M2 tag. INDEX uses INT64 storage; keep its logical identity in
# the prepared request. Do not widen c_pointer's supported dtype set here.
_TMR_DTYPES = {
    **_DTYPES,
    DataType.INDEX: "INT64",
    DataType.FP8E4M3FN: "FP8E4M3FN",
    DataType.FP8E8M0: "FP8E8M0",
}


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", value, re.ASCII) is None
    ):
        raise ContractError(f"CANNBotDSL adapter: {label} must be a qualified C++ identifier, got {value!r}")
    return value


def _type_name(value: Any, label: str) -> str:
    name = _identifier(value, label)
    # Builtins cannot carry a global qualifier. Exported typedefs/structs must
    # resolve outside our generated namespace even if they share helper names.
    return name if name in ("bool", "char", "short", "int", "long", "float", "double") else f"::{name}"


def _manifest(prepared: PreparedKernel) -> dict[str, Any]:
    validate_prepared(prepared)
    _validate_inputs(prepared.request)
    if (prepared.native_abi.name, prepared.native_abi.version) != (ABI_NAME, 1):
        raise ContractError("CANNBotDSL adapter: expected device_callable v1 ABI")
    try:
        files = {f.path: f.content for f in prepared.artifact.files}
        manifest = json.loads(files["device_manifest.json"])
        expected = _contracts(manifest, prepared.request)
        if expected != (prepared.contract, prepared.native_abi, prepared.execution):
            raise ContractError("CANNBotDSL adapter: prepared metadata differs from the owned manifest")
        if (prepared.artifact.entry, prepared.artifact.symbol, prepared.artifact.include_dirs) != (
            "device_callable.hpp",
            manifest["symbol"],
            ("include",),
        ):
            raise ContractError(
                "CANNBotDSL adapter: artifact entry, symbol or include paths differ from the export"
            )
        digests = {"device_callable.hpp": manifest["source_sha256"]}
        digests.update(
            {
                f"include/{name}": digest
                for name, digest in manifest["build_requirements"]["dsl_headers"].items()
            }
        )
        if set(files) != {*digests, "device_manifest.json"}:
            raise ContractError("CANNBotDSL adapter: owned source/header set differs from the manifest")
        for path, digest in digests.items():
            if hashlib.sha256(files[path]).hexdigest() != digest:
                raise ContractError(f"CANNBotDSL adapter: manifest digest mismatch for {path!r}")
        for position, item in enumerate(manifest["parameters"]):
            _identifier(item["cpp_type"], f"parameter {position} type")
            if item["kind"] == "tensor":
                _identifier(item["element_cpp_type"], f"parameter {position} element type")
            elif item["scalar_kind"] == "integer" and type(item["signed"]) is not bool:
                raise ContractError("CANNBotDSL adapter: native scalar signedness must be boolean")
        return manifest
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"CANNBotDSL adapter: invalid owned manifest: {error}") from error


def _assert_cpp(condition: str, message: str) -> str:
    return f"    static_assert({condition}, {json.dumps(message)});"


def _descriptor_lines(
    arg: ABIParameter, spec: TensorSpec, item: dict[str, Any], typ: str, name: str
) -> list[str]:
    element = _type_name(item["element_cpp_type"], f"{name} element type")
    logical_element = _type_name(
        _TENSOR_CPP_TYPES.get(spec.dtype, spec.dtype.to_c_type_string()), f"{name} logical element type"
    )
    tensor = f"t{arg.parameter}"
    lines = [
        _assert_cpp(
            f"std::is_standard_layout_v<{typ}> && std::is_trivial_v<{typ}>", "descriptor must be POD"
        ),
        _assert_cpp(
            f"offsetof({typ}, ptr) == 0 && sizeof({name}.ptr) == 8", "descriptor pointer layout mismatch"
        ),
        _assert_cpp(
            f"std::is_same_v<decltype({name}.ptr), __gm__ {element}*>", "descriptor pointer type mismatch"
        ),
        _assert_cpp(f"sizeof({element}) == {spec.dtype.get_byte()}", "tensor element width mismatch"),
        _assert_cpp(f"std::is_same_v<{element}, {logical_element}>", "tensor element dtype mismatch"),
        # TMR's start_offset is in elements. Apply it once, including for empty
        # views; callers must prove address representability and storage bounds.
        f"    {name}.ptr = reinterpret_cast<decltype({name}.ptr)>("
        f"{tensor}.buffer.addr + {tensor}.start_offset * {spec.dtype.get_byte()}ULL);",
    ]
    for field in ("shape", "stride"):
        fields = [f for f in item["dynamic_fields"] if f["field"] == field]
        if not fields:
            continue
        lines.append(
            _assert_cpp(
                f"std::extent_v<decltype({typ}::{field})> == {len(fields)}", "dynamic array length mismatch"
            )
        )
        for entry in fields:
            slot, axis, offset = entry["slot"], entry["dimension"], entry["offset"]
            lines += [
                _assert_cpp(
                    f"std::is_same_v<std::remove_reference_t<decltype({name}.{field}[{slot}])>, int64_t>",
                    "dynamic field type mismatch",
                ),
                _assert_cpp(
                    f"offsetof({typ}, {field}) + {slot} * sizeof(int64_t) == {offset}",
                    "dynamic field offset mismatch",
                ),
                f"    {name}.{field}[{slot}] = {tensor}.{field}s[{axis}];",
            ]
    return lines


def _scalar_lines(item: dict[str, Any], typ: str, name: str, slot: int) -> list[str]:
    kind = item["scalar_kind"]
    if kind == "integer":
        condition = (
            f"std::is_integral_v<{typ}> && !std::is_same_v<{typ}, bool> && "
            f"std::is_signed_v<{typ}> == {str(item['signed']).lower()}"
        )
    else:
        condition = f"std::is_same_v<{typ}, bool>" if kind == "bool" else f"std::is_floating_point_v<{typ}>"
    bits = f"bits_{name}"
    lines = [
        _assert_cpp(condition, "scalar kind mismatch"),
        f"    const uint64_t {bits} = uint64_t(args[{slot}]);",
    ]
    if kind == "bool":
        # Copying a noncanonical bool byte is undefined behavior. Normalize it;
        # numeric scalars preserve all payload bits, including NaNs/signed zero.
        lines.append(f"    {name} = ({bits} & 255ULL) != 0;")
    else:
        lines += [
            f"    for (unsigned b = 0; b < sizeof({typ}); ++b)",
            f"        reinterpret_cast<unsigned char*>(&{name})[b] = ({bits} >> (8 * b)) & 255;",
        ]
    return lines


def _validate_call(prepared: PreparedKernel) -> dict[str, Any]:
    manifest = _manifest(prepared)
    symbol = _identifier(prepared.artifact.symbol, "entry symbol")
    if symbol == "kernel_entry":
        raise ContractError("CANNBotDSL adapter: kernel_entry is reserved for runtime dispatch")
    tensors, scalars = _slots(prepared)
    if len(tensors) > 32 or len(scalars) > 16:
        raise ContractError("CANNBotDSL adapter: TMR supports at most 32 tensors and 16 scalars")
    for parameter in prepared.contract.parameters:
        spec = parameter.type
        if isinstance(spec, TensorSpec):
            if not 1 <= len(spec.shape) <= 5:
                raise ContractError("CANNBotDSL adapter: TMR tensor rank must be in 1..5")
        elif spec.dtype in (DataType.FP16, DataType.BF16):
            raise ContractError("CANNBotDSL adapter: half-width runtime scalars are not supported")
    return manifest


def build_binding_checker(prepared: PreparedKernel) -> SourceArtifact:
    """Generate a host checker for each invocation's actual TMR arguments.

    The caller must run the exported function on CoreTaskArgs before packing
    or submitting those same bindings, including on artifact cache hits. The
    checker proves metadata/bounds and conservative element non-overlap, not
    concurrent cache-line isolation, resource ownership or task completion.
    """
    _validate_call(prepared)
    _validate_parameters(prepared, dtypes=_TMR_DTYPES)
    for parameter in prepared.contract.parameters:
        if not isinstance(parameter.type, TensorSpec) and parameter.type.dtype in _TENSOR_CPP_TYPES:
            raise ContractError("CANNBotDSL adapter: FP8 runtime scalars are not supported")
    symbol = f"pypto_cannbotdsl_check_{prepared_digest(prepared)}"
    # Pair checkers embed this header under a digest-based path. A named guard
    # also permits callers to include a standalone copy in the same TU.
    guard = f"{symbol.upper()}_HPP"
    source = (
        f'#pragma once\n#ifndef {guard}\n#define {guard}\n#include "types.h"\n'
        + _preflight(prepared, symbol, dtypes=_TMR_DTYPES, view_pointer=True, non_overlapping=True)
        + "#endif\n"
    )
    return SourceArtifact((_file("binding_check.hpp", source.encode()),), "binding_check.hpp", symbol)


def build_conflict_checker(first: PreparedKernel, second: PreparedKernel) -> SourceArtifact:
    """Generate a host predicate for cache-line conflicts between two invocations.

    The exported function takes the first and second CoreTaskArgs, validates
    both, and returns true when their conservative access ranges require
    serialization. Binding metadata and allocation addresses must remain fixed,
    and storage must stay alive until execution completes. A false result
    addresses only this pair's GM conflicts; it does
    not establish resource ownership or authorize runtime submission.
    """
    checkers = [build_binding_checker(p) for p in (first, second)]
    for prepared in (first, second):
        isolation = _manifest(prepared)["resources"].get("gm_write_isolation")
        policy = cast(dict[str, Any], isolation)
        if (
            not isinstance(isolation, dict)
            or set(policy) != {"granularity_bytes", "rule"}
            or type(policy["granularity_bytes"]) is not int
            or policy["granularity_bytes"] != 64
            or policy["rule"] != "serialize_conflicting_cache_lines"
        ):
            raise ContractError(
                "CANNBotDSL conflict checker: expected 64-byte GM cache-line isolation "
                f"with serialize_conflicting_cache_lines, got {isolation!r}"
            )
    digests = [prepared_digest(p) for p in (first, second)]
    digest = hashlib.sha256(json.dumps(digests).encode()).hexdigest()
    symbol = f"pypto_cannbotdsl_conflict_{digest}"
    # The same kernel can appear on both sides. Emit its checker only once,
    # using digest-based paths so multiple pair artifacts can share headers.
    files = {f"checks/{digest}.hpp": checker.files[0].content for digest, checker in zip(digests, checkers)}
    lines = ["#pragma once", *(f'#include "{path}"' for path in files)]
    lines += [
        f"template<class FirstArgs, class SecondArgs> inline bool {symbol}(",
        "        const FirstArgs& first, const SecondArgs& second) {",
        f"    {checkers[0].symbol}(first);",
        f"    {checkers[1].symbol}(second);",
        # Both checkers run before any early conflict return. Recompute spans
        # only from validated metadata; there is no cached binding/address.
        "    struct Lines { uint64_t first, last; bool empty; };",
        "    auto cache_lines = [](const auto& tensor, uint64_t itemsize) -> Lines {",
        "        for (unsigned axis = 0; axis < tensor.ndims; ++axis)",
        "            if (tensor.shapes[axis] == 0) return {0, 0, true};",
        "        uint64_t last = tensor.start_offset;",
        "        for (unsigned axis = 0; axis < tensor.ndims; ++axis)",
        "            last += uint64_t(tensor.shapes[axis] - 1) * tensor.strides[axis];",
        # Inclusive last-line indices avoid rounding end up past UINT64_MAX.
        # Binding checks proved addr + (last + 1) * itemsize representable.
        "        return {(tensor.buffer.addr + tensor.start_offset * itemsize) / 64,",
        "                (tensor.buffer.addr + (last + 1) * itemsize - 1) / 64, false};",
        "    };",
    ]
    accesses: list[list[tuple[str, bool]]] = []
    for side, prepared in zip(("first", "second"), (first, second)):
        effects = {e.parameter: e.access for e in prepared.contract.effects}
        side_accesses: list[tuple[str, bool]] = []
        for index, slot in _slots(prepared)[0].items():
            if effects.get(index, Access.NONE) == Access.NONE:
                continue
            spec = prepared.contract.parameters[index].type
            name = f"{side}_{index}"
            lines.append(
                f"    const auto {name} = cache_lines({side}.tensor({slot}).ref(), {spec.dtype.get_byte()});"
            )
            side_accesses.append((name, effects[index] in (Access.WRITE, Access.READ_WRITE)))
        accesses.append(side_accesses)
    for left, left_writes in accesses[0]:
        for right, right_writes in accesses[1]:
            if left_writes or right_writes:
                lines += [
                    f"    if (!{left}.empty && !{right}.empty &&",
                    f"        {left}.first <= {right}.last && {right}.first <= {left}.last) return true;",
                ]
    lines += ["    return false;", "}", ""]
    files["conflict_check.hpp"] = "\n".join(lines).encode()
    return SourceArtifact(
        tuple(_file(path, content) for path, content in files.items()), "conflict_check.hpp", symbol
    )


def build_call_adapter(prepared: PreparedKernel) -> SourceArtifact:
    """Own a compile-only device call adapter and its native dependencies.

    The caller must supply valid TMR packed arguments (tensors first, scalars
    second), and run build_binding_checker's function on every invocation.
    This function validates declarations and the compiled ABI only; returning
    a SourceArtifact does not authorize runtime submission.
    """
    manifest = _validate_call(prepared)
    symbol = prepared.artifact.symbol
    tensors, scalars = _slots(prepared)
    namespace = f"pypto_cannbotdsl_{prepared_digest(prepared)}"
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <tuple>",
        "#include <type_traits>",
        '#include "tensormap_and_ringbuffer/tensor.h"',
        '#include "native/device_callable.hpp"',
        f"namespace {namespace} {{",
        "template<class> struct Signature;",
        "template<class R, class... A> struct Signature<R(A...)> {",
        "    using Result = R; using Args = std::tuple<A...>;",
        "};",
        f"using Native = Signature<decltype(::{symbol})>;",
        'static_assert(std::is_same_v<Native::Result, void>, "native entry must return void");',
        _assert_cpp(
            f"std::tuple_size_v<Native::Args> == {len(manifest['parameters'])}",
            "native argument count mismatch",
        ),
        "__aicore__ inline void invoke(__gm__ int64_t* args) {",
    ]
    for index, slot in tensors.items():
        lines.append(
            f"    const auto& t{index} = *reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[{slot}]);"
        )
    arguments: list[str] = []
    for position, (arg, item) in enumerate(zip(prepared.native_abi.parameters, manifest["parameters"])):
        typ, name = f"A{position}", f"a{position}"
        native_type = _type_name(item["cpp_type"], f"parameter {position} type")
        spec = prepared.contract.parameters[arg.parameter].type
        lines += [
            f"    using {typ} = std::tuple_element_t<{position}, Native::Args>;",
            f'    static_assert(std::is_same_v<{typ}, {native_type}>, "native type mismatch");',
            _assert_cpp(
                f"sizeof({typ}) == {arg.size_bytes} && alignof({typ}) == {arg.alignment}",
                "native size/alignment mismatch",
            ),
            f"    {typ} {name}{{}};",
        ]
        if arg.value == ABIValue.DESCRIPTOR:
            assert isinstance(spec, TensorSpec)
            lines += _descriptor_lines(arg, spec, item, typ, name)
        else:
            lines += _scalar_lines(item, typ, name, len(tensors) + scalars[arg.parameter])
        arguments.append(name)
    lines += [f"    ::{symbol}({', '.join(arguments)});", "}", "}", ""]
    files = tuple(_file(f"native/{f.path}", f.content) for f in prepared.artifact.files)
    return SourceArtifact(
        (*files, _file("call_adapter.hpp", "\n".join(lines).encode())),
        "call_adapter.hpp",
        f"{namespace}::invoke",
        tuple(f"native/{p}" for p in prepared.artifact.include_dirs),
    )
