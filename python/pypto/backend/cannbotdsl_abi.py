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
from typing import Any

from pypto.jit.external.contracts import ABIParameter, ABIValue, PreparedKernel, SourceArtifact, TensorSpec
from pypto.jit.external.identity import prepared_digest
from pypto.jit.external.providers.cannbotdsl import ABI_NAME, _contracts, _validate_inputs
from pypto.jit.external.validation import ContractError, validate_prepared
from pypto.pypto_core import DataType

from .tmr_bridge import _file, _slots

# CANN C API spellings differ from PTO's float8_* names. Other supported
# element types use the shared IR C++ spelling, including INDEX as int64_t.
_TENSOR_CPP_TYPES = {
    DataType.FP8E4M3FN: "fp8_e4m3fn_t",
    DataType.FP8E5M2: "fp8_e5m2_t",
    DataType.FP8E8M0: "fp8_e8m0_t",
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


def build_call_adapter(prepared: PreparedKernel) -> SourceArtifact:
    """Own a compile-only device call adapter and its native dependencies.

    The caller must supply valid TMR packed arguments (tensors first, scalars
    second), and prove bounds, dtype, strides, alignment and aliasing on every
    invocation. This function validates declarations and the compiled ABI only;
    returning a SourceArtifact does not authorize runtime submission.
    """
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
