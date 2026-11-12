# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 TMR bridge for the internal ``c_pointer`` v1 native ABI.

Physical arguments can reorder/repeat logical parameters. Metadata arguments
are unsigned integers; pointers name allocation bases and the native function
applies its separately declared offset. Scalars preserve their bit patterns.
The callable must be an inline device function, not a host launch stub.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import (
    ABIParameter,
    ABIValue,
    Access,
    Core,
    PreparedKernel,
    ScalarSpec,
    SourceArtifact,
    SourceFile,
    TensorSpec,
    Unit,
)
from pypto.jit.external.identity import prepared_digest
from pypto.jit.external.validation import ContractError
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout

from .external_bridge import ExternalTask, RuntimeTarget, validate_target

if TYPE_CHECKING:
    from pypto.jit.specializer import SpecializeContext


# Runtime dtype spellings are distinct from both IR and native C++ spellings.
_DTYPES = {
    DataType.FP32: "FLOAT32",
    DataType.FP16: "FLOAT16",
    DataType.BF16: "BFLOAT16",
    DataType.INT8: "INT8",
    DataType.INT16: "INT16",
    DataType.INT32: "INT32",
    DataType.INT64: "INT64",
    DataType.UINT8: "UINT8",
    DataType.UINT16: "UINT16",
    DataType.UINT32: "UINT32",
    DataType.UINT64: "UINT64",
    DataType.BOOL: "BOOL",
}
_CAPABILITIES = frozenset(
    {
        "core:aic",
        "core:aiv",
        "aiv_lanes:0",
        "aiv_lanes:1",
        "aiv_lanes:2",
        "completion:entry_return_visible",
        "parameter_returns",
        "tensor_writeback",
        "partial_write",
        "aliasing",
    }
)


def _file(path: str, content: bytes) -> SourceFile:
    return SourceFile(path, hashlib.sha256(content).hexdigest(), content)


def _slots(prepared: PreparedKernel) -> tuple[dict[int, int], dict[int, int]]:
    """TMR packs tensors first, then scalars, independently of native ABI order."""
    tensors = [i for i, p in enumerate(prepared.contract.parameters) if isinstance(p.type, TensorSpec)]
    scalars = [i for i, p in enumerate(prepared.contract.parameters) if isinstance(p.type, ScalarSpec)]
    return dict(zip(tensors, range(len(tensors)))), dict(zip(scalars, range(len(scalars))))


def _metadata(arg: ABIParameter, tensor: str, spec: TensorSpec) -> str:
    if arg.value == ABIValue.OFFSET:
        value = f"{tensor}.start_offset"
    elif arg.value == ABIValue.DIMENSION:
        value = f"{tensor}.shapes[{arg.axis}]"
    else:
        value = f"{tensor}.strides[{arg.axis}]"
    multiplier = spec.dtype.get_byte() if arg.unit == Unit.BYTES else 1
    return f"(uint64_t({value}) * {multiplier}ULL)"


def _validate(prepared: PreparedKernel, target: RuntimeTarget) -> None:
    validate_target(prepared, target)
    if re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", prepared.artifact.symbol, re.ASCII) is None:
        raise ContractError("TMR bridge: native entry symbol must be a qualified C++ identifier")
    if prepared.artifact.symbol == "kernel_entry":
        raise ContractError("TMR bridge: kernel_entry is reserved for the runtime wrapper")
    if any(c in prepared.artifact.entry for c in '\n\r"'):
        raise ContractError("TMR bridge: entry path cannot contain quotes or newlines")
    if (target.runtime, target.version) != ("tmr", "1"):
        raise ContractError(
            f"TMR bridge: expected runtime ('tmr', '1'), got {(target.runtime, target.version)!r}"
        )
    if target.compile_target.architecture != "a5" or target.compile_target.compiler != "ccec":
        raise ContractError("TMR bridge: only the a5/ccec compile target is supported")
    if (prepared.native_abi.name, prepared.native_abi.version) != ("c_pointer", 1):
        raise ContractError("TMR bridge: expected native ABI ('c_pointer', 1)")
    unsupported = prepared.contract.required_capabilities - _CAPABILITIES
    if unsupported:
        raise ContractError(f"TMR bridge: unsupported capabilities {sorted(unsupported)}")
    for name in ("full_residency", "sync_start", "logical_launch_context"):
        if getattr(prepared.execution, name):
            raise ContractError(f"TMR bridge: {name} is not implemented")
    tensors, scalars = _slots(prepared)
    if len(tensors) > 32 or len(scalars) > 16:
        raise ContractError("TMR bridge: at most 32 tensors and 16 scalars are supported")
    _validate_parameters(prepared)


def _validate_parameters(prepared: PreparedKernel) -> None:
    for p in prepared.contract.parameters:
        if p.type.dtype not in _DTYPES:
            raise ContractError(f"TMR bridge: unsupported dtype for {p.name!r}: {p.type.dtype}")
        if isinstance(p.type, TensorSpec):
            if p.type.layout != TensorLayout.ND or not 1 <= len(p.type.shape) <= 5:
                raise ContractError(f"TMR bridge: {p.name!r} requires ND layout and rank 1..5")
            for value in (*p.type.shape, *p.type.strides):
                if isinstance(value, int) and value > (1 << 32) - 1:
                    raise ContractError(f"TMR bridge: {p.name!r} extent/stride exceeds uint32")
            if any(value == 0 for value in p.type.strides):
                raise ContractError(f"TMR bridge: {p.name!r} requires positive strides")
            if isinstance(p.type.offset, int) and p.type.offset > (1 << 64) - 1:
                raise ContractError(f"TMR bridge: {p.name!r} offset exceeds uint64")
            if p.type.alignment > 1 << 63:
                raise ContractError(f"TMR bridge: {p.name!r} alignment exceeds address width")
        elif p.type.dtype in (DataType.FP16, DataType.BF16):
            raise ContractError(f"TMR bridge: half-width scalar {p.name!r} is not supported")


def _validate_abi(prepared: PreparedKernel) -> None:
    for arg in prepared.native_abi.parameters:
        if arg.value == ABIValue.DESCRIPTOR:
            raise ContractError("TMR bridge: c_pointer v1 does not define descriptor layout")
        if arg.value == ABIValue.DATA_POINTER:
            if (arg.size_bytes, arg.alignment) != (8, 8):
                raise ContractError("TMR bridge: data pointers require size/alignment 8/8")
        elif arg.size_bytes not in (1, 2, 4, 8) or arg.alignment != arg.size_bytes:
            raise ContractError("TMR bridge: arguments require natural alignment and 1, 2, 4, or 8 bytes")


def _wrapper(prepared: PreparedKernel) -> str:
    tensors, scalars = _slots(prepared)
    symbol = prepared.artifact.symbol
    lines = [
        "#include <cstdint>",
        "#include <type_traits>",
        "#include <tuple>",
        "#ifndef __aicore__",
        "#define __aicore__ [aicore]",
        "#endif",
        '#include "tensormap_and_ringbuffer/tensor.h"',
        f'#include "native/{prepared.artifact.entry}"',
        "template<class> struct NativeSignature;",
        "template<class R, class... A> struct NativeSignature<R(A...)> {",
        "    using Result = R; using Args = std::tuple<A...>;",
        "};",
        f"using Signature = NativeSignature<decltype({symbol})>;",
        'static_assert(std::is_same_v<Signature::Result, void>, "native entry must return void");',
        f"static_assert(std::tuple_size_v<Signature::Args> == {len(prepared.native_abi.parameters)}, "
        '"native argument count mismatch");',
        'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {',
    ]
    for index, slot in tensors.items():
        lines.append(
            f"    const auto& t{index} = *reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[{slot}]);"
        )
    arguments: list[str] = []
    for position, arg in enumerate(prepared.native_abi.parameters):
        index = arg.parameter
        spec = prepared.contract.parameters[index].type
        name, typ = f"a{position}", f"A{position}"
        lines += [
            f"    using {typ} = std::tuple_element_t<{position}, Signature::Args>;",
            f"    static_assert(sizeof({typ}) == {arg.size_bytes} && alignof({typ}) == {arg.alignment}, "
            '"native argument size/alignment mismatch");',
        ]
        if arg.value == ABIValue.SCALAR:
            lines += [
                f"    static_assert(std::is_same_v<{typ}, {spec.dtype.to_c_type_string()}>,"
                ' "scalar dtype mismatch");',
                f"    {typ} {name}{{}};",
                f"    uint64_t bits{position} = uint64_t(args[{len(tensors) + scalars[index]}]);",
                # Byte copying avoids numeric conversion and union/aliasing UB,
                # including signed zero and NaN payloads. v1 is little-endian.
                f"    for (unsigned b = 0; b < sizeof({typ}); ++b)",
                f"        reinterpret_cast<unsigned char*>(&{name})[b] = (bits{position} >> (8 * b)) & 255;",
            ]
        elif arg.value == ABIValue.DATA_POINTER:
            lines += [
                f'    static_assert(std::is_pointer_v<{typ}>, "native data argument must be a pointer");',
                f"    {typ} {name} = reinterpret_cast<{typ}>(t{index}.buffer.addr);",
            ]
        else:
            assert isinstance(spec, TensorSpec)
            lines += [
                f"    static_assert(std::is_same_v<{typ}, uint{8 * arg.size_bytes}_t>,"
                ' "metadata must be unsigned");',
                f"    {typ} {name} = static_cast<{typ}>({_metadata(arg, f't{index}', spec)});",
            ]
        arguments.append(name)
    lines += [
        # Atomic enable/type occupies CTRL[8:6]. Snapshot only after resetting
        # it, so both the cube initialization and exit restore keep it disabled.
        "    set_atomic_none();",
        "    uint64_t saved_ctrl = get_ctrl();",
        "#if defined(__DAV_CUBE__)",
        # A5 CTRL bit 46 is MAD_MODE_BIT (HF32). Match the PTO cube wrapper
        # without requiring PTO headers in a native source bundle. AIV leaves
        # the bit alone; both variants restore the post-reset saved state.
        "    // Start cube tasks in FP32 mode, independent of the previous task.",
        "    set_ctrl(saved_ctrl & ~(1ULL << 46));",
        "#endif",
        f"    {symbol}({', '.join(arguments)});",
        # The task scheduler observes entry return. Drain both instruction
        # pipelines and cached writes before restoring the caller's state.
        "    pipe_barrier(PIPE_ALL);",
        "    dcci((__gm__ int64_t*)0, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);",
        "    dsb(DSB_DDR);",
        "    set_atomic_none();",
        "    set_ctrl(saved_ctrl);",
        "}",
        "",
    ]
    return "\n".join(lines)


def _preflight(prepared: PreparedKernel, symbol: str) -> str:
    tensors, scalars = _slots(prepared)
    lines = [
        "#include <cstdint>",
        '#include "assert_compat.h"',
        f"template<class Args> inline void {symbol}(const Args& args) {{",
    ]

    def check(condition: str, message: str) -> None:
        lines.append(f"    always_assert(({condition}) && {json.dumps('external binding: ' + message)});")

    check(
        f"args.tensor_count() == {len(tensors)} && args.scalar_count() == {len(scalars)}", "argument counts"
    )
    symbols: dict[str, str] = {}

    def constraint(value: int | str | None, expr: str, label: str) -> None:
        if isinstance(value, int):
            check(f"{expr} == {value}ULL", label)
        elif isinstance(value, str):
            if value in symbols:
                check(f"{expr} == {symbols[value]}", f"{label}: shared symbol {value}")
            else:
                symbols[value] = expr

    for index, slot in tensors.items():
        parameter = prepared.contract.parameters[index]
        spec = parameter.type
        assert isinstance(spec, TensorSpec)
        t, n = f"t{index}", parameter.name
        # OUTPUT slots hold TensorCreateInfo, not a Tensor. Check the tag before
        # touching the union; this bridge requires storage before submission.
        check(f"args.tag({slot}) != TensorArgType::OUTPUT", f"{n}: runtime-allocated outputs unsupported")
        lines.append(f"    const auto& {t} = args.tensor({slot}).ref();")
        check(f"{t}.ndims == {len(spec.shape)}", f"{n}: rank")
        check(f"{t}.dtype == DataType::{_DTYPES[spec.dtype]}", f"{n}: dtype")
        itemsize = spec.dtype.get_byte()
        lines.append(f"    uint64_t end{index} = {t}.start_offset;")
        # Empty tensors have no touched bytes, but pointer arithmetic and native
        # metadata still have to be representable. All strides stay positive.
        empty = " || ".join(f"{t}.shapes[{axis}] == 0" for axis in range(len(spec.shape)))
        lines.append(f"    const bool empty{index} = {empty};")
        for axis, dimension in enumerate(spec.shape):
            constraint(dimension, f"{t}.shapes[{axis}]", f"{n}: dimension {axis}")
            check(f"{t}.strides[{axis}] > 0", f"{n}: positive stride {axis}")
            if spec.strides:
                constraint(spec.strides[axis], f"{t}.strides[{axis}]", f"{n}: stride {axis}")
            lines += [
                f"    if (!empty{index}) {{",
                f"        const uint64_t extent = uint64_t({t}.shapes[{axis}] - 1) * {t}.strides[{axis}];",
            ]
            check(f"end{index} <= UINT64_MAX - extent", f"{n}: extent overflow")
            lines += [f"        end{index} += extent;", "    }"]
        constraint(spec.offset, f"{t}.start_offset", f"{n}: offset")
        check(f"end{index} <= UINT64_MAX / {itemsize} - (!empty{index})", f"{n}: byte extent overflow")
        lines += [
            f"    end{index} = (end{index} + !empty{index}) * {itemsize};",
            f"    const uint64_t begin{index} = {t}.start_offset * {itemsize};",
        ]
        check(f"end{index} <= {t}.buffer.size", f"{n}: storage bounds")
        check(f"{t}.buffer.addr <= UINT64_MAX - end{index}", f"{n}: address overflow")
        check(f"empty{index} || {t}.buffer.addr != 0", f"{n}: null storage")
        # c_pointer passes the allocation base unchanged; the separate offset
        # must not hide a misaligned pointer or impose a view-alignment promise.
        check(f"{t}.buffer.addr % {spec.alignment} == 0", f"{n}: alignment")
    for arg in prepared.native_abi.parameters:
        if arg.value in (ABIValue.DIMENSION, ABIValue.STRIDE, ABIValue.OFFSET):
            spec = prepared.contract.parameters[arg.parameter].type
            assert isinstance(spec, TensorSpec)
            # Bounds above make offset multiplication safe; uint32 shape/stride
            # times the supported element size fits in uint64 without overflow.
            check(
                f"{_metadata(arg, f't{arg.parameter}', spec)} <= {(1 << (8 * arg.size_bytes)) - 1}ULL",
                f"parameter {arg.parameter}: native {arg.value.value} width",
            )
    written = {
        e.parameter for e in prepared.contract.effects if e.access in (Access.WRITE, Access.READ_WRITE)
    }
    permissions = {frozenset((p.first, p.second)) for p in prepared.contract.alias_permissions}
    for first in tensors:
        for second in tensors:
            if (
                first >= second
                or not ({first, second} & written)
                or frozenset((first, second)) in permissions
            ):
                continue
            # Bounding intervals are deliberately conservative for strided
            # views: uncertainty rejects, rather than inventing alias safety.
            check(
                f"empty{first} || empty{second} || "
                f"t{first}.buffer.addr + end{first} <= t{second}.buffer.addr + begin{second} || "
                f"t{second}.buffer.addr + end{second} <= t{first}.buffer.addr + begin{first}",
                f"parameters {first}/{second}: overlap needs alias permission",
            )
    lines += ["}", ""]
    return "\n".join(lines)


class TMRBridge:
    """Lower supported native functions to existing AIC/AIV/MIX task entries."""

    def lower(self, prepared: PreparedKernel, target: RuntimeTarget) -> ExternalTask:
        _validate(prepared, target)
        _validate_abi(prepared)
        symbol = f"pypto_external_check_{prepared_digest(prepared)}"
        files = tuple(_file(f"native/{f.path}", f.content) for f in prepared.artifact.files)
        source = SourceArtifact(
            (*files, _file("bridge.cpp", _wrapper(prepared).encode())),
            "bridge.cpp",
            "kernel_entry",
            tuple(f"native/{p}" for p in prepared.artifact.include_dirs),
        )
        return ExternalTask(source, prepared, target, "tmr-c-pointer-1", _preflight(prepared, symbol), symbol)


@dataclass(frozen=True)
class MaterializedTask:
    """Private source paths valid only for the surrounding materialize() lease."""

    task: ExternalTask
    source: str
    include_dirs: tuple[str, ...]

    def configure_context(self, ctx: "SpecializeContext") -> None:
        """Attach the checked bridge output to the existing specialization path."""
        if ctx.func_type != "extern":
            raise ContractError("TMR bridge: configure_context requires an extern declaration")
        prepared = self.task.prepared
        actual = request_from_context(ctx, prepared.request.target, options=dict(prepared.request.options))
        if actual != prepared.request:
            raise ContractError("TMR bridge: context differs from the prepared request; prepare it again")
        execution = prepared.execution
        ctx.external_core_type = "mixed" if len(execution.cores) == 2 else execution.cores[0].value
        ctx.external_aic_source = self.source if Core.AIC in execution.cores else None
        ctx.external_aiv_source = self.source if Core.AIV in execution.cores else None
        ctx.external_dual_aiv_dispatch = execution.aiv_lanes == 2
        ctx.external_include_dirs = self.include_dirs
        ctx.external_contract = self.task.prepared.contract
        ctx.external_preflight_source = self.task.preflight_source
        ctx.external_preflight_symbol = self.task.preflight_symbol


@contextmanager
def materialize(task: ExternalTask) -> Iterator[MaterializedTask]:
    """Own a private tree until compilation/runtime source publication finishes.

    Only TMRBridge output is accepted: rebuilding it verifies provenance and all
    paths before disk access. A fresh private directory has no inherited links.
    Persistent artifact storage and provider caching are separate concerns.
    """
    if task != TMRBridge().lower(task.prepared, task.target):
        raise ContractError("TMR bridge: materialize requires unmodified bridge output")
    with TemporaryDirectory(prefix="pypto-tmr-bridge-") as directory:
        root = Path(directory)
        for source in task.source.files:
            path = root / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(source.content)
        yield MaterializedTask(
            task, str(root / task.source.entry), tuple(str(root / p) for p in task.source.include_dirs)
        )
