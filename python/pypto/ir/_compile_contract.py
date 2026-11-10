# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time checks for the internal borrowed-tensor kernel contract."""

import re
from collections.abc import Sequence
from dataclasses import replace

from pypto.backend._host_tensor_args import host_tensor_indices
from pypto.pypto_core import ir

from .param_info import (
    ParamInfo,
    _to_torch_dtype,
    block_nz_shape,
    validate_l2_capacity,
    validate_parameter_names,
    validate_return_aliases,
)


def entry_function(program: ir.Program) -> ir.Function:
    entries = [f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration]
    if len(entries) == 1:
        return entries[0]
    functions = list(program.functions.values())
    if not entries and len(functions) == 1:
        return functions[0]
    raise ValueError("Kernel contract requires exactly one single-chip entry")


def return_aliases(program: ir.Program, entry: ir.Function) -> tuple[int, ...] | None:
    # Return aliases describe the logical caller signature, before lowering.
    indices = ir.returned_param_indices(entry, program)
    if len(indices) != len(entry.return_types) or any(i is None for i in indices):
        return None
    parameters = entry.params
    if any(not isinstance(parameters[i].type, ir.TensorType) for i in indices if i is not None):
        return None
    return tuple(i for i in indices if i is not None)


def validate_kernel_parameters(params: Sequence[ParamInfo]) -> None:
    """Validate signature restrictions on compilation and metadata restoration."""
    validate_parameter_names(params)
    validate_l2_capacity(params)
    for param in params:
        if param.shape is None and param.direction != ir.ParamDirection.In:
            raise ValueError(f"Kernel scalar {param.name!r} must have direction In")


def validate_kernel_program(program: ir.Program, params: Sequence[ParamInfo]) -> tuple[int, ...]:
    if any(f.level is not None and ir.level_to_linqu_level(f.level) >= 3 for f in program.functions.values()):
        raise ValueError("Kernel contract requires a single chip, not a distributed program")
    entry = entry_function(program)
    validate_kernel_parameters(params)
    aliases = return_aliases(program, entry)
    if aliases is None:
        raise ValueError("Kernel returns must alias external tensor parameters; supply explicit Out/InOut")
    validate_return_aliases(params, aliases)
    return aliases


def validate_kernel_host_access(program: ir.Program) -> None:
    # Run on lowered entry code: submit predicates stay on the device, while
    # tensor.read/write in orchestration would require a host transfer.
    if host_tensor_indices(entry_function(program)):
        raise ValueError("Kernel contract forbids HBG Host tensor data access")


def external_parameter_names(logical: Sequence[ParamInfo], lowered: Sequence[ParamInfo]) -> list[ParamInfo]:
    """Restore caller names after SSA versioning, preserving order checks.

    SSA appends a version to the same external slot. Other renames and reordered
    slots remain visible to contract validation; new program outputs retain their
    generated names. Assembly binds slots, not the compiler's SSA identifiers.
    """
    result = list(lowered)
    for index, (before, after) in enumerate(zip(logical, lowered)):
        if re.fullmatch(re.escape(before.name) + r"__ssa_v\d+", after.name):
            result[index] = replace(after, name=before.name)
    return result


def validate_lowered_signature(logical: Sequence[ParamInfo], lowered: Sequence[ParamInfo]) -> None:
    """Prove assembly still consumes the caller's external buffers and scalars.

    NZ is a reshape of unpadded storage in element units; logical carrier
    dtype is unchanged, hence equal element counts also imply equal bytes.
    Dynamic NZ permits only the batch dimension supported by codegen.
    """
    validate_l2_capacity(lowered)
    if len(logical) != len(lowered):
        raise ValueError("Lowering changed external parameter count; kernel requires explicit Out/InOut")
    for before, after in zip(logical, lowered, strict=True):
        if (before.name, before.direction, before.dtype) != (after.name, after.direction, after.dtype):
            raise ValueError(f"Lowering changed external parameter {before.name!r}")
        expected = before.shape
        if before.layout == "NZ" and before.shape is not None:
            dtype = _to_torch_dtype(before.dtype)
            if dtype is None:
                raise ValueError(f"Unsupported NZ dtype for {before.name!r}: {before.dtype}")
            if any(d < 0 for d in before.shape) and not (
                len(before.shape) == 3 and before.shape[0] == -1 and all(d > 0 for d in before.shape[1:])
            ):
                raise ValueError(f"Unsupported dynamic NZ shape {before.shape}")
            expected = tuple(block_nz_shape(before.shape, dtype))
        if expected != after.shape or before.layout != after.layout:
            raise ValueError(
                f"Lowering changed external storage for {before.name!r}: {before.shape} -> {after.shape}"
            )


def analyze_contract(
    program: ir.Program, contract: object
) -> tuple[list[ParamInfo] | None, tuple[int, ...] | None]:
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415

    from .param_info import _extract_func_param_infos  # noqa: PLC0415

    if type(contract) is not ArtifactExecutionMode:
        raise TypeError(f"Expected ArtifactExecutionMode, got {contract!r}")
    if contract is ArtifactExecutionMode.KERNEL:
        logical, _, _ = _extract_func_param_infos(entry_function(program))
        return logical, validate_kernel_program(program, logical)
    try:
        entry = entry_function(program)
        logical, _, _ = _extract_func_param_infos(entry)
        return logical, return_aliases(program, entry)
    except (ValueError, TypeError):
        return None, None
