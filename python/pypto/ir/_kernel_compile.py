# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal kernel artifact producer; public compile() remains a program entry."""

import math
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pypto._kernel_abi import KernelABI, validate_kernel_signature
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Function, FunctionType, Program, level_to_linqu_level

from .compiled_program import _extract_func_param_infos, write_kernel_metadata
from .param_info import _ParamInfo, kernel_abi_from_params, kernel_parameters_from_params

if TYPE_CHECKING:
    from pypto.runtime._kernel_artifact import KernelArtifact


def _entry(program: Program) -> Function:
    """Select one chip entry; distributed and ambiguous entry sets are unsupported."""
    functions = list(program.functions.values())
    if any(f.level is not None and level_to_linqu_level(f.level) >= 3 for f in functions):
        raise ValueError("Kernel artifacts require a single chip entry, not a distributed program")
    entries = [f for f in functions if f.func_type == FunctionType.Orchestration]
    if not entries and len(functions) == 1:
        entries = functions
    if len(entries) != 1:
        raise ValueError("Kernel artifacts require exactly one orchestration entry")
    return entries[0]


def kernel_signature_for_program(program: Program) -> tuple[list[_ParamInfo], tuple[int, ...]]:
    """Derive external parameters and return aliases without choosing an execution target.

    Dispatcher schemas depend only on this signature, so registration can run
    before ``pypto.torch.init`` binds a platform and runtime.
    """
    entry = _entry(program)
    params, _, returns = _extract_func_param_infos(entry)
    aliases = codegen._returned_param_indices(entry, program)
    if len(aliases) != len(returns) or any(index is None for index in aliases):
        raise ValueError(
            "Kernel returns must alias external tensor parameters; pass every Out/InOut explicitly"
        )
    _, checked = validate_kernel_signature(
        kernel_parameters_from_params(params), [i for i in aliases if i is not None]
    )
    return params, checked


def kernel_abi_for_program(program: Program, *, platform: str, runtime: str) -> KernelABI:
    """Derive logical pools and external return aliases before lowering rewrites IR."""
    params, aliases = kernel_signature_for_program(program)
    return kernel_abi_from_params(params, platform=platform, runtime=runtime, return_aliases=aliases)


def validate_hbg_kernel_orchestration(program: Program) -> None:
    """Reject Host tensor-data access, including reachable orchestration helpers.

    Walk each Host function once, before optimization can erase branches. Device
    function bodies and in-core scopes are excluded; their call arguments are
    still evaluated by Host orchestration. This is O(nodes + call edges).
    """
    functions = {func.name: func for func in program.functions.values()}
    pending = [_entry(program)]
    visited: set[str] = set()
    device_types = {FunctionType.InCore, FunctionType.AIC, FunctionType.AIV, FunctionType.Group}
    read_op = _ir.get_op("tensor.read").name
    write_op = _ir.get_op("tensor.write").name

    class HostAccessVisitor(_ir.IRVisitor):
        def __init__(self, function_name: str) -> None:
            super().__init__()
            self.function_name = function_name
            self.expressions: list[_ir.Expr] = []
            self.statements: list[_ir.Stmt] = []
            self.predicates: list[_ir.Expr] = []

        def run(self, body: _ir.Stmt) -> None:
            # Defer recursive dispatch until the Python override returns.
            # Nanobind suppresses re-entry into an active visitor override.
            self.statements.append(body)
            while self.expressions or self.statements or self.predicates:
                if self.expressions:
                    self.visit_expr(self.expressions.pop())
                elif self.predicates:
                    self.visit_predicate(self.predicates.pop())
                else:
                    self.visit_stmt(self.statements.pop())

        def check_call(self, op: _ir.Call | _ir.Submit) -> None:
            if isinstance(op.op, _ir.GlobalVar):
                callee = functions.get(op.op.name)
                if callee is not None:
                    pending.append(callee)
                return
            if op.op.name not in (read_op, write_op):
                return
            guidance = (
                "Pass the required Host value as an explicit scalar argument."
                if op.op.name == read_op
                else "Express the update as a Device task."
            )
            raise ValueError(
                f"{op.span.to_string()}: HBG kernel Host orchestration "
                f"'{self.function_name}' cannot use {op.op.name} on Tensor storage. {guidance}"
            )

        def visit_call(self, op: _ir.Call) -> None:
            self.check_call(op)
            self.expressions.extend(op.args)
            self.visit_attributes(op.attrs)

        def visit_submit(self, op: _ir.Submit) -> None:
            self.check_call(op)
            self.expressions.extend(op.args)
            self.expressions.extend(op.deps)
            if op.core_num is not None:
                self.expressions.append(op.core_num)
            if op.predicate is not None:
                self.predicates.append(op.predicate)
            self.visit_attributes(op.attrs)

        def visit_predicate(self, predicate: _ir.Expr) -> None:
            host = self

            class DevicePredicateVisitor(_ir.IRVisitor):
                def visit_call(self, op: _ir.Call) -> None:
                    # EmitPredicateHint encodes this read as Device scheduler
                    # metadata, not get_tensor_data. Its indices are Host values.
                    if not isinstance(op.op, _ir.GlobalVar) and op.op.name == read_op:
                        host.expressions.extend(op.args)
                    else:
                        host.expressions.append(op)

            DevicePredicateVisitor().visit_expr(predicate)

        def visit_attributes(self, attrs: Mapping[str, object]) -> None:
            for key, value in attrs.items():
                if key == "predicate" and isinstance(value, _ir.Expr):
                    self.predicates.append(value)
                else:
                    self.visit_attribute_value(value)

        def visit_attribute_value(self, value: object) -> None:
            if isinstance(value, _ir.Expr):
                self.expressions.append(value)
            elif isinstance(value, (list, tuple)):
                for element in value:
                    self.visit_attribute_value(element)

        def visit_in_core_scope_stmt(self, op: _ir.InCoreScopeStmt) -> None:
            # Only launch inputs run on Host; the body runs on Device.
            self.visit_attributes(op.attrs)

        def visit_hierarchy_scope_stmt(self, op: _ir.HierarchyScopeStmt) -> None:
            self.visit_attributes(op.attrs)
            if level_to_linqu_level(op.level) > 1:
                self.statements.append(op.body)

        def visit_spmd_scope_stmt(self, op: _ir.SpmdScopeStmt) -> None:
            self.expressions.append(op.core_num)
            self.visit_attributes(op.attrs)
            self.statements.append(op.body)

    while pending:
        func = pending.pop()
        if func.name in visited or func.func_type in device_types:
            continue
        visited.add(func.name)
        HostAccessVisitor(func.name).run(func.body)


def _external_pool_key(param: "KernelParameter") -> tuple:
    """Identity of one externally supplied pool: dtype, direction, element count.

    The shape itself is deliberately not part of the key. A lowering pass may
    re-describe the same buffer with a different shape: ``BlockNzTensorViews``
    rewrites an ``NZ`` tensor's GM view from its logical shape to the rank-5
    blocked shape ``[C/c0, R/16, 16, c0]`` (e.g. ``(8, 1024, 4096)`` becomes
    ``(8, 256, 64, 16, 16)``), which is the same memory, the same byte count, at
    the same pool index. What this check exists to catch is lowering *introducing,
    dropping or reordering* the external pools, and that is still caught: the
    element count, dtype, direction and position all have to line up.

    The caller-facing contract stays the logical shape, because ``abi`` is derived
    from the original program and that is the shape the framework passes tensors
    in. A dynamic extent falls back to a per-dimension comparison, since an
    element count cannot be formed from it.
    """
    if param.shape is None:
        return (param.dtype, param.direction, None)
    if -1 in param.shape:
        return (param.dtype, param.direction, tuple(param.shape))
    return (param.dtype, param.direction, math.prod(param.shape))


def finish_kernel_artifact(
    original: Program, lowered: Program, directory: Path, abi: KernelABI
) -> "KernelArtifact":
    """Stamp freshly generated code only after its chip signature matches the contract."""
    from pypto.runtime._artifact_sources import read_kernel_config  # noqa: PLC0415
    from pypto.runtime._kernel_artifact import KernelArtifact, validate_kernel_config  # noqa: PLC0415

    kernel_abi_for_program(original, platform=abi.platform, runtime=abi.runtime).require_compatible(abi)
    # A lowered entry may not introduce or reorder the externally supplied pools.
    params, _, _ = _extract_func_param_infos(_entry(lowered))
    actual = kernel_abi_from_params(
        params, platform=abi.platform, runtime=abi.runtime, return_aliases=abi.return_aliases
    )
    expected = [_external_pool_key(p) for p in abi.parameters]
    if [_external_pool_key(p) for p in actual.parameters] != expected:
        # Name what actually moved: a bare "the ABI changed" leaves the caller to
        # guess which parameter, and the shape-only cases (a pass re-describing the
        # same buffer) look identical to a genuinely reordered pool.
        detail = []
        if len(actual.parameters) != len(abi.parameters):
            detail.append(f"parameter count {len(abi.parameters)} -> {len(actual.parameters)}")
        for index, (want, got) in enumerate(zip(abi.parameters, actual.parameters)):
            if (want.dtype, want.direction, want.shape) != (got.dtype, got.direction, got.shape):
                detail.append(
                    f"#{index} {want.name!r}: {want.dtype}/{want.direction}/{want.shape}"
                    f" -> {got.name!r}: {got.dtype}/{got.direction}/{got.shape}"
                )
        raise ValueError("Lowering changed the kernel entry parameter ABI: " + "; ".join(detail))
    config = read_kernel_config(directory / "kernel_config.py")
    validate_kernel_config(config, abi)
    source = Path(config.ORCHESTRATION["source"])
    if directory.resolve() not in source.resolve().parents:
        raise ValueError("Kernel orchestration must be generated inside its private artifact directory")
    # Simpler converts the wire ChipStorageTaskArgs to ChipTaskArgs before entry.
    # Both runtime modes use this function type; keep a distinct binary contract.
    tag = json.dumps(abi.binary_tag().decode("ascii").rstrip("\0"))
    with source.open("a") as stream:
        stream.write(
            "\n#include <type_traits>\n"
            "static_assert(std::is_same_v<decltype(&aicpu_orchestration_entry), "
            "void (*)(const ChipTaskArgs&)>);\n"
            'extern "C" __attribute__((visibility("default")))\n'
            f"const char* pypto_kernel_abi_descriptor() {{ return {tag}; }}\n"
        )
    logical_params, _, _ = _extract_func_param_infos(_entry(original))
    write_kernel_metadata(directory, logical_params, abi)
    return KernelArtifact(directory, abi)
