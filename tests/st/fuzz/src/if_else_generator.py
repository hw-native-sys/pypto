# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
If/else kernel code generation for fuzz testing.

Generates InCore kernel functions with if/else branching (scf.if).
Each branch contains an independent op chain. The condition is controlled
by a ``branch_cond: pl.Scalar[pl.BOOL]`` parameter read from a config tensor
at runtime.

Generated kernel pattern::

    if branch_cond:
        ...ops...
        result = pl.store(<last_op>, ..., output)
    else:
        ...ops...
        result = pl.store(<last_op>, ..., output)
    return result

The store is placed inside each branch so that no Tile-type variable
crosses the if/else boundary.  The ConvertToSSA pass's phi-node
creation has two limitations that make Tile-type phi variables
unusable at the fuzz level:

1. ``convert_to_ssa_pass.cpp:226-227`` skips phi creation for variables
   not defined before the if/else.
2. ``init_memref.cpp`` does not assign a MemRef to IfStmt return_vars,
   leaving the phi variable's TileType with a DDR memory space that
   the CCE codegen rejects (``type_converter.cpp:76``).

By storing inside each branch, only the ``result`` variable (TensorType)
survives after the if/else.  Both branches write to the same ``output``
tensor, so the then-branch's ``result`` reference is semantically correct
regardless of which branch executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_generator import KernelGenerator


def generate_if_else_kernel_code(
    gen: KernelGenerator,
    kernel_name: str,
    inputs: list[tuple[str, tuple[int, int]]],
    scalars: list[tuple[str, str]],
    output_shape: tuple[int, int],
    scalar_value_to_param: dict[str, str],
    then_chain: list[dict[str, Any]],
    else_chain: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Generate kernel code with if/else branching (if-only mode, no for loop).

    Args:
        gen: KernelGenerator instance (provides op code generation helpers)
        kernel_name: Kernel function name
        inputs: Tensor input list [(name, tile_shape), ...]
        scalars: Scalar parameter list [(param_name, value), ...]
        output_shape: Tile shape (rows, cols)
        scalar_value_to_param: Mapping from scalar values to parameter names
        then_chain: Op chain for the then-branch
        else_chain: Op chain for the else-branch

    Returns:
        Tuple of (generated code string, loop_info dict).
    """
    tile_rows, tile_cols = output_shape

    # Build function signature
    params = []
    for name, (r, c) in inputs:
        params.append(f"{name}: pl.Tensor[[{r}, {c}], pl.FP32]")

    for scalar_name, _ in scalars:
        params.append(f"{scalar_name}: pl.Scalar[pl.FP32]")

    params.append(f"output: pl.Out[pl.Tensor[[{tile_rows}, {tile_cols}], pl.FP32]]")
    # branch_cond scalar for if/else condition (last param, after output)
    params.append("branch_cond: pl.Scalar[pl.BOOL]")

    code_lines = [
        "    @pl.function(type=pl.FunctionType.InCore)",
        f"    def {kernel_name}(self, {', '.join(params)})"
        f" -> pl.Tensor[[{tile_rows}, {tile_cols}], pl.FP32]:",
    ]

    # If/else branches use basic_ops_only, so no matmul or reduction ops.
    # This keeps branch code simple and avoids complex buffer allocation patterns.
    has_matmul = False

    # Generate input loads (no offset — if-only has no loop)
    load_lines = gen._generate_input_loads(inputs, has_matmul, row_offset_expr=None)
    code_lines.extend(load_lines)

    # Generate then-branch (ops + store inside branch)
    code_lines.append("        if branch_cond:")
    then_lines = _build_branch_lines(gen, then_chain, output_shape, scalar_value_to_param)
    code_lines.extend(then_lines)
    code_lines.append("            result = pl.store(branch_out, offsets=[0, 0], output_tensor=output)")

    # Generate else-branch (ops + store inside branch)
    code_lines.append("        else:")
    else_lines = _build_branch_lines(gen, else_chain, output_shape, scalar_value_to_param)
    code_lines.extend(else_lines)
    code_lines.append("            result = pl.store(branch_out, offsets=[0, 0], output_tensor=output)")

    code_lines.append("        return result")

    loop_info = {
        "iterations": 0,
        "tiling": False,
        "split_point": 0,
    }
    return "\n".join(code_lines), loop_info


def _build_branch_lines(
    gen: KernelGenerator,
    op_chain: list[dict[str, Any]],
    output_shape: tuple[int, int],
    scalar_value_to_param: dict[str, str],
    indent: str = "            ",
) -> list[str]:
    """Generate code lines for one branch of an if/else block.

    The last op in the chain gets its output renamed to ``branch_out`` with a
    ``pl.Tile`` type annotation.  ``branch_out`` is only used inside the same
    branch (for the store) and never crosses the if/else boundary.

    Branches use basic_ops_only (no matmul, no reductions requiring create_tile)
    to keep generated branch code simple and predictable.

    Args:
        gen: KernelGenerator instance (provides op code generation helpers)
        op_chain: Operation chain for this branch
        output_shape: Tile shape (rows, cols)
        scalar_value_to_param: Mapping from scalar values to parameter names
        indent: Indentation prefix for each generated line

    Returns:
        List of indented code lines for the branch body
    """
    if not op_chain:
        return []

    tile_rows, tile_cols = output_shape
    lines: list[str] = []

    for idx, op_dict in enumerate(op_chain):
        is_last = idx == len(op_chain) - 1

        # Only regular ops in if/else branches (basic_ops_only)
        op_lines = [gen._generate_regular_op(op_dict, scalar_value_to_param)]

        if is_last:
            # Replace output variable with branch_out + type annotation
            last_output = op_dict["output"]
            for line in op_lines:
                replaced = line.replace(
                    f"        {last_output} =",
                    f"        branch_out: pl.Tile[[{tile_rows}, {tile_cols}], pl.FP32] =",
                )
                lines.append(indent + replaced.lstrip())
        else:
            for line in op_lines:
                lines.append(indent + line.lstrip())

    return lines


def generate_if_else_golden_lines(
    kernel: dict[str, Any],
) -> list[str]:
    """Generate Torch reference function code for an if/else kernel.

    The golden function takes ``branch_cond`` as an extra parameter and
    computes the then-branch or else-branch accordingly.

    Args:
        kernel: Kernel metadata dict with ``if_else_info``.

    Returns:
        List of code lines for the golden function.
    """
    from .golden_generator import _build_op_lines  # noqa: PLC0415 (circular import)

    kernel_name = kernel["name"]
    input_names = [inp[0] for inp in kernel["inputs"]]
    if_else_info = kernel["if_else_info"]
    then_chain = if_else_info["then_chain"]
    else_chain = if_else_info["else_chain"]

    code_lines: list[str] = []
    code_lines.append(f"    def _torch_{kernel_name}({', '.join(input_names)}, branch_cond):")
    code_lines.append(f'        """Torch reference for {kernel_name} (if/else)"""')
    code_lines.append("        env = {}")
    for name in input_names:
        code_lines.append(f"        env['tile_{name}'] = {name}.clone()")
    code_lines.append("")

    # Then-branch
    then_op_lines, then_last = _build_op_lines(then_chain)
    code_lines.append("        if branch_cond:")
    for line in then_op_lines:
        code_lines.append("    " + line)
    code_lines.append(f"            env['branch_out'] = env['{then_last}']")

    # Else-branch
    else_op_lines, else_last = _build_op_lines(else_chain)
    code_lines.append("        else:")
    for line in else_op_lines:
        code_lines.append("    " + line)
    code_lines.append(f"            env['branch_out'] = env['{else_last}']")

    code_lines.append("        return env['branch_out']")
    code_lines.append("")
    return code_lines
