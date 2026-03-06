# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Recursive code generation from body AST.

Generates PTO kernel code lines from a composable body AST tree.
Each node type (OpBlock, ForBlock, IfElseBlock) is handled recursively,
with indentation computed from the nesting depth.

The generated code is loop-agnostic at the body level — tiling mode
(loads + body + store inside a for loop) is handled by the caller
(``kernel_generator._generate_kernel_code``), not by this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ast import BodyNode, ForBlock, IfElseBlock, OpBlock

if TYPE_CHECKING:
    from ..kernel_generator import KernelGenerator

# Indentation unit (4 spaces)
_INDENT = "    "

# Loop variable names for nested for blocks (avoids variable shadowing)
_LOOP_VARS = ("i", "j", "k", "l", "m", "n")


def _loop_var_name(depth: int) -> str:
    """Get loop variable name for the given nesting depth."""
    if depth < len(_LOOP_VARS):
        return _LOOP_VARS[depth]
    return f"i_{depth}"


def generate_body_code(
    gen: KernelGenerator,
    body: list[BodyNode],
    output_shape: tuple[int, int],
    indent_level: int = 2,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate PTO kernel code lines from a body AST.

    Recursively walks the body tree and generates code at the appropriate
    indentation level. Each BodyNode type is dispatched to its handler.

    Args:
        gen: KernelGenerator instance (provides op code generation helpers).
        body: List of BodyNode instances forming the body.
        output_shape: Tile shape (rows, cols) for store operations.
        indent_level: Indentation nesting level (default 2 = 8 spaces).

    Returns:
        Tuple of (code_lines, last_output_var).
    """
    code_lines: list[str] = []
    last_output = ""

    for node in body:
        if isinstance(node, OpBlock):
            lines, last_output = _generate_op_block(gen, node, output_shape, indent_level)
            code_lines.extend(lines)
        elif isinstance(node, ForBlock):
            lines, last_output = _generate_for_block(gen, node, output_shape, indent_level, loop_depth)
            code_lines.extend(lines)
        elif isinstance(node, IfElseBlock):
            lines, last_output = _generate_if_else_block(gen, node, output_shape, indent_level, loop_depth)
            code_lines.extend(lines)

    return code_lines, last_output


def _generate_op_block(
    gen: KernelGenerator,
    node: OpBlock,
    output_shape: tuple[int, int],
    indent_level: int,
) -> tuple[list[str], str]:
    """Generate code lines for an OpBlock (plain operations).

    Handles regular ops, matmul ops (with memory moves), and reduction ops
    (with temporary tiles).
    """
    code_lines: list[str] = []
    has_matmul = any(op_dict["op"].name == "block.matmul" for op_dict in node.op_chain)
    moved_tiles: dict[str, str] = {}
    l0c_vars: set[str] = {
        op_dict["output"] for op_dict in node.op_chain if op_dict["op"].name == "block.matmul"
    }

    for op_dict in node.op_chain:
        op = op_dict["op"]
        if op.name == "block.matmul":
            code_lines.extend(
                gen._generate_matmul_op(
                    op_dict,
                    has_matmul,
                    moved_tiles,
                    l0c_vars,
                    indent_level,
                )
            )
        elif op.constraints.get("requires_tmp_tile", False):
            code_lines.extend(
                gen._generate_reduction_op(
                    op_dict,
                    output_shape,
                    indent_level,
                )
            )
        else:
            code_lines.append(
                gen._generate_regular_op(
                    op_dict,
                    node.scalar_value_to_param,
                    indent_level,
                )
            )

    last_output = node.op_chain[-1]["output"] if node.op_chain else ""
    return code_lines, last_output


def _generate_for_block(
    gen: KernelGenerator,
    node: ForBlock,
    output_shape: tuple[int, int],
    indent_level: int,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate code lines for a ForBlock (for loop with optional accumulation).

    Structure::

        for i in pl.range(N):
            <body at indent_level+1>
            [accum_var = pl.add(last_body_output, accum_var)]
    """
    ind = _INDENT * indent_level
    inner_ind = _INDENT * (indent_level + 1)
    code_lines: list[str] = []

    # For loop header
    loop_var = _loop_var_name(loop_depth)
    code_lines.append(f"{ind}for {loop_var} in pl.range({node.iterations}):")

    # Recurse into body at indent_level + 1, next loop depth
    body_lines, body_last = generate_body_code(gen, node.body, output_shape, indent_level + 1, loop_depth + 1)
    code_lines.extend(body_lines)

    # Accumulation line
    last_output = body_last
    if node.accum_var and node.last_body_output:
        code_lines.append(f"{inner_ind}{node.accum_var} = pl.add({node.last_body_output}, {node.accum_var})")
        last_output = node.accum_var

    return code_lines, last_output


def _generate_if_else_block(
    gen: KernelGenerator,
    node: IfElseBlock,
    output_shape: tuple[int, int],
    indent_level: int,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate code lines for an IfElseBlock.

    The last output of each branch is renamed to ``branch_out`` so both
    branches converge to the same variable (function scoping).

    Structure::

        if branch_cond:
            <then_body at indent_level+1>
            branch_out = <then_last_output>    (via rename)
        else:
            <else_body at indent_level+1>
            branch_out = <else_last_output>    (via rename)
    """
    ind = _INDENT * indent_level
    inner_ind = _INDENT * (indent_level + 1)
    code_lines: list[str] = []

    # Then-branch
    code_lines.append(f"{ind}if branch_cond:")
    then_lines, then_last = generate_body_code(
        gen, node.then_body, output_shape, indent_level + 1, loop_depth
    )
    then_lines = _rename_last_output(then_lines, then_last, "branch_out")
    if not then_lines:
        code_lines.append(f"{inner_ind}pass")
    else:
        code_lines.extend(then_lines)

    # Else-branch
    code_lines.append(f"{ind}else:")
    else_lines, else_last = generate_body_code(
        gen, node.else_body, output_shape, indent_level + 1, loop_depth
    )
    else_lines = _rename_last_output(else_lines, else_last, "branch_out")
    if not else_lines:
        code_lines.append(f"{inner_ind}pass")
    else:
        code_lines.extend(else_lines)

    return code_lines, "branch_out"


def _rename_last_output(
    lines: list[str],
    old_name: str,
    new_name: str,
) -> list[str]:
    """Rename the last output assignment in code lines.

    Finds the last line containing ``old_name =`` and renames it to
    ``new_name =``. This ensures both branches of an if/else converge
    to the same output variable.
    """
    if not lines or not old_name or old_name == new_name:
        return lines

    # Find the last line that assigns to old_name
    result = list(lines)
    for i in range(len(result) - 1, -1, -1):
        if f"{old_name} =" in result[i]:
            result[i] = result[i].replace(f"{old_name} =", f"{new_name} =", 1)
            break

    return result
