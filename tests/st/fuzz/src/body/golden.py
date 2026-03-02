# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Recursive Torch golden reference generation from body AST.

Generates per-kernel Torch reference functions that model the expected
output for each kernel, supporting arbitrary nesting of control flow.

The golden code uses an ``env`` dict pattern::

    env = {}
    env['tile_a'] = a.clone()
    env['tmp_0'] = env['tile_a'] + env['tile_b']
    if branch_cond:
        env['tmp_1'] = torch.relu(env['tmp_0'])
        env['branch_out'] = env['tmp_1']
    else:
        ...
    for _i in range(N):
        env['tmp_2'] = env['branch_out'] * env['tile_a']
        env['accum'] = env['tmp_2'] + env['accum']
"""

from __future__ import annotations

from typing import Any

from ..golden_generator import TORCH_OP_MAP
from .ast import BodyNode, ForBlock, IfElseBlock, OpBlock
from .generator import body_needs_branch_cond

# Indentation unit (4 spaces)
_INDENT = "    "

# Loop variable names for nested for blocks (avoids variable shadowing)
_LOOP_VARS = ("i", "j", "k", "l", "m", "n")


def _golden_loop_var(depth: int) -> str:
    """Get golden loop variable name (underscore-prefixed) for the given depth."""
    if depth < len(_LOOP_VARS):
        return f"_{_LOOP_VARS[depth]}"
    return f"_i_{depth}"


def generate_body_golden_lines(
    kernel: dict[str, Any],
    body: list[BodyNode],
) -> list[str]:
    """Generate Torch golden reference function for a kernel with body AST.

    Handles three top-level modes based on ``for_loop_info``:
    - No loop: single-pass body execution
    - Tiling: for-loop iterating over tile slices
    - Non-tiling (accumulation-compatible): handled by body AST's ForBlocks

    Args:
        kernel: Kernel metadata dict.
        body: Body AST nodes.

    Returns:
        List of code lines for the golden function.
    """
    kernel_name = kernel["name"]
    input_names = [inp[0] for inp in kernel["inputs"]]
    loop_info = kernel.get("for_loop_info", {"iterations": 0, "tiling": False})
    iterations = loop_info["iterations"]
    use_tiling = loop_info["tiling"]
    tile_rows, tile_cols = kernel.get("tile_shape", kernel["output_shape"])

    has_branch = body_needs_branch_cond(body)

    # Function signature
    params = list(input_names)
    if has_branch:
        params.append("branch_cond")

    code_lines: list[str] = []
    code_lines.append(f"    def _torch_{kernel_name}({', '.join(params)}):")
    code_lines.append(f'        """Torch reference for {kernel_name}"""')

    if iterations > 0 and use_tiling:
        _generate_tiling_golden(
            code_lines,
            input_names,
            body,
            iterations,
            tile_rows,
            tile_cols,
        )
    else:
        _generate_plain_golden(code_lines, input_names, body)

    code_lines.append("")
    return code_lines


def _generate_plain_golden(
    code_lines: list[str],
    input_names: list[str],
    body: list[BodyNode],
) -> None:
    """Generate golden code for non-tiling mode (single pass or body-managed loops)."""
    code_lines.append("        env = {}")
    for name in input_names:
        code_lines.append(f"        env['tile_{name}'] = {name}.clone()")
    code_lines.append("")

    body_lines, last_output = generate_body_golden(body, indent_level=2, loop_depth=0)
    code_lines.extend(body_lines)
    code_lines.append(f"        return env['{last_output}']")


def _generate_tiling_golden(
    code_lines: list[str],
    input_names: list[str],
    body: list[BodyNode],
    iterations: int,
    tile_rows: int,
    tile_cols: int,
) -> None:
    """Generate golden code for tiling mode (for loop over tile slices)."""
    full_rows = iterations * tile_rows
    code_lines.append(f"        _tile_rows = {tile_rows}")
    code_lines.append(f"        _output = torch.zeros(({full_rows}, {tile_cols}), dtype=torch.float32)")
    code_lines.append(f"        for _i in range({iterations}):")
    code_lines.append("            env = {}")
    for name in input_names:
        code_lines.append(f"            env['tile_{name}'] = {name}[_i*_tile_rows:(_i+1)*_tile_rows, :]")
    code_lines.append("")

    body_lines, last_output = generate_body_golden(body, indent_level=3, loop_depth=1)
    code_lines.extend(body_lines)
    code_lines.append(f"            _output[_i*_tile_rows:(_i+1)*_tile_rows, :] = env['{last_output}']")
    code_lines.append("        return _output")


def generate_body_golden(
    body: list[BodyNode],
    indent_level: int = 2,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate Torch golden reference lines from a body AST.

    Recursively walks the body tree and generates golden code using
    the ``env`` dict pattern.

    Args:
        body: List of BodyNode instances.
        indent_level: Indentation nesting level.

    Returns:
        Tuple of (code_lines, last_output_var).
    """
    code_lines: list[str] = []
    last_output = ""

    for node in body:
        if isinstance(node, OpBlock):
            lines, last_output = _golden_op_block(node, indent_level)
            code_lines.extend(lines)
        elif isinstance(node, ForBlock):
            lines, last_output = _golden_for_block(node, indent_level, loop_depth)
            code_lines.extend(lines)
        elif isinstance(node, IfElseBlock):
            lines, last_output = _golden_if_else_block(node, indent_level, loop_depth)
            code_lines.extend(lines)

    return code_lines, last_output


def _golden_op_block(
    node: OpBlock,
    indent_level: int,
) -> tuple[list[str], str]:
    """Generate golden code for an OpBlock."""
    ind = _INDENT * indent_level
    code_lines: list[str] = []

    for op_dict in node.op_chain:
        op = op_dict["op"]
        inputs = op_dict["inputs"]
        output = op_dict["output"]

        input_vals = []
        for inp in inputs:
            if inp.startswith(("tile_", "tmp_", "branch_", "pre_")):
                input_vals.append(f"env['{inp}']")
            else:
                input_vals.append(inp)

        if op.np_equivalent:
            torch_expr = _get_torch_operation(op.name, input_vals)
            code_lines.append(f"{ind}env['{output}'] = {torch_expr}")

    last_output = node.op_chain[-1]["output"] if node.op_chain else ""
    return code_lines, last_output


def _golden_for_block(
    node: ForBlock,
    indent_level: int,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate golden code for a ForBlock."""
    ind = _INDENT * indent_level
    inner_ind = _INDENT * (indent_level + 1)
    code_lines: list[str] = []

    loop_var = _golden_loop_var(loop_depth)
    code_lines.append(f"{ind}for {loop_var} in range({node.iterations}):")

    # Recurse into body at indent_level + 1, next loop depth
    body_lines, body_last = generate_body_golden(node.body, indent_level + 1, loop_depth + 1)
    code_lines.extend(body_lines)

    # Accumulation line
    last_output = body_last
    if node.accum_var and node.last_body_output:
        code_lines.append(
            f"{inner_ind}env['{node.accum_var}'] = env['{node.last_body_output}'] + env['{node.accum_var}']"
        )
        last_output = node.accum_var

    return code_lines, last_output


def _golden_if_else_block(
    node: IfElseBlock,
    indent_level: int,
    loop_depth: int = 0,
) -> tuple[list[str], str]:
    """Generate golden code for an IfElseBlock."""
    ind = _INDENT * indent_level
    inner_ind = _INDENT * (indent_level + 1)
    code_lines: list[str] = []

    # Then-branch
    code_lines.append(f"{ind}if branch_cond:")
    then_lines, then_last = generate_body_golden(node.then_body, indent_level + 1, loop_depth)
    if then_lines:
        code_lines.extend(then_lines)
        if then_last:
            code_lines.append(f"{inner_ind}env['branch_out'] = env['{then_last}']")
    else:
        code_lines.append(f"{inner_ind}pass")

    # Else-branch
    code_lines.append(f"{ind}else:")
    else_lines, else_last = generate_body_golden(node.else_body, indent_level + 1, loop_depth)
    if else_lines:
        code_lines.extend(else_lines)
        if else_last:
            code_lines.append(f"{inner_ind}env['branch_out'] = env['{else_last}']")
    else:
        code_lines.append(f"{inner_ind}pass")

    return code_lines, "branch_out"


def _get_torch_operation(op_name: str, input_vals: list[str]) -> str:
    """Convert a PyPTO op name to a Torch expression string."""
    op_func = TORCH_OP_MAP.get(op_name)
    if op_func:
        return op_func(input_vals)
    return f"# Unsupported operation: {op_name}"
