# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Torch golden reference code generator for fuzz test cases.

Generates per-kernel Torch reference functions that model the expected output
using the composable body AST path.
"""

from typing import Any

# PyPTO op name -> torch expression builder
TORCH_OP_MAP: dict[str, Any] = {
    # Binary arithmetic operations (scalar second arg uses same expression via parser auto-dispatch)
    "block.add": lambda v: f"{v[0]} + {v[1]}",
    "block.sub": lambda v: f"{v[0]} - {v[1]}",
    "block.mul": lambda v: f"{v[0]} * {v[1]}",
    "block.div": lambda v: f"{v[0]} / {v[1]}",
    # Binary comparison operations
    "block.maximum": lambda v: f"torch.maximum({v[0]}, {v[1]})",
    "block.minimum": lambda v: f"torch.minimum({v[0]}, {v[1]})",
    # Unary operations
    "block.sqrt": lambda v: f"torch.sqrt({v[0]})",
    "block.rsqrt": lambda v: f"torch.rsqrt({v[0]})",
    "block.exp": lambda v: f"torch.exp({v[0]})",
    "block.neg": lambda v: f"-{v[0]}",
    "block.recip": lambda v: f"torch.reciprocal({v[0]})",
    "block.log": lambda v: f"torch.log({v[0]})",
    "block.abs": lambda v: f"torch.abs({v[0]})",
    "block.relu": lambda v: f"torch.relu({v[0]})",
    # Row expand operations (broadcast [M, 1] to [M, N])
    "block.row_expand_add": lambda v: f"{v[0]} + {v[1]}",
    "block.row_expand_sub": lambda v: f"{v[0]} - {v[1]}",
    "block.row_expand_mul": lambda v: f"{v[0]} * {v[1]}",
    "block.row_expand_div": lambda v: f"{v[0]} / {v[1]}",
    # Row reduction operations (produce [M, 1] output)
    "block.row_sum": lambda v: f"torch.sum({v[0]}, dim=1, keepdim=True)",
    "block.row_max": lambda v: f"torch.max({v[0]}, dim=1, keepdim=True)[0]",
    "block.row_min": lambda v: f"torch.min({v[0]}, dim=1, keepdim=True)[0]",
    # Column expand operations (broadcast [1, N] to [M, N])
    "block.col_expand_mul": lambda v: f"{v[0]} * {v[1]}",
    "block.col_expand_div": lambda v: f"{v[0]} / {v[1]}",
    "block.col_expand_sub": lambda v: f"{v[0]} - {v[1]}",
    # Column reduction operations (produce [1, N] output)
    "block.col_sum": lambda v: f"torch.sum({v[0]}, dim=0, keepdim=True)",
    "block.col_max": lambda v: f"torch.max({v[0]}, dim=0, keepdim=True)[0]",
    "block.col_min": lambda v: f"torch.min({v[0]}, dim=0, keepdim=True)[0]",
    # Matrix operations
    "block.matmul": lambda v: f"torch.matmul({v[0]}, {v[1]})",
}


def get_torch_operation(op_name: str, input_vals: list[str]) -> str:
    """Convert a PyPTO op name to a Torch expression string.

    Args:
        op_name: PyPTO operation name (e.g., "block.add")
        input_vals: Input value expression strings (e.g., ["env['tmp_0']", "env['tmp_1']"])

    Returns:
        Torch expression string, or a comment if unsupported.
    """
    op_func = TORCH_OP_MAP.get(op_name)
    if op_func:
        return op_func(input_vals)
    return f"# Unsupported operation: {op_name}"


def _build_op_lines(
    op_chain: list[dict[str, Any]],
    indent_level: int = 2,
) -> tuple[list[str], str]:
    """Build torch expression lines for each op in the chain.

    Args:
        op_chain: Operation chain from kernel metadata.
        indent_level: Indentation nesting level (default 2 = 8 spaces).

    Returns:
        Tuple of (list of code lines, name of the last output variable).
    """
    if not op_chain:
        return [], ""
    ind = "    " * indent_level
    op_lines = []
    for op_dict in op_chain:
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
            torch_expr = get_torch_operation(op.name, input_vals)
            op_lines.append(f"{ind}env['{output}'] = {torch_expr}")
    last_output = op_chain[-1]["output"]
    return op_lines, last_output


def generate_kernel_torch_ref(kernel: dict[str, Any]) -> list[str]:
    """Generate Torch reference function code lines for a single kernel.

    Uses the composable body AST path for golden reference generation.

    Args:
        kernel: Kernel metadata dict (from KernelGenerator).

    Returns:
        List of code lines (indented with 4 spaces for embedding in a class body).
    """
    from .body.golden import generate_body_golden_lines  # noqa: PLC0415

    body = kernel.get("body")
    if body is not None:
        return generate_body_golden_lines(kernel, body)

    # Fallback for simple kernels without body AST
    kernel_name = kernel["name"]
    input_names = [inp[0] for inp in kernel["inputs"]]
    op_chain = kernel["op_chain"]

    op_lines, last_output = _build_op_lines(op_chain)

    code_lines: list[str] = []
    code_lines.append(f"    def _torch_{kernel_name}({', '.join(input_names)}):")
    code_lines.append(f'        """Torch reference for {kernel_name}"""')
    code_lines.append("        env = {}")
    for name in input_names:
        code_lines.append(f"        env['tile_{name}'] = {name}.clone()")
    code_lines.append("")
    code_lines.extend(op_lines)
    code_lines.append(f"        return env['{last_output}']")
    code_lines.append("")
    return code_lines
