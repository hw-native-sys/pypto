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
for each kernel mode (no loop, tiling, middle).
"""

from typing import Any

from .if_else_generator import generate_if_else_golden_lines

# PyPTO op name -> torch expression builder
TORCH_OP_MAP: dict[str, Any] = {
    # Binary arithmetic operations (scalar second arg uses same expression via parser auto-dispatch)
    "tile.add": lambda v: f"{v[0]} + {v[1]}",
    "tile.sub": lambda v: f"{v[0]} - {v[1]}",
    "tile.mul": lambda v: f"{v[0]} * {v[1]}",
    "tile.div": lambda v: f"{v[0]} / {v[1]}",
    # Binary comparison operations
    "tile.maximum": lambda v: f"torch.maximum({v[0]}, {v[1]})",
    "tile.minimum": lambda v: f"torch.minimum({v[0]}, {v[1]})",
    # Unary operations
    "tile.sqrt": lambda v: f"torch.sqrt({v[0]})",
    "tile.rsqrt": lambda v: f"torch.rsqrt({v[0]})",
    "tile.exp": lambda v: f"torch.exp({v[0]})",
    "tile.neg": lambda v: f"-{v[0]}",
    "tile.recip": lambda v: f"torch.reciprocal({v[0]})",
    "tile.log": lambda v: f"torch.log({v[0]})",
    "tile.abs": lambda v: f"torch.abs({v[0]})",
    "tile.relu": lambda v: f"torch.relu({v[0]})",
    # Row expand operations (broadcast [M, 1] to [M, N])
    "tile.row_expand_add": lambda v: f"{v[0]} + {v[1]}",
    "tile.row_expand_sub": lambda v: f"{v[0]} - {v[1]}",
    "tile.row_expand_mul": lambda v: f"{v[0]} * {v[1]}",
    "tile.row_expand_div": lambda v: f"{v[0]} / {v[1]}",
    # Row reduction operations (produce [M, 1] output)
    "tile.row_sum": lambda v: f"torch.sum({v[0]}, dim=1, keepdim=True)",
    "tile.row_max": lambda v: f"torch.max({v[0]}, dim=1, keepdim=True)[0]",
    "tile.row_min": lambda v: f"torch.min({v[0]}, dim=1, keepdim=True)[0]",
    # Column expand operations (broadcast [1, N] to [M, N])
    "tile.col_expand_mul": lambda v: f"{v[0]} * {v[1]}",
    "tile.col_expand_div": lambda v: f"{v[0]} / {v[1]}",
    "tile.col_expand_sub": lambda v: f"{v[0]} - {v[1]}",
    # Column reduction operations (produce [1, N] output)
    "tile.col_sum": lambda v: f"torch.sum({v[0]}, dim=0, keepdim=True)",
    "tile.col_max": lambda v: f"torch.max({v[0]}, dim=0, keepdim=True)[0]",
    "tile.col_min": lambda v: f"torch.min({v[0]}, dim=0, keepdim=True)[0]",
    # Matrix operations
    "tile.matmul": lambda v: f"torch.matmul({v[0]}, {v[1]})",
}


def get_torch_operation(op_name: str, input_vals: list[str]) -> str:
    """Convert a PyPTO op name to a Torch expression string.

    Args:
        op_name: PyPTO operation name (e.g., "tile.add")
        input_vals: Input value expression strings (e.g., ["env['tmp_0']", "env['tmp_1']"])

    Returns:
        Torch expression string, or a comment if unsupported.
    """
    op_func = TORCH_OP_MAP.get(op_name)
    if op_func:
        return op_func(input_vals)
    return f"# Unsupported operation: {op_name}"


def _build_op_lines(op_chain: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Build torch expression lines for each op in the chain.

    Args:
        op_chain: Operation chain from kernel metadata.

    Returns:
        Tuple of (list of code lines, name of the last output variable).
    """
    op_lines = []
    for op_dict in op_chain:
        op = op_dict["op"]
        inputs = op_dict["inputs"]
        output = op_dict["output"]
        input_vals = []
        for inp in inputs:
            if inp.startswith("tile_") or inp.startswith("tmp_"):
                input_vals.append(f"env['{inp}']")
            else:
                input_vals.append(inp)
        if op.np_equivalent:
            torch_expr = get_torch_operation(op.name, input_vals)
            op_lines.append(f"        env['{output}'] = {torch_expr}")
    last_output = op_chain[-1]["output"]
    return op_lines, last_output


def generate_kernel_torch_ref(kernel: dict[str, Any]) -> list[str]:
    """Generate Torch reference function code lines for a single kernel.

    Four modes are supported:
    - If/else (if_else_info present): branch based on branch_cond parameter.
    - No loop (iterations == 0): process the full input tensor in one pass.
    - Tiling mode (use_tiling=True): loop over N input tiles, each processed
      independently; output tiles are assembled into the full output tensor.
    - Middle mode (use_tiling=False): compute from the first input tile once,
      then broadcast the result to N output tiles via torch.cat.

    Args:
        kernel: Kernel metadata dict (from KernelGenerator).

    Returns:
        List of code lines (indented with 4 spaces for embedding in a class body).
    """
    # If/else kernel: delegate to if_else_generator
    if_else_info = kernel.get("if_else_info")
    if if_else_info and if_else_info.get("enabled"):
        return generate_if_else_golden_lines(kernel)

    kernel_name = kernel["name"]
    input_names = [inp[0] for inp in kernel["inputs"]]
    op_chain = kernel["op_chain"]
    loop_info = kernel.get("for_loop_info", {"iterations": 0, "tiling": False})
    iterations = loop_info["iterations"]
    use_tiling = loop_info["tiling"]
    tile_rows, tile_cols = kernel.get("tile_shape", kernel["output_shape"])

    op_lines, last_output = _build_op_lines(op_chain)

    code_lines: list[str] = []
    code_lines.append(f"    def _torch_{kernel_name}({', '.join(input_names)}):")
    code_lines.append(f'        """Torch reference for {kernel_name}"""')

    if iterations > 0 and use_tiling:
        # Tiling: each iteration reads a different input tile and writes to the
        # corresponding output tile.
        full_rows = iterations * tile_rows
        code_lines.append(f"        _tile_rows = {tile_rows}")
        code_lines.append(f"        _output = torch.zeros(({full_rows}, {tile_cols}), dtype=torch.float32)")
        code_lines.append(f"        for _i in range({iterations}):")
        code_lines.append("            env = {}")
        for name in input_names:
            code_lines.append(f"            env['tile_{name}'] = {name}[_i*_tile_rows:(_i+1)*_tile_rows, :]")
        code_lines.append("")
        for line in op_lines:
            code_lines.append("    " + line)  # extra indent inside the for loop
        code_lines.append(f"            _output[_i*_tile_rows:(_i+1)*_tile_rows, :] = env['{last_output}']")
        code_lines.append("        return _output")

    elif iterations > 0 and not use_tiling:
        # Accumulation mode: load first tile, run all ops (pre-loop + loop body).
        # Since there is no loop-carried state, all loop iterations produce the same
        # result — the golden simply runs the ops once and returns the single tile.
        code_lines.append(f"        _tile_rows = {tile_rows}")
        code_lines.append("        env = {}")
        for name in input_names:
            code_lines.append(f"        env['tile_{name}'] = {name}[:_tile_rows, :]")
        code_lines.append("")
        code_lines.extend(op_lines)
        code_lines.append(f"        return env['{last_output}']")

    else:
        # No loop: process the full tensor in one pass.
        code_lines.append("        env = {}")
        for name in input_names:
            code_lines.append(f"        env['tile_{name}'] = {name}.clone()")
        code_lines.append("")
        code_lines.extend(op_lines)
        code_lines.append(f"        return env['{last_output}']")

    code_lines.append("")
    return code_lines
