# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
InCore kernel function generator

This module is responsible for generating @pl.function(type=pl.FunctionType.InCore) kernel functions.
Each kernel contains a chain of randomly generated operator operations.
"""

import random
from typing import Any

from .body.ast import BodyNode, ForBlock, IfElseBlock, OpBlock
from .body.codegen import generate_body_code
from .body.generator import (
    BodyGenerator,
    body_needs_branch_cond,
    collect_all_op_chains,
    collect_all_scalars,
)
from .core.config import ControlFlowConfig
from .core.fuzzer import OpFuzzer
from .core.shape_utils import generate_aligned_shape, is_shape_aligned

# Indentation unit (4 spaces)
_INDENT = "    "


def _generate_tiling_loop(
    iterations: int,
    load_lines: list[str],
    body_lines: list[str],
    store_lines: list[str],
    indent_level: int = 2,
) -> list[str]:
    """Wrap load/body/store lines inside a ``for i in pl.range(N)`` tiling loop."""
    ind = _INDENT * indent_level
    all_body = load_lines + body_lines + store_lines
    code_lines = [f"{ind}for i in pl.range({iterations}):"]
    for line in all_body:
        code_lines.append(_INDENT + line)
    return code_lines


def _body_has_matmul(body: list[BodyNode]) -> bool:
    """Check if any OpBlock in the body tree contains a matmul operation."""
    for node in body:
        if isinstance(node, OpBlock):
            if any(op_dict["op"].name == "tile.matmul" for op_dict in node.op_chain):
                return True
        elif isinstance(node, ForBlock):
            if _body_has_matmul(node.body):
                return True
        elif isinstance(node, IfElseBlock):
            if _body_has_matmul(node.then_body) or _body_has_matmul(node.else_body):
                return True
    return False


def _get_last_op_chain(body: list[BodyNode]) -> list[dict[str, Any]]:
    """Extract the op_chain from the body's last OpBlock (for store generation).

    Follows the last node recursively: for ForBlock, looks at the loop body;
    for IfElseBlock, returns empty (branch_out is not an op chain).
    """
    if not body:
        return []
    last = body[-1]
    if isinstance(last, OpBlock):
        return last.op_chain
    if isinstance(last, ForBlock):
        return _get_last_op_chain(last.body)
    return []  # IfElseBlock → branch_out, no direct op_chain


def _analyze_input_usage(
    body: list[BodyNode], inputs: list[tuple[str, tuple[int, int]]]
) -> dict[str, dict[str, bool]]:
    """Analyze how each input is used in the operation chain.

    Args:
        body: Body AST nodes to analyze
        inputs: List of (name, shape) tuples for kernel inputs

    Returns:
        Dictionary mapping input names to usage flags:
        {
            "a": {"used_in_matmul": True, "used_in_other_ops": False},
            "b": {"used_in_matmul": False, "used_in_other_ops": True},
        }
    """
    # Initialize usage tracking for all inputs
    usage = {name: {"used_in_matmul": False, "used_in_other_ops": False} for name, _ in inputs}
    input_names = {name for name, _ in inputs}

    def _analyze_node(node: BodyNode) -> None:
        """Recursively analyze a body node."""
        if isinstance(node, OpBlock):
            for op_dict in node.op_chain:
                op_name = op_dict["op"].name
                is_matmul = op_name == "tile.matmul"

                # Check inputs used in this operation
                for input_var in op_dict.get("inputs", []):
                    # Extract base input name (e.g., "tile_a" -> "a")
                    base_name = input_var.replace("tile_", "")
                    if base_name in input_names:
                        if is_matmul:
                            usage[base_name]["used_in_matmul"] = True
                        else:
                            usage[base_name]["used_in_other_ops"] = True

        elif isinstance(node, ForBlock):
            for child in node.body:
                _analyze_node(child)

        elif isinstance(node, IfElseBlock):
            for child in node.then_body:
                _analyze_node(child)
            for child in node.else_body:
                _analyze_node(child)

    # Analyze all nodes in body
    for node in body:
        _analyze_node(node)

    return usage


class KernelGenerator:
    """Generator for InCore kernel functions with random operator chains.

    This class generates @pl.function(type=pl.FunctionType.InCore) kernels containing
    chains of randomly selected operators. Each kernel includes input loading, operator
    operations, and output storing.
    """

    # Minimum for-loop iterations (avoids trivial single-iteration loops)
    MIN_FOR_LOOP_ITERATIONS = 2
    # Maximum allowed for-loop iterations to avoid excessive runtime
    MAX_FOR_LOOP_ITERATIONS = 4
    # Maximum ops for pre-if and post-if segments
    MAX_BRANCH_OPS = 3

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
        control_flow: ControlFlowConfig | None = None,
    ):
        """Initialize kernel generator

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operators (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
            control_flow: Control flow configuration (for loops, if/else, nesting).
                Defaults to ControlFlowConfig() (no control flow).
        """
        cf = control_flow or ControlFlowConfig()
        self.rng = random.Random(seed)
        self.enable_for_loop = cf.enable_for_loop
        self.enable_if_else = cf.enable_if_else
        self.for_loop_probability = max(0.0, min(1.0, cf.for_loop_probability))
        self.if_else_probability = max(0.0, min(1.0, cf.if_else_probability))
        self.max_for_loop_iterations = max(
            self.MIN_FOR_LOOP_ITERATIONS,
            min(cf.max_for_loop_iterations, self.MAX_FOR_LOOP_ITERATIONS),
        )
        self.max_depth = cf.max_depth
        self.fuzzer = OpFuzzer(
            seed=seed,
            enable_advanced_ops=enable_advanced_ops,
            advanced_ops_probability=advanced_ops_probability,
        )
        self.body_generator = BodyGenerator(
            rng=self.rng,
            fuzzer=self.fuzzer,
            max_depth=cf.max_depth,
            depth_decay=cf.depth_decay,
            for_loop_prob=cf.for_loop_probability if cf.enable_for_loop else 0.0,
            if_else_prob=cf.if_else_probability if cf.enable_if_else else 0.0,
            min_for_iterations=self.MIN_FOR_LOOP_ITERATIONS,
            max_for_iterations=self.max_for_loop_iterations,
        )

    def generate_kernel(
        self,
        kernel_name: str,
        num_inputs: int = 2,
        num_ops: int = 5,
        shape: tuple[int, int] = (128, 128),
        input_shapes: list[tuple[int, int]] | None = None,
        output_shape: tuple[int, int] | None = None,
        for_loop_iterations: int | None = None,
        for_loop_tiling: bool | None = None,
    ) -> dict[str, Any]:
        """Generate an InCore kernel using the composable body AST.

        Uses BodyGenerator to create a composable body structure with
        nested/parallel control flow, then generates code from the AST.

        Args:
            kernel_name: Kernel function name.
            num_inputs: Number of input tensors.
            num_ops: Total op budget for the kernel body.
            shape: Default shape for inputs.
            input_shapes: Explicit input shapes (overrides num_inputs/shape).
            output_shape: Output shape (defaults to first input shape).
            for_loop_iterations: Explicit iteration count for top-level tiling.
            for_loop_tiling: Explicit tiling mode for top-level loop.

        Returns:
            Kernel metadata dictionary with ``body`` field containing the AST.
        """
        # Resolve input shapes
        if input_shapes is not None:
            actual_num_inputs = len(input_shapes)
            actual_shapes = list(input_shapes)
        else:
            actual_num_inputs = num_inputs
            actual_shapes = [shape] * num_inputs

        # Validate alignment
        dtype = "FP32"
        for i, s in enumerate(actual_shapes):
            if not is_shape_aligned(s, dtype):
                actual_shapes[i] = generate_aligned_shape(self.rng, dtype)

        if output_shape is not None:
            actual_output_shape = output_shape
            if not is_shape_aligned(actual_output_shape, dtype):
                actual_output_shape = generate_aligned_shape(self.rng, dtype)
        else:
            actual_output_shape = actual_shapes[0]

        input_names = [chr(97 + i) for i in range(actual_num_inputs)]
        inputs = [(name, actual_shapes[i]) for i, name in enumerate(input_names)]

        # Resolve top-level tiling first (needed to adjust body depth)
        iterations, use_tiling = self._resolve_tiling(
            for_loop_iterations,
            for_loop_tiling,
        )

        # Generate body AST
        # When tiling is active, start at depth=1 so total nesting respects max_depth
        body_start_depth = 1 if (iterations > 0 and use_tiling) else 0
        body = self.body_generator.generate_body(
            num_ops=num_ops,
            num_inputs=actual_num_inputs,
            output_shape=actual_output_shape,
            depth=body_start_depth,
        )

        # Collect and unify scalar mappings across the body tree
        scalar_value_to_param = collect_all_scalars(body)
        scalars = [
            (param, value)
            for value, param in sorted(
                scalar_value_to_param.items(),
                key=lambda x: x[1],
            )
        ]

        # Generate kernel code from body AST
        code = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            scalars=scalars,
            body=body,
            output_shape=actual_output_shape,
            iterations=iterations,
            use_tiling=use_tiling,
        )

        # Scale shapes for tiling
        if iterations > 0 and use_tiling:
            scaled_inputs = [(name, (iterations * r, c)) for name, (r, c) in inputs]
            scaled_output_shape = (iterations * actual_output_shape[0], actual_output_shape[1])
        else:
            scaled_inputs = inputs
            scaled_output_shape = actual_output_shape

        has_branch = body_needs_branch_cond(body)
        op_chain = collect_all_op_chains(body)

        result: dict[str, Any] = {
            "name": kernel_name,
            "inputs": scaled_inputs,
            "scalars": scalars,
            "output_shape": scaled_output_shape,
            "tile_shape": actual_output_shape,
            "for_loop_info": {
                "iterations": iterations,
                "tiling": use_tiling,
            },
            "op_chain": op_chain,
            "body": body,
            "code": code,
        }

        if has_branch:
            result["has_config_scalar"] = True
            result["config_scalar_dtype"] = "BOOL"

        return result

    def _resolve_tiling(
        self,
        for_loop_iterations: int | None,
        for_loop_tiling: bool | None,
    ) -> tuple[int, bool]:
        """Resolve top-level tiling configuration.

        Returns:
            Tuple of (iterations, use_tiling).
        """
        if for_loop_iterations is not None:
            iterations = for_loop_iterations
        elif self.enable_for_loop and self.rng.random() < self.for_loop_probability:
            iterations = self.rng.randint(
                self.MIN_FOR_LOOP_ITERATIONS,
                self.max_for_loop_iterations,
            )
        else:
            iterations = 0

        if iterations > 0:
            use_tiling = for_loop_tiling if for_loop_tiling is not None else self.rng.choice([True, False])
        else:
            use_tiling = False

        return iterations, use_tiling

    def _generate_kernel_code(
        self,
        kernel_name: str,
        inputs: list[tuple[str, tuple[int, int]]],
        scalars: list[tuple[str, str]],
        body: list[BodyNode],
        output_shape: tuple[int, int],
        iterations: int = 0,
        use_tiling: bool = False,
    ) -> str:
        """Generate kernel function code from a body AST.

        Handles function signature, loads, body code generation, store,
        and optional top-level tiling loop wrapping.
        """
        tile_rows, tile_cols = output_shape
        has_branch = body_needs_branch_cond(body)
        has_matmul = _body_has_matmul(body)

        # Tensor shapes
        tensor_rows = iterations * tile_rows if (iterations > 0 and use_tiling) else tile_rows
        tensor_cols = tile_cols

        # Build function signature
        params = []
        for name, (r, c) in inputs:
            tr = iterations * r if (iterations > 0 and use_tiling) else r
            params.append(f"{name}: pl.Tensor[[{tr}, {c}], pl.FP32]")

        for scalar_name, _ in scalars:
            params.append(f"{scalar_name}: pl.Scalar[pl.FP32]")

        params.append(f"output: pl.Out[pl.Tensor[[{tensor_rows}, {tensor_cols}], pl.FP32]]")

        if has_branch:
            params.append("branch_cond: pl.Scalar[pl.BOOL]")

        code_lines = [
            "    @pl.function(type=pl.FunctionType.InCore)",
            f"    def {kernel_name}(self, {', '.join(params)})"
            f" -> pl.Tensor[[{tensor_rows}, {tensor_cols}], pl.FP32]:",
        ]

        # Generate body code from AST
        body_loop_depth = 1 if (iterations > 0 and use_tiling) else 0
        body_code, body_last = generate_body_code(self, body, output_shape, loop_depth=body_loop_depth)

        # Determine store function and variable
        last_op_chain = _get_last_op_chain(body)
        store_var = body_last if body_last else None

        if iterations > 0 and use_tiling:
            # Tiling: wrap loads + body + store in for loop, return after loop
            row_offset = f"i * {tile_rows}"
            load_lines = self._generate_input_loads(inputs, has_matmul, row_offset)
            store_lines = self._generate_store_from_body(
                last_op_chain,
                inputs,
                output_shape,
                row_offset,
                store_var,
            )
            return_line = f"{_INDENT * 2}return result"

            loop_lines = _generate_tiling_loop(
                iterations,
                load_lines,
                body_code,
                store_lines,
            )
            code_lines.extend(loop_lines)
            code_lines.append(return_line)
        else:
            # No tiling: loads + body + store + return
            load_lines = self._generate_input_loads(inputs, has_matmul)
            store_lines = self._generate_store_from_body(
                last_op_chain,
                inputs,
                output_shape,
                store_var=store_var,
            )
            return_line = f"{_INDENT * 2}return result"

            code_lines.extend(load_lines)
            code_lines.extend(body_code)
            code_lines.extend(store_lines)
            code_lines.append(return_line)

        return "\n".join(code_lines)

    def _generate_store_from_body(
        self,
        last_op_chain: list[dict[str, Any]],
        inputs: list[tuple[str, tuple[int, int]]],
        output_shape: tuple[int, int],
        row_offset_expr: str | None = None,
        store_var: str | None = None,
        indent_level: int = 2,
    ) -> list[str]:
        """Generate store operation for body AST output.

        Uses store for all cases (l0c_store was consolidated into store).
        """
        # If last op is matmul, let _generate_store_op handle it via op_chain
        if last_op_chain and last_op_chain[-1]["op"].name == "tile.matmul":
            return self._generate_store_op(
                last_op_chain,
                inputs,
                output_shape,
                row_offset_expr,
                indent_level=indent_level,
            )
        # Otherwise, use explicit store_var
        return self._generate_store_op(
            [],
            inputs,
            output_shape,
            row_offset_expr,
            store_var=store_var,
            indent_level=indent_level,
        )

    def _generate_matmul_memory_moves(
        self,
        input_var: str,
        target_memory: int,
        has_matmul: bool,
        moved_tiles: dict[str, str] | None = None,
        l0c_vars: set[str] | None = None,
        indent_level: int = 2,
    ) -> tuple[str, list[str]]:
        """Generate memory move operations for matmul inputs.

        Args:
            input_var: Input variable name (e.g., "tile_a")
            target_memory: Target memory type (3 for Left, 4 for Right)
            has_matmul: Whether the kernel contains matmul operations
            moved_tiles: Cache of already-moved tiles {(var, target): result_var}.
                If provided, reuses existing move results to avoid duplicate move.
            l0c_vars: Set of variable names that are L0C results from prior matmuls.
                These need a direct move to L0A/L0B without the _l1 indirection.
            indent_level: Indentation nesting level (default 2 = 8 spaces).
        """
        ind = _INDENT * indent_level
        # Map integer target_memory to Mem enum names (pl.Mem is short for pl.MemorySpace)
        memory_enum_map = {3: "pl.Mem.Left", 4: "pl.Mem.Right"}
        code_lines = []
        memory_suffix = "l0a" if target_memory == 3 else "l0b"
        memory_enum = memory_enum_map.get(target_memory, str(target_memory))

        # L0C matmul result: move directly to L0A or L0B (no _l1 indirection needed)
        if l0c_vars and input_var in l0c_vars:
            cache_key = (input_var, target_memory)
            if moved_tiles is not None and cache_key in moved_tiles:
                return moved_tiles[cache_key], []
            output_var = f"{input_var}_{memory_suffix}"
            code_lines.append(f"{ind}{output_var} = pl.move({input_var}, target_memory={memory_enum})")
            if moved_tiles is not None:
                moved_tiles[cache_key] = output_var
            return output_var, code_lines

        if input_var.startswith("tile_") and not input_var.endswith(("_l0a", "_l0b", "_l0c")):
            cache_key = (input_var, target_memory)
            if moved_tiles is not None and cache_key in moved_tiles:
                # Tile already moved to this memory space — reuse existing variable
                return moved_tiles[cache_key], []

            input_l1 = f"{input_var}_l1" if has_matmul else input_var
            output_var = f"{input_var}_{memory_suffix}"
            code_lines.append(f"{ind}{output_var} = pl.move({input_l1}, target_memory={memory_enum})")
            if moved_tiles is not None:
                moved_tiles[cache_key] = output_var
            return output_var, code_lines
        else:
            return input_var, code_lines

    def _generate_input_loads(
        self,
        inputs: list[tuple[str, tuple[int, int]]],
        has_matmul: bool,
        row_offset_expr: str | None = None,
        indent_level: int = 2,
    ) -> list[str]:
        """Generate input load operations.

        Args:
            inputs: Input tensor list [(name, tile_shape), ...]
            has_matmul: Whether the kernel contains matmul operations
            row_offset_expr: Expression for row offset (e.g. "i * 64") for tiling loops
            indent_level: Indentation nesting level (default 2 = 8 spaces).

        Note:
            When has_matmul is True, only generates Mat space loads (tile_{name}_l1).
            Matmul kernels don't mix with other Vec operations, so Vec loads are unnecessary.
        """
        ind = _INDENT * indent_level
        code_lines = []
        for name, (r, c) in inputs:
            offset = f"[{row_offset_expr}, 0]" if row_offset_expr else "[0, 0]"
            if has_matmul:
                # Only generate Mat space load for matmul kernels
                code_lines.append(
                    f"{ind}tile_{name}_l1 = pl.load({name}, offsets={offset}, "
                    f"shapes=[{r}, {c}], target_memory=pl.MemorySpace.Mat)"
                )
            else:
                # Generate default Vec space load for non-matmul kernels
                code_lines.append(f"{ind}tile_{name} = pl.load({name}, offsets={offset}, shapes=[{r}, {c}])")
        return code_lines

    def _generate_matmul_op(
        self,
        op_dict: dict[str, Any],
        has_matmul: bool,
        moved_tiles: dict[str, str] | None = None,
        l0c_vars: set[str] | None = None,
        indent_level: int = 2,
    ) -> list[str]:
        """Generate matmul operation with memory moves."""
        ind = _INDENT * indent_level
        code_lines = []
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]

        input_a_l0a, move_lines_a = self._generate_matmul_memory_moves(
            inputs_list[0], 3, has_matmul, moved_tiles, l0c_vars, indent_level
        )
        code_lines.extend(move_lines_a)

        input_b_l0b, move_lines_b = self._generate_matmul_memory_moves(
            inputs_list[1], 4, has_matmul, moved_tiles, l0c_vars, indent_level
        )
        code_lines.extend(move_lines_b)

        code_lines.append(f"{ind}{output} = pl.matmul({input_a_l0a}, {input_b_l0b})")
        return code_lines

    def _generate_reduction_op(
        self,
        op_dict: dict[str, Any],
        output_shape: tuple[int, int],
        indent_level: int = 2,
    ) -> list[str]:
        """Generate reduction operation with temporary tile.

        For row_sum/row_max/row_min operations, the tmp_tile must have the same shape
        as the input (e.g., [M, N]), not the output shape ([M, 1]).
        """
        ind = _INDENT * indent_level
        code_lines = []
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        op_name = op.name.split(".")[-1]

        # Use input shape for tmp_tile, not output shape
        input_shapes = op_dict.get("input_shapes", [])
        if input_shapes:
            tmp_shape = input_shapes[0]
        else:
            tmp_shape = op_dict.get("output_shape", (output_shape[0], output_shape[1]))

        tmp_tile_var = f"tmp_tile_{output}"
        code_lines.append(
            f"{ind}{tmp_tile_var} = pl.create_tile([{tmp_shape[0]}, {tmp_shape[1]}], dtype=pl.FP32)"
        )
        code_lines.append(f"{ind}{output} = pl.{op_name}({inputs_list[0]}, {tmp_tile_var})")
        return code_lines

    def _generate_regular_op(
        self,
        op_dict: dict[str, Any],
        scalar_value_to_param: dict[str, str],
        indent_level: int = 2,
    ) -> str:
        """Generate regular operation."""
        ind = _INDENT * indent_level
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        params = op_dict.get("params")
        op_name = op.name.split(".")[-1]

        # Replace scalar literals with parameter references
        processed_inputs = []
        for inp in inputs_list:
            if inp in scalar_value_to_param:
                processed_inputs.append(scalar_value_to_param[inp])
            else:
                processed_inputs.append(inp)

        inputs_str = ", ".join(processed_inputs)
        if params:
            # Special handling for reshape: target_shape is a positional argument
            if op_name == "reshape" and "target_shape" in params:
                target_shape = params["target_shape"]
                shape_str = f"[{', '.join(map(str, target_shape))}]"
                return f"{ind}{output} = pl.{op_name}({inputs_str}, {shape_str})"

            # Format other parameters as keyword arguments
            formatted_params = []
            for k, v in params.items():
                if isinstance(v, tuple):
                    # Convert tuple to list format for shapes
                    formatted_params.append(f"{k}=[{', '.join(map(str, v))}]")
                else:
                    formatted_params.append(f"{k}={v}")
            params_str = ", ".join(formatted_params)
            return f"{ind}{output} = pl.{op_name}({inputs_str}, {params_str})"
        return f"{ind}{output} = pl.{op_name}({inputs_str})"

    def _generate_store_op(
        self,
        op_chain: list[dict[str, Any]],
        inputs: list[tuple[str, tuple[int, int]]],
        output_shape: tuple[int, int],
        row_offset_expr: str | None = None,
        store_var: str | None = None,
        indent_level: int = 2,
    ) -> list[str]:
        """Generate store operation (without return statement).

        Args:
            op_chain: Operation chain
            inputs: Input tensor list
            output_shape: Tile shape (per-iteration block size)
            row_offset_expr: Expression for row offset (e.g. "i * 64") for tiling loops
            store_var: Explicit variable name to store. Overrides op_chain[-1]["output"].
            indent_level: Indentation nesting level (default 2 = 8 spaces).
        """
        ind = _INDENT * indent_level
        code_lines = []
        offset = f"[{row_offset_expr}, 0]" if row_offset_expr else "[0, 0]"

        if store_var:
            code_lines.append(f"{ind}result = pl.store({store_var}, offsets={offset}, output_tensor=output)")
        elif op_chain:
            last_output = op_chain[-1]["output"]
            code_lines.append(
                f"{ind}result = pl.store({last_output}, offsets={offset}, output_tensor=output)"
            )
        else:
            first_input = inputs[0][0]
            code_lines.append(
                f"{ind}result = pl.store(tile_{first_input}, offsets={offset}, output_tensor=output)"
            )

        return code_lines

    def generate_multiple_kernels(
        self,
        num_kernels: int = 3,
        num_inputs_range: tuple[int, int] = (2, 3),
        num_ops_range: tuple[int, int] | None = None,
        num_ops: int | tuple[int, int] = (3, 7),
        shape: tuple[int, int] = (128, 128),
        input_shapes_list: list[list[tuple[int, int]]] | None = None,
        output_shapes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate multiple InCore kernel functions.

        Args:
            num_kernels: Number of kernels to generate
            num_inputs_range: Range for number of inputs (min, max)
            num_ops_range: Deprecated alias for num_ops (tuple form).
            num_ops: Number of operations per kernel. Can be an int (fixed)
                or a tuple (min, max) for random selection.
            shape: Default shape for inputs
            input_shapes_list: List of input shapes for each kernel,
                              e.g., [[(128,128), (64,64)], [(256,256)], ...]
            output_shapes: Output shapes for each kernel (optional)

        Returns:
            List of kernel metadata dictionaries
        """
        # Handle backward-compatible num_ops_range alias
        if num_ops_range is not None:
            num_ops = num_ops_range

        kernels = []
        for i in range(num_kernels):
            # Resolve num_ops for this kernel
            if isinstance(num_ops, tuple):
                kernel_num_ops = self.rng.randint(*num_ops)
            else:
                kernel_num_ops = num_ops

            if input_shapes_list and i < len(input_shapes_list):
                kernel_input_shapes = input_shapes_list[i]
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None
            else:
                kernel_input_shapes = None
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None

            num_inputs = self.rng.randint(*num_inputs_range) if kernel_input_shapes is None else None

            kernel = self.generate_kernel(
                kernel_name=f"kernel_{i}",
                num_inputs=num_inputs or 2,
                num_ops=kernel_num_ops,
                shape=shape,
                input_shapes=kernel_input_shapes,
                output_shape=kernel_output_shape,
            )
            kernels.append(kernel)

        return kernels
