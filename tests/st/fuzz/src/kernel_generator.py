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

from .fuzzer import OpFuzzer, generate_aligned_shape, is_shape_aligned
from .if_else_generator import generate_if_else_kernel_code


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

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
        enable_for_loop: bool = False,
        max_for_loop_iterations: int = 4,
        enable_if_else: bool = False,
        for_loop_probability: float = 1.0,
    ):
        """Initialize kernel generator

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operators (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
            enable_for_loop: Wrap kernel body in a for loop (scf.for)
            max_for_loop_iterations: Upper bound for random iteration count (min..max, capped
                at MAX_FOR_LOOP_ITERATIONS=4)
            enable_if_else: Generate if/else branching in kernel (scf.if).
                Mutually exclusive with enable_for_loop for now.
            for_loop_probability: Probability that a generated kernel batch actually uses a for
                loop when enable_for_loop=True. Defaults to 1.0 (always). Set below 1.0 to
                produce a mix of loop and no-loop test cases.
        """
        self.rng = random.Random(seed)
        self.enable_for_loop = enable_for_loop
        self.enable_if_else = enable_if_else
        self.for_loop_probability = max(0.0, min(1.0, for_loop_probability))
        self.max_for_loop_iterations = max(
            self.MIN_FOR_LOOP_ITERATIONS,
            min(max_for_loop_iterations, self.MAX_FOR_LOOP_ITERATIONS),
        )
        self.fuzzer = OpFuzzer(
            seed=seed,
            enable_advanced_ops=enable_advanced_ops,
            advanced_ops_probability=advanced_ops_probability,
        )

    def generate_kernel(
        self,
        kernel_name: str,
        num_inputs: int = 2,
        num_ops: int = 5,
        shape: tuple[int, int] = (128, 128),
        allow_scalars: bool = True,
        input_shapes: list[tuple[int, int]] | None = None,
        output_shape: tuple[int, int] | None = None,
        for_loop_iterations: int | None = None,
        for_loop_tiling: bool | None = None,
        use_if_else: bool | None = None,
    ) -> dict[str, Any]:
        """Generate an InCore kernel function.

        Args:
            kernel_name: Kernel function name
            num_inputs: Number of input tensors (ignored if input_shapes is provided)
            num_ops: Number of operations in the chain
            shape: Default shape for inputs (ignored if input_shapes is provided)
            allow_scalars: Whether to allow scalar operations
            input_shapes: List of input shapes, overrides num_inputs and shape
            output_shape: Output shape, defaults to first input shape
            for_loop_iterations: Explicit iteration count override for for loop
            for_loop_tiling: Explicit tiling mode override. True = top-level loop
                with i-based offsets; False = middle placement, no tiling.
            use_if_else: Explicit override for if/else mode. None means use
                self.enable_if_else setting.

        Returns:
            Kernel metadata dictionary containing:
            - name: Kernel function name
            - inputs: Input tensor list [(name, full_tensor_shape), ...]
            - scalars: Scalar parameter list [(scalar_name, value), ...]
            - output_shape: Full output tensor shape (scaled by iterations)
            - tile_shape: Per-iteration tile shape
            - for_loop_info: Loop configuration info
            - op_chain: Operation chain (then-chain for if/else)
            - code: Generated PyPTO kernel code
            - if_else_info: (optional) If/else branch info
            - has_config_scalar: (optional) True if kernel needs config tensor
            - config_scalar_dtype: (optional) PL dtype name for the config scalar (e.g. "BOOL")
        """
        if input_shapes is not None:
            actual_num_inputs = len(input_shapes)
            actual_shapes = input_shapes
        else:
            actual_num_inputs = num_inputs
            actual_shapes = [shape] * num_inputs

        # Validate input shape alignment
        dtype = "FP32"  # Currently only FP32 is supported
        for i, input_shape in enumerate(actual_shapes):
            if not is_shape_aligned(input_shape, dtype):
                print(
                    f"Warning: Input shape {input_shape} is not 32-byte aligned. Regenerating aligned shape."
                )
                actual_shapes[i] = generate_aligned_shape(self.rng, dtype)

        # Validate output shape alignment
        if output_shape is not None:
            actual_output_shape = output_shape
            if not is_shape_aligned(actual_output_shape, dtype):
                print(
                    f"Warning: Output shape {actual_output_shape} is not 32-byte aligned. "
                    f"Regenerating aligned shape."
                )
                actual_output_shape = generate_aligned_shape(self.rng, dtype)
        else:
            actual_output_shape = actual_shapes[0]

        # Resolve whether to use if/else
        do_if_else = use_if_else if use_if_else is not None else self.enable_if_else

        input_names = [chr(97 + i) for i in range(actual_num_inputs)]  # a, b, c, ...
        inputs = [(name, actual_shapes[i]) for i, name in enumerate(input_names)]

        if do_if_else:
            return self._generate_if_else_kernel(
                kernel_name=kernel_name,
                num_ops=num_ops,
                actual_num_inputs=actual_num_inputs,
                actual_output_shape=actual_output_shape,
                allow_scalars=allow_scalars,
                inputs=inputs,
            )

        op_chain = self.fuzzer.generate_op_chain(
            num_ops=num_ops,
            input_count=actual_num_inputs,
            allow_scalars=allow_scalars,
            track_shapes=True,
            default_shape=actual_output_shape,
        )

        # Collect unique scalar values used in op_chain
        scalar_values = set()
        for op_dict in op_chain:
            if op_dict.get("scalar_value"):
                scalar_values.add(op_dict["scalar_value"])

        # Create scalar parameter list: [(param_name, value), ...]
        scalars = []
        scalar_value_to_param = {}
        for idx, value in enumerate(sorted(scalar_values)):
            param_name = f"scalar_{idx}"
            scalars.append((param_name, value))
            scalar_value_to_param[value] = param_name

        # Generate kernel code
        code, loop_info = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            scalars=scalars,
            op_chain=op_chain,
            output_shape=actual_output_shape,
            scalar_value_to_param=scalar_value_to_param,
            for_loop_iterations=for_loop_iterations,
            for_loop_tiling=for_loop_tiling,
        )

        iterations = loop_info["iterations"]
        use_tiling = loop_info["tiling"]

        # Scale shapes for tiling mode only; accumulation mode uses tile-sized tensors
        if iterations > 0 and use_tiling:
            scaled_inputs = [(name, (iterations * r, c)) for name, (r, c) in inputs]
            scaled_output_shape = (iterations * actual_output_shape[0], actual_output_shape[1])
        else:
            scaled_inputs = inputs
            scaled_output_shape = actual_output_shape

        return {
            "name": kernel_name,
            "inputs": scaled_inputs,
            "scalars": scalars,
            "output_shape": scaled_output_shape,
            "tile_shape": actual_output_shape,
            "for_loop_info": loop_info,
            "op_chain": op_chain,
            "code": code,
        }

    def _generate_if_else_kernel(
        self,
        kernel_name: str,
        num_ops: int,
        actual_num_inputs: int,
        actual_output_shape: tuple[int, int],
        allow_scalars: bool,
        inputs: list[tuple[str, tuple[int, int]]],
    ) -> dict[str, Any]:
        """Generate an if/else kernel with two branched op chains.

        Args:
            kernel_name: Kernel function name
            num_ops: Total number of ops (split between branches)
            actual_num_inputs: Number of input tensors
            actual_output_shape: Output tile shape
            allow_scalars: Whether to allow scalar operations
            inputs: Input tensor list [(name, shape), ...]

        Returns:
            Kernel metadata dictionary with if_else_info.
        """
        ops_per_branch = max(1, num_ops // 2)
        branches = self.fuzzer.generate_branched_op_chains(
            num_ops_per_branch=ops_per_branch,
            input_count=actual_num_inputs,
            allow_scalars=allow_scalars,
            track_shapes=True,
            default_shape=actual_output_shape,
            basic_ops_only=True,
        )
        then_chain, else_chain = branches[0], branches[1]

        # Collect scalars from both branches
        scalar_values = set()
        for op_dict in then_chain + else_chain:
            if op_dict.get("scalar_value"):
                scalar_values.add(op_dict["scalar_value"])

        scalars = []
        scalar_value_to_param = {}
        for idx, value in enumerate(sorted(scalar_values)):
            param_name = f"scalar_{idx}"
            scalars.append((param_name, value))
            scalar_value_to_param[value] = param_name

        if_else_info: dict[str, Any] = {
            "enabled": True,
            "then_chain": then_chain,
            "else_chain": else_chain,
        }

        code, loop_info = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            scalars=scalars,
            op_chain=then_chain,
            output_shape=actual_output_shape,
            scalar_value_to_param=scalar_value_to_param,
            for_loop_iterations=0,
            for_loop_tiling=False,
            if_else_info=if_else_info,
        )

        return {
            "name": kernel_name,
            "inputs": inputs,  # No shape scaling for if-only
            "scalars": scalars,
            "output_shape": actual_output_shape,
            "tile_shape": actual_output_shape,
            "for_loop_info": loop_info,
            "op_chain": then_chain,
            "if_else_info": if_else_info,
            "has_config_scalar": True,
            "config_scalar_dtype": "BOOL",
            "code": code,
        }

    def _generate_matmul_memory_moves(
        self,
        input_var: str,
        target_memory: int,
        has_matmul: bool,
        moved_tiles: dict[str, str] | None = None,
        l0c_vars: set[str] | None = None,
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

        Returns:
            Tuple of (final_var_name, list_of_code_lines)
        """
        # Map integer target_memory to MemorySpace enum names
        memory_enum_map = {3: "pl.MemorySpace.Left", 4: "pl.MemorySpace.Right"}
        code_lines = []
        memory_suffix = "l0a" if target_memory == 3 else "l0b"
        memory_enum = memory_enum_map.get(target_memory, str(target_memory))

        # L0C matmul result: move directly to L0A or L0B (no _l1 indirection needed)
        if l0c_vars and input_var in l0c_vars:
            cache_key = (input_var, target_memory)
            if moved_tiles is not None and cache_key in moved_tiles:
                return moved_tiles[cache_key], []
            output_var = f"{input_var}_{memory_suffix}"
            code_lines.append(f"        {output_var} = pl.move({input_var}, target_memory={memory_enum})")
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
            code_lines.append(f"        {output_var} = pl.move({input_l1}, target_memory={memory_enum})")
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
    ) -> list[str]:
        """Generate input load operations.

        Args:
            inputs: Input tensor list [(name, tile_shape), ...]
            has_matmul: Whether the kernel contains matmul operations
            row_offset_expr: Expression for row offset (e.g. "i * 64") for tiling loops
        """
        code_lines = []
        for name, (r, c) in inputs:
            offset = f"[{row_offset_expr}, 0]" if row_offset_expr else "[0, 0]"
            if has_matmul:
                code_lines.append(
                    f"        tile_{name}_l1 = pl.load({name}, offsets={offset}, "
                    f"shapes=[{r}, {c}], target_memory=pl.MemorySpace.Mat)"
                )
            else:
                code_lines.append(
                    f"        tile_{name} = pl.load({name}, offsets={offset}, shapes=[{r}, {c}])"
                )
        return code_lines

    def _generate_matmul_op(
        self,
        op_dict: dict[str, Any],
        has_matmul: bool,
        moved_tiles: dict[str, str] | None = None,
        l0c_vars: set[str] | None = None,
    ) -> list[str]:
        """Generate matmul operation with memory moves."""
        code_lines = []
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]

        input_a_l0a, move_lines_a = self._generate_matmul_memory_moves(
            inputs_list[0], 3, has_matmul, moved_tiles, l0c_vars
        )
        code_lines.extend(move_lines_a)

        input_b_l0b, move_lines_b = self._generate_matmul_memory_moves(
            inputs_list[1], 4, has_matmul, moved_tiles, l0c_vars
        )
        code_lines.extend(move_lines_b)

        code_lines.append(f"        {output} = pl.matmul({input_a_l0a}, {input_b_l0b})")
        return code_lines

    def _generate_reduction_op(
        self,
        op_dict: dict[str, Any],
        output_shape: tuple[int, int],
    ) -> list[str]:
        """Generate reduction operation with temporary tile.

        For row_sum/row_max/row_min operations, the tmp_tile must have the same shape
        as the input (e.g., [M, N]), not the output shape ([M, 1]).
        """
        code_lines = []
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        op_name = op.name.replace("block.", "")

        # Use input shape for tmp_tile, not output shape
        # For row_sum: input is [M, N], output is [M, 1], tmp_tile should be [M, N]
        input_shapes = op_dict.get("input_shapes", [])
        if input_shapes:
            tmp_shape = input_shapes[0]  # Use first input's shape
        else:
            # Fallback: use output_shape (this maintains backward compatibility)
            tmp_shape = op_dict.get("output_shape", (output_shape[0], output_shape[1]))

        tmp_tile_var = f"tmp_tile_{output}"
        code_lines.append(
            f"        {tmp_tile_var} = pl.create_tile([{tmp_shape[0]}, {tmp_shape[1]}], "
            f"dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)"
        )
        code_lines.append(f"        {output} = pl.{op_name}({inputs_list[0]}, {tmp_tile_var})")
        return code_lines

    def _generate_regular_op(self, op_dict: dict[str, Any], scalar_value_to_param: dict[str, str]) -> str:
        """Generate regular operation."""
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        params = op_dict.get("params")
        op_name = op.name.replace("block.", "")

        # Replace scalar literals with parameter references
        processed_inputs = []
        for inp in inputs_list:
            if inp in scalar_value_to_param:
                processed_inputs.append(scalar_value_to_param[inp])
            else:
                processed_inputs.append(inp)

        inputs_str = ", ".join(processed_inputs)
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"        {output} = pl.{op_name}({inputs_str}, {params_str})"
        return f"        {output} = pl.{op_name}({inputs_str})"

    def _generate_store_op(
        self,
        op_chain: list[dict[str, Any]],
        inputs: list[tuple[str, tuple[int, int]]],
        output_shape: tuple[int, int],
        row_offset_expr: str | None = None,
    ) -> list[str]:
        """Generate store operation.

        Args:
            op_chain: Operation chain
            inputs: Input tensor list
            output_shape: Tile shape (per-iteration block size)
            row_offset_expr: Expression for row offset (e.g. "i * 64") for tiling loops
        """
        code_lines = []
        rows, cols = output_shape
        offset = f"[{row_offset_expr}, 0]" if row_offset_expr else "[0, 0]"

        if op_chain:
            last_output = op_chain[-1]["output"]
            last_op = op_chain[-1]["op"]

            if last_op.name == "block.matmul":
                code_lines.append(
                    f"        result = pl.store({last_output}, offsets={offset}, "
                    f"shapes=[{rows}, {cols}], output_tensor=output)"
                )
            else:
                code_lines.append(
                    f"        result = pl.store({last_output}, offsets={offset}, "
                    f"shapes=[{rows}, {cols}], output_tensor=output)"
                )
        else:
            first_input = inputs[0][0]
            code_lines.append(
                f"        result = pl.store(tile_{first_input}, offsets={offset}, "
                f"shapes=[{rows}, {cols}], output_tensor=output)"
            )

        code_lines.append("        return result")
        return code_lines

    def _resolve_loop_config(
        self,
        for_loop_iterations: int | None,
        for_loop_tiling: bool | None,
        for_loop_split_point: int | None,
        op_chain: list[dict[str, Any]],
    ) -> tuple[int, bool, int]:
        """Resolve for-loop configuration from explicit overrides or random choices.

        Returns:
            Tuple of (iterations, use_tiling, split_point).
        """
        if for_loop_iterations is not None:
            iterations = for_loop_iterations
        elif self.enable_for_loop:
            iterations = self.rng.randint(self.MIN_FOR_LOOP_ITERATIONS, self.max_for_loop_iterations)
        else:
            iterations = 0

        if iterations > 0:
            use_tiling = for_loop_tiling if for_loop_tiling is not None else self.rng.choice([True, False])
        else:
            use_tiling = False

        if iterations > 0 and not use_tiling:
            split_point = (
                for_loop_split_point
                if for_loop_split_point is not None
                else self.rng.randint(1, max(1, len(op_chain) - 1))
            )
        else:
            split_point = 0

        return iterations, use_tiling, split_point

    def _generate_if_else_kernel_code(
        self,
        kernel_name: str,
        inputs: list[tuple[str, tuple[int, int]]],
        scalars: list[tuple[str, str]],
        output_shape: tuple[int, int],
        scalar_value_to_param: dict[str, str],
        then_chain: list[dict[str, Any]],
        else_chain: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Delegate to if_else_generator for if/else kernel code generation."""
        return generate_if_else_kernel_code(
            gen=self,
            kernel_name=kernel_name,
            inputs=inputs,
            scalars=scalars,
            output_shape=output_shape,
            scalar_value_to_param=scalar_value_to_param,
            then_chain=then_chain,
            else_chain=else_chain,
        )

    def _generate_kernel_code(
        self,
        kernel_name: str,
        inputs: list[tuple[str, tuple[int, int]]],
        scalars: list[tuple[str, str]],
        op_chain: list[dict[str, Any]],
        output_shape: tuple[int, int],
        scalar_value_to_param: dict[str, str],
        for_loop_iterations: int | None = None,
        for_loop_tiling: bool | None = None,
        for_loop_split_point: int | None = None,
        if_else_info: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate kernel function code.

        Args:
            kernel_name: Kernel function name
            inputs: Tensor input list (tile shapes per iteration)
            scalars: Scalar parameter list [(param_name, value), ...]
            op_chain: Operation chain (used for non-if/else modes, and as then-chain
                when if_else_info is provided without explicit chains)
            output_shape: Tile shape (per-iteration block size)
            scalar_value_to_param: Mapping from scalar values to parameter names
            for_loop_iterations: Explicit iteration count override. None means
                determine from settings (random if enable_for_loop, else 0).
            for_loop_tiling: Explicit tiling mode override. None means random.
                True = top-level loop with i-based offsets.
                False = middle placement, no tiling.
            for_loop_split_point: Explicit op split point override for middle mode.
                None means random. Number of ops placed before the for loop.
            if_else_info: If/else configuration dict with keys:
                - enabled (bool): Whether if/else is active
                - then_chain (list): Op chain for the then-branch
                - else_chain (list): Op chain for the else-branch

        Returns:
            Tuple of (generated code, loop_info dict).
            loop_info keys: iterations (int), tiling (bool), split_point (int).
        """
        # Check if this is an if-else kernel
        if if_else_info and if_else_info.get("enabled"):
            return self._generate_if_else_kernel_code(
                kernel_name=kernel_name,
                inputs=inputs,
                scalars=scalars,
                output_shape=output_shape,
                scalar_value_to_param=scalar_value_to_param,
                then_chain=if_else_info["then_chain"],
                else_chain=if_else_info["else_chain"],
            )

        tile_rows, tile_cols = output_shape

        iterations, use_tiling, split_point = self._resolve_loop_config(
            for_loop_iterations, for_loop_tiling, for_loop_split_point, op_chain
        )

        # Tensor shapes: tiling mode scales by iterations; accumulation mode uses tile size
        if iterations > 0 and use_tiling:
            tensor_rows = iterations * tile_rows
        else:
            tensor_rows = tile_rows
        tensor_cols = tile_cols

        # Build function signature
        params = []
        for name, (r, c) in inputs:
            tr = iterations * r if (iterations > 0 and use_tiling) else r
            params.append(f"{name}: pl.Tensor[[{tr}, {c}], pl.FP32]")

        for scalar_name, _ in scalars:
            params.append(f"{scalar_name}: pl.Scalar[pl.FP32]")

        params.append(f"output: pl.Out[pl.Tensor[[{tensor_rows}, {tensor_cols}], pl.FP32]]")

        code_lines = [
            "    @pl.function(type=pl.FunctionType.InCore)",
            f"    def {kernel_name}(self, {', '.join(params)})"
            f" -> pl.Tensor[[{tensor_rows}, {tensor_cols}], pl.FP32]:",
        ]

        has_matmul = any(op_dict["op"].name == "block.matmul" for op_dict in op_chain)

        # Load offset: only tiling mode uses i-based offsets (accumulation mode loads from fixed [0,0])
        load_row_offset = f"i * {tile_rows}" if (use_tiling and iterations > 0) else None
        # Store offset: only tiling mode writes to i-th tile; accumulation mode stores once at [0,0]
        store_row_offset = f"i * {tile_rows}" if (use_tiling and iterations > 0) else None

        # Generate code lines for loads, ops, and store
        load_lines = self._generate_input_loads(inputs, has_matmul, load_row_offset)

        op_line_groups: list[list[str]] = []
        moved_tiles: dict[str, str] = {}  # Cache of already-moved tiles to avoid duplicate move
        # Track variables that are L0C results from prior matmuls (need pl.move to Left/Right)
        l0c_vars: set[str] = {
            op_dict["output"] for op_dict in op_chain if op_dict["op"].name == "block.matmul"
        }
        for op_dict in op_chain:
            op = op_dict["op"]
            if op.name == "block.matmul":
                op_line_groups.append(self._generate_matmul_op(op_dict, has_matmul, moved_tiles, l0c_vars))
            elif op.constraints.get("requires_tmp_tile", False):
                op_line_groups.append(self._generate_reduction_op(op_dict, output_shape))
            else:
                op_line_groups.append([self._generate_regular_op(op_dict, scalar_value_to_param)])

        store_lines = self._generate_store_op(op_chain, inputs, output_shape, store_row_offset)

        # Flatten op groups into a list of lines
        all_op_lines = [line for group in op_line_groups for line in group]

        if iterations > 0 and use_tiling:
            # Tiling mode: entire body inside for loop
            body_lines = load_lines + all_op_lines + store_lines
            return_line = body_lines.pop()
            code_lines.append(f"        for i in pl.range({iterations}):")
            for line in body_lines:
                code_lines.append("    " + line)
            code_lines.append(return_line)

        elif iterations > 0 and not use_tiling:
            # Accumulation mode: loads + pre-loop ops outside; loop body applies ops repeatedly
            # on the same tile; store happens ONCE after the loop at [0, 0] (no i-based offsets).
            pre_op_lines = [line for group in op_line_groups[:split_point] for line in group]
            loop_op_lines = [line for group in op_line_groups[split_point:] for line in group]

            # Pre-loop section
            code_lines.extend(load_lines)
            code_lines.extend(pre_op_lines)

            # For loop: accumulate ops repeatedly (no store, no i-based offsets)
            if loop_op_lines:
                code_lines.append(f"        for i in pl.range({iterations}):")
                for line in loop_op_lines:
                    code_lines.append("    " + line)

            # Store ONCE after loop (store_lines includes both store and return)
            code_lines.extend(store_lines)

        else:
            # No for loop
            code_lines.extend(load_lines + all_op_lines + store_lines)

        loop_info = {
            "iterations": iterations,
            "tiling": use_tiling,
            "split_point": split_point,
        }
        return "\n".join(code_lines), loop_info

    def generate_multiple_kernels(
        self,
        num_kernels: int = 3,
        num_inputs_range: tuple[int, int] = (2, 3),
        num_ops_range: tuple[int, int] = (3, 7),
        shape: tuple[int, int] = (128, 128),
        input_shapes_list: list[list[tuple[int, int]]] | None = None,
        output_shapes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate multiple InCore kernel functions.

        Args:
            num_kernels: Number of kernels to generate
            num_inputs_range: Range for number of inputs (min, max)
            num_ops_range: Range for number of operations (min, max)
            shape: Default shape for inputs
            input_shapes_list: List of input shapes for each kernel,
                              e.g., [[(128,128), (64,64)], [(256,256)], ...]
            output_shapes: Output shapes for each kernel (optional)

        Returns:
            List of kernel metadata dictionaries
        """
        # Pick for-loop iterations and tiling mode once for all kernels in the chain
        # If/else mode is mutually exclusive with for-loop for now
        shared_if_else = self.enable_if_else
        if self.enable_for_loop and not shared_if_else and self.rng.random() < self.for_loop_probability:
            shared_iterations = self.rng.randint(self.MIN_FOR_LOOP_ITERATIONS, self.max_for_loop_iterations)
            shared_tiling = self.rng.choice([True, False])
        else:
            shared_iterations = 0
            shared_tiling = False

        kernels = []
        for i in range(num_kernels):
            num_ops = self.rng.randint(*num_ops_range)

            if input_shapes_list and i < len(input_shapes_list):
                kernel_input_shapes = input_shapes_list[i]
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None
                kernel = self.generate_kernel(
                    kernel_name=f"kernel_{i}",
                    num_ops=num_ops,
                    shape=shape,
                    input_shapes=kernel_input_shapes,
                    output_shape=kernel_output_shape,
                    for_loop_iterations=shared_iterations,
                    for_loop_tiling=shared_tiling,
                    use_if_else=shared_if_else,
                )
            else:
                num_inputs = self.rng.randint(*num_inputs_range)
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None
                kernel = self.generate_kernel(
                    kernel_name=f"kernel_{i}",
                    num_inputs=num_inputs,
                    num_ops=num_ops,
                    shape=shape,
                    output_shape=kernel_output_shape,
                    for_loop_iterations=shared_iterations,
                    for_loop_tiling=shared_tiling,
                    use_if_else=shared_if_else,
                )
            kernels.append(kernel)

        return kernels
