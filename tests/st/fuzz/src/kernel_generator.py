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


class KernelGenerator:
    """Generator for InCore kernel functions with random operator chains.

    This class generates @pl.function(type=pl.FunctionType.InCore) kernels containing
    chains of randomly selected operators. Each kernel includes input loading, operator
    operations, and output storing.
    """

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
    ):
        """Initialize kernel generator

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operators (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
        """
        self.rng = random.Random(seed)
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

        Returns:
            Kernel metadata dictionary containing:
            - name: Kernel function name
            - inputs: Input tensor list [(name, shape), ...]
            - scalars: Scalar parameter list [(scalar_name, value), ...]
            - output_shape: Output tensor shape
            - op_chain: Operation chain
            - code: Generated PyPTO kernel code
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
                # If not aligned, regenerate aligned shape
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

        op_chain = self.fuzzer.generate_op_chain(
            num_ops=num_ops,
            input_count=actual_num_inputs,
            allow_scalars=allow_scalars,
            track_shapes=True,  # Track shapes for operations like row_expand
            default_shape=actual_output_shape,
        )

        input_names = [chr(97 + i) for i in range(actual_num_inputs)]  # a, b, c, ...
        inputs = [(name, actual_shapes[i]) for i, name in enumerate(input_names)]

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
        code = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            scalars=scalars,
            op_chain=op_chain,
            output_shape=actual_output_shape,
            scalar_value_to_param=scalar_value_to_param,
        )

        return {
            "name": kernel_name,
            "inputs": inputs,
            "scalars": scalars,
            "output_shape": actual_output_shape,
            "op_chain": op_chain,
            "code": code,
        }

    def _generate_matmul_memory_moves(
        self,
        input_var: str,
        target_memory: int,
        has_matmul: bool,
    ) -> tuple[str, list[str]]:
        """Generate memory move operations for matmul inputs.

        Args:
            input_var: Input variable name (e.g., "tile_a")
            target_memory: Target memory type (3 for L0A, 4 for L0B)
            has_matmul: Whether the kernel contains matmul operations

        Returns:
            Tuple of (final_var_name, list_of_code_lines)
        """
        code_lines = []

        if input_var.startswith("tile_") and not input_var.endswith(("_l0a", "_l0b", "_l0c")):
            input_l1 = f"{input_var}_l1" if has_matmul else input_var
            memory_suffix = "l0a" if target_memory == 3 else "l0b"
            output_var = f"{input_var}_{memory_suffix}"
            code_lines.append(
                f"        {output_var} = pl.block.move({input_l1}, target_memory={target_memory})"
            )
            return output_var, code_lines
        else:
            return input_var, code_lines

    def _generate_input_loads(
        self,
        inputs: list[tuple[str, tuple[int, int]]],
        has_matmul: bool,
    ) -> list[str]:
        """Generate input load operations."""
        code_lines = []
        for name, (r, c) in inputs:
            if has_matmul:
                code_lines.append(
                    f"        tile_{name}_l1 = pl.block.load({name}, offsets=[0, 0], "
                    f"shapes=[{r}, {c}], target_memory=2)"
                )
            else:
                code_lines.append(f"        tile_{name} = pl.load({name}, offsets=[0, 0], shapes=[{r}, {c}])")
        return code_lines

    def _generate_matmul_op(
        self,
        op_dict: dict[str, Any],
        has_matmul: bool,
    ) -> list[str]:
        """Generate matmul operation with memory moves."""
        code_lines = []
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]

        input_a_l0a, move_lines_a = self._generate_matmul_memory_moves(inputs_list[0], 3, has_matmul)
        code_lines.extend(move_lines_a)

        input_b_l0b, move_lines_b = self._generate_matmul_memory_moves(inputs_list[1], 4, has_matmul)
        code_lines.extend(move_lines_b)

        code_lines.append(f"        {output} = pl.block.matmul({input_a_l0a}, {input_b_l0b})")
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
            f"        {tmp_tile_var} = pl.block.create_tile([{tmp_shape[0]}, {tmp_shape[1]}], "
            f"dtype=pl.FP32, target_memory=1)"
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
    ) -> list[str]:
        """Generate store operation."""
        code_lines = []
        rows, cols = output_shape

        if op_chain:
            last_output = op_chain[-1]["output"]
            last_op = op_chain[-1]["op"]

            if last_op.name == "block.matmul":
                code_lines.append(
                    f"        result = pl.block.l0c_store({last_output}, offsets=[0, 0], "
                    f"shapes=[{rows}, {cols}], output_tensor=output)"
                )
            else:
                code_lines.append(
                    f"        result = pl.store({last_output}, offsets=[0, 0], "
                    f"shapes=[{rows}, {cols}], output_tensor=output)"
                )
        else:
            first_input = inputs[0][0]
            code_lines.append(
                f"        result = pl.store(tile_{first_input}, offsets=[0, 0], "
                f"shapes=[{rows}, {cols}], output_tensor=output)"
            )

        code_lines.append("        return result")
        return code_lines

    def _generate_kernel_code(
        self,
        kernel_name: str,
        inputs: list[tuple[str, tuple[int, int]]],
        scalars: list[tuple[str, str]],
        op_chain: list[dict[str, Any]],
        output_shape: tuple[int, int],
        scalar_value_to_param: dict[str, str],
    ) -> str:
        """Generate kernel function code.

        Args:
            kernel_name: Kernel function name
            inputs: Tensor input list
            scalars: Scalar parameter list [(param_name, value), ...]
            op_chain: Operation chain
            output_shape: Output tensor shape
            scalar_value_to_param: Mapping from scalar values to parameter names

        Returns:
            Generated PyPTO kernel code
        """
        rows, cols = output_shape

        params = []
        for name, (r, c) in inputs:
            params.append(f"{name}: pl.Tensor[[{r}, {c}], pl.FP32]")

        # Add scalar parameters
        for scalar_name, _ in scalars:
            params.append(f"{scalar_name}: pl.Scalar[pl.FP32]")

        params.append(f"output: pl.Tensor[[{rows}, {cols}], pl.FP32]")

        code_lines = [
            "    @pl.function(type=pl.FunctionType.InCore)",
            f"    def {kernel_name}(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        has_matmul = any(op_dict["op"].name == "block.matmul" for op_dict in op_chain)

        code_lines.extend(self._generate_input_loads(inputs, has_matmul))

        for op_dict in op_chain:
            op = op_dict["op"]

            if op.name == "block.matmul":
                code_lines.extend(self._generate_matmul_op(op_dict, has_matmul))
            elif op.constraints.get("requires_tmp_tile", False):
                code_lines.extend(self._generate_reduction_op(op_dict, output_shape))
            else:
                code_lines.append(self._generate_regular_op(op_dict, scalar_value_to_param))

        code_lines.extend(self._generate_store_op(op_chain, inputs, output_shape))

        return "\n".join(code_lines)

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
                )
            kernels.append(kernel)

        return kernels
