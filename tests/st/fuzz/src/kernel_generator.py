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
from typing import Any, Dict, List, Optional, Tuple

from .fuzzer import OpFuzzer, generate_aligned_shape, is_shape_aligned


class KernelGenerator:
    """InCore kernel functions"""

    def __init__(
        self,
        seed: Optional[int] = None,
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
        shape: Tuple[int, int] = (128, 128),
        allow_scalars: bool = True,
        input_shapes: Optional[List[Tuple[int, int]]] = None,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """InCore kernels

        Args:
            kernel_name: kernel functions
            num_inputs: （ input_shapes）
            num_ops:
            shape: default（ input_shapes）
            allow_scalars:
            input_shapes: ， num_inputs  shape
            output_shape: ，default

        Returns:
            kernels:
            - name: kernels
            - inputs:  [(name, shape), ...]
            - output_shape:
            - op_chain:
            - code:  PyPTO
        """
        if input_shapes is not None:
            actual_num_inputs = len(input_shapes)
            actual_shapes = input_shapes
        else:
            actual_num_inputs = num_inputs
            actual_shapes = [shape] * num_inputs

        # aligned
        dtype = "FP32"  #  FP32
        for i, input_shape in enumerate(actual_shapes):
            if not is_shape_aligned(input_shape, dtype):
                # aligned，aligned
                print(
                    f"Warning: Input shape {input_shape} is not 32-byte aligned. Regenerating aligned shape."
                )
                actual_shapes[i] = generate_aligned_shape(self.rng, dtype)

        # aligned
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
            track_shapes=True,  # （row_expand）
            default_shape=actual_output_shape,
        )

        input_names = [chr(97 + i) for i in range(actual_num_inputs)]  # a, b, c, ...
        inputs = [(name, actual_shapes[i]) for i, name in enumerate(input_names)]

        # kernels
        code = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            op_chain=op_chain,
            output_shape=actual_output_shape,
        )

        return {
            "name": kernel_name,
            "inputs": inputs,
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
        inputs: List[Tuple[str, Tuple[int, int]]],
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
        output_shape: Tuple[int, int],
    ) -> list[str]:
        """Generate reduction operation with temporary tile."""
        code_lines = []
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        op_name = op.name.replace("block.", "")

        tmp_shape = op_dict.get("output_shape", (output_shape[0], 1))
        tmp_tile_var = f"tmp_tile_{output}"
        code_lines.append(
            f"        {tmp_tile_var} = pl.block.create_tile([{tmp_shape[0]}, {tmp_shape[1]}], "
            f"dtype=pl.FP32, target_memory=1)"
        )
        code_lines.append(f"        {output} = pl.{op_name}({inputs_list[0]}, {tmp_tile_var})")
        return code_lines

    def _generate_regular_op(self, op_dict: dict[str, Any]) -> str:
        """Generate regular operation."""
        op = op_dict["op"]
        inputs_list = op_dict["inputs"]
        output = op_dict["output"]
        params = op_dict.get("params")
        op_name = op.name.replace("block.", "")

        inputs_str = ", ".join(inputs_list)
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"        {output} = pl.{op_name}({inputs_str}, {params_str})"
        return f"        {output} = pl.{op_name}({inputs_str})"

    def _generate_store_op(
        self,
        op_chain: List[Dict[str, Any]],
        inputs: List[Tuple[str, Tuple[int, int]]],
        output_shape: Tuple[int, int],
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
        inputs: List[Tuple[str, Tuple[int, int]]],
        op_chain: List[Dict[str, Any]],
        output_shape: Tuple[int, int],
    ) -> str:
        """kernel functions

        Args:
            kernel_name: kernels
            inputs:
            op_chain:
            output_shape:

        Returns:
             PyPTO
        """
        rows, cols = output_shape

        params = []
        for name, (r, c) in inputs:
            params.append(f"{name}: pl.Tensor[[{r}, {c}], pl.FP32]")
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
                code_lines.append(self._generate_regular_op(op_dict))

        code_lines.extend(self._generate_store_op(op_chain, inputs, output_shape))

        return "\n".join(code_lines)

    def generate_multiple_kernels(
        self,
        num_kernels: int = 3,
        num_inputs_range: Tuple[int, int] = (2, 3),
        num_ops_range: Tuple[int, int] = (3, 7),
        shape: Tuple[int, int] = (128, 128),
        input_shapes_list: Optional[List[List[Tuple[int, int]]]] = None,
        output_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """multiple InCore kernels

        Args:
            num_kernels: kernels
            num_inputs_range:  (min, max)
            num_ops_range:  (min, max)
            shape: default
            input_shapes_list: kernels，
                              : [[(128,128), (64,64)], [(256,256)], ...]
            output_shapes: kernels（）

        Returns:
            kernels
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
