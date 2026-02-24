# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Orchestration function generator.

This module generates @pl.function(type=pl.FunctionType.Orchestration) functions that
coordinate multiple InCore kernels. Supported orchestration modes:
- Sequential: kernels execute one after another
- Branching: kernels execute in parallel branches
- Mixed: combination of parallel and sequential execution
"""

import random
from typing import Any


class OrchestratorGenerator:
    """Generates orchestration functions for coordinating multiple InCore kernels."""

    def __init__(self, seed: int | None = None):
        """Initialize the orchestrator generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    @staticmethod
    def _collect_input_shapes(kernels: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        """Collect unified input shapes across all kernels.

        When multiple kernels use the same input name with different shapes,
        the larger shape (by total element count) is kept.

        Args:
            kernels: List of kernel info dicts

        Returns:
            Mapping from input names to their unified shapes
        """
        input_shapes_map: dict[str, tuple[int, int]] = {}
        for kernel in kernels:
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in input_shapes_map:
                    input_shapes_map[inp_name] = inp_shape
                elif inp_shape != input_shapes_map[inp_name]:
                    existing_size = input_shapes_map[inp_name][0] * input_shapes_map[inp_name][1]
                    new_size = inp_shape[0] * inp_shape[1]
                    if new_size > existing_size:
                        input_shapes_map[inp_name] = inp_shape
        return input_shapes_map

    @staticmethod
    def _build_params(
        input_shapes_map: dict[str, tuple[int, int]],
    ) -> list[str]:
        """Build PL-typed parameter strings from an input shapes map.

        Args:
            input_shapes_map: Mapping from input names to shapes

        Returns:
            List of parameter strings (e.g., "x: pl.Tensor[[128, 128], pl.FP32]")
        """
        input_params = sorted(input_shapes_map.keys())
        params = []
        for name in input_params:
            inp_shape = input_shapes_map[name]
            params.append(f"{name}: pl.Tensor[[{inp_shape[0]}, {inp_shape[1]}], pl.FP32]")
        return params

    def generate_sequential(
        self,
        kernels: list[dict[str, Any]],
        shape: tuple[int, int] = (128, 128),
    ) -> dict[str, Any]:
        """Generate a sequential orchestration function.

        Chains kernels so that each kernel's output feeds into the next kernel's first input.

        Args:
            kernels: List of kernel info dicts
            shape: Default tensor shape (rows, cols)

        Returns:
            Orchestration function info dict
        """
        if not kernels:
            raise ValueError("At least one kernel is required")

        input_shapes_map = self._collect_input_shapes(kernels)
        input_params = sorted(input_shapes_map.keys())
        params = self._build_params(input_shapes_map)

        # Output shape is determined by the last kernel
        output_shape = kernels[-1]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # Call kernels sequentially - each returns a tensor
        result_var = None
        for i, kernel in enumerate(kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]

            # For subsequent kernels, use previous kernel's output as first input
            if i > 0 and result_var:
                kernel_inputs[0] = result_var

            # Add scalar arguments
            scalar_args = [value for _, value in kernel.get("scalars", [])]

            # InCore kernels return a tensor
            result_var = f"result_{i}"
            all_args = kernel_inputs + scalar_args
            inputs_str = ", ".join(all_args)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        code_lines.append(f"        return {result_var}")

        return {
            "mode": "sequential",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
        }

    def generate_branching(
        self,
        kernels: list[dict[str, Any]],
        shape: tuple[int, int] = (128, 128),
    ) -> dict[str, Any]:
        """Generate a branching orchestration function.

        Runs multiple kernels in parallel branches, then merges the results.

        Args:
            kernels: List of kernel info dicts
            shape: Default tensor shape (rows, cols)

        Returns:
            Orchestration function info dict
        """
        if not kernels:
            raise ValueError("At least one kernel is required")

        input_shapes_map = self._collect_input_shapes(kernels)
        input_params = sorted(input_shapes_map.keys())
        params = self._build_params(input_shapes_map)

        # In branching mode, all kernels must produce the same output shape for merging
        output_shape = kernels[0]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # Run all kernels in parallel - each returns a tensor
        result_vars = []
        for i, kernel in enumerate(kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]
            result_var = f"branch_{i}"
            result_vars.append(result_var)

            # Add scalar arguments
            scalar_args = [value for _, value in kernel.get("scalars", [])]
            all_args = kernel_inputs + scalar_args
            inputs_str = ", ".join(all_args)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        if len(result_vars) == 1:
            code_lines.append(f"        return {result_vars[0]}")
        else:
            # Merge branch results via add
            code_lines.append("        # Merge branch results")
            merged = result_vars[0]
            for i in range(1, len(result_vars)):
                new_merged = f"merged_{i}"
                code_lines.append(f"        {new_merged} = self.merge_results({merged}, {result_vars[i]})")
                merged = new_merged
            code_lines.append(f"        return {merged}")

        return {
            "mode": "branching",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
            "needs_merge_kernel": len(result_vars) > 1,
        }

    def generate_mixed(
        self,
        kernels: list[dict[str, Any]],
        shape: tuple[int, int] = (128, 128),
    ) -> dict[str, Any]:
        """Generate a mixed orchestration function.

        Combines parallel branches with sequential execution.

        Args:
            kernels: List of kernel info dicts
            shape: Default tensor shape (rows, cols)

        Returns:
            Orchestration function info dict
        """
        if len(kernels) < 2:
            # Need at least 2 kernels for mixed mode; fall back to sequential
            return self.generate_sequential(kernels, shape)

        input_shapes_map = self._collect_input_shapes(kernels)
        input_params = sorted(input_shapes_map.keys())
        params = self._build_params(input_shapes_map)

        # Output shape is determined by the last kernel
        output_shape = kernels[-1]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # Split kernels: first half runs in parallel, second half runs sequentially
        mid = len(kernels) // 2
        parallel_kernels = kernels[:mid]
        sequential_kernels = kernels[mid:]

        # Run parallel kernels - each returns a tensor
        branch_results = []
        for i, kernel in enumerate(parallel_kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]
            result_var = f"parallel_{i}"
            branch_results.append(result_var)

            # Add scalar arguments
            scalar_args = [value for _, value in kernel.get("scalars", [])]
            all_args = kernel_inputs + scalar_args
            inputs_str = ", ".join(all_args)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        if len(branch_results) > 1:
            code_lines.append("        # Merge parallel results")
            merged = branch_results[0]
            for i in range(1, len(branch_results)):
                new_merged = f"merged_parallel_{i}"
                code_lines.append(f"        {new_merged} = self.merge_results({merged}, {branch_results[i]})")
                merged = new_merged
            current_result = merged
        else:
            current_result = branch_results[0]

        for i, kernel in enumerate(sequential_kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]

            kernel_inputs[0] = current_result

            result_var = f"sequential_{i}"
            # Add scalar arguments
            scalar_args = [value for _, value in kernel.get("scalars", [])]
            all_args = kernel_inputs + scalar_args
            inputs_str = ", ".join(all_args)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")
            current_result = result_var

        code_lines.append(f"        return {current_result}")

        return {
            "mode": "mixed",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
            "needs_merge_kernel": len(branch_results) > 1,
        }

    def generate_merge_kernel(self, shape: tuple[int, int] = (128, 128)) -> str:
        """Generate a merge kernel that adds two tensors element-wise.

        Args:
            shape: Tensor shape (rows, cols)

        Returns:
            Generated merge kernel code string
        """
        rows, cols = shape
        code = f"""    @pl.function(type=pl.FunctionType.InCore)
    def merge_results(self, a: pl.Tensor[[{rows}, {cols}], pl.FP32],
                      b: pl.Tensor[[{rows}, {cols}], pl.FP32],
                      output: pl.Tensor[[{rows}, {cols}], pl.FP32]) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[{rows}, {cols}])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[{rows}, {cols}])
        result_tile = pl.add(tile_a, tile_b)
        result = pl.store(result_tile, offsets=[0, 0], shapes=[{rows}, {cols}], output_tensor=output)
        return result"""
        return code
