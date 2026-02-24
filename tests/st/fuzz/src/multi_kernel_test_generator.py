# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Multi-kernel test case generator

This module is responsible for generating:
- Multiple InCore kernels
- Orchestration function
- PyTorch reference implementation
- PTOTestCase test class
"""

from typing import Any

import numpy as np
import torch

from .fuzzer import OpFuzzer
from .kernel_generator import KernelGenerator
from .orchestrator_generator import OrchestratorGenerator


class MultiKernelTestGenerator:
    """kernels"""

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
        tensor_init_type: str = "constant",
        validate_golden: bool = True,
    ):
        """Initialize test generator

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operators (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
            tensor_init_type: Tensor initialization type (constant, random, range, normal)
            validate_golden: Validate golden output (check for NaN/Inf)
        """
        self.seed = seed
        self.enable_advanced_ops = enable_advanced_ops
        self.advanced_ops_probability = advanced_ops_probability
        self.tensor_init_type = tensor_init_type
        self.validate_golden = validate_golden
        self.kernel_gen = KernelGenerator(
            seed=seed,
            enable_advanced_ops=enable_advanced_ops,
            advanced_ops_probability=advanced_ops_probability,
        )
        self.orch_gen = OrchestratorGenerator(seed=seed)
        self.fuzzer = OpFuzzer(
            seed=seed,
            enable_advanced_ops=enable_advanced_ops,
            advanced_ops_probability=advanced_ops_probability,
        )

    def _generate_tensor_init_value(self, tensor_index: int, init_type: str | None = None) -> str:
        """

        Args:
            tensor_index: （）
            init_type: ，Noneself.tensor_init_type

        Returns:

        """
        if init_type is None:
            init_type = self.tensor_init_type

        if init_type == "constant":
            # ：
            init_val = 2.0 + tensor_index * 0.5
            return f"init_value={init_val}"
        elif init_type == "random":
            # ：torch.randn，fuzzer
            return "init_value=torch.randn"
        elif init_type == "range":
            # ：01torch.rand
            return "init_value=torch.rand"
        elif init_type == "normal":
            # ：torch.randn，fuzzer
            return "init_value=torch.randn"
        elif init_type == "ones":
            # 1
            return "init_value=1.0"
        elif init_type == "zeros":
            # 0（，）
            return "init_value=0.0"
        else:
            # default
            init_val = 2.0 + tensor_index * 0.5
            return f"init_value={init_val}"

    def _compute_output_shapes_for_sequential(  # noqa: PLR0912
        self,
        num_kernels: int,
        default_shape: tuple[int, int],
        input_shapes_list: list[list[tuple[int, int]]] | None,
        mode: str,
    ) -> list[tuple[int, int]]:
        """kernels，

        Args:
            num_kernels: kernels
            default_shape: default
            input_shapes_list:
            mode:

        Returns:
            kernels
        """
        output_shapes = []

        if mode == "sequential":
            # ：kernel_i  kernel_{i+1}
            for i in range(num_kernels):
                if i == num_kernels - 1:
                    # kernels：
                    if input_shapes_list and i < len(input_shapes_list):
                        output_shapes.append(input_shapes_list[i][0])
                    else:
                        output_shapes.append(default_shape)
                # kernels：kernels
                elif input_shapes_list and i + 1 < len(input_shapes_list):
                    next_kernel_first_input = input_shapes_list[i + 1][0]
                    output_shapes.append(next_kernel_first_input)
                else:
                    output_shapes.append(default_shape)

        elif mode == "branching":
            # ：kernels（）
            # kernels
            if input_shapes_list and len(input_shapes_list) > 0:
                unified_output_shape = input_shapes_list[0][0]
            else:
                unified_output_shape = default_shape

            for i in range(num_kernels):
                output_shapes.append(unified_output_shape)

        elif mode == "mixed":
            # ：，
            mid = num_kernels // 2

            # ：kernels
            if input_shapes_list and len(input_shapes_list) > 0:
                parallel_output_shape = input_shapes_list[0][0]
            else:
                parallel_output_shape = default_shape

            for i in range(num_kernels):
                if i < mid:
                    # ：
                    output_shapes.append(parallel_output_shape)
                elif i == mid:
                    # kernels：kernels（）
                    if i == num_kernels - 1:
                        # ，
                        if input_shapes_list and i < len(input_shapes_list):
                            output_shapes.append(input_shapes_list[i][0])
                        else:
                            output_shapes.append(default_shape)
                    # kernels
                    elif input_shapes_list and i + 1 < len(input_shapes_list):
                        output_shapes.append(input_shapes_list[i + 1][0])
                    else:
                        output_shapes.append(default_shape)
                # kernels
                elif i == num_kernels - 1:
                    # kernels
                    if input_shapes_list and i < len(input_shapes_list):
                        output_shapes.append(input_shapes_list[i][0])
                    else:
                        output_shapes.append(default_shape)
                # kernels
                elif input_shapes_list and i + 1 < len(input_shapes_list):
                    output_shapes.append(input_shapes_list[i + 1][0])
                else:
                    output_shapes.append(default_shape)

        return output_shapes

    def _regenerate_kernel_code_with_unified_shapes(
        self,
        kernel: dict[str, Any],
        input_shapes_map: dict[str, tuple[int, int]],
    ) -> str:
        """Regenerate kernel code with unified input shapes

        Args:
            kernel: Kernel information
            input_shapes_map: Mapping from input names to unified shapes

        Returns:
            Regenerated kernel code
        """
        # Update kernel inputs with unified shapes
        unified_inputs = [(inp_name, input_shapes_map[inp_name]) for inp_name, _ in kernel["inputs"]]

        # Get scalars from kernel
        scalars = kernel.get("scalars", [])

        # Build scalar value to param mapping
        scalar_value_to_param = {}
        for param_name, value in scalars:
            scalar_value_to_param[value] = param_name

        # Reuse the kernel generator's code generation logic
        return self.kernel_gen._generate_kernel_code(
            kernel_name=kernel["name"],
            inputs=unified_inputs,
            scalars=scalars,
            op_chain=kernel["op_chain"],
            output_shape=kernel["output_shape"],
            scalar_value_to_param=scalar_value_to_param,
        )

    def generate_test_case(
        self,
        test_name: str,
        num_kernels: int = 3,
        orchestration_mode: str = "sequential",
        shape: tuple[int, int] = (128, 128),
        num_ops_range: tuple[int, int] = (3, 7),
        input_shapes_list: list[list[tuple[int, int]]] | None = None,
        tensor_init_type: str | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> str:
        """

        Args:
            test_name:
            num_kernels: kernels
            orchestration_mode:  ("sequential", "branching", "mixed")
            shape:
            num_ops_range: kernels
            input_shapes_list: kernels（）
            tensor_init_type: （，）
            atol: Absolute error tolerance
            rtol: Relative error tolerance

        Returns:

        """
        #  sequential、branching  mixed ，
        if orchestration_mode in ["sequential", "branching", "mixed"]:
            output_shapes = self._compute_output_shapes_for_sequential(
                num_kernels, shape, input_shapes_list, orchestration_mode
            )
        else:
            output_shapes = None

        # multiplekernels
        kernels = self.kernel_gen.generate_multiple_kernels(
            num_kernels=num_kernels,
            num_inputs_range=(2, 3),
            num_ops_range=num_ops_range,
            shape=shape,
            input_shapes_list=input_shapes_list,
            output_shapes=output_shapes,
        )

        #  Orchestration
        if orchestration_mode == "sequential":
            orch_info = self.orch_gen.generate_sequential(kernels, shape)
        elif orchestration_mode == "branching":
            orch_info = self.orch_gen.generate_branching(kernels, shape)
        elif orchestration_mode == "mixed":
            orch_info = self.orch_gen.generate_mixed(kernels, shape)
        else:
            raise ValueError(f": {orchestration_mode}")

        #  Torch reference implementation
        torch_code = self._generate_torch_reference(kernels, orch_info)

        # test class
        test_code = self._generate_test_class(
            test_name=test_name,
            kernels=kernels,
            orch_info=orch_info,
            torch_code=torch_code,
            shape=shape,
            tensor_init_type=tensor_init_type,
            atol=atol,
            rtol=rtol,
        )

        #  golden （ NaN/Inf）
        if self.validate_golden:
            self._validate_golden_output(kernels, orch_info, shape, tensor_init_type or self.tensor_init_type)

        return test_code

    def _generate_torch_reference(
        self,
        kernels: list[dict[str, Any]],
        orch_info: dict[str, Any],
    ) -> str:
        """Torch reference implementation

        Args:
            kernels: kernels
            orch_info: Orchestration

        Returns:
            Torch reference implementation
        """
        code_lines = []

        # kernels Torch
        for kernel in kernels:
            kernel_name = kernel["name"]
            input_names = [inp[0] for inp in kernel["inputs"]]
            op_chain = kernel["op_chain"]

            #  self
            code_lines.append(f"    def _torch_{kernel_name}({', '.join(input_names)}):")
            code_lines.append(f'        """Torch reference implementation for {kernel_name}"""')

            code_lines.append("        env = {}")
            for name in input_names:
                code_lines.append(f"        env['tile_{name}'] = {name}.clone()")

            code_lines.append("")
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
                    torch_expr = self._get_torch_operation(op.name, input_vals)
                    code_lines.append(f"        env['{output}'] = {torch_expr}")

            code_lines.append(f"        return env['{op_chain[-1]['output']}']")
            code_lines.append("")

        return "\n".join(code_lines)

    # Operation mapping: PyPTO op name -> torch expression template or callable
    _TORCH_OP_MAP = {
        # Binary arithmetic operations
        "block.add": lambda v: f"{v[0]} + {v[1]}",
        "block.sub": lambda v: f"{v[0]} - {v[1]}",
        "block.mul": lambda v: f"{v[0]} * {v[1]}",
        "block.div": lambda v: f"{v[0]} / {v[1]}",
        "block.adds": lambda v: f"{v[0]} + {v[1]}",
        "block.subs": lambda v: f"{v[0]} - {v[1]}",
        "block.muls": lambda v: f"{v[0]} * {v[1]}",
        "block.divs": lambda v: f"{v[0]} / {v[1]}",
        # Binary comparison operations
        "block.maximum": lambda v: f"torch.maximum({v[0]}, {v[1]})",
        "block.minimum": lambda v: f"torch.minimum({v[0]}, {v[1]})",
        # Unary operations
        "block.sqrt": lambda v: f"torch.sqrt({v[0]})",
        "block.rsqrt": lambda v: f"torch.rsqrt({v[0]})",
        "block.exp": lambda v: f"torch.exp(torch.clamp({v[0]}, -10, 10))",
        "block.neg": lambda v: f"-{v[0]}",
        "block.recip": lambda v: f"torch.reciprocal({v[0]})",
        "block.log": lambda v: f"torch.log({v[0]})",
        "block.abs": lambda v: f"torch.abs({v[0]})",
        "block.relu": lambda v: f"torch.relu({v[0]})",
        # Row expand operations (broadcasting)
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

    def _get_torch_operation(self, op_name: str, input_vals: list[str]) -> str:
        """Convert PyPTO operation to Torch expression.

        Args:
            op_name: PyPTO operation name (e.g., "block.add")
            input_vals: Input value expressions

        Returns:
            Torch expression string
        """
        op_func = self._TORCH_OP_MAP.get(op_name)
        if op_func:
            return op_func(input_vals)
        return f"# Unsupported operation: {op_name}"

    def _generate_test_class(  # noqa: PLR0912, PLR0915
        self,
        test_name: str,
        kernels: list[dict[str, Any]],
        orch_info: dict[str, Any],
        torch_code: str,
        shape: tuple[int, int],
        tensor_init_type: str | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> str:
        """test class

        Args:
            test_name:
            kernels: kernels
            orch_info: Orchestration
            torch_code: Torch reference implementation
            shape:
            tensor_init_type: （，）
            atol: Absolute error tolerance
            rtol: Relative error tolerance

        Returns:
            test class
        """
        rows, cols = shape
        class_name = f"Test{test_name.replace('_', ' ').title().replace(' ', '')}"

        input_shapes_map = {}  # {input_name: shape}
        for kernel in kernels:
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in input_shapes_map:
                    input_shapes_map[inp_name] = inp_shape
                # kernels，
                elif inp_shape != input_shapes_map[inp_name]:
                    existing_size = input_shapes_map[inp_name][0] * input_shapes_map[inp_name][1]
                    new_size = inp_shape[0] * inp_shape[1]
                    if new_size > existing_size:
                        input_shapes_map[inp_name] = inp_shape

        input_list = sorted(input_shapes_map.keys())

        # kernels
        output_shape = kernels[-1]["output_shape"] if kernels else shape

        code_lines = [
            f"class {class_name}(PTOTestCase):",
            '    """',
            f"    : {test_name}",
            f"    : {orch_info['mode']}",
            f"    kernels: {len(kernels)}",
            '    """',
            "",
            f"    rows = {rows}",
            f"    cols = {cols}",
            "",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.config.atol = {atol}",
            f"        self.config.rtol = {rtol}",
            "",
            "    def get_name(self) -> str:",
            f"        return '{test_name}'",
            "",
            "    def define_tensors(self) -> List[TensorSpec]:",
            "        return [",
        ]

        #  -
        for idx, inp_name in enumerate(input_list):
            inp_shape = input_shapes_map[inp_name]
            init_code = self._generate_tensor_init_value(idx, tensor_init_type)
            code_lines.append(
                f"            TensorSpec('{inp_name}', [{inp_shape[0]}, {inp_shape[1]}], "
                f"DataType.FP32, {init_code}),"
            )

        #  -
        code_lines.append(
            f"            TensorSpec('output', [{output_shape[0]}, {output_shape[1]}], "
            f"DataType.FP32, is_output=True),"
        )
        code_lines.append("        ]")
        code_lines.append("")

        #  PyPTO
        code_lines.append("    def get_program(self) -> Any:")
        code_lines.append("        import pypto.language as pl")
        code_lines.append("")
        code_lines.append("        @pl.program")
        code_lines.append(f"        class {test_name.replace('_', ' ').title().replace(' ', '')}Program:")

        # kernels（）
        for kernel in kernels:
            #  kernel
            regenerated_code = self._regenerate_kernel_code_with_unified_shapes(kernel, input_shapes_map)
            # kernels8（4get_program，4@pl.program）
            kernel_lines = regenerated_code.split("\n")
            for line in kernel_lines:
                code_lines.append(f"        {line}")
            code_lines.append("")

        # kernels（）
        if orch_info.get("needs_merge_kernel", False):
            merge_code = self.orch_gen.generate_merge_kernel(shape)
            merge_lines = merge_code.split("\n")
            for line in merge_lines:
                code_lines.append(f"        {line}")
            code_lines.append("")

        #  Orchestration
        orch_lines = orch_info["code"].split("\n")
        for line in orch_lines:
            code_lines.append(f"        {line}")
        code_lines.append("")

        code_lines.append(f"        return {test_name.replace('_', ' ').title().replace(' ', '')}Program")
        code_lines.append("")

        #  Torch reference implementation
        code_lines.append("    def compute_expected(self, tensors, params=None):")
        code_lines.append('        """Compute expected output using Torch reference implementation"""')
        code_lines.append("        torch_tensors = {}")
        code_lines.append("        for name, arr in tensors.items():")
        code_lines.append("            if not name.endswith('output'):")
        code_lines.append("                if isinstance(arr, torch.Tensor):")
        code_lines.append("                    torch_tensors[name] = arr")
        code_lines.append("                else:")
        code_lines.append("                    torch_tensors[name] = torch.from_numpy(arr)")
        code_lines.append("")
        # torch_code ， compute_expected ，
        torch_lines = torch_code.split("\n")
        for line in torch_lines:
            if line.strip():
                code_lines.append(f"    {line}")  # 4
            else:
                code_lines.append(line)
        code_lines.append("")

        if orch_info["mode"] == "sequential":
            result_var = None
            for i, kernel in enumerate(kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]

                if i > 0 and result_var:
                    kernel_inputs[0] = result_var
                    inputs_parts = [kernel_inputs[0]]
                    for inp in kernel_inputs[1:]:
                        inputs_parts.append(f"torch_tensors['{inp}']")
                    inputs_str = ", ".join(inputs_parts)
                else:
                    inputs_str = ", ".join([f"torch_tensors['{inp}']" for inp in kernel_inputs])

                result_var = f"result_{i}"
                code_lines.append(f"        {result_var} = _torch_{kernel_name}({inputs_str})")

            code_lines.append("        if isinstance(tensors['output'], torch.Tensor):")
            code_lines.append(f"            tensors['output'][:] = {result_var}")
            code_lines.append("        else:")
            code_lines.append(f"            tensors['output'][:] = {result_var}.numpy()")

        elif orch_info["mode"] == "branching":
            branch_results = []
            for i, kernel in enumerate(kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                result_var = f"branch_{i}"
                branch_results.append(result_var)

                inputs_str = ", ".join([f"torch_tensors['{inp}']" for inp in kernel_inputs])
                code_lines.append(f"        {result_var} = _torch_{kernel_name}({inputs_str})")

            if len(branch_results) == 1:
                code_lines.append("        if isinstance(tensors['output'], torch.Tensor):")
                code_lines.append(f"            tensors['output'][:] = {branch_results[0]}")
                code_lines.append("        else:")
                code_lines.append(f"            tensors['output'][:] = {branch_results[0]}.numpy()")
            else:
                merged = branch_results[0]
                for i in range(1, len(branch_results)):
                    new_merged = f"merged_{i}"
                    code_lines.append(f"        {new_merged} = {merged} + {branch_results[i]}")
                    merged = new_merged
                code_lines.append("        if isinstance(tensors['output'], torch.Tensor):")
                code_lines.append(f"            tensors['output'][:] = {merged}")
                code_lines.append("        else:")
                code_lines.append(f"            tensors['output'][:] = {merged}.numpy()")

        elif orch_info["mode"] == "mixed":
            mid = len(kernels) // 2
            parallel_kernels = kernels[:mid]
            sequential_kernels = kernels[mid:]

            branch_results = []
            for i, kernel in enumerate(parallel_kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                result_var = f"parallel_{i}"
                branch_results.append(result_var)

                inputs_str = ", ".join([f"torch_tensors['{inp}']" for inp in kernel_inputs])
                code_lines.append(f"        {result_var} = _torch_{kernel_name}({inputs_str})")

            if len(branch_results) > 1:
                merged = branch_results[0]
                for i in range(1, len(branch_results)):
                    new_merged = f"merged_parallel_{i}"
                    code_lines.append(f"        {new_merged} = {merged} + {branch_results[i]}")
                    merged = new_merged
                current_result = merged
            else:
                current_result = branch_results[0]

            for i, kernel in enumerate(sequential_kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                kernel_inputs[0] = current_result

                result_var = f"sequential_{i}"
                inputs_parts = [kernel_inputs[0]]
                for inp in kernel_inputs[1:]:
                    inputs_parts.append(f"torch_tensors['{inp}']")
                inputs_str = ", ".join(inputs_parts)
                code_lines.append(f"        {result_var} = _torch_{kernel_name}({inputs_str})")
                current_result = result_var

            code_lines.append("        if isinstance(tensors['output'], torch.Tensor):")
            code_lines.append(f"            tensors['output'][:] = {current_result}")
            code_lines.append("        else:")
            code_lines.append(f"            tensors['output'][:] = {current_result}.numpy()")

        code_lines.append("")

        return "\n".join(code_lines)

    def _validate_golden_output(  # noqa: PLR0912
        self,
        kernels: list[dict[str, Any]],
        orch_info: dict[str, Any],
        shape: tuple[int, int],
        tensor_init_type: str,
    ) -> None:
        """golden ， NaN/Inf

        Args:
            kernels: kernels
            orch_info: Orchestration
            shape:
            tensor_init_type:

        Raises:
            ValueError:  golden  NaN  Inf
        """
        tensors = {}
        for i, kernel in enumerate(kernels):
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in tensors:
                    if tensor_init_type == "constant":
                        value = 2.0 + i * 0.5
                        tensors[inp_name] = torch.full(inp_shape, value, dtype=torch.float32)
                    elif tensor_init_type == "random":
                        tensors[inp_name] = torch.randn(inp_shape, dtype=torch.float32)
                    elif tensor_init_type == "range":
                        tensors[inp_name] = torch.rand(inp_shape, dtype=torch.float32)
                    elif tensor_init_type == "normal":
                        tensors[inp_name] = torch.randn(inp_shape, dtype=torch.float32)
                    elif tensor_init_type == "ones":
                        tensors[inp_name] = torch.ones(inp_shape, dtype=torch.float32)
                    elif tensor_init_type == "zeros":
                        tensors[inp_name] = torch.zeros(inp_shape, dtype=torch.float32)
                    else:
                        tensors[inp_name] = torch.full(inp_shape, 2.0, dtype=torch.float32)

        # kernels Torch
        kernel_results = {}
        for kernel in kernels:
            kernel_name = kernel["name"]
            input_names = [inp[0] for inp in kernel["inputs"]]
            op_chain = kernel["op_chain"]

            env = {}
            for inp_name in input_names:
                env[f"tile_{inp_name}"] = tensors[inp_name].clone()

            for op_dict in op_chain:
                op = op_dict["op"]
                inputs = op_dict["inputs"]
                output = op_dict["output"]

                input_vals = []
                for inp in inputs:
                    if inp in env:
                        val = env[inp]
                    else:
                        try:
                            val = float(inp)
                        except ValueError:
                            val = env.get(inp, torch.tensor(0.0))
                    input_vals.append(val)

                if "avoid_zero" in op.constraints and op.constraints["avoid_zero"]:
                    for i, val in enumerate(input_vals):
                        if isinstance(val, torch.Tensor):
                            input_vals[i] = torch.where(torch.abs(val) < 0.01, torch.tensor(1.0), val)

                if "positive_only" in op.constraints and op.constraints["positive_only"]:
                    for i, val in enumerate(input_vals):
                        if isinstance(val, torch.Tensor):
                            input_vals[i] = torch.abs(val) + 1e-6

                # （，）
                try:
                    if op.np_equivalent:
                        #  numpy
                        np_inputs = [v.numpy() if isinstance(v, torch.Tensor) else v for v in input_vals]
                        result = op.np_equivalent(*np_inputs)
                        env[output] = (
                            torch.from_numpy(result)
                            if isinstance(result, np.ndarray)
                            else torch.tensor(result)
                        )
                except Exception as e:
                    print(f"Warning: Failed to execute {op.name}: {e}")
                    env[output] = torch.zeros_like(
                        input_vals[0] if isinstance(input_vals[0], torch.Tensor) else torch.tensor(0.0)
                    )

            if op_chain:
                kernel_results[kernel_name] = env[op_chain[-1]["output"]]

        # Check
        if kernel_results:
            final_result = list(kernel_results.values())[-1]
            if torch.isnan(final_result).any():
                raise ValueError("Golden output contains NaN! This test case is invalid.")
            if torch.isinf(final_result).any():
                raise ValueError("Golden output contains Inf! This test case is invalid.")

            #  golden ，
            # print(f"✓ Golden validation passed (no NaN/Inf detected)")
            # print(f"  Golden output shape: {final_result.shape}")
            # print(f"  Golden output sample values: min={final_result.min().item():.6f}, "
            #       f"max={final_result.max().item():.6f}, mean={final_result.mean().item():.6f}")
            # print(f"  Golden output[0,0] = {final_result[0,0].item():.6f}")
