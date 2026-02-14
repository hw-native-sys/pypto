# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test case base classes and data structures.

Provides the foundation for defining PTO test cases that can be
executed on both simulation and hardware platforms.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pypto.ir.pass_manager import OptimizationStrategy


class DataType(Enum):
    """Supported data types for tensors."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get corresponding torch dtype."""
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.INT32: torch.int32,
            DataType.INT64: torch.int64,
            DataType.BOOL: torch.bool,
        }
        return mapping[self]


@dataclass
class TensorSpec:
    """Specification for a test tensor.

    Attributes:
        name: Tensor name, used as parameter name in IR and C++ code.
        shape: Tensor shape as list of integers.
        dtype: Data type of tensor elements.
        init_value: Initial value for the tensor. Can be:
            - None: Will be zero-initialized
            - Scalar: All elements set to this value
            - torch.Tensor: Use this tensor directly
            - Callable: Function that returns a tensor given the shape
        is_output: Whether this tensor is an output (result to validate).
    """

    name: str
    shape: list[int]
    dtype: DataType
    init_value: int | float | torch.Tensor | Callable | None = None
    is_output: bool = False

    def create_array(self) -> torch.Tensor:
        """Create a torch tensor based on this specification."""
        if self.init_value is None:
            return torch.zeros(self.shape, dtype=self.dtype.torch_dtype)
        elif isinstance(self.init_value, torch.Tensor):
            return self.init_value.to(dtype=self.dtype.torch_dtype)
        elif callable(self.init_value):
            return torch.tensor(self.init_value(self.shape), dtype=self.dtype.torch_dtype)
        else:
            return torch.full(self.shape, self.init_value, dtype=self.dtype.torch_dtype)


@dataclass
class TestConfig:
    """Configuration for test execution.

    Attributes:
        platform: Target platform ("a2a3sim" or "a2a3").
        device_id: Device ID for hardware platform.
        atol: Absolute tolerance for result comparison.
        rtol: Relative tolerance for result comparison.
        block_dim: Number of blocks for parallel execution.
        aicpu_thread_num: Number of AICPU scheduler threads.
        save_kernels: If True, save generated kernels to persistent directory.
        save_kernels_dir: Directory to save generated kernels.
                          If None, defaults to build/outputs/output_{timestamp}/
                          Structure:
                            {save_dir}/{test_name}/
                              ├── kernels/aiv/
                              ├── kernels/orchestration/
                              ├── pass_dump/  (if dump_passes=True)
                              └── metadata.json
        dump_passes: If True, dump intermediate IR after each pass.
        codegen_only: If True, only generate code without executing runtime.
    """

    __test__ = False  # Not a pytest test class

    platform: str = "a2a3sim"
    device_id: int = 0
    atol: float = 1e-5
    rtol: float = 1e-5
    block_dim: int = 1
    aicpu_thread_num: int = 1
    save_kernels: bool = False
    save_kernels_dir: str | None = None
    dump_passes: bool = False
    codegen_only: bool = False

    def __post_init__(self):
        if self.platform not in ("a2a3sim", "a2a3"):
            raise ValueError(f"Invalid platform: {self.platform}")


@dataclass
class TestResult:
    """Result of a test execution.

    Attributes:
        passed: Whether the test passed.
        test_name: Name of the test case.
        error: Error message if test failed.
        max_abs_error: Maximum absolute error observed.
        max_rel_error: Maximum relative error observed.
        mismatch_count: Number of mismatched elements.
        mismatch_indices: Sample of indices with mismatches.
        execution_time: Time taken to execute (in seconds).
    """

    __test__ = False  # Not a pytest test class

    passed: bool
    test_name: str
    error: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    mismatch_count: int = 0
    mismatch_indices: list[tuple] | None = None
    execution_time: float | None = None

    def __str__(self) -> str:
        if self.passed:
            return f"PASS: {self.test_name}"
        else:
            msg = f"FAIL: {self.test_name}"
            if self.error:
                msg += f" - {self.error}"
            if self.max_abs_error is not None:
                msg += f" (max_abs_err={self.max_abs_error:.6e})"
            return msg


class PTOTestCase(ABC):
    """Abstract base class for PTO test cases.

    Subclasses must implement:
        - get_name(): Return the test case name
        - define_tensors(): Define input/output tensors
        - get_program(): Return a @pl.program class or ir.Program
        - compute_expected(): Compute expected results with NumPy (in-place)

    Optional overrides:
        - get_strategy(): Return optimization strategy (default: Default)

    Example:
        import pypto.language as pl

        class TestTileAdd(PTOTestCase):
            def get_name(self):
                return "tile_add_128x128"

            def define_tensors(self):
                return [
                    TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
                    TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
                    TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
                ]

            def get_program(self):
                @pl.program
                class TileAddProgram:
                    @pl.function(type=pl.FunctionType.InCore)
                    def tile_add(self, a: pl.Tensor[[128, 128], pl.FP32],
                                 b: pl.Tensor[[128, 128], pl.FP32],
                                 c: pl.Tensor[[128, 128], pl.FP32]):
                        tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                        tile_b = pl.block.load(b, offsets=[0, 0], shapes=[128, 128])
                        tile_c = pl.block.add(tile_a, tile_b)
                        pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return TileAddProgram
                @pl.function(type=pl.FunctionType.Orchestration)
                def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32],
                                 b: pl.Tensor[[128, 128], pl.FP32],
                                 c: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                    return self.tile_add(a, b, c)
                # if orchestration function is not implemented, will be auto-generated

            def compute_expected(self, tensors, params=None):
                tensors["c"][:] = tensors["a"] + tensors["b"]
    """

    def __init__(self, config: TestConfig | None = None):
        """Initialize test case.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or TestConfig()
        self._tensor_specs: list[TensorSpec] | None = None

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name for this test case."""
        pass

    @abstractmethod
    def define_tensors(self) -> list[TensorSpec]:
        """Define all input and output tensors for this test.

        Returns:
            List of TensorSpec objects defining the tensors.
        """
        pass

    @abstractmethod
    def get_program(self) -> Any:
        """Return a PyPTO Program for kernel code generation.

        Returns:
            PyPTO Program object (from @pl.program decorator or ir.Program).
        """
        pass

    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy for the pass pipeline.

        Override to use a different strategy (e.g., PTOAS).
        Default is OptimizationStrategy.Default.

        Returns:
            OptimizationStrategy enum value.
        """
        return OptimizationStrategy.Default

    @abstractmethod
    def compute_expected(
        self, tensors: dict[str, torch.Tensor], params: dict[str, Any] | None = None
    ) -> None:
        """Compute expected outputs using torch (modifies tensors in-place).

        This method should compute the expected outputs and write them directly
        to the output tensors in the tensors dict. This signature matches the
        compute_golden() function in generated golden.py files.

        Args:
            tensors: Dict mapping all tensor names (inputs and outputs) to torch tensors.
                     Modify output tensors in-place.
            params: Optional dict of parameters (for parameterized tests).

        Example:
            def compute_expected(self, tensors, params=None):
                # Simple computation
                tensors["c"][:] = tensors["a"] + tensors["b"]

            def compute_expected(self, tensors, params=None):
                # Complex multi-step computation
                temp = torch.exp(tensors["a"])
                result = torch.maximum(temp * tensors["b"], torch.tensor(0.0))
                tensors["output"][:] = torch.sqrt(result)
        """
        pass

    @property
    def tensor_specs(self) -> list[TensorSpec]:
        """Get cached tensor specifications."""
        if self._tensor_specs is None:
            self._tensor_specs = self.define_tensors()
        return self._tensor_specs
