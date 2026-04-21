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

import pytest
import torch
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig
from pypto.runtime.tensor_spec import ScalarSpec

# ---------------------------------------------------------------------------
# Pre-defined platform parameter lists for @pytest.mark.parametrize.
#
# A platform string is one of "a2a3", "a5", "a2a3sim", "a5sim".  The architecture
# prefix selects the backend (Ascend910B / Ascend950) and the optional ``sim``
# suffix toggles between simulator and on-board (real chip) execution.
#
# Filtering happens in two layers:
#     1. CLI ``--platform`` accepts a comma-separated subset (default ``a2a3``,
#        matching legacy on-NPU CI behaviour); non-matching parametrize
#        variants are deselected during collection.
#     2. ``@pytest.mark.platforms("a5", "a5sim")`` on a test function further
#        restricts that test to the listed platforms.
#
# Two parallel families of constants exist here on purpose:
#     - ``*_PLATFORM_IDS`` are plain string tuples used by code that needs
#       to *compare* against requested platform ids (e.g. the runtime
#       hardware-availability gate or the CLI parser).
#     - ``PLATFORMS`` / ``SIM_PLATFORMS`` / ``ONBOARD_PLATFORMS`` are
#       ``pytest.param`` lists used by ``@pytest.mark.parametrize``; they
#       cannot be put in a ``set()`` and therefore must not be used for
#       string membership checks.
#
# Usage:
#     @pytest.mark.parametrize("platform", PLATFORMS)
#     def test_foo(self, test_runner, platform):
#         result = test_runner.run(MyTestCase(platform=platform))
#         assert result.passed
# ---------------------------------------------------------------------------

SIM_PLATFORM_IDS: tuple[str, ...] = ("a2a3sim", "a5sim")
ONBOARD_PLATFORM_IDS: tuple[str, ...] = ("a2a3", "a5")
ALL_PLATFORM_IDS: tuple[str, ...] = (*SIM_PLATFORM_IDS, *ONBOARD_PLATFORM_IDS)

PLATFORMS = [pytest.param(p, id=p) for p in ALL_PLATFORM_IDS]
SIM_PLATFORMS = [pytest.param(p, id=p) for p in SIM_PLATFORM_IDS]
ONBOARD_PLATFORMS = [pytest.param(p, id=p) for p in ONBOARD_PLATFORM_IDS]

_PLATFORM_TO_BACKEND: dict[str, BackendType] = {
    "a2a3": BackendType.Ascend910B,
    "a2a3sim": BackendType.Ascend910B,
    "a5": BackendType.Ascend950,
    "a5sim": BackendType.Ascend950,
}


def platform_to_backend(platform: str) -> BackendType:
    """Return the BackendType corresponding to *platform* string."""
    try:
        return _PLATFORM_TO_BACKEND[platform]
    except KeyError as exc:
        raise ValueError(f"Unknown platform '{platform}'. Expected one of {ALL_PLATFORM_IDS}.") from exc


class DataType(Enum):
    """Supported data types for tensors."""

    BF16 = "bf16"
    FP32 = "fp32"
    FP16 = "fp16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    BOOL = "bool"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get corresponding torch dtype."""
        mapping = {
            DataType.BF16: torch.bfloat16,
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.INT32: torch.int32,
            DataType.UINT32: torch.int32,  # PyTorch has no uint32; use int32 (same bits)
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
                        tile_a = pl.tile.load(a, offsets=[0, 0], shapes=[128, 128])
                        tile_b = pl.tile.load(b, offsets=[0, 0], shapes=[128, 128])
                        tile_c = pl.tile.add(tile_a, tile_b)
                        pl.tile.store(tile_c, offsets=[0, 0], output_tensor=c)
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

    def __init__(
        self,
        config: RunConfig | None = None,
        *,
        platform: str | None = None,
        backend_type: BackendType | None = None,
        strategy: OptimizationStrategy | None = None,
    ):
        """Initialize test case.

        Args:
            config: Test configuration. If None, uses default config.
            platform: Override the target platform string ("a2a3", "a5",
                "a2a3sim", "a5sim").  If None, falls back to the class-level
                ``get_platform()`` (which defaults to ``None``, deferring to
                the session-wide ``--platform`` CLI value, currently
                ``a2a3``).  Pass explicitly to run the same test case on a
                different platform without subclassing.
            backend_type: (Legacy) Override the backend type for code
                generation.  Prefer ``platform``.  If both are given, the
                value derived from ``platform`` wins.
            strategy: Override the optimization strategy.  If None, falls
                back to the class-level ``get_strategy()`` default (Default).
        """
        self.config = config or RunConfig()
        self._override_platform = platform
        self._override_backend = backend_type
        self._override_strategy = strategy
        self._tensor_specs: list[TensorSpec] | None = None
        self._scalar_specs: list[ScalarSpec] | None = None

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

        If *strategy* was passed to the constructor, that value takes
        precedence.  Otherwise falls back to ``OptimizationStrategy.Default``.
        Subclasses may still override this method; the constructor override
        only applies when the subclass does **not** redefine the method.

        Returns:
            OptimizationStrategy enum value.
        """
        if self._override_strategy is not None:
            return self._override_strategy
        return OptimizationStrategy.Default

    def get_platform(self) -> str | None:
        """Return the target platform string ("a2a3"/"a5"/"a2a3sim"/"a5sim").

        Resolution order:
            1. The ``platform`` constructor arg, if set, wins.
            2. ``None`` otherwise, signalling that the runner should fall
               back to the session-wide ``--platform`` CLI value.

        Subclasses may still override this method to hard-pin a platform; the
        constructor override only applies when the subclass does **not**
        redefine the method.
        """
        return self._override_platform

    def get_backend_type(self) -> BackendType:
        """Return the backend type for code generation.

        Resolution order:
            1. The ``platform`` constructor arg, if set, decides the backend
               via :data:`_PLATFORM_TO_BACKEND` (preferred path).
            2. The legacy ``backend_type`` constructor arg, if set.
            3. ``BackendType.Ascend910B`` as the global default.

        Subclasses may still override this method; the constructor override
        only applies when the subclass does **not** redefine the method.
        """
        if self._override_platform is not None:
            return platform_to_backend(self._override_platform)
        if self._override_backend is not None:
            return self._override_backend
        return BackendType.Ascend910B

    def define_scalars(self) -> list[ScalarSpec]:
        """Define scalar TaskArg parameters for this test.

        Override to provide scalar values that are passed to the orchestration
        function via TaskArg scalar slots (after all tensor slots).

        Returns:
            List of ScalarSpec objects.  Empty by default.
        """
        return []

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

    @property
    def scalar_specs(self) -> list[ScalarSpec]:
        """Get cached scalar specifications."""
        if self._scalar_specs is None:
            self._scalar_specs = self.define_scalars()
        return self._scalar_specs
