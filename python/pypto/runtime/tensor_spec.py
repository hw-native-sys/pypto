# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tensor specification for the PyPTO runtime module.

Provides TensorSpec, which describes a single tensor's name, shape, dtype,
initialisation strategy, and whether it is an output to be validated.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # type: ignore[import]


@dataclass
class TensorSpec:
    """Specification for a runtime tensor.

    Attributes:
        name: Tensor name, used as the parameter name in generated C++ and golden.py.
        shape: Tensor shape as a list of integers.
        dtype: PyTorch dtype (e.g. ``torch.float32``, ``torch.bfloat16``).
        init_value: Initial value strategy.  Can be:

            - ``None`` — zero-initialised.
            - ``int`` or ``float`` — every element set to this constant.
            - ``torch.Tensor`` — use this tensor directly (must have matching shape/dtype).
            - ``Callable`` — a no-argument callable that returns a ``torch.Tensor``, or
              one of the supported ``torch`` factory functions
              (``torch.randn``, ``torch.rand``, ``torch.zeros``, ``torch.ones``)
              that will be called with ``(shape, dtype=dtype)``.
        is_output: If ``True``, the tensor is an output to be validated against the
            golden reference.

    Example:
        >>> import torch
        >>> TensorSpec("query", [32, 128], torch.bfloat16, init_value=torch.randn)
        >>> TensorSpec("out", [32, 128], torch.float32, is_output=True)
    """

    name: str
    shape: list[int]
    dtype: "torch.dtype"
    init_value: "int | float | torch.Tensor | Callable | None" = field(default=None)
    is_output: bool = False

    def create_tensor(self) -> "torch.Tensor":
        """Create and return a ``torch.Tensor`` based on this specification.

        Returns:
            Initialised tensor with the requested shape and dtype.
        """
        import torch  # type: ignore[import]  # noqa: PLC0415 — optional dependency

        if self.init_value is None or self.is_output:
            return torch.zeros(self.shape, dtype=self.dtype)
        if isinstance(self.init_value, (int, float)):
            return torch.full(self.shape, self.init_value, dtype=self.dtype)
        if isinstance(self.init_value, torch.Tensor):
            return self.init_value.to(dtype=self.dtype)
        if callable(self.init_value):
            # Support the standard torch factory functions used as callables
            fn = self.init_value
            if fn in (torch.randn, torch.rand, torch.zeros, torch.ones):
                return fn(self.shape, dtype=self.dtype)
            # Generic callable: call with no arguments, then cast
            result: Any = fn()
            return torch.as_tensor(result, dtype=self.dtype)
        raise TypeError(f"Unsupported init_value type {type(self.init_value)!r} for tensor {self.name!r}")
