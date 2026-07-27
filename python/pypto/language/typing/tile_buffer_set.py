# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tuple-like DSL wrapper for homogeneous physical tile buffer sets."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemorySpace

if TYPE_CHECKING:
    from pypto.language.typing.scalar import Scalar
    from pypto.language.typing.tile import Tile


class TileBufferSetMeta(type):
    """Enable ``pl.TileBufferSet[[shape], dtype, count, memory_space]`` syntax."""

    def __getitem__(cls, item: tuple[Any, ...]) -> "TileBufferSet":
        if not isinstance(item, tuple) or len(item) != 4:
            raise TypeError("TileBufferSet requires [shape, dtype, count, memory_space] notation")
        shape, dtype, count, memory_space = item
        return cls(shape, dtype, count, memory_space, _annotation_only=True)


class TileBufferSet(metaclass=TileBufferSetMeta):
    """Homogeneous allocation group with runtime-selectable tile slots."""

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        count: int | None = None,
        memory_space: MemorySpace | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ) -> None:
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.count = count
            self.memory_space = memory_space
            self._expr: Expr | None = None
        elif expr is not None and count is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self.count = count
            self.memory_space = None
        else:
            raise ValueError("TileBufferSet runtime wrappers require expr and count")

    def unwrap(self) -> Expr:
        """Return the underlying buffer-set expression."""
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only TileBufferSet")
        return self._expr

    def __len__(self) -> int:
        """Return the compile-time number of physical slots."""
        if self.count is None:
            raise ValueError("TileBufferSet count is unavailable")
        return self.count

    def __getitem__(self, index: "int | Expr | Scalar") -> "Tile":
        """Select a static or dynamic physical slot."""
        from pypto.language.op import tile as _tile_dsl  # noqa: PLC0415

        return _tile_dsl.buffer_slot(self, index)

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType, int, MemorySpace]) -> "TileBufferSet":
        """Support static type checkers for the annotation syntax."""
        return type(cls).__getitem__(cls, item)


__all__ = ["TileBufferSet"]
