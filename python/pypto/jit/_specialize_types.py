# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor metadata shared by JIT analysis and source specialization.

The public import paths in specializer remain aliases of these definitions;
body rewriting and metadata inference must use the same DynDim identity.
"""

from dataclasses import dataclass, field

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout


@dataclass(frozen=True)
class DynDim:
    """A dynamic dimension bound to a tensor axis.

    Attributes:
        name:        Python variable name used in user source (e.g. ``"M"``).
        literal:     String passed to ``pl.dynamic("...")``; usually equals
                     ``name`` but may diverge (e.g. ``rows = pl.dynamic("M")``).
        static_bound: Concrete extent at this specialization. Replaced by
                     ``None`` in the cache key so different bounds reuse the
                     same compilation, but still available as the static-shape
                     fallback wherever a numeric dim is required (e.g.
                     ``pl.slice`` parent-dim inheritance, ``ir.compile``).
        synthesized: True when ``@pl.jit`` invented this symbol for an extent it
                     could not resolve statically — a runtime-valued dim such as
                     ``pld.world_size()`` or ``pl.tensor.read(...)``. Such a dim
                     is a placeholder: a dep that declares the same dim with
                     ``pl.dynamic`` replaces it with the user's symbol, so a
                     kernel body referencing that name stays bound. A DynDim the
                     caller actually derived is never replaced.
    """

    name: str
    literal: str
    static_bound: int
    synthesized: bool = False


ShapeDim = int | DynDim


@dataclass
class TensorMeta:
    """Runtime metadata for a single tensor parameter.

    Attributes:
        shape: Per-dim entries. Each entry is either a static ``int`` or a
            :class:`DynDim` capturing a JIT-time-bound dynamic dim.
        dtype: DataType resolved from the torch tensor.
        layout: Layout declared by an annotation or inferred from a producing
            operation. None means an omitted layout that a dependency's
            annotation may fill; explicit ND means the layout is known and
            must not be overwritten. Non-ND layouts keep their third slot in
            the regenerated ``@pl.function`` source.
    """

    shape: tuple[ShapeDim, ...]
    dtype: DataType
    layout: TensorLayout | None = None
    # Bounds retained by local inference are useful for compatibility, but are
    # not exact dimensions and must never establish a constant-width proof.
    inexact_axes: frozenset[int] = field(default_factory=frozenset[int], compare=False, repr=False)
    # A fallback bound in shape is not the view's descriptor extent. Keep that
    # extent's independent identity so descriptor reads and full views can share
    # it without asserting equality with the parent bound.
    runtime_dims: dict[int, DynDim] = field(default_factory=dict[int, DynDim], compare=False, repr=False)

    def actual_dim(self, axis: int) -> ShapeDim | None:
        """Return a proven static/symbolic extent, never an inexact fallback bound."""
        if axis in self.inexact_axes:
            return self.runtime_dims.get(axis)
        return self.shape[axis]

    def static_shape(self) -> tuple[int, ...]:
        """Concrete shape tuple — DynDim dims collapse to their static_bound."""
        return tuple(d.static_bound if isinstance(d, DynDim) else d for d in self.shape)

    def dynamic_dim_indices(self) -> set[int]:
        """Indices of dims that are DynDim-bound at this specialization."""
        return {i for i, d in enumerate(self.shape) if isinstance(d, DynDim)}
