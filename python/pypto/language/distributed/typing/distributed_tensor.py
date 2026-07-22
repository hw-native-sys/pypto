# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.DistributedTensor`` — DSL annotation for window-bound tensors.

Function-signature type annotation for chip_orch / InCore parameters that
slice a HCCL window buffer carved by a CommDomainScopeStmt. Behaves identically to :class:`pl.Tensor`
at the DSL surface (same ``[shape, dtype, layout|memref|view]`` subscript
forms); the only difference is the IR-level ``ObjectKind``
(:class:`ir.DistributedTensorType`), which lets cross-rank op verifiers
(``pld.tile.remote_load`` / ``pld.system.notify`` / ``pld.system.wait``, added
in later milestones) reject plain ``Tensor`` arguments.

Use::

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pld.DistributedTensor[[256], pl.FP32],
        bias: pld.DistributedTensor[[256], pl.FP32, pl.NZ],
    ): ...
"""

from collections.abc import Sequence
from typing import Any, cast

from pypto.language.typing.tensor import Tensor, TensorMeta


class DistributedTensorMeta(TensorMeta):
    """Metaclass enabling ``pld.DistributedTensor[...]`` syntax.

    Inherits :class:`TensorMeta`'s subscript dispatch, except it drops a trailing
    ``"window_buffer=<name>"`` debug marker before delegating. ``PassDumpLevel.
    EXPLICIT`` dumps append that marker as an extra subscript element to surface a
    distributed tensor's window-buffer back-reference (issue #2088); it is
    informational only — the real reference re-derives from ``pld.tensor.window``
    — so ignoring it on parse keeps EXPLICIT pass dumps reparseable, which
    ``validate_ir`` relies on (it reloads every dump via ``pl.loads``).
    """

    def __getitem__(cls, item: Any) -> "DistributedTensor":
        if (
            isinstance(item, tuple)
            and item
            and isinstance(item[-1], str)
            and item[-1].startswith("window_buffer=")
        ):
            item = item[:-1]
        return cast("DistributedTensor", super().__getitem__(item))


class DistributedTensor(Tensor, metaclass=DistributedTensorMeta):
    """Tensor backed by a per-rank slice of a HCCL window buffer carved by a CommDomainScopeStmt.

    Same DSL surface as :class:`pl.Tensor` — supports memref / layout /
    tensor_view in the third (and fourth) subscript slot. The IR-level type
    (:class:`ir.DistributedTensorType`) is a precise-``ObjectKind`` subclass of
    ``TensorType``: cross-rank op verifiers dispatch on it to reject plain
    ``Tensor`` arguments. ``pl.load`` / ``pl.store`` etc. operate
    transparently on a DistributedTensor (local rank's slice).
    """

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[Any], Any]) -> "DistributedTensor":
        return type(cls).__getitem__(cls, item)


__all__ = ["DistributedTensor"]
