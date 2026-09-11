# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared IR construction for the tensor and tile img2col operators."""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, Span

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def create_img2col_call(
    op_name: str,
    src: Expr,
    position: tuple[int | Expr, int | Expr],
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    image_shape: Sequence[int | Expr],
    kernel_size: Sequence[int | Expr],
    stride: Sequence[int | Expr] = (1, 1),
    padding: Sequence[int | Expr] = (0, 0, 0, 0),
    dilation: Sequence[int | Expr] = (1, 1),
    span: Span | None = None,
) -> Call:
    """Normalize the shared geometry and construct the requested IR operator."""
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, int] = {}
    for names, values in (
        (("fmap_h", "fmap_w"), image_shape),
        (("kernel_h", "kernel_w"), kernel_size),
        (("stride_h", "stride_w"), stride),
        (("pad_top", "pad_bottom", "pad_left", "pad_right"), padding),
        (("dilation_h", "dilation_w"), dilation),
    ):
        if len(values) != len(names):
            raise ValueError(f"{op_name} {names} require {len(names)} compile-time integers")
        for name, value in zip(names, values):
            constant = value.value if isinstance(value, ConstInt) else value
            if not isinstance(constant, int) or isinstance(constant, bool):
                raise ValueError(f"{op_name} {name} requires a compile-time integer")
            kwargs[name] = constant
    return _ir_core.create_op_call(
        op_name,
        [
            src,
            _normalize_expr(position[0], actual_span, int_dtype=DataType.INDEX),
            _normalize_expr(position[1], actual_span, int_dtype=DataType.INDEX),
            _to_make_tuple(shape, actual_span),
        ],
        kwargs,
        actual_span,
    )
