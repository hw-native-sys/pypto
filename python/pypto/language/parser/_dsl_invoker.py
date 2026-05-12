# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parse-time invocation of DSL op wrappers.

Bridges the parser's ``ir.Expr`` arguments to the DSL wrappers, which expect
DSL types (``Tensor`` / ``Tile`` / ``Scalar``). The parser:

  1. Parses each AST argument to an ``ir.Expr``.
  2. Calls :func:`invoke_dsl` with the wrapper, the parsed args, and the
     call-site span. ``invoke_dsl`` wraps each ``ir.Expr`` to its matching
     DSL type (by inspecting ``expr.type``), pins the span via the
     ``_PARSER_SPAN`` contextvar so IR builders inside the wrapper pick it
     up, calls the wrapper, and unwraps the returned DSL object back to an
     ``ir.Expr``.

This collapses the dispatch the parser used to do via hand-maintained tables
into ordinary Python attribute lookup: ``pl.add`` is just
``pypto.language.op.add``, and the wrapper itself owns the type-based
dispatch logic that ``_TILE_SCALAR_OPS`` / ``_SCALAR_BINARY_OPS`` etc. used
to duplicate.
"""

from collections.abc import Callable
from typing import Any

from pypto.ir.utils import use_parser_span
from pypto.language.typing import Array, Scalar, Tensor, Tile
from pypto.pypto_core import ir


def _wrap_arg(arg: Any) -> Any:
    """Wrap a parsed ``ir.Expr`` as the matching DSL type.

    Rules:

    - ``MakeTuple`` (the parser's representation of a Python list literal in
      source) is unwrapped to a Python list of its elements so wrappers that
      expect ``Sequence[IntLike]`` (e.g. shape / offset / indices) accept it.
    - ``ConstInt`` / ``ConstFloat`` literals stay as raw ``Expr``. They
      carry the parser's chosen dtype (``INDEX`` for plain ``int`` literals,
      ``FP32`` for plain ``float``, or whatever the user specified via
      ``pl.const``) — extracting them to Python ``int`` / ``float`` would
      lose that dtype and let downstream IR-builder defaults pick something
      different (e.g. tensor.muls' rhs defaults to ``FP32``, which would
      silently change the result dtype). Wrappers that need a Python
      ``int`` (axis-style arguments) explicitly extract via the IR layer's
      ``ConstInt.value`` field.
    - ``ir.Expr`` whose ``type`` is ``TensorType`` / ``TileType`` /
      ``ScalarType`` (and which is not a ``Const*`` literal) is wrapped in
      the matching DSL class so type-dispatch in the wrapper sees the right
      ``isinstance`` answer.
    - Anything else (non-Expr Python values, or Expr with no matching DSL
      class) is returned unchanged.
    """
    if isinstance(arg, ir.MakeTuple):
        return list(arg.elements)
    if not isinstance(arg, ir.Expr):
        return arg
    if isinstance(arg, (ir.ConstInt, ir.ConstFloat)):
        return arg
    t = arg.type
    if isinstance(t, ir.TensorType):
        return Tensor(expr=arg)
    if isinstance(t, ir.TileType):
        return Tile(expr=arg)
    if isinstance(t, ir.ArrayType):
        return Array(expr=arg)
    if isinstance(t, ir.ScalarType):
        return Scalar(expr=arg)
    return arg


def _unwrap_result(value: Any) -> Any:
    """Unwrap a DSL return value to an ``ir.Expr`` for the parser to consume.

    Multi-output DSL wrappers (e.g. ``pl.tile.gather_compare``) return a
    ``tuple`` of DSL objects whose underlying expressions are
    ``TupleGetItemExpr(call, 0)``, ``TupleGetItemExpr(call, 1)``, ... all
    referencing the same tuple-typed Call. The parser's tuple-unpacking path
    expects the bare Call so it can rebind ``_tuple_tmp`` and re-emit the
    ``TupleGetItemExpr``s; here we recover that Call.
    """
    if isinstance(value, (Tensor, Tile, Scalar, Array)):
        return value.unwrap()
    if isinstance(value, tuple) and value and all(isinstance(v, (Tensor, Tile, Scalar)) for v in value):
        unwrapped = tuple(v.unwrap() for v in value)
        common_call: ir.Expr | None = None
        for i, expr in enumerate(unwrapped):
            if not isinstance(expr, ir.TupleGetItemExpr) or expr.index != i:
                common_call = None
                break
            if common_call is None:
                common_call = expr.tuple
            elif common_call is not expr.tuple:
                common_call = None
                break
        if common_call is not None:
            return common_call
        return unwrapped
    return value


def invoke_dsl(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
    span: ir.Span,
) -> Any:
    """Wrap parsed args, invoke a DSL wrapper under a pinned span, unwrap the result."""
    wrapped_args = [_wrap_arg(a) for a in args]
    wrapped_kwargs = {k: _wrap_arg(v) for k, v in kwargs.items()}
    with use_parser_span(span):
        result = fn(*wrapped_args, **wrapped_kwargs)
    return _unwrap_result(result)


__all__ = ["invoke_dsl"]
