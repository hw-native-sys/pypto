# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.comm_ctx.*`` — CommContext scalar accessor DSL wrappers.

Exposes the ``rank`` / ``nranks`` scalar reads on a :class:`pld.CommCtx`
handle (output of :func:`pld.get_comm_ctx`). The op verifier (C++)
rejects any argument that is not :class:`ir.CommCtxType`.

These wrappers are routed by the parser through the 3-segment dispatch
``pld.comm_ctx.<op>`` (parallel to ``pld.tile.<op>``); the call-site
parser invokes them via ``invoke_dsl``, which unwraps the
:class:`pld.CommCtx` argument back to ``ir.Expr``.
"""

from pypto.ir.op.distributed import system_ops as _ir_system
from pypto.language.typing import Scalar

from ..typing.comm_ctx import CommCtx
from .memory_ops import _unwrap


def rank(ctx: CommCtx) -> Scalar:
    """Return the local rank as an ``INT32`` :class:`Scalar`.

    Codegen lowers each call site to a scalar load of the runtime
    ``CommContext::rankId`` field.

    Args:
        ctx: A :class:`pld.CommCtx` handle from :func:`pld.get_comm_ctx`.

    Returns:
        :class:`Scalar` wrapping an :class:`ir.Expr` of type
        ``ScalarType(INT32)``.
    """
    return Scalar(expr=_ir_system.comm_ctx_rank(_unwrap(ctx)))


def nranks(ctx: CommCtx) -> Scalar:
    """Return the rank count of the comm group as an ``INT32`` :class:`Scalar`.

    Codegen lowers each call site to a scalar load of the runtime
    ``CommContext::rankNum`` field.

    Args:
        ctx: A :class:`pld.CommCtx` handle from :func:`pld.get_comm_ctx`.

    Returns:
        :class:`Scalar` wrapping an :class:`ir.Expr` of type
        ``ScalarType(INT32)``.
    """
    return Scalar(expr=_ir_system.comm_ctx_nranks(_unwrap(ctx)))


__all__ = ["nranks", "rank"]
