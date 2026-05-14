# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builder for ``pld.world_size``.

Mirror of :mod:`pypto.ir.op.system_ops` for the distributed namespace —
exposes the registered C++ op as a Python builder. The DSL layer in
:mod:`pypto.language.distributed.op.system_ops` wraps this for symmetry
with the rest of the ``pld.*`` surface, even though the op has no DSL
type to wrap.
"""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Span

from ...utils import _get_span_or_capture


def world_size(*, span: Span | None = None) -> Call:
    """Build a ``pld.world_size()`` Call returning ``ScalarType(INT64)``.

    Host-only — the parser already validates the call site, so this builder
    is unconditional.
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.world_size", [], {}, actual_span)


__all__ = ["world_size"]
