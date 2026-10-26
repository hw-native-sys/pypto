# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared type parameters for tile operation wrappers."""

from typing import TypeVar

from ..typing import Tensor

# Bound TypeVar lets store / mscatter propagate the caller's concrete tensor
# class (Tensor or its DistributedTensor subclass) through to the return type,
# so ``data = pl.store(local, [0, 0], data)`` type-checks when ``data`` is a
# DistributedTensor. The runtime polymorphism comes from
# ``output_tensor.__class__(expr=call_expr)``; no DistributedTensor import is
# needed here (which avoids a circular dependency on ``..distributed``).
_TensorT = TypeVar("_TensorT", bound=Tensor)
