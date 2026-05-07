# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ManualDep marker for tensor annotations.

Usage::

    x: pl.Tensor[[64, 64], pl.FP32, pl.ManualDep]
    y: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.NZ, pl.ManualDep]]

When set, the tensor opts out of automatic dependency tracking by the
runtime (skips OverlapMap lookup and TensorMap insert). The user is
responsible for ordering by other means: disjoint writes guaranteed by
the kernel, manual scope plus explicit add_dep, or pl.no_dep at call
sites for per-arg overrides.

This is a marker class with no instances or methods — its presence in a
Tensor annotation is detected at parse time and threaded into the
underlying ir.TensorType's manual_dep field.
"""


class ManualDep:
    """Marker class signalling a tensor opts out of runtime auto-dep tracking."""


__all__ = ["ManualDep"]
