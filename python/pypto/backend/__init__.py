# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Backend module - SoC hierarchy and backend implementations."""

from pypto.pypto_core.backend import (
    # Backend
    Backend,
    Backend910B,
    Cluster,
    Core,
    Die,
    # Components
    Mem,
    SoC,
)

__all__ = [
    "Mem",
    "Core",
    "Cluster",
    "Die",
    "SoC",
    "Backend",
    "Backend910B",
]
