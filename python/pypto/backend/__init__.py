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
    # Backend type and implementations
    Backend,
    Backend910B_CCE,
    Backend910B_PTO,
    BackendType,
    Cluster,
    Core,
    Die,
    # Components
    Mem,
    SoC,
    # Backend configuration
    get_backend_type,
    is_backend_configured,
    reset_for_testing,
    set_backend_type,
)

__all__ = [
    "Mem",
    "Core",
    "Cluster",
    "Die",
    "SoC",
    "Backend",
    "Backend910B_CCE",
    "Backend910B_PTO",
    "BackendType",
    "set_backend_type",
    "get_backend_type",
    "is_backend_configured",
    "reset_for_testing",
]
