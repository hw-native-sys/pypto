# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser diagnostics and error reporting."""

from .error_codes import ErrorCategory, ErrorCode, get_error_code
from .exceptions import (
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    ScopeIsolationError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from .renderer import ErrorRenderer

__all__ = [
    # Exceptions
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "UndefinedVariableError",
    "SSAViolationError",
    "UnsupportedFeatureError",
    "InvalidOperationError",
    "ScopeIsolationError",
    # Renderer
    "ErrorRenderer",
    # Error codes
    "ErrorCategory",
    "ErrorCode",
    "get_error_code",
]
