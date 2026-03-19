# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Error code categorization for parser diagnostics.

Error codes are for internal organization and are not displayed to users
unless PTO_BACKTRACE is enabled.
"""

from enum import Enum

from .exceptions import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


class ErrorCategory(Enum):
    """Error categories for parser diagnostics."""

    # E001-E099: Syntax errors (loops, conditionals, assignments)
    SYNTAX = "E0"

    # E100-E199: Type errors (annotations, type mismatches)
    TYPE = "E1"

    # E200-E299: Name/scope errors (undefined variables, SSA violations)
    NAME = "E2"

    # E300-E399: Operation errors (unknown ops, invalid arguments)
    OPERATION = "E3"


class ErrorCode(Enum):
    """Specific error codes for diagnostics."""

    # Syntax errors (E001-E099)
    UNSUPPORTED_STATEMENT = "E001"
    INVALID_ASSIGNMENT = "E002"
    INVALID_LOOP_TARGET = "E003"
    INVALID_RANGE_USAGE = "E004"
    MISSING_RANGE_CALL = "E005"
    TUPLE_UNPACKING_ERROR = "E006"
    ITER_ARG_MISMATCH = "E007"

    # Type errors (E100-E199)
    MISSING_TYPE_ANNOTATION = "E100"
    INVALID_TYPE_ANNOTATION = "E101"
    UNSUPPORTED_TYPE = "E102"
    INCOMPLETE_TYPE = "E103"
    TENSOR_TYPE_ERROR = "E104"
    DTYPE_ERROR = "E105"

    # Name/scope errors (E200-E299)
    UNDEFINED_VARIABLE = "E200"
    SSA_VIOLATION = "E201"
    SCOPE_ISOLATION_ERROR = "E202"

    # Operation errors (E300-E399)
    UNKNOWN_OPERATION = "E300"
    UNSUPPORTED_OPERATION = "E301"
    INVALID_OPERATION_ARGS = "E302"
    UNSUPPORTED_BINARY_OP = "E303"
    UNSUPPORTED_UNARY_OP = "E304"
    UNSUPPORTED_COMPARISON = "E305"


def get_error_code(exception_type: type) -> ErrorCode | None:
    """Get error code for exception type.

    Args:
        exception_type: Exception class

    Returns:
        Corresponding error code or None
    """

    mapping = {
        ParserSyntaxError: ErrorCode.UNSUPPORTED_STATEMENT,
        ParserTypeError: ErrorCode.INVALID_TYPE_ANNOTATION,
        UndefinedVariableError: ErrorCode.UNDEFINED_VARIABLE,
        SSAViolationError: ErrorCode.SSA_VIOLATION,
        UnsupportedFeatureError: ErrorCode.UNSUPPORTED_OPERATION,
        InvalidOperationError: ErrorCode.UNKNOWN_OPERATION,
    }

    return mapping.get(exception_type)


__all__ = ["ErrorCategory", "ErrorCode", "get_error_code"]
