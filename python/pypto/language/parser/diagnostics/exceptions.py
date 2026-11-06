# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Forwarding module for the pre-move import spelling of the DSL exceptions.

The definitions live in :mod:`pypto.language.diagnostics`; this path keeps
``pypto.language.parser.diagnostics.exceptions`` importable for external
consumers.
"""

from pypto.language.diagnostics import (
    BUG_CLASS_EXCEPTIONS,
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    ScopeIsolationError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
    concise_error_message,
)

__all__ = [
    "BUG_CLASS_EXCEPTIONS",
    "InvalidOperationError",
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "SSAViolationError",
    "ScopeIsolationError",
    "UndefinedVariableError",
    "UnsupportedFeatureError",
    "concise_error_message",
]
