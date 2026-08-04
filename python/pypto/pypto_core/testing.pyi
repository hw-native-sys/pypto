# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Type stubs for pypto.testing submodule

Internal testing utilities (do not use in production)
"""

from typing import Literal, NoReturn, TypedDict

from .ir import Call, Function

class DsaReusePenaltyEdge(TypedDict):
    """One internal pre-solver DSA-RP recognizer result."""

    first_interval: int
    second_interval: int
    first_name: str
    second_name: str
    cost: int

def raise_value_error(message: str) -> NoReturn:
    """Raise a ValueError from C++ for testing error handling"""

def raise_type_error(message: str) -> NoReturn:
    """Raise a TypeError from C++ for testing error handling"""

def raise_runtime_error(message: str) -> NoReturn:
    """Raise a RuntimeError from C++ for testing error handling"""

def raise_not_implemented_error(message: str) -> NoReturn:
    """Raise a NotImplementedError from C++ for testing error handling"""

def raise_index_error(message: str) -> NoReturn:
    """Raise an IndexError from C++ for testing error handling"""

def raise_generic_error(message: str) -> NoReturn:
    """Raise a generic Error from C++ for testing error handling"""

def raise_assertion_error(message: str) -> NoReturn:
    """Raise an AssertionError from C++ for testing purposes"""

def raise_internal_error(message: str) -> NoReturn:
    """Raise an InternalError from C++ for testing error handling"""

def raise_internal_error_with_span(message: str, filename: str, line: int, col: int) -> NoReturn:
    """Raise an InternalError with IR source span for testing"""

def recognize_dsa_reuse_penalties(function: Function) -> list[DsaReusePenaltyEdge]:
    """Return recognized DSA-RP edges without running placement."""

def try_infer_pipe(call: Call) -> int | None:
    """Return the exact backend pipe for a Call, or None."""

def get_execution_memory_access_evidence(op_name: str) -> Literal["unknown", "functional", "no_access"]:
    """Return an operation's execution-memory-access evidence."""
