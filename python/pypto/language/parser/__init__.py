# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Language Parser module for converting high-level DSL code to IR structures.

This module (pypto.language.parser) provides a decorator-based system for
parsing Python functions with DSL annotations and converting them to IR
builder programs.

Part of the pypto.language package - use via:
    import pypto.language as pl

    @pl.function
    def my_func(...):
        ...
"""

import os
import sys
from types import TracebackType
from typing import Optional

# Import DSL helpers from parent language module
from ..dsl_api import range, yield_
from ..typing import Scalar, Tensor, Tile
from .decorator import function, program
from .diagnostics import ErrorRenderer, ParserError
from .text_parser import loads, loads_program, parse, parse_program

__all__ = [
    "function",
    "program",
    "parse",
    "loads",
    "parse_program",
    "loads_program",
    "range",
    "yield_",
    "Tensor",
    "Tile",
    "Scalar",
]


def _install_parser_excepthook():
    """Install custom exception hook to pretty-print uncaught ParserErrors."""
    # Save the original excepthook
    original_excepthook = sys.excepthook

    def parser_excepthook(
        exc_type: type,
        exc_value: BaseException,
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """Custom exception hook that pretty-prints ParserError exceptions."""
        if isinstance(exc_value, ParserError):
            # Check environment variable for debug mode
            debug_level = os.environ.get("PTO_BACKTRACE", "0")

            if debug_level == "0":
                # Show only pretty error (no Python traceback)
                renderer = ErrorRenderer()
                error_message = renderer.render(exc_value)
                print(error_message, file=sys.stderr)
                print(
                    "\nnote: run with `PTO_BACKTRACE=1` environment variable to display a backtrace.",
                    file=sys.stderr,
                )
            elif debug_level == "1":
                # Show both pretty error and Python traceback
                renderer = ErrorRenderer()
                error_message = renderer.render(exc_value)
                print(error_message, file=sys.stderr)
                print("\n--- Python Traceback (PTO_BACKTRACE=1) ---", file=sys.stderr)
                original_excepthook(exc_type, exc_value, exc_traceback)
            else:
                raise ValueError(
                    "Invalid value of `PTO_BACKTRACE` environment variable. Only `0` and `1` are allowed."
                )
        else:
            # Not a ParserError, use the original excepthook
            original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = parser_excepthook


_install_parser_excepthook()
