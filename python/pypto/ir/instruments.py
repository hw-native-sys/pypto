# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass instruments for IR verification beyond the built-in VerificationInstrument."""

import warnings

from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes as _passes


def make_roundtrip_instrument() -> _passes.CallbackInstrument:
    """Create a CallbackInstrument that verifies print→parse roundtrip after each pass.

    After every pass, the instrument:
    1. Prints the resulting IR to Python DSL text (``python_print``).
    2. Parses the text back to an IR Program (``parse``).
    3. Asserts structural equality between the original and re-parsed programs.

    A failure means the printer or parser cannot faithfully represent the IR
    produced by that pass, which is a bug in the printer/parser layer.

    Known non-failures (instrument emits a warning instead):

    - **Printer InternalError**: Some transitional IR states (e.g. ``ForKind::Unroll``
      with SSA ``iter_args`` after ``ConvertToSSA``) have no valid Python DSL syntax.
      The instrument cannot roundtrip what it cannot print; it warns and skips.

    Returns:
        A ``CallbackInstrument`` named ``"RoundtripInstrument"``.
    """

    def _after_pass(pass_obj: _passes.Pass, program: _ir.Program) -> None:
        # Lazy imports to avoid circular imports at module load time.
        from pypto.ir.printer import python_print  # noqa: PLC0415
        from pypto.language.parser.text_parser import parse  # noqa: PLC0415

        pass_name = pass_obj.get_name()

        # --- Step 1: print ---
        try:
            printed = python_print(program, format=False)
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
            # Only suppress known transitional IR states that have no valid DSL syntax.
            # Currently: ForKind::Unroll with SSA iter_args (created when UnrollLoops is
            # skipped and ConvertToSSA adds loop-carried values to an unroll loop).
            # All other printer failures are propagated so regressions are visible.
            if "does not support iter_args" in first_line:
                warnings.warn(
                    f"[RoundtripInstrument] IR not printable after '{pass_name}' — "
                    f"skipping roundtrip: {first_line}",
                    stacklevel=2,
                )
                return
            raise RuntimeError(
                f"[RoundtripInstrument] Printer failed after pass '{pass_name}'.\n\nError: {first_line}"
            ) from exc

        # --- Step 2: parse ---
        try:
            reparsed = parse(printed, filename="<roundtrip>")
        except Exception as exc:
            from pypto.language.parser.diagnostics import ErrorRenderer, ParserError  # noqa: PLC0415

            if isinstance(exc, ParserError):
                error_detail = ErrorRenderer(use_color=False).render(exc)
            else:
                error_detail = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"[RoundtripInstrument] Parse failed after pass '{pass_name}'.\n\n{error_detail}"
            ) from exc

        if not isinstance(reparsed, _ir.Program):
            raise RuntimeError(
                f"[RoundtripInstrument] Parse returned {type(reparsed).__name__}, "
                f"expected Program, after pass '{pass_name}'."
            )

        # --- Step 3: structural equality ---
        try:
            _ir.assert_structural_equal(program, reparsed)
        except Exception as exc:
            error_msg = str(exc)
            raise RuntimeError(
                f"[RoundtripInstrument] Structural equality failed after pass '{pass_name}'.\n"
                f"\n"
                f"Error: {error_msg}\n"
                f"\n"
                f"--- Printed IR ---\n{printed}"
            ) from exc

    return _passes.CallbackInstrument(
        after_pass=_after_pass,
        name="RoundtripInstrument",
    )
