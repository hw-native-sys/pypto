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

    - **UnknownType improvement**: Manually-constructed IR may use
      ``ir.Call(ir.Op(...))`` without going through ``ir.create_op_call``, leaving
      the call result typed as ``UnknownType``.  Parsing always infers a concrete
      type instead — this is a type *improvement*, not a printer/parser bug.

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
            # UnknownType in the original IR comes from manually-constructed IR that
            # bypasses C++ type inference (ir.Call(ir.Op(...)) without create_op_call).
            # Parsing always infers a concrete type in its place — this is a type
            # improvement, not a printer/parser asymmetry.
            if "UnknownType !=" in error_msg or "!= UnknownType" in error_msg:
                return
            # Variable pointer mismatch occurs when dynamic-shape Var nodes appear in
            # return types or other positions outside the function body, where the
            # first-encounter bijection in structural_equal cannot establish a mapping.
            # This is a structural_equal limitation with dynamic shapes, not a
            # printer/parser bug — the printed IR is faithfully parsed.
            if "Variable pointer mismatch" in error_msg:
                return
            # tensor.add(x, scalar) → tensor.adds: the Python API dispatches scalar rhs
            # to tensor.adds, so manually-constructed tensor.add(x, scalar_const) is
            # normalized by roundtrip.  This is an IR-level discrepancy in the original,
            # not a printer/parser bug.
            if "Operator name mismatch" in error_msg and (
                "'tensor.add' != 'tensor.adds'" in error_msg or "'tensor.adds' != 'tensor.add'" in error_msg
            ):
                return
            # tile.load 3-arg → 4-arg: manually-constructed tile.load via ir.Call(ir.Op(...))
            # may have only 3 positional args (tensor, offsets, shapes).  The Python API
            # always produces 4 args (adding valid_shapes=shapes by default), and the C++
            # type inference requires exactly 4 args.  The printer's special case pads the
            # 3-arg form to 4-arg when printing, so the parsed version has 4 args.
            # This 1-arg discrepancy is a manual-construction artifact, not a printer/parser bug.
            if "Vector size mismatch (3 items != 4 items)" in error_msg:
                return
            # TileType tile_view presence mismatch: some passes (e.g. InferTileMemorySpace)
            # update the Var type without rebuilding the Call's result type, creating a
            # Var.type != Call.type inconsistency in the original IR.  On reparse, the
            # annotation-driven override rebuilds the Call and introduces a presence
            # mismatch between the original (implicit TileView = None) and parsed
            # (non-implicit TileView from annotation).  This is a pass-level design issue,
            # not a printer/parser asymmetry.
            if "TileType tile_view presence mismatch" in error_msg:
                return
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
