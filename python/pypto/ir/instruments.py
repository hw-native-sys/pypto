# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass instruments for IR verification beyond the built-in VerificationInstrument."""

from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes as _passes


def _assert_structural_hash_equal(original: _ir.IRNode, restored: _ir.IRNode, pass_name: str) -> None:
    """Enforce the equal-IR/equal-hash contract after a roundtrip."""
    try:
        original_hash = _ir.structural_hash(original)
        restored_hash = _ir.structural_hash(restored)
    except Exception as exc:
        raise RuntimeError(
            f"[RoundtripInstrument] Structural hash computation failed after pass '{pass_name}'.\n{exc}"
        ) from exc
    if original_hash != restored_hash:
        raise RuntimeError(
            f"[RoundtripInstrument] Structural hash mismatch after pass '{pass_name}'.\n"
            f"Original hash: {original_hash}\nRestored hash: {restored_hash}"
        )


def make_roundtrip_instrument() -> _passes.CallbackInstrument:
    """Create a CallbackInstrument that verifies IR roundtrip after each pass.

    After every pass, the instrument:
    1. Prints the resulting IR to Python DSL text (``python_print``).
    2. Parses the text back to an IR Program (``parse``).
    3. Asserts structural equality between the original and re-parsed programs.
    4. Checks that their structural hashes are equal, using the same mapping policy.

    A failure means the printer or parser cannot faithfully represent the IR
    produced by that pass, or structural hashing violates the equal-IR/equal-hash contract.

    Buffer-stage programs use binary serialization instead: their Python output
    is diagnostic text, not executable DSL. The complete program, including
    orchestration and device stage markers, must still be structurally equal and
    have equal structural hashes.

    Transitional Unroll loops with SSA ``iter_args`` use the same text roundtrip
    as other functional IR. Printer and parser failures are always errors.

    Returns:
        A ``CallbackInstrument`` named ``"RoundtripInstrument"``.
    """

    def _after_pass(pass_obj: _passes.Pass, program: _ir.Program) -> None:
        # Lazy imports to avoid circular imports at module load time.
        from pypto.ir.printer import python_print  # noqa: PLC0415
        from pypto.language.parser.text_parser import parse  # noqa: PLC0415

        pass_name = pass_obj.get_name()

        if any(func.ir_stage == _ir.FunctionIRStage.Buffer for func in program.functions.values()):
            try:
                restored = _ir.deserialize(_ir.serialize(program))
                _ir.assert_structural_equal(program, restored)
            except Exception as exc:
                raise RuntimeError(
                    f"[RoundtripInstrument] Binary roundtrip failed after pass '{pass_name}'.\n{exc}"
                ) from exc
            _assert_structural_hash_equal(program, restored, pass_name)
            return

        # --- Step 1: print ---
        try:
            printed = python_print(program, format=False)
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
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

        _assert_structural_hash_equal(program, reparsed, pass_name)

    return _passes.CallbackInstrument(
        after_pass=_after_pass,
        name="RoundtripInstrument",
    )
