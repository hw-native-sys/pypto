# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR pass instrumentation helpers."""

from difflib import unified_diff

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

from .printer import python_print


def _roundtrip_program(program: _ir_core.Program) -> tuple[str, _ir_core.Program, str]:
    """Print and re-parse a program, returning both text forms."""
    from pypto.language import parse  # noqa: PLC0415

    try:
        printed = python_print(program)
    except Exception as exc:
        raise RuntimeError(f"Roundtrip verification failed: could not print IR.\nPrint error: {exc}") from exc

    try:
        reparsed = parse(printed)
    except Exception as exc:
        raise RuntimeError(
            "Roundtrip verification failed: could not parse printed IR.\n"
            f"Parse error: {exc}\n\n"
            f"Printed IR:\n{printed}"
        ) from exc

    if not isinstance(reparsed, _ir_core.Program):
        raise RuntimeError(
            f"Roundtrip verification failed: expected Program after re-parse, got {type(reparsed).__name__}."
        )

    return printed, reparsed, python_print(reparsed)


def verify_roundtrip(program: _ir_core.Program) -> None:
    """Verify that a program survives a strict print-parse roundtrip."""
    printed, reparsed, reprinted = _roundtrip_program(program)

    try:
        _ir_core.assert_structural_equal(program, reparsed, enable_auto_mapping=False)
    except ValueError as structural_exc:
        diff = "".join(
            unified_diff(
                printed.splitlines(keepends=True),
                reprinted.splitlines(keepends=True),
                fromfile="original_printed",
                tofile="reprinted_after_parse",
            )
        )
        diff_text = diff if diff else "<no textual diff>"
        raise ValueError(
            "Roundtrip verification failed: structural mismatch after print→parse.\n"
            f"Structural mismatch:\n{structural_exc}\n\n"
            f"Printed IR:\n{printed}\n\n"
            f"Reprinted IR:\n{reprinted}\n\n"
            f"Diff:\n{diff_text}"
        ) from structural_exc


def _run_roundtrip_instrument_check(pass_name: str, program: _ir_core.Program, phase: str) -> None:
    """Run strict roundtrip verification for a specific pass phase."""
    try:
        verify_roundtrip(program)
    except (RuntimeError, ValueError) as exc:
        raise type(exc)(f"Roundtrip verification failed {phase} pass '{pass_name}':\n{exc}") from exc


def RoundtripInstrument(
    mode: _passes.VerificationMode = _passes.VerificationMode.AFTER,
) -> _passes.CallbackInstrument:
    """Create a pass instrument that checks strict print-parse roundtrip stability."""

    def _before(pass_obj: _passes.Pass, program: _ir_core.Program) -> None:
        _run_roundtrip_instrument_check(pass_obj.get_name(), program, "before")

    def _after(pass_obj: _passes.Pass, program: _ir_core.Program) -> None:
        _run_roundtrip_instrument_check(pass_obj.get_name(), program, "after")

    before_cb = None
    after_cb = None
    if mode in (_passes.VerificationMode.BEFORE, _passes.VerificationMode.BEFORE_AND_AFTER):
        before_cb = _before
    if mode in (_passes.VerificationMode.AFTER, _passes.VerificationMode.BEFORE_AND_AFTER):
        after_cb = _after

    return _passes.CallbackInstrument(
        before_pass=before_cb,
        after_pass=after_cb,
        name="RoundtripInstrument",
    )


__all__ = ["verify_roundtrip", "RoundtripInstrument"]
