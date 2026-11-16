# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The corpus calls must match the intrinsics `tile_std` declares.

Three captured calls did not, and nothing was comparing them. Each is seeded
below as a case the checker must flag, because a conformance gate that reports
zero findings on a clean corpus is indistinguishable from one that cannot read.

The benign shapes are pinned just as hard. The corpus uses the
destination-passing idiom -- `(%x, %x, %eps, %r, %cl)`, the same tile as both
`dst` and `src` -- in 19 places, and an earlier version of the `dst` rule
flagged every one of them. Nineteen false findings would have been reported as
defects if the rule had not been checked against a sample of its own hits.
"""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
sys.path.insert(0, str(BENCH / "shapes"))

from check_call_conformance import (  # noqa: E402
    captures,
    check_file,
    declarations,
    fixed_variants,
    split_top_level,
    tile_files,
)

DECLS = declarations()


def probe(body):
    """Run the checker over one synthetic module."""
    d = Path(tempfile.mkdtemp())
    f = d / "probe.tile"
    f.write_text(body, encoding="utf-8")
    return check_file(f, DECLS)


def module(call, extra_consts="", loads=None):
    loads = loads or [
        "%ta = llvm.call @__tile_load_f32(%p0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32",
        "%tb = llvm.call @__tile_load_f32(%p1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32",
    ]
    return (
        "module {\n"
        "  llvm.func @probe(%p0: !llvm.ptr<1>, %p1: !llvm.ptr<1>, %p2: !llvm.ptr<1>)"
        " attributes {hacc.entry} {\n"
        "    %c0 = llvm.mlir.constant(0 : i32) : i32\n"
        "    %r = llvm.mlir.constant(16 : i32) : i32\n"
        "    %c = llvm.mlir.constant(64 : i32) : i32\n"
        "    %eps = llvm.mlir.constant(1e-20 : f32) : f32\n"
        f"{extra_consts}" + "".join(f"    {ln}\n" for ln in loads) + f"    {call}\n"
        "    llvm.return\n"
        "  }\n"
        "}\n"
    )


def test_the_declarations_were_actually_read():
    """Guard against every other test passing over an empty table."""
    assert len(DECLS) > 100, f"only {len(DECLS)} declarations parsed"
    assert DECLS["__tile_matmul_f16"] == [
        ("dst", "u32"),
        ("a", "u32"),
        ("b", "u32"),
        ("m", "u32"),
        ("k", "u32"),
        ("n", "u32"),
    ]
    assert len(tile_files()) > 40


def test_every_fixed_variant_conforms():
    """A `tile/fixed/` file exists to be correct, so any finding in one is a defect."""
    findings = []
    for path in fixed_variants():
        findings.extend(check_file(path, DECLS))
    assert not findings, "\n".join(findings)


def test_every_non_conforming_capture_has_a_conforming_sibling():
    """The specification, and it is deliberately NOT "every capture conforms".

    The top-level `tile/` files are faithful captures of a real lowering. A
    malformed one is DATA -- it records what the lowering actually produced --
    and overwriting it would destroy that record. An earlier version of this
    gate did exactly that to three captures before the `tile/fixed/`
    convention was noticed, and one of the three corrections it wrote already
    existed, byte for byte, one directory down.

    What a malformed capture may not do is go unanswered. Each one must have a
    conforming sibling under `tile/fixed/`, so the corpus holds both the defect
    and the kernel the source meant.
    """
    unanswered = []
    for path in captures():
        if not check_file(path, DECLS):
            continue
        sibling = path.parent / "fixed" / path.name
        if not sibling.is_file():
            unanswered.append(f"{path.name}: non-conforming with no tile/fixed/ sibling")
        elif check_file(sibling, DECLS):
            unanswered.append(f"{path.name}: its fixed sibling does not conform either")
    assert not unanswered, "\n".join(unanswered)


def test_there_are_non_conforming_captures_to_answer():
    """The gate above is vacuous if no capture is non-conforming.

    FOUR are, and each is a measured defect written up in its fixed sibling's
    header. If this ever drops to zero the captures were probably edited, which
    is the thing the specification exists to prevent.

    The fourth arrived 2026-11-09 with the checker's fourth rule, not with a new
    capture: `quant_i8_v41_8x5120` passes a 1x1 TILE into
    `__tile_quantize_f32_i8`'s `scale: f32`. Arity matches at five and the leading
    `dst` is the in-place form rule 2 exempts, so three rules saw nothing while the
    call meant something the declaration does not allow -- and the kernel's refusal
    blamed a missing arm for three iterations. One violation in 2541 operand slots,
    which is the whole corpus, so the count rising here is a rule arriving rather
    than a corpus getting worse.
    """
    bad = [p.name for p in captures() if check_file(p, DECLS)]
    assert len(bad) == 4, f"expected 4 non-conforming captures, found {bad}"


def test_a_missing_operand_is_caught():
    """`matmul.tile`'s real defect: 5 operands where 6 are declared.

    It did not fail at the time, because the PTO emitter carries a SECOND
    contract at 5 operands and silently reinterpreted the call under it.
    """
    out = probe(
        module("%res = llvm.call @__tile_matmul_f16(%ta, %tb, %r, %c, %c) : (i32, i32, i32, i32, i32) -> i32")
    )
    assert len(out) == 1
    assert "takes 6 operand(s)" in out[0] and "passes 5" in out[0]


def test_an_operand_shift_at_CORRECT_arity_is_caught():
    """`attn_gqa.tile`'s defect, and the reason arity alone is not enough.

    8 operands against 8 declared, so an arity check reports it clean. Every
    operand is shifted by one and a `causal` operand is appended that the
    surviving declaration does not carry.
    """
    out = probe(
        module(
            "%res = llvm.call @__tile_attention_gqa_f32"
            "(%ta, %tb, %tc, %r, %c, %r, %c, %c0)"
            " : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32",
            loads=[
                "%ta = llvm.call @__tile_load_f32(%p0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32",
                "%tb = llvm.call @__tile_load_f32(%p1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32",
                "%tc = llvm.call @__tile_load_f32(%p2, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32",
            ],
        )
    )
    assert len(out) == 1, out
    assert "declares operand 0 as `dst`" in out[0]
    assert "shifted by one" in out[0]


def test_a_pointer_operand_given_a_scalar_is_caught():
    """The i8 matmul's `scale: *const f32` passed as an `i32`."""
    out = probe(
        module(
            "%res = llvm.call @__tile_matmul_i8_acc_i32_dequant_f16"
            "(%c0, %ta, %tb, %r, %r, %c, %c)"
            " : (i32, i32, i32, i32, i32, i32, i32) -> i32"
        )
    )
    assert len(out) == 1, out
    assert "scale: *const f32" in out[0] and "not a pointer" in out[0]


def test_an_undeclared_intrinsic_is_caught():
    out = probe(
        module("%res = llvm.call @__tile_invented_f32(%c0, %ta, %r, %c) : (i32, i32, i32, i32) -> i32")
    )
    assert len(out) == 1
    assert "not declared in tile_std" in out[0]


@pytest.mark.parametrize(
    "call",
    [
        # Destination-passing: the same tile as `dst` and as `src`. 19 real
        # corpus calls look like this and every one is correct.
        "%o = llvm.call @__tile_rms_norm_f32(%ta, %ta, %eps, %r, %c) : (i32, i32, f32, i32, i32) -> i32",
        "%o = llvm.call @__tile_rope_inplace_f32(%ta, %ta, %r, %c) : (i32, i32, i32, i32) -> i32",
        # A constant `dst`, meaning "allocate a fresh tile".
        "%o = llvm.call @__tile_matmul_f16(%c0, %ta, %tb, %r, %c, %c)"
        " : (i32, i32, i32, i32, i32, i32) -> i32",
    ],
)
def test_the_benign_idioms_stay_silent(call):
    """The counterweight, and it is the half that nearly went wrong.

    A `dst` rule that demanded a constant flags all 19 destination-passing
    calls in this corpus. Checking the rule against a sample of its own hits is
    what caught that; without it, 19 correct calls would have been reported as
    defects.
    """
    assert probe(module(call)) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a, b, c", ["a", "b", "c"]),
        ("%a, %b", ["%a", "%b"]),
        # The reason this is not `str.split(",")`: an MLIR type carries commas
        # inside its own brackets, and splitting on every comma miscounts the
        # operands -- which would make the arity rule fire at random.
        ("!llvm.ptr<1>, i32, i32", ["!llvm.ptr<1>", "i32", "i32"]),
        ("!pto.tile_buf<rows=1, cols=64>, i32", ["!pto.tile_buf<rows=1, cols=64>", "i32"]),
    ],
)
def test_split_top_level_respects_brackets(text, expected):
    assert split_top_level(text) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
