# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every checked-in `.pto` must actually assemble.

The corpus already had a byte-identity gate over its PTO: re-run the emitter and
compare against the stored file. That gate is sound and it was green while three
of the stored artefacts did not parse. It could not see them, because it
compares the emitter's output to the file the emitter produced -- when an
emitter writes something invalid, both sides hold the SAME invalid text and the
comparison succeeds. Reproducibility is not validity.

So this runs each stored `.pto` through the assembler and records what it says.

# What it found

`KNOWN_INVALID` is now EMPTY: all 40 stored artefacts assemble. Getting there
took seven emitter fixes, and the three entries that were recorded named the
FIRST blocker of each, not the whole of it. Each fix uncovered the next:

`pto.tci` (two artefacts)
    1. `{descending = false}` sat inside the `ins(...)` operand list, where MLIR
       wants `:`. It belongs on the op, after `outs(...)`.
    2. the index tile was `ui32` while the start scalar is
       `arith.constant : i32`, and the op's verifier requires them to match.

the topk composition (behind the same two artefacts)
    3. `pto.tgather` with a mask pattern requires `axis`; `row` is the only
       direction the verifier accepts for a 2:1 gather.
    4. the `1×k` result tile at `k=6` is 24 bytes and cannot be allocated at
       all, because a row must be 32-byte aligned. Padded, `v_col` keeping the
       logical width.
    5. `pto.tmov` refuses `1×cols → 1×k`; the head extract is `pto.textract`.

`attention_gqa` (one artefact)
    6. the arm reused the whole-operand partition view for a per-head use, so
       one SSA value carried `<32x64xf32>` and `<16x64xf32>`.
    7. with that fixed, the store moved the whole output view out of an `s×d`
       tile; the result now records that it covers its own extent only.

None of that made the emitted kernels correct: fixes 3-5 are forced by the
assembler rather than chosen, and the topk arm's `P1010` polarity is still
unresolved. This gate answers "does the text parse", and only that.

# Why the list stays

Deleting `KNOWN_INVALID` along with the defects would delete the mechanism that
retires an entry. It is kept, and the gate asserts that every recorded artefact
still fails with its recorded message, so a fix flips this gate and tells
whoever fixed it to drop the entry. A declared exception with no test behind it
outlives the defect it was written for.

# The toolchain caveat, stated rather than buried

The assembler here is whatever `pto-to-ascendc` finds, and this repository pins
`ptoas` v0.70 while the machine this was measured on has 0.68. The gate prints
the assembler version it used so a reader on 0.70 can see whether a verdict
moved. For the defects above it cannot be a version difference: an attribute
inside an operand list, one value at two types, a missing verifier attribute and
an unaligned row are each invalid on their face. All were reproduced on 0.68 and
none of the fixes depends on a 0.70-only rule.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
# bench lives at rust/bench, so the REPOSITORY root is one more level up.
ROOT = BENCH.parent.parent
EMITTER = ROOT / "rust/target/release/pto-to-ascendc"

# path -> (the defect it demonstrates, assembler text that proves it is still
# that defect). Empty: the corpus assembles.
KNOWN_INVALID = {}


def stored_pto():
    """Every `.pto` the repository tracks under `bench/`."""
    out = subprocess.run(
        ["git", "ls-files", "rust/bench/*/pto/*.pto", "rust/bench/*/pto/fixed/*.pto"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def assembler_version():
    if not EMITTER.is_file():
        return None
    run = subprocess.run([str(EMITTER), "--version"], check=False, capture_output=True, text=True)
    return (run.stdout + run.stderr).strip().splitlines()[0] if run.returncode == 0 else "unknown"


def assemble(rel):
    """`(ok, message)` for one stored artefact."""
    run = subprocess.run(
        [str(EMITTER), rel, "-o", "/dev/null"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if run.returncode == 0:
        return True, ""
    text = run.stdout + run.stderr
    hit = re.search(r"error: (.*)", text)
    return False, (hit.group(1).strip() if hit else text.strip().splitlines()[-1])


def classify(files):
    """Partition the corpus into (unexpected_fail, unexpected_pass, expected_fail)."""
    unexpected_fail, unexpected_pass, expected_fail = [], [], []
    for rel in files:
        ok, message = assemble(rel)
        known = rel in KNOWN_INVALID
        if ok and known:
            unexpected_pass.append(rel)
        elif ok:
            pass
        elif known:
            expected_fail.append((rel, message))
        else:
            unexpected_fail.append((rel, message))
    return unexpected_fail, unexpected_pass, expected_fail


def chunk_loop_problems(files) -> list[tuple[str, str]]:
    """Every chunk loop must COVER its width. Assembling does not say that.

    This is the hole that let a wrong loop through: `scf.for %n_i = 0 to
    (width/chunk) step chunk` steps past the range on its first iteration, so the
    loop runs once at offset zero and the kernel computes a fraction of its output
    while still assembling. Nothing checked the arithmetic, so nothing noticed.

    The emitter now builds its loops through a constructor that refuses a chunk which
    does not divide the width; this checks the ARTEFACTS, which is the half a
    constructor cannot cover -- a file can be hand-edited, or produced before the
    constructor existed.
    """
    import re

    problems: list[tuple[str, str]] = []
    # `stored_pto` returns repository-relative PATHS, not Path objects
    for rel in files:
        p = ROOT / rel
        text = p.read_text(encoding="utf-8", errors="replace")
        # CHUNK loops only. The blocked-matmul path also iterates `%n_i`, with a
        # per-core bound and a different partner constant; matching on the induction
        # variable alone reported those as broken chunk loops, which is a gate
        # inventing a defect.
        if "%n_chunk = arith.constant" not in text:
            continue
        if "to %n_iters step %n_one" not in text:
            problems.append((str(rel), "the chunk loop does not step by one over its count"))
            continue

        def const(name: str):
            m = re.search(rf"%{name} = arith\.constant (\d+) : index", text)
            return int(m.group(1)) if m else None

        iters, chunk = const("n_iters"), const("n_chunk")
        if iters is None or chunk is None:
            problems.append((str(rel), "the chunk loop's constants are not literal"))
            continue
        widths = [int(m.group(1)) for m in re.finditer(r"shape = \[%c\d+, %c(\d+)\]", text)]
        if not widths:
            problems.append((str(rel), "no tensor view to compare the loop against"))
            continue
        width = max(widths)
        if iters * chunk != width:
            problems.append(
                (str(rel), f"{iters} chunks of {chunk} cover {iters * chunk}, not the width {width}")
            )
    return problems


def report(files, unexpected_fail, unexpected_pass, expected_fail) -> int:
    """Print what was found and return the exit status."""
    if expected_fail:
        print(f"{len(expected_fail)} known-invalid artefact(s), still failing as recorded:")
        for rel, message in expected_fail:
            print(f"  {rel}\n      {KNOWN_INVALID[rel][0]}\n      assembler: {message[:140]}")
        print()

    if unexpected_pass:
        print(f"{len(unexpected_pass)} artefact(s) are listed as invalid but now ASSEMBLE:")
        for rel in unexpected_pass:
            print(f"  {rel}\n      the emitter defect looks fixed -- drop it from KNOWN_INVALID")
        print()

    if unexpected_fail:
        print(f"{len(unexpected_fail)} artefact(s) do NOT assemble and are not recorded:")
        for rel, message in unexpected_fail:
            print(f"  {rel}\n      {message[:160]}")
        print()

    chunk_problems = chunk_loop_problems(files)
    if chunk_problems:
        print("Chunk loops that do not cover their width:")
        for rel, message in chunk_problems:
            print(f"  {rel}\n      {message}")
        print()

    bad = len(unexpected_fail) + len(unexpected_pass) + len(chunk_problems)
    if bad:
        print(f"Results: {bad} problem(s)")
        return 1
    if expected_fail:
        print(
            f"Results: {len(files) - len(expected_fail)} of {len(files)} assemble; "
            f"the other {len(expected_fail)} are recorded defects that still fail"
        )
    else:
        print(f"Results: all {len(files)} stored artefact(s) assemble")
    return 0


def main() -> int:
    if not EMITTER.is_file():
        print(
            f"The assembler driver is not built: {EMITTER.relative_to(ROOT)}\n"
            "Build it with `cargo build --release -p pto_to_ascendc` in `rust/`.\n"
            "Refusing to report a pass: a gate that cannot run has not run."
        )
        return 1

    files = stored_pto()
    if not files:
        print("No stored .pto artefacts found; the reader or the corpus broke")
        return 1

    print(f"Assembling {len(files)} stored artefact(s) with {assembler_version()}\n")
    return report(files, *classify(files))


if __name__ == "__main__":
    sys.exit(main())
