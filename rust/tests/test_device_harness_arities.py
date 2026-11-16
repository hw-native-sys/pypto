# Copyright (c) PyPTO contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every kernel arity in the corpus must have a path in the device harness.

This is the guard for the most expensive repeated mistake in this repository. The
harness `rust/tools/direct-invoke/op_host/main.asc` switches on `KERNEL_ARGS`, and
four separate bugs came out of that cascade:

  * the second input copied only when `KERNEL_ARGS == 3` -- a five-argument kernel
    ran with `k = 0`, and the whole `beta*(q.k)*v` term vanished;
  * the third input copied only when `KERNEL_ARGS == 5` -- a four-argument kernel
    ran with `wu = 0`;
  * a ninth-argument branch allocated `p3` but filled it in the `== 5` branch, so
    `x3` was zero and the result was short by exactly one product;
  * the comparison tolerance was passed as the seventh argument, which is `eps`,
    so every "re-run with a looser tolerance" was the same run.

Every one of those presented as an emitter bug first. The arity is the only fact
the harness may branch on, and a kernel shape the harness cannot express must fail
HERE, at `cargo test` speed, rather than on a board forty minutes away.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
HARNESS = ROOT / "rust" / "tools" / "direct-invoke" / "op_host" / "main.asc"

PTR = "!llvm.ptr<1>"


def harness_arities() -> set[int]:
    """The arities `main.asc` can actually INVOKE, read from its `KERNEL_DO` calls.

    Not from the `KERNEL_ARGS` comparisons: the first version of this guard
    expanded a `>= 4` branch into 4..9 and declared every arity supported, so
    hiding the nine-argument branch changed nothing and the negative control
    passed. The calls are the truth -- a branch that dispatches to nothing is a
    comment.

    `KERNEL_DO(blockDim, stream, ...)` -- the arity is its argument count minus two.
    """
    text = HARNESS.read_text(encoding="utf-8")
    found: set[int] = set()
    for m in re.finditer(r"KERNEL_DO\(([^;]*?)\);", text, re.S):
        args = [a.strip() for a in m.group(1).split(",") if a.strip()]
        if len(args) >= 4:
            found.add(len(args) - 2)
    return found


def entry_arity(path: Path) -> int | None:
    """Number of GM pointer arguments on the `.tile`'s entry function."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"llvm\.func @\w+\((.*?)\)\s*attributes", text, re.S)
    if not m:
        return None
    return m.group(1).count(PTR)


def test_every_corpus_arity_has_a_harness_path() -> None:
    supported = harness_arities()
    assert supported, "could not read any KERNEL_ARGS branch from main.asc"

    missing: list[tuple[str, int]] = []
    seen: set[int] = set()
    for tile in sorted(BENCH.glob("*/tile/**/*.tile")):
        arity = entry_arity(tile)
        if arity is None:
            continue
        seen.add(arity)
        if arity not in supported:
            missing.append((str(tile.relative_to(BENCH)), arity))

    assert not missing, (
        "these kernels need an arity the device harness has no branch for, and a "
        "missing branch presents as an emitter bug rather than a harness bug:\n  "
        + "\n  ".join(f"{name}: {n} pointer arguments" for name, n in missing)
        + f"\nsupported: {sorted(supported)}"
    )
    assert seen, "no kernel arities were found at all -- the parse is broken"


def test_the_harness_is_the_single_source_of_arity_truth() -> None:
    """A count of `KERNEL_ARGS ==` branches, so adding a shape is a visible edit."""
    text = HARNESS.read_text(encoding="utf-8")
    eq = sorted({int(n) for n in re.findall(r"KERNEL_ARGS\s*==\s*(\d+)", text)})
    # If a new equality branch appears, this list should grow with it: the number
    # is not the point, noticing the edit is.
    assert len(eq) >= 4, f"expected at least four equality branches, found {eq}"


def test_every_recorded_run_sized_its_buffers_for_the_whole_tile() -> None:
    """`n` is the DEVICE ALLOCATION, not the row width, and a many-row kernel that
    is given the row width writes past it.

    `main.asc` allocates `n * sizeof(float)` per pointer and the oracles compare
    `n` elements. So for a kernel declaring R rows of C columns, every recorded
    `verify_args` must pass `n >= R*C`: anything less and the kernel reads
    uninitialised device memory for the rows the harness never filled, writes
    past the allocation for the rows the oracle never reads, and the PASS it earns
    covers one row of R.

    MEASURED 2026-11-08, one case in the tree: `rmsnorm_v41_8x1280_quarter` was
    banked `verified` off `n = 1280` against a kernel that touches 8*1280 = 10240
    floats -- under-sized 8x, with 7 rows of 1280 written past a 5120-byte
    allocation. Re-run at n=10240 it reports `wrong=10240 max_rel=7.058e+00`, and
    re-run at n=10240 on ROW-PERIODIC input -- eight identical rows, which is the
    one shape that makes the single-row ORACLE 1 a valid oracle for an eight-row
    kernel -- it reports `wrong=0 max_rel=2.933e-08`. So the artefact was right
    and the run was under-sized; the evidence, not the kernel, was the defect.

    Nothing compared the two numbers, because they live in different files: the
    extents are in the `.tile` and `n` is in the manifest's `verify_args`. This
    does.
    """
    import json

    bad: list[str] = []
    checked = 0
    for manifest in sorted(BENCH.glob("*/manifest.json")):
        corpus = manifest.parent.name
        for stem, entry in json.loads(manifest.read_text())["kernels"].items():
            args = (entry.get("verify_args") or "").split()
            if len(args) < 3:
                continue
            try:
                n = int(args[2])
            except ValueError:
                continue
            tile = manifest.parent / "tile" / f"{stem}.tile"
            if not tile.is_file():
                continue
            text = tile.read_text()
            load = re.search(r"__tile_load_f32\((%\w+), (%\w+), (%\w+)\)", text)
            if not load:
                continue

            def const(name: str, text: str = text) -> int | None:
                m = re.search(
                    rf"{re.escape(name)} = llvm\.mlir\.constant\((\d+)", text
                )
                return int(m.group(1)) if m else None

            rows, cols = const(load.group(2)), const(load.group(3))
            if not rows or not cols or rows == 1:
                continue
            checked += 1
            if n < rows * cols:
                bad.append(
                    f"{corpus}/{stem}: declares {rows}x{cols} = {rows * cols} "
                    f"elements and was run at n={n}"
                )
    assert checked, "no many-row kernel carries verify_args -- this gate is asleep"
    assert not bad, (
        "these recorded runs allocated for less than the kernel touches, so their "
        "PASS covers one row:\n  " + "\n  ".join(bad)
    )
