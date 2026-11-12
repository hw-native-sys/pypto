# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every accepted kernel's live tile set must fit the buffer it is planned against.

`rust/bench/shapes/live_set.py` is the report; this is the gate. The report exists
because the same question was answered three times by reasoning instead of by
measurement (see `rust/bench/rust/README.md`, lesson 1), and each time the answer was
"the order of the chain", not "the emitter cannot express this".

The gate is deliberately narrow: it fails only when a kernel the emitter ACCEPTS
has a live peak over the buffer, which would mean the budget and the emitted body
disagree -- a defect in the emitter, not a tight fit. Headroom is printed and not
asserted, because headroom is a warning about the next edit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "rust" / "bench" / "shapes" / "live_set.py"
LESSONS = ROOT / "rust" / "README.md"


def test_no_accepted_kernel_exceeds_the_buffer_it_was_planned_against() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT
    )
    assert proc.returncode == 0, (
        "a kernel whose live peak exceeds the 188416-byte buffer was accepted by the "
        "emitter, so the budget check and the emitted body disagree:\n"
        f"{proc.stdout[-3000:]}{proc.stderr[-2000:]}"
    )
    assert "every stored kernel fits" in proc.stdout, (
        f"the report did not run to completion:\n{proc.stdout[-2000:]}"
    )


def test_the_lessons_file_names_a_guard_for_each_lesson() -> None:
    """A lessons file without guards is a diary. Each lesson must cite one."""
    text = LESSONS.read_text(encoding="utf-8")
    assert "## Where each guard lives" in text, (
        "rust/README.md must end with the table mapping each lesson to the check that "
        "now catches it, or it is prose nobody will act on"
    )
    for lesson in ("A kernel that refuses on UB", "per-arity code", "write the rule first"):
        assert lesson in text, f"rust/README.md no longer covers: {lesson}"
    # and the file must be honest that repetition is the signal
    assert "missing check or a missing sentence" in text, (
        "the rule at the bottom of rust/README.md is the point of the file"
    )
