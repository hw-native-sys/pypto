# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Report the live tile set of every stored kernel, against the buffer it must fit.

This exists because the same question was answered three times by reasoning instead
of by measurement, and each time the answer was "the order of the chain", not "the
emitter cannot express this":

  * the situ gate/up pair materialised a constant TILE where a scalar add would do;
  * the delta-net needed a broadcast that the emitters already had a name for;
  * the K3 convolution loaded eight operands up front and kept nine tiles live,
    where interleaving a tap holds four.

`rust/bench/LESSONS.md` is the prose; this is the number. For each emitted `.cpp`
the live interval of every buffer is read out of the body, and the peak is compared
with the 188416-byte usable buffer.

Exit status is 1 only if a kernel that the emitter ACCEPTS has a peak over the
buffer -- which would mean the budget and the body disagree, a real defect. A
kernel whose peak is close to the limit is printed and not failed: headroom is a
warning about the next edit, not an error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent

# The usable vector budget the AscendC emitter plans against: 192 KiB minus the
# 8 KiB it reserves for TPipe management.
UB_USABLE = 188416

BUF_RE = re.compile(r"(?<![A-Za-z0-9_])(?:buf|tbuf)(\d+)\b")


def live_peak(text: str) -> tuple[int, int]:
    """(peak live buffers, peak live bytes) over the Process() body."""
    body = text.split("Process()", 1)[1] if "Process()" in text else text
    sizes: dict[int, int] = {}
    for m in re.finditer(r"InitBuffer\((?:tbuf|inQue|outQue)(\d+),\s*(?:(\d+),\s*)?(\d+)\)", text):
        idx = int(m.group(1))
        sizes[idx] = int(m.group(3))

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, line in enumerate(body.splitlines()):
        if "InitBuffer" in line or "LocalTensor" in line:
            continue
        for m in BUF_RE.finditer(line):
            idx = int(m.group(1))
            first.setdefault(idx, i)
            last[idx] = i

    peak_bufs = 0
    peak_bytes = 0
    if first:
        for i in range(min(first.values()), max(last.values()) + 1):
            live = [k for k in first if first[k] <= i <= last[k]]
            peak_bufs = max(peak_bufs, len(live))
            peak_bytes = max(peak_bytes, sum(sizes.get(k, 0) for k in live))
    return peak_bufs, peak_bytes


def main() -> int:
    rows = []
    over = []
    for cpp in sorted(BENCH.glob("*/ascendc/**/*.cpp")):
        text = cpp.read_text(encoding="utf-8", errors="replace")
        if "Process()" not in text:
            continue
        nbufs, nbytes = live_peak(text)
        rel = cpp.relative_to(BENCH)
        rows.append((nbytes, nbufs, str(rel)))
        if nbytes > UB_USABLE:
            over.append((rel, nbufs, nbytes))

    rows.sort(reverse=True)
    print(f"{'kernel':52s} {'live':>5s} {'bytes':>9s} {'headroom':>9s}")
    for nbytes, nbufs, rel in rows:
        print(f"{rel:52s} {nbufs:5d} {nbytes:9d} {UB_USABLE - nbytes:9d}")

    print()
    if over:
        print(f"{len(over)} accepted kernel(s) over the {UB_USABLE}-byte buffer:")
        for rel, nbufs, nbytes in over:
            print(f"  {rel}: {nbufs} live buffers, {nbytes} B")
        return 1
    print(f"every stored kernel fits: {len(rows)} checked against {UB_USABLE} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
