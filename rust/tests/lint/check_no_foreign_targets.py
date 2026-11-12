# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the Rust trees consistent with the repository licence.

CANN Open Software License Agreement Version 2.0, section 3.1: the Software and
its derivative works may not be used to develop or distribute software for use
in systems with processors other than Huawei AI Processors. So a Rust file whose
*code* targets another vendor's processor does not belong in this tree.

Two design points, both of which came out of measuring a real tree rather than
from taste:

1. **Code and comments are judged differently.** A comment that contrasts with
   CUDA, or explains why an Ascend buffer is sized the way it is "unlike a GPU's
   shared memory", is documentation and is allowed. Over the candidate Rust
   sources the counts were 42 non-comment hits against 194 comment hits; a check
   that conflated them would report 236 findings where 42 matter, and would be
   switched off within a week.

2. **Process vocabulary is refused outright, in code or comment.** Milestone
   tags and tracker ids ("M126:", "task #7") describe how work was scheduled
   somewhere else. They were the same lines as the foreign-target references in
   the worst file measured, which is why one check covers both.

Scope is the Rust trees only. The C++/Python side of this repository is not
touched, and `3rdparty/` is excluded.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCANNED_PREFIXES = ("rust/",)
# `rust/bench/` carries the kernel corpus in several representations, and those files
# are as capable of naming another vendor's target as source is -- the captured
# MLIR arrived with a provenance header naming one. JSON is included because a
# rustc target specification is JSON, and the one this project uses for its
# device names an entirely different architecture.
SCANNED_SUFFIXES = (".rs", ".toml", ".tile", ".pto", ".mlir", ".cpp", ".h", ".json")

# Processor targets that are not Huawei AI Processors. Word-bounded: `bang` must
# not fire on "banging", and `metal` must not fire on "metallurgy".
FOREIGN_TARGET = re.compile(
    r"\b(cuda|nvidia|nvptx|ptx|metal|msl|spirv|vulkan|rocm|hip|gfx\d+|"
    r"trainium|nki|gaudi|tpc|cambricon|bang|tenstorrent|ttmetal|hexagon|"
    r"riscv|rvv|musa|moore[-_ ]?threads|cerebras|csl)\b",
    re.IGNORECASE,
)

# Milestone tags, tracker ids and phase labels.
#
# `M1 Ultra` and `M5 Max` are Apple silicon generations and legitimate prose, so
# the tag form requires a colon or a ` #`. That is not sufficient on its own:
# measured over this project's sources, `M5:` occurs twice and means DIFFERENT
# things -- "Apple M5: upstream measured ..." is hardware provenance, while
# "deferred to M5:" is a milestone. Digit count does not separate them either,
# since the real tags run from M4 to M150. Only the preceding word does, hence
# the one narrow exception for `Apple`.
# A trailing `[a-z]?` catches the sub-milestone spelling (`M102a`, `M104b`).
#
# The second alternative is the one this check was missing. Requiring a colon
# made it blind to a BARE reference -- `sibling of M94`, `the M72 sink fold-in`
# -- which is the same process vocabulary and, worse, is unresolvable: a reader
# of this tree has no way to find out what M94 was. Measured when `tile_std`
# was first scanned: the colon form found 23 lines and the bare form found 85
# more, in the same file. A two- or three-digit `M` cannot be an Apple silicon
# generation, so the bare form is safe to refuse at that width; `M1`..`M9` are
# left to the colon rule, where only the preceding word separates `Apple M5:`
# from `deferred to M5:`.
PROCESS_VOCAB = re.compile(
    r"((?<!Apple )\bM\d{1,3}[a-z]?:|(?<!Apple )\bM\d{2,3}[a-z]?\b|"
    r"\bM\d{1,3}\s+#|\btask\s+#\d+|\bissue\s+#\d+|"
    r"\bPhase\s+[A-Z0-9]+\s+[-—]\s)",
)

# A private document reference. The digit class is load-bearing: a pattern
# written `docs/[A-Za-z_]*\.md` reports ZERO hits on `docs/DS4_THING.md`, which
# is exactly the filename shape that was actually present.
PRIVATE_DOC = re.compile(r"docs/[A-Za-z0-9_]+\.md")

AUTHOR_PATH = re.compile(r"/(Users|home)/[a-z0-9_.-]+/")


def is_comment(line: str) -> bool:
    """True when the line carries no code -- only a comment or doc comment."""
    s = line.strip()
    return (
        s.startswith("//")
        or s.startswith("#")
        or s.startswith("*")
        or s.startswith("/*")
        or s.startswith("*/")
    )


def tracked_files(root: Path) -> list[Path]:
    """Git-tracked files inside the scanned prefixes."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Error: cannot list git files: {exc}", file=sys.stderr)
        sys.exit(1)
    files = []
    for line in out.splitlines():
        if not line.startswith(SCANNED_PREFIXES):
            continue
        if not line.endswith(SCANNED_SUFFIXES):
            continue
        path = root / line
        if path.is_file():
            files.append(path)
    return files


def check_file(path: Path, root: Path) -> list[str]:
    """Return one message per finding in `path`."""
    findings = []
    rel = path.relative_to(root)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{rel}: cannot read: {exc}"]

    for number, line in enumerate(text.splitlines(), start=1):
        comment = is_comment(line)

        if not comment:
            hit = FOREIGN_TARGET.search(line)
            if hit:
                findings.append(
                    f"{rel}:{number}: code names the non-Huawei target "
                    f"`{hit.group(0)}` -- CANN OSL 3.1 does not permit it here"
                )

        hit = PROCESS_VOCAB.search(line)
        if hit:
            findings.append(
                f"{rel}:{number}: process vocabulary `{hit.group(0).strip()}` -- "
                "write the rationale, not the tracking id"
            )

        hit = PRIVATE_DOC.search(line)
        # A `docs/...md` inside a URL is a reference to someone else's published
        # documentation, which is a citation rather than a dangling pointer. The
        # first version of this check flagged the GGUF format specification.
        if hit and "://" not in line:
            findings.append(
                f"{rel}:{number}: points at `{hit.group(0)}`, which this repository does not publish"
            )

        hit = AUTHOR_PATH.search(line)
        if hit:
            findings.append(
                f"{rel}:{number}: author-absolute path `{hit.group(0)}` -- "
                "resolve an environment variable instead"
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse non-Huawei processor targets and process vocabulary in the Rust trees"
    )
    parser.add_argument("path", nargs="?", default=".", help="repository root")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not (root / ".git").exists():
        print(f"Error: '{root}' is not a git repository", file=sys.stderr)
        return 1

    files = tracked_files(root)
    if not files:
        # Not a pass to celebrate: before the first Rust PR lands there is
        # nothing to check, and saying so is more useful than printing OK.
        print(f"No files under {', '.join(SCANNED_PREFIXES)} to check yet")
        return 0

    print(f"Checking {len(files)} file(s) under {', '.join(SCANNED_PREFIXES)}...")
    findings = []
    for path in files:
        found = check_file(path, root)
        findings.extend(found)
        if args.verbose and not found:
            print(f"OK {path.relative_to(root)}")

    if findings:
        print()
        for message in findings:
            print(f"  {message}")
        print(f"\nResults: {len(findings)} finding(s) in {len(files)} file(s)")
        return 1

    print(f"\nResults: 0 findings in {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
