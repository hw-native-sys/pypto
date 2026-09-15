# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute the runnable code blocks embedded in the user manual.

Teaching code that is never executed rots silently: a kernel can compile, run,
write nothing, and still look right on the page. This runner makes the manual
itself the executable artifact, so a doc block that stops working fails CI.

Opt-in, by an HTML comment on the line before the fence — invisible in the
rendered page, so it costs the reader nothing:

    <!-- doctest: setup -->      imports / constants / helpers for the page;
                                 prepended to every run block on that page
    <!-- doctest: run -->        executed as (setup + this block)

Unmarked blocks are ignored, which is what the fragments and illustrative
snippets throughout the manual need.

Each run block asserts its own result, so "it executed" and "it computed the
right thing" are the same check.

Usage:
    python tests/docs/run_doc_examples.py [--pages docs/en/user/performance] [-p a2a3sim]
    python tests/docs/run_doc_examples.py --jobs 16     # run N blocks at once
    python tests/docs/run_doc_examples.py --list        # show what would run
    python tests/docs/run_doc_examples.py --check-parity  # zh code == en code
"""

import argparse
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where a block's generated script is written. `--keep` retains it here rather
#: than in the repository root, which AGENTS.md keeps free of stray scripts.
_SCRATCH_DIR = Path(__file__).resolve().parent / "generated"

_TIMEOUT_SECONDS = 1800
_FAILURE_TAIL_LINES = 12

_BLOCK = re.compile(
    r"<!--\s*doctest:\s*(?P<kind>setup|run)\s*-->\s*\n```python\n(?P<code>.*?)\n```",
    re.S,
)


class Block:
    """One marked code block: where it came from and what it holds."""

    def __init__(self, page: Path, kind: str, code: str, line: int):
        self.page = page
        self.kind = kind
        self.code = code
        self.line = line

    @property
    def label(self) -> str:
        return f"{self.page.relative_to(REPO_ROOT)}:{self.line}"


def extract(page: Path) -> list[Block]:
    """Return the marked blocks of *page*, in source order."""
    text = page.read_text(encoding="utf-8")
    out = []
    for m in _BLOCK.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        out.append(Block(page, m.group("kind"), m.group("code"), line))
    return out


def page_program(blocks: list[Block], run_block: Block) -> str:
    """Compose the setup blocks of a page with one run block."""
    setup = "\n".join(b.code for b in blocks if b.kind == "setup")
    return f"{setup}\n\n{run_block.code}\n" if setup else f"{run_block.code}\n"


def _tail(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    """Join the last few lines of a finished or abandoned process's output."""
    parts = []
    for stream in (stdout, stderr):
        if stream is None:
            continue
        parts.append(stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream)
    lines = "".join(parts).strip().splitlines()
    return "\n".join(lines[-_FAILURE_TAIL_LINES:])


def run_block(blocks: list[Block], block: Block, platform: str, keep: bool) -> tuple[bool, str]:
    """Execute one run block; return (passed, output tail)."""
    program = page_program(blocks, block).replace("__PLATFORM__", platform)
    # `--keep` leaves the generated file behind, so it is written under this
    # directory rather than the repository root, which AGENTS.md keeps free of
    # stray scripts. The block still runs with the repository root as its
    # working directory; no doc block imports anything by repository-relative
    # path, so the script's own directory never has to be the root.
    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=not keep, dir=_SCRATCH_DIR) as fh:
        fh.write(program)
        fh.flush()
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, path from tempfile
                [sys.executable, fh.name],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            # A timeout is a failed block, not a reason to abandon the run:
            # letting it escape the worker takes down every other result and
            # the summary with it. Keep whatever the block printed before it
            # hung -- that is the most useful part of the report.
            captured = _tail(expired.stdout, expired.stderr)
            return False, (
                f"TIMED OUT after {_TIMEOUT_SECONDS}s -- output before the timeout:\n{captured}"
                if captured
                else f"TIMED OUT after {_TIMEOUT_SECONDS}s -- no output captured"
            )
    return proc.returncode == 0, _tail(proc.stdout, proc.stderr)


def check_parity(pages: list[Path]) -> list[str]:
    """Report zh pages whose marked code differs from their en counterpart.

    Code is not translated, so the two must stay byte-identical; otherwise only
    one language is covered by the runs above.
    """
    problems = []
    for en in pages:
        if "/en/" not in en.as_posix():
            continue
        zh = Path(en.as_posix().replace("/en/", "/zh/", 1))
        if not zh.exists():
            continue
        en_code = [b.code for b in extract(en)]
        zh_code = [b.code for b in extract(zh)]
        if en_code != zh_code:
            rel = zh.relative_to(REPO_ROOT)
            if len(en_code) != len(zh_code):
                problems.append(f"{rel}: {len(zh_code)} marked blocks, en has {len(en_code)}")
            else:
                for i, (a, b) in enumerate(zip(en_code, zh_code)):
                    if a != b:
                        problems.append(f"{rel}: marked block {i + 1} differs from en")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the runnable code blocks in the user manual.")
    parser.add_argument(
        "--pages",
        default="docs/en/user",
        help="directory to scan for markdown pages (default: docs/en/user)",
    )
    parser.add_argument("-p", "--platform", default="a2a3sim", help="substituted for __PLATFORM__")
    parser.add_argument("--list", action="store_true", help="list runnable blocks and exit")
    parser.add_argument("--check-parity", action="store_true", help="also verify zh code matches en")
    parser.add_argument("--keep", action="store_true", help="keep the generated .py files")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="how many blocks to run at once (default: 1). Each block is an independent process.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        print(f"--jobs must be at least 1, got {args.jobs}", file=sys.stderr)
        return 1

    root = (REPO_ROOT / args.pages).resolve()
    pages = [root] if root.is_file() else sorted(root.rglob("*.md"))
    if not pages:
        print(f"no markdown pages under {args.pages}", file=sys.stderr)
        return 1

    if args.check_parity:
        problems = check_parity(pages)
        for p in problems:
            print(f"FAIL parity  {p}")
        if problems:
            return 1
        print("parity OK: every zh marked block matches its en counterpart")

    # A run block is composed with its own page's setup blocks and executed as
    # its own process, so no two of them share anything -- `--jobs` just runs N
    # at once. The width belongs to the caller, who knows what else the machine
    # is doing, so it defaults to 1 and CI passes it explicitly.
    todo: list[tuple[list[Block], Block]] = []
    for page in pages:
        blocks = extract(page)
        todo.extend((blocks, b) for b in blocks if b.kind == "run")
    total = len(todo)

    if args.list:
        for _, block in todo:
            print(f"  would run  {block.label}")
        print(f"\n{total} runnable block(s)")
        return 0

    # Printed as they land so a long run shows progress; the failing tails are
    # replayed below, where they are not buried under later passes.
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_block, blocks, block, args.platform, args.keep): block for blocks, block in todo
        }
        for future in as_completed(futures):
            block = futures[future]
            ok, tail = future.result()
            print(f"{'PASS' if ok else 'FAIL'}  {block.label}", flush=True)
            if not ok:
                failures.append((block.label, tail))

    for label, tail in failures:
        print(f"\n=== {label}\n{tail}", file=sys.stderr)
    print(f"\n{total - len(failures)}/{total} runnable doc blocks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
