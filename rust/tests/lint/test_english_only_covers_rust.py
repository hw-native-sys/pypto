# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`check_english_only` now scans Rust, and can say so when it allows something.

The gap this closes was silent in the worst way: the check reported "1830
passed" over a repository holding 175 tracked `.rs` files, none of which it had
opened, because `.rs` was missing from `source_extensions`. A green lint whose
scope excludes the language you just added says nothing about that language.
With `.rs` included the count is 2005.

The allowance mechanism is tested as hard as the check, because an opt-out is
the part that rots: the next person to hit a failure reaches for the marker, and
if a bare marker works then the lint has quietly stopped meaning anything.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LINT = ROOT / "rust" / "lint" / "check_english_only.py"

def _git_env() -> dict:
    """`os.environ` with every `GIT_*` variable removed.

    WHY, measured 2026-11-08. `git commit <path>` and `git commit -o <path>` build
    the commit from a TEMPORARY index and export `GIT_INDEX_FILE` (and `GIT_DIR`)
    to hooks. This repository's `pre-commit` hook runs `pytest rust/tests`, so any
    test here that shells out to `git` inherits those variables and answers a
    different question than the one it asks: a `git init` + `git add -A` in a tmp
    directory writes into the OUTER repository's temporary index instead of the
    throwaway one, so the lint sees no tracked files and reports "found 0"
    allowances. The failure then reads as a corpus problem rather than an
    environment one.

    PASS THIS TO EVERY CHILD, not only to the `git` calls. The lint driver runs
    `git ls-files` itself, so invoking it with an unscrubbed environment leaves the
    same bug one process further down. Measured: scrubbing only the `git init` and
    `git add` calls fixed the two tests that read the repository and broke the two
    that build a throwaway one -- which then failed under the hook while passing
    when run by hand, the same signature as the original defect.

    The consequence is worse than a red gate: the hook refused a legitimate
    single-path commit, deterministically, three times in a row, and the obvious
    next move when a hook refuses for no visible reason is to bypass the hook --
    the one thing it exists to prevent.
    """
    env = dict(os.environ)
    for k in [k for k in env if k.startswith("GIT_")]:
        del env[k]
    return env


sys.path.insert(0, str(LINT.parent))

from check_english_only import allowance, contains_non_english  # noqa: E402

# Built from codepoints rather than written literally, so this file does not
# itself carry the characters the lint refuses -- it is scanned too.
IDEOGRAPH = chr(0x4E2D)
FULLWIDTH_BAR = chr(0xFF5C)


def test_rust_is_in_the_scanned_extensions():
    text = LINT.read_text(encoding="utf-8")
    assert '".rs"' in text, (
        "`.rs` is not in the lint's source_extensions, so every Rust file in this "
        "repository is unscanned and the pass count means nothing about them"
    )


def test_the_checker_fires_on_a_rust_file(tmp_path):
    f = tmp_path / "kernel.rs"
    f.write_text(f"// a comment with {IDEOGRAPH} in it\n", encoding="utf-8")
    bad, hits = contains_non_english(f.read_text(encoding="utf-8"))
    assert bad, "the checker did not flag an ideograph in a .rs file"
    assert hits and hits[0][0] == 1


def test_a_clean_rust_file_passes(tmp_path):
    f = tmp_path / "kernel.rs"
    f.write_text("/// Lower a tile-rs kernel to PTO.\npub fn emit() {}\n", encoding="utf-8")
    bad, _ = contains_non_english(f.read_text(encoding="utf-8"))
    assert not bad


def test_em_dashes_and_arrows_are_still_fine(tmp_path):
    # The counterweight. This project's comment style uses them heavily, and a
    # check that refused them would be turned off within a week.
    f = tmp_path / "kernel.rs"
    f.write_text("// MLIR → PTO — the one route that needs nothing.\n", encoding="utf-8")
    bad, hits = contains_non_english(f.read_text(encoding="utf-8"))
    assert not bad, f"em dash or arrow refused: {hits}"


def test_an_allowance_must_carry_a_reason():
    assert allowance("// nothing here\n") is None
    assert allowance("// check-english-only: allow -- the model's own tokens\n") == ("the model's own tokens")
    # A bare marker parses to the empty string, which the driver treats as a
    # FAILURE rather than an allowance. That distinction is the whole point.
    assert allowance("// check-english-only: allow\n") == ""


def test_a_bare_allowance_marker_fails_the_lint(tmp_path):
    # End to end, through the real driver, because the empty-reason path is in
    # `main` and not in `allowance`.
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=_git_env())
    bad = repo / "src" / "x.rs"
    bad.write_text(f"// check-english-only: allow\n// {IDEOGRAPH}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=_git_env())
    out = subprocess.run(
        [sys.executable, str(LINT), str(repo)],
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    assert out.returncode != 0, (
        "a marker with no reason was accepted. A bare opt-out is how this lint "
        f"stops meaning anything.\n{out.stdout}"
    )
    assert "without a reason" in out.stdout


def test_a_reasoned_allowance_passes_and_is_reported(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=_git_env())
    f = repo / "src" / "x.rs"
    f.write_text(
        f"// check-english-only: allow -- a model's token spelling\n"
        f'pub const T: &str = "{FULLWIDTH_BAR}";\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=_git_env())
    out = subprocess.run(
        [sys.executable, str(LINT), str(repo)],
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    assert out.returncode == 0, f"a reasoned allowance was refused:\n{out.stdout}"
    assert "allowed" in out.stdout and "token spelling" in out.stdout, (
        "the allowance was granted silently. It has to be printed, or it becomes "
        f"invisible the moment it is written.\n{out.stdout}"
    )


def test_this_repository_declares_exactly_one_allowance():
    # Named and bounded. One file legitimately spells a model's special tokens;
    # a second allowance appearing is a question, not a detail.
    out = subprocess.run(
        [sys.executable, str(LINT), str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    assert out.returncode == 0, f"the lint fails on this repository:\n{out.stdout}"
    granted = [line for line in out.stdout.splitlines() if line.startswith("allowed ")]
    assert len(granted) == 1, (
        "expected exactly one allowance (the tokenizer's special-token "
        f"spellings), found {len(granted)}:\n" + "\n".join(granted)
    )
    assert "tokenizer.rs" in granted[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
