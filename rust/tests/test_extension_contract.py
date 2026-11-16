# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One extension per representation, across every tool.

The convention is the thing a user actually relies on when wiring these tools
into a build: `.rs` is tile-rs source, `.tile` is the tile-dialect MLIR the
tools pass between stages, `.pto` is PTO assembly, `.cpp` is AscendC. A tool
that spelled the same representation differently would make the pipeline
unscriptable, and a tool that accepted a representation it cannot handle would
make a wrong answer look like a working one.

So the contract is asserted per tool rather than described once in the README.
Read from the help text and the sources, so it needs no toolchain and runs
anywhere the rest of the Python suite does.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRATES = ROOT / "rust" / "crates"

def _git_env() -> dict:
    """`os.environ` with every `GIT_*` variable removed.

    WHY, measured 2026-11-08. `git commit <path>` and `git commit -o <path>` build
    the commit from a TEMPORARY index and export `GIT_INDEX_FILE` (and `GIT_DIR`)
    to hooks. This repository's `pre-commit` hook runs `pytest rust/tests`, so any
    test here that shells out to `git` inherits those variables and answers a
    different question than the one it asks -- `git ls-files` reads the temporary
    index and lists nothing, and a `git init` + `git add -A` in a tmp directory
    writes into the OUTER repository's temporary index instead of the throwaway
    one. Both then fail through their own "this cannot happen" guards, which makes
    the cause read as a corpus problem rather than an environment one.

    The consequence was worse than a red gate: the hook refused a legitimate
    single-path commit, deterministically, three times in a row, and the obvious
    next move is to bypass the hook -- the one thing it exists to prevent.
    """
    env = dict(os.environ)
    for k in [k for k in env if k.startswith("GIT_")]:
        del env[k]
    return env


# tool -> (crate, what it reads, what it writes)
TOOLS = {
    "rspto": ("rspto", {".tile", ".mlir", ".rs"}, {".pto", ".tile"}),
    "ptoas-rs": ("ptoas_rs", {".pto"}, {".rs"}),
    "tile-search": ("tile_search", {".tile", ".mlir", ".rs"}, {".tile", ".rs"}),
    "pto-to-ascendc": ("pto_to_ascendc", {".tile", ".mlir", ".pto"}, {".cpp"}),
}


def help_text(crate: str, tool: str) -> str:
    """The tool's user-facing text, read from everywhere in its crate.

    Not from `src/bin/<tool>.rs` alone: two of these crates keep their HELP
    constant in `cli.rs` and leave the binary a thin main, which is the better
    layout and which the first version of this test did not survive.
    """
    root = CRATES / crate / "src"
    assert root.is_dir(), f"missing {root.relative_to(ROOT)}"
    bin_src = root / "bin" / f"{tool}.rs"
    assert bin_src.is_file(), f"missing {bin_src.relative_to(ROOT)}"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.rs")))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_every_tool_documents_the_extension_it_reads(tool):
    crate, reads, _ = TOOLS[tool]
    text = help_text(crate, tool)
    for ext in reads:
        assert ext in text, f"{tool} does not mention reading {ext}"


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_no_tool_invents_a_second_spelling_for_mlir(tool):
    """`.tile` is the project's name for the MLIR; `.mlir` is accepted because
    that is what the codegen backend writes. Any THIRD spelling would be a
    private convention."""
    crate, _, _ = TOOLS[tool]
    text = help_text(crate, tool)
    for banned in (".mlirtext", ".tiledialect", ".ir", ".tile-mlir"):
        assert banned not in text, f"{tool} invents the spelling {banned}"


def test_tile_rs_source_is_always_dot_rs():
    """Both directions: the lift WRITES .rs, and the tools that take tile-rs
    source READ .rs. A pipeline where one end says .rs and the other says
    something else cannot be scripted."""
    lift = help_text("ptoas_rs", "ptoas-rs")
    assert "output.rs" in lift or "out.rs" in lift, "the lift does not write .rs"
    for tool in ("rspto", "tile-search"):
        crate, _, _ = TOOLS[tool]
        assert ".rs" in help_text(crate, tool), f"{tool} does not read .rs"


def test_the_rust_input_mode_is_refused_uniformly_not_silently():
    """Every tool that documents `.rs` but cannot lower it must say so and use
    the same exit code, so a caller can branch on one number.

    This is the property that keeps the convention honest while half the
    pipeline is absent: a tool that accepted .rs and produced nothing, or that
    failed with a usage error, would send a user looking for a typo."""
    for tool in ("rspto", "tile-search"):
        crate, _, _ = TOOLS[tool]
        text = help_text(crate, tool)
        assert "NOT built into this release" in text or "not built into this release" in text, (
            f"{tool} documents .rs input without saying it is unavailable"
        )
        assert re.search(r"exit code 4|4 = |4 ok|0 ok", text), (
            f"{tool} does not document the exit code for an unavailable route"
        )


def test_all_three_tools_can_emit_the_tile_dialect():
    """The request is explicit that (1), (2) and (3) all produce the MLIR as
    `.tile` files, so this checks the clause rather than one tool.

    This test used to assert the OPPOSITE for `ptoas-rs` -- that `--emit tile`
    was "accepted and refused", because getting MLIR was thought to require
    lifting to Rust and compiling it with the codegen backend. It is not: the
    `.tile` form is `llvm.func` + `hacc.entry` + `llvm.call @__tile_*`, which is
    the intrinsic-level information the lift already computes, and
    `emitter_mlir` writes it straight from the AST.

    Worth recording how the stale version survived: it asserted
    `"4 = " in text`, which the exit-code TABLE satisfies on its own. The claim
    it was really making -- that the flag is refused -- was never checked, so the
    test kept passing after the refusal was removed."""
    for crate, tool in (("rspto", "rspto"), ("ptoas_rs", "ptoas-rs"), ("tile_search", "tile-search")):
        text = help_text(crate, tool)
        assert "--emit" in text, f"{tool} has no --emit flag"
        assert "tile" in text, f"{tool} does not mention the tile form"


def test_ptoas_rs_does_not_let_verify_and_emit_tile_be_combined():
    """`--verify` compiles the lifted RUST. There is nothing for rustc or Miri
    to say about a file of MLIR, and verifying the Rust while writing the MLIR
    would report a proof about a file the user never received."""
    text = help_text("ptoas_rs", "ptoas-rs")
    assert "--verify" in text
    low = text.lower()
    assert "not combinable with" in low or "cannot be combined" in low, (
        "the help does not say the two cannot be combined; a user who tries it "
        "gets an exit code and no explanation"
    )


def test_the_readme_states_the_same_convention():
    """One table, and it has to agree with the tools."""
    readme = (ROOT / "rust" / "README.md").read_text(encoding="utf-8")
    for ext in (".rs", ".tile", ".pto", ".cpp"):
        assert f"`{ext}`" in readme, f"the README does not name {ext}"


def test_the_bench_uses_exactly_these_extensions():
    """The corpus is the convention's largest consumer, so it is also the place
    a drifted spelling would show up first.

    Scoped to GIT-TRACKED files, like every lint in `tests/lint/`. It used to
    walk the filesystem, and then any local build artefact failed it: importing
    the kernel estimator from `rust/bench/shapes/` leaves a `__pycache__/*.pyc`
    there, which is gitignored and was never part of the corpus. A convention
    test that fails on a file nobody committed is reporting on the developer's
    working directory rather than on the convention.
    """
    allowed = {".tile", ".pto", ".cpp", ".rs", ".json", ".py"}
    tracked = subprocess.run(
        ["git", "ls-files", "rust/bench"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    ).stdout.split()
    assert tracked, "git lists no files under rust/bench/"
    seen = {Path(f).suffix for f in tracked if Path(f).suffix}
    unexpected = seen - allowed
    assert not unexpected, f"the bench carries unexpected extensions: {unexpected}"
    # And the representations the request named are all present somewhere.
    for ext in (".tile", ".pto", ".cpp", ".rs"):
        assert any(f.endswith(ext) for f in tracked), f"no {ext} anywhere in the bench"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
