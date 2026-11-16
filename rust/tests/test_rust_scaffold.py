# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pin the facts about the Rust tree that are stated in more than one place.

Each test here exists because the same value is written down twice, and the two
copies can drift silently: a toolchain channel named in a manifest and in a
workflow, a licence named in a manifest and in the repository root. A reader
cannot tell which copy is stale, so these assert they agree.

They deliberately do not invoke cargo. That belongs in the `rust` CI job, which
has a toolchain; this file has to pass on a machine with no Rust installed at all.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "rust" / "Cargo.toml"
TOOLCHAIN = ROOT / "rust" / "rust-toolchain.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
README = ROOT / "rust" / "README.md"
INSTALLER = ROOT / "rust" / "install.sh"

LICENSE_ID = "LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0"


def test_workspace_manifest_exists():
    assert WORKSPACE.is_file(), f"missing {WORKSPACE.relative_to(ROOT)}"


def test_workspace_declares_the_repository_licence():
    # The root licence inherits to the subprojects, so the Rust workspace must
    # not declare a different one.
    #
    # Read the `license =` field, do not substring-search the file: the CANN
    # header itself contains the words "BUT NOT LIMITED TO", and `"MIT" in text`
    # matches inside LIMITED. That false positive failed this test on its first
    # run, which is the whole argument for parsing the field.
    text = WORKSPACE.read_text(encoding="utf-8")
    declared = re.findall(r'^\s*license\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert declared, "the Rust workspace declares no licence"
    for value in declared:
        assert value == LICENSE_ID, f"unexpected licence {value!r}"


def _pinned_channel() -> str:
    channel = re.search(r'channel\s*=\s*"([^"]+)"', TOOLCHAIN.read_text(encoding="utf-8"))
    assert channel, "rust-toolchain.toml declares no channel"
    return channel.group(1)


def test_the_workspace_is_pinned_to_stable():
    # Measured on rustc 1.98.1: the emitter, the lift and both CLIs build and
    # pass on stable, and the stable-built emitter produces byte-identical PTO.
    # A nightly pin here would impose nightly on contributors who never touch
    # the two components that need it, so the default must stay stable.
    pinned = _pinned_channel()
    assert not pinned.startswith("nightly"), (
        f"the workspace is pinned to {pinned}; stable is sufficient for every "
        "member and nightly belongs only to tile_std and Miri"
    )
    assert re.fullmatch(r"(stable|\d+\.\d+(\.\d+)?)", pinned), pinned


def test_the_manifest_rust_version_matches_the_pin():
    pinned = _pinned_channel()
    if pinned == "stable":
        pytest.skip("a floating stable pin has no version to match")
    declared = re.search(
        r'^\s*rust-version\s*=\s*"([^"]+)"',
        WORKSPACE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert declared, "the workspace declares no rust-version"
    assert pinned.startswith(declared.group(1)), (
        f"rust-toolchain.toml pins {pinned} but rust-version is {declared.group(1)}"
    )


def test_the_ci_job_does_not_install_a_nightly_toolchain():
    # The job must not quietly reintroduce what the pin removed.
    job = WORKFLOW.read_text(encoding="utf-8")
    job = job[job.index("  rust:\n") : job.index("  clang-tidy:\n")]
    assert "nightly" in job, (
        "the rust job should still EXPLAIN why it is not nightly, so the next reader does not add it back"
    )
    assert "rustup toolchain install nightly" not in job
    assert "cargo +nightly" not in job


def test_tile_std_is_excluded_from_the_workspace():
    # One nightly member would impose nightly on the whole workspace build,
    # because cargo applies a single toolchain per build.
    text = WORKSPACE.read_text(encoding="utf-8")
    excluded = re.search(r"^\s*exclude\s*=\s*\[([^\]]*)\]", text, re.MULTILINE)
    assert excluded, "the workspace declares no exclude list"
    assert "tile_std" in excluded.group(1)


def test_the_readme_documents_both_tools_and_the_soc_flag():
    text = README.read_text(encoding="utf-8")
    for tool in ("rspto", "pto-to-ascendc"):
        assert tool in text, f"the README does not name {tool}"
    # The one flag whose absence silently breaks every kernel.
    assert "--soc" in text


def test_the_readme_says_stable_is_enough():
    # The claim a reader most needs up front, and the one this project measured
    # rather than assumed.
    text = README.read_text(encoding="utf-8")
    assert "Stable Rust is enough" in text
    assert "No nightly" in text or "no nightly" in text


# One README per utility, plus a main one that indexes them all. Nothing else:
# any other markdown here is a second place for the same facts to go stale in.
EXPECTED_MARKDOWN = sorted(
    [
        "rust/README.md",
        "rust/crates/rspto/README.md",
        "rust/crates/ptoas_rs/README.md",
        "rust/crates/tile_search/README.md",
        "rust/crates/pto_to_ascendc/README.md",
        "rust/ds4-rs-ascend/README.md",
    ]
)


def test_only_the_per_utility_readmes_and_the_main_one_exist():
    found = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "rust").rglob("*.md"))
    assert found == EXPECTED_MARKDOWN, f"unexpected markdown: {set(found) ^ set(EXPECTED_MARKDOWN)}"
    assert not list((ROOT / "rust" / "bench").rglob("*.md")), "bench/ must carry no markdown"
    assert not list((ROOT / "rust" / "demo").rglob("*.md")), "rust/demo/ must carry no markdown"


def test_every_utility_page_embeds_its_own_demo():
    """A page whose recording is missing, or is another tool's, is worse than
    none: a reader takes the picture for the tool."""
    pages = {
        "rust/crates/rspto/README.md": "rspto",
        "rust/crates/ptoas_rs/README.md": "ptoas-rs",
        "rust/crates/tile_search/README.md": "tile-search",
        "rust/crates/pto_to_ascendc/README.md": "pto-to-ascendc",
        "rust/ds4-rs-ascend/README.md": "ds4-rs-ascend",
    }
    for page, scene in pages.items():
        text = (ROOT / page).read_text(encoding="utf-8")
        assert f"demo/media/{scene}.gif" in text, f"{page} does not embed {scene}.gif"
        assert f"demo/media/{scene}.mp4" in text, f"{page} does not link {scene}.mp4"
        for other in set(pages.values()) - {scene}:
            assert f"media/{other}.gif" not in text, f"{page} embeds {other}'s recording"


def test_the_main_readme_carries_every_utility():
    text = (ROOT / "rust" / "README.md").read_text(encoding="utf-8")
    for scene in ("rspto", "ptoas-rs", "tile-search", "pto-to-ascendc", "ds4-rs-ascend"):
        assert f"demo/media/{scene}.gif" in text, f"the main README omits {scene}'s gif"
        assert f"demo/media/{scene}.mp4" in text, f"the main README omits {scene}'s mp4"
    for page in EXPECTED_MARKDOWN[1:]:
        link = page.removeprefix("rust/")
        assert link in text, f"the main README does not link {page}"
    assert "ds4-rs-ascend/README.md" in text


def test_every_referenced_recording_exists():
    """Relative links that 404 are worse than no demo at all."""

    missing = []
    for page in EXPECTED_MARKDOWN:
        src = ROOT / page
        for rel in re.findall(
            r"\]\(([^)]*demo/(?:media|narration)/[^)]+)\)", src.read_text(encoding="utf-8")
        ):
            if not (src.parent / rel).resolve().is_file():
                missing.append(f"{page} -> {rel}")
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", src.read_text(encoding="utf-8")):
            if not (src.parent / rel).resolve().is_file():
                missing.append(f"{page} -> {rel}")
    assert not missing, "broken media links:\n" + "\n".join(missing)


def test_the_readme_covers_the_four_sections_it_is_for():
    text = README.read_text(encoding="utf-8")
    for heading in ("## Installation", "## Usage", "## Features", "## Benchmarks"):
        assert heading in text, f"the README has no {heading!r} section"


def test_the_readme_install_line_points_at_the_installer_that_exists():
    # A one-line installer in a README is the first thing anyone runs, so the
    # path it names has to be a file in this repository.
    text = README.read_text(encoding="utf-8")
    assert "rust/install.sh" in text, "the README does not name the installer"
    assert INSTALLER.is_file(), "the installer the README names does not exist"
    assert INSTALLER.stat().st_mode & 0o111, "the installer is not executable"


def test_the_readme_names_the_extensions_and_the_exit_codes():
    text = README.read_text(encoding="utf-8")
    for ext in (".rs", ".tile", ".pto", ".cpp"):
        assert f"`{ext}`" in text, f"the README does not name {ext}"
    # Exit code 4 is the one a caller has to handle and cannot guess.
    assert "`4`" in text, "the README does not document the toolchain-unavailable code"


DEFERRED_TESTS = {
    # cargo test name -> file it lives in, and the thing that restores it
    #
    # This registry held NINETEEN entries, every one of them blocked on the
    # same thing: the kernel-side DSL was not in this tree, so a test that
    # read or compiled against it could not run. The DSL is here now, and
    # eighteen of the nineteen run.
    #
    # Two of them had to be more than un-ignored, which is why the count is
    # worth recording rather than just deleting:
    #
    #   * Thirteen only ever read the DSL as TEXT, and would have run as soon
    #     as it arrived -- except that three test files resolved a
    #     `crates/<other>/` path into their own fixture directory, so the
    #     reason "needs tile_std" was hiding a second, independent bug.
    #   * The two compile gates shelled out to the AMBIENT cargo, which is
    #     this workspace's stable toolchain. The DSL is `#![no_core]`, so
    #     stable rejects it before reading the kernel; they now invoke the
    #     pinned nightly. The first run of `every_artifact_the_lift_accepts
    #     _compiles` then failed on 9 of 32 artifacts with E0308 -- a real
    #     emitter defect that nothing had caught, because nothing had ever
    #     compiled the lift's output.
    #
    # One remains, and NOT for the reason the blanket label gave:
    "the_mirror_still_makes_the_claim_being_checked": (
        "rust/crates/ptoas_rs/tests/typestate_mirror_matches_tile_std.rs",
        "the Coq development's mirror under proofs/, which is not part of this tree",
    ),
    # Reached only once the serving utility moved under rust/: the scanner walks
    # rust/ alone, so a deferral outside it was invisible rather than approved.
    "dump_laguna_config": (
        "rust/ds4-rs-ascend/crates/engine_core/src/gguf.rs",
        "Prints the parsed Laguna config for an arbitrary GGUF",
    ),
    # A THIRD, and the only one deferred because the thing it checked STOPPED
    # EXISTING rather than not yet existing. It asserted that the AscendC oracle
    # refuses an extent the PTO oracle accepts -- the reason there are two oracles --
    # and carried that existential claim on a named fixture that had to move three
    # times as the emitter improved. The third move ended it: every remaining AscendC
    # refusal in the model corpora is a missing arm or a vocabulary gather, which no
    # choice of extent reaches, so the search has nothing left to disagree about.
    # Measured over all four candidates the manifests record a `searched` entry for.
    #
    # Registered rather than deleted because the second oracle's REASON survives its
    # last instance, and the test becomes a test again the day an arm lands here
    # whose PTO twin is missing -- which bench/tile_rs_acceptance.json says twelve
    # kernels are waiting for.
    "the_two_oracles_do_not_agree": (
        "rust/crates/tile_search/tests/cli_contract.rs",
        "a kernel whose captured extent the AscendC emitter refuses and the PTO one accepts",
    ),
}


def test_every_deferred_test_is_still_present_and_still_explains_itself():
    """A deferred test must not quietly become a deleted test.

    The one registered here guards a DISAGREEMENT between tile_std's declared
    intrinsic signatures and the emitter's dispatch -- a defect invisible from
    either side alone, which once stayed green while the emitter read a float
    scalar as a row count. It is ignored only because tile_std is not in this
    tree yet. If someone removes it instead of restoring it, this fails.
    """
    for name, (path, needs) in DEFERRED_TESTS.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        assert f"fn {name}(" in text, f"{name} was removed from {path}"
        i = text.index(f"fn {name}(")
        preamble = text[max(0, i - 2000) : i]
        assert "#[ignore" in preamble, f"{name} is no longer marked ignored"
        assert needs in preamble, f"{name}'s ignore reason no longer says it needs {needs}"


def test_no_test_is_ignored_without_being_registered():
    """A new `#[ignore]` must be a decision, not a drive-by.

    Counting the attribute across the Rust sources and matching it against the
    register is crude, but it is the property that matters: an ignored test
    reports success, so one added quietly is indistinguishable from one that
    passes. Doc-tests are excluded -- `cargo test` marks illustrative examples
    ignored by design and they are not assertions about behaviour.
    """
    found = []
    for path in sorted((ROOT / "rust").rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines()):
            if line.strip().startswith("#[ignore"):
                # The name follows within a few lines, and the attribute may sit
                # either side of `#[test]` -- both orders occur in this tree.
                window = "\n".join(text.splitlines()[i : i + 5])
                found.append((path.relative_to(ROOT).as_posix(), window))

    assert len(found) == len(DEFERRED_TESTS), (
        f"{len(found)} ignored test(s) but {len(DEFERRED_TESTS)} registered. "
        "Register the deferral with its reason, or do not defer it.\n" + "\n".join(f"  {p}" for p, _ in found)
    )
    for name in DEFERRED_TESTS:
        assert any(name in window for _, window in found), f"{name} is not ignored"


def test_foreign_target_lint_is_registered_with_pre_commit():
    # The lint is only a gate if pre-commit runs it; CI invokes pre-commit with
    # --all-files, so registration is what puts it in the pipeline.
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check_no_foreign_targets.py" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
