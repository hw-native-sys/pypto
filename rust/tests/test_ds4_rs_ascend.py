# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The serving utility carries no entitlement machinery, and nothing restores it.

Removing a licence gate is easy to do incompletely: a verb comes out of the
parser while the module stays, or the module goes while a build flag that
suppressed its output remains. These tests assert the absence of each piece by
name, so a partial restoration fails rather than passing quietly.

Needs no Rust toolchain -- it reads sources -- so it runs anywhere the rest of
the Python suite does.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTIL = ROOT / "rust" / "ds4-rs-ascend"


def sources():
    return sorted(p for p in UTIL.rglob("*.rs") if "/target/" not in p.as_posix())


def manifests():
    return sorted(p for p in UTIL.rglob("Cargo.toml") if "/target/" not in p.as_posix())


def test_the_utility_is_present():
    assert UTIL.is_dir(), "rust/ds4-rs-ascend/ is missing"
    assert sources(), "no Rust sources under rust/ds4-rs-ascend/"


@pytest.mark.parametrize(
    "token,why",
    [
        ("ds4_license", "the entitlement crate"),
        ("require_pro", "the Pro gate"),
        ("is_dev_build", "the build-edition probe"),
        ("pubkey_hex()[", "the licence key banner"),
        ("trusted_client", "the partner build-flag exemption"),
        ("DS4_TRUSTED_CLIENT", "the exemption's build flag"),
        ("edition_name", "the edition label"),
        ("LicenseCmd", "the licence subcommand"),
    ],
)
def test_no_entitlement_machinery_remains(token, why):
    hits = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in sources()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        # A comment recording that something was REMOVED is the point, not a
        # violation; only live code or an identifier counts.
        if token in line and not line.strip().startswith(("//", "///", "//!"))
    ]
    assert not hits, f"{why} is still referenced in code: {hits}"


def test_no_tier_vocabulary_survives_in_user_facing_text():
    """The licensing MECHANISM was removed and its VOCABULARY was not.

    `require_pro` -- the gate -- is asserted absent above, and was. But four
    `(Pro)` markers survived in `cli.rs`: two doc comments and two lines of the
    `--advanced` help, labelling `subscribe` and `share` as paid-tier commands.
    Measured by running them: both are ungated, and refuse only because no
    publisher endpoint is configured. So the label was not merely stale -- it
    told a user they needed a tier this build has no concept of, to run a
    command that would otherwise have worked.

    Same shape as any gate that checks a mechanism and misses the residue:
    removing the enforcement is the easy half, and the half a test naturally
    reaches for."""
    banned = ("(Pro)", "(pro)", "Pro only", "Pro tier", "pro-only", "requires Pro")
    hits = []
    for p in sources():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            # A comment RECORDING a removal is the point, not a violation.
            if "removed" in low or "no longer" in low or "survived" in low:
                continue
            for b in banned:
                if b in line:
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not hits, (
        "tier vocabulary survives in user-facing text, which tells a user a command "
        "needs a licence tier this build has no concept of:\n" + "\n".join(hits)
    )


def test_the_publisher_rename_left_no_mangled_prose():
    """A blanket string substitution renamed an internal service and broke the
    English around it.

    Replacing the old name with "the publisher" turned "the <name>" into "the
    the publisher", "a <name> set" into "a the publisher set", and -- worst --
    the HOSTNAME "<name>.ai" into "the publisher.ai". Eight occurrences across
    two files, every one in text a user or a reviewer reads.

    The identifier damage from the same substitution (`<name>_pubkey_hex()`
    becoming `the publisher_pubkey_hex()`) broke the build and was fixed at the
    time, which is precisely why the prose damage was not: a compiler error gets
    attention and a grammar error does not."""
    bad = (
        "the the ",
        "a the publisher",
        "an the publisher",
        "The the ",
        "the publisher.ai",
        "the publisher.com",
    )
    hits = []
    for p in sources():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            if "mangled" in low or "substitution" in low:
                continue
            for b in bad:
                if b in line:
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not hits, "the publisher rename left mangled text behind:\n" + "\n".join(hits)


def test_the_license_verb_is_not_in_the_command_surface():
    cli = (UTIL / "crates/controller/src/cli.rs").read_text(encoding="utf-8")
    # The parser must not have an arm for it. Checked on the match arms rather
    # than the whole file, so the test that asserts it now ERRORS is allowed.
    assert '"license" | "licence" =>' not in cli
    assert "Command::License" not in cli


def test_no_manifest_depends_on_an_entitlement_or_apple_crate():
    for p in manifests():
        text = p.read_text(encoding="utf-8")
        for dep in ("ds4_license", "metal", "objc", "cocoa", "core-foundation", "security-framework"):
            # Match a dependency line, not prose in a comment.
            bad = [line for line in text.splitlines() if re.match(rf"^\s*{re.escape(dep)}\s*=", line)]
            assert not bad, f"{p.relative_to(ROOT)} depends on {dep}: {bad}"


def test_there_is_no_build_script_linking_apple_frameworks():
    # The question is APPLE FRAMEWORKS, not the existence of a build script. `engine_ascend` now
    # carries one that points the linker at CANN for its `ascend` feature -- a Linux-only concern
    # that has nothing to do with the entitlement/framework coupling this test exists to prevent.
    for b in UTIL.rglob("build.rs"):
        text = b.read_text(encoding="utf-8")
        assert "Security" not in text and "Metal" not in text, (
            f"{b} links an Apple framework; that is what this test forbids"
        )
    for p in sources():
        text = p.read_text(encoding="utf-8")
        assert 'link(name = "Security"' not in text
        assert 'link(name = "Metal"' not in text


def test_the_integrity_verifier_is_still_there():
    """The counterpart. Removing entitlement must not remove integrity.

    These answer different questions -- "may you run this" versus "is this what
    was published" -- and only the first one was a gate. A change that deleted
    both would pass every test above.
    """
    netchan = (UTIL / "crates/controller/src/netchan.rs").read_text(encoding="utf-8")
    assert "verify_strict" in netchan, "the signed-index check is gone"
    assert "VerifyingKey" in netchan
    ctrl = (UTIL / "crates/controller/Cargo.toml").read_text(encoding="utf-8")
    assert "ed25519-dalek" in ctrl


def test_the_separate_workspace_declares_the_repository_licence():
    ws = (UTIL / "Cargo.toml").read_text(encoding="utf-8")
    declared = re.findall(r'^\s*license\s*=\s*"([^"]+)"', ws, re.MULTILINE)
    assert declared, "the workspace declares no licence"
    for v in declared:
        assert v == "LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0", v


def test_the_toolchain_is_stable():
    pin = (UTIL / "rust-toolchain.toml").read_text(encoding="utf-8")
    channel = re.search(r'channel\s*=\s*"([^"]+)"', pin)
    assert channel, "no channel pinned"
    assert not channel.group(1).startswith("nightly"), channel.group(1)


def test_the_readme_records_what_the_removal_cost():
    """The correctness oracle that went with the Apple bridge is owed back.

    Recording a debt is the only thing that keeps it from being forgotten, so
    the README has to carry it.
    """
    readme = (ROOT / "rust" / "README.md").read_text(encoding="utf-8")
    assert "ds4-rs-ascend" in readme
    assert "no equivalent yet" in readme or "is owed" in readme


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
