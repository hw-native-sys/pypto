# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the temporary A2/A3 TREM PTO-ISA fallback."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from pypto.runtime import pto_isa


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in pto_isa._GIT_LOCATION_ENV}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=_git_env()
    ).stdout.strip()


def _init_checkout(root: Path, content: str) -> None:
    header = root / pto_isa._TREM_PINNED_HEADER
    header.parent.mkdir(parents=True)
    header.write_text(content, encoding="utf-8")
    subprocess.run(
        ["git", "init", "-q", str(root)], check=True, capture_output=True, text=True, env=_git_env()
    )
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--no-gpg-sign",
        "--no-verify",
        "-q",
        "-m",
        "pin",
    )


def test_patch_targets_the_remainder_headers_and_marks_the_fix():
    text = pto_isa._TREM_PATCH_PATH.read_text(encoding="utf-8")
    assert "include/pto/npu/a2a3/TRem.hpp" in text
    assert "include/pto/npu/a2a3/TRemS.hpp" in text
    assert pto_isa._TREM_FIX_MARKER in text
    # Local result-finiteness guard for the upstream HpFusedSub overflow (#435).
    assert pto_isa._TREM_GUARD_MARKER in text
    assert "vabs(c1, c0" in text


def test_fix_already_present_leaves_the_pin_untouched(tmp_path):
    pin = tmp_path / "pin"
    _init_checkout(pin, f"// {pto_isa._TREM_FIX_MARKER}\n// {pto_isa._TREM_GUARD_MARKER}\n")

    assert pto_isa._ensure_trem_patch(str(pin)) == str(pin)


def test_partial_fix_is_not_treated_as_fixed(tmp_path):
    pin = tmp_path / "pin"
    _init_checkout(pin, f"// {pto_isa._TREM_FIX_MARKER}\n")

    assert not pto_isa._trem_fix_present(pin)


def test_patch_creates_a_clean_reusable_worktree(tmp_path, monkeypatch):
    pin = tmp_path / "pin"
    _init_checkout(pin, "original\n")
    pin_revision = _git(pin, "rev-parse", "HEAD")

    patch = tmp_path / "fix.patch"
    patch.write_text(
        "diff --git a/include/pto/npu/a2a3/TRem.hpp b/include/pto/npu/a2a3/TRem.hpp\n"
        "--- a/include/pto/npu/a2a3/TRem.hpp\n"
        "+++ b/include/pto/npu/a2a3/TRem.hpp\n"
        "@@ -1 +1,3 @@\n"
        " original\n"
        f"+// {pto_isa._TREM_FIX_MARKER}\n"
        f"+// {pto_isa._TREM_GUARD_MARKER}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pto_isa, "_TREM_PATCH_PATH", patch)

    patched = Path(pto_isa._ensure_trem_patch(str(pin)))

    assert patched != pin
    patched_text = (patched / pto_isa._TREM_PINNED_HEADER).read_text(encoding="utf-8")
    assert pto_isa._TREM_FIX_MARKER in patched_text
    assert pto_isa._TREM_GUARD_MARKER in patched_text
    assert _git(patched, "status", "--porcelain") == ""
    assert _git(patched, "rev-parse", "HEAD") != pin_revision

    # The patched tree is reused rather than rebuilt.
    assert pto_isa._ensure_trem_patch(str(pin)) == str(patched)

    # A committed extra change invalidates the cached worktree.
    (patched / pto_isa._TREM_PINNED_HEADER).write_text(patched_text + "// tamper\n", encoding="utf-8")
    _git(patched, "add", ".")
    _git(
        patched,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--no-gpg-sign",
        "--no-verify",
        "-q",
        "-m",
        "tamper",
    )
    patch_id = hashlib.sha256(patch.read_bytes()).hexdigest()[:12]
    assert not pto_isa._is_patched_worktree(patched, pin_revision, patch_id)

    restored = Path(pto_isa._ensure_trem_patch(str(pin)))
    restored_text = (restored / pto_isa._TREM_PINNED_HEADER).read_text(encoding="utf-8")
    assert pto_isa._TREM_GUARD_MARKER in restored_text
    assert "// tamper" not in restored_text


def test_disabled_fallback_returns_the_pin(tmp_path, monkeypatch):
    pin = tmp_path / "pin"
    _init_checkout(pin, "original\n")
    monkeypatch.setenv(pto_isa._TREM_PATCH_DISABLE_ENV, "1")

    assert pto_isa._resolve_effective_pto_isa_root(str(pin)) == str(pin)


@pytest.mark.parametrize("missing", ["not-a-checkout"])
def test_non_checkout_is_left_untouched(tmp_path, missing):
    root = tmp_path / missing
    root.mkdir()

    assert pto_isa._ensure_trem_patch(str(root)) == str(root)


def test_effective_revision_appends_the_patch_id(monkeypatch):
    monkeypatch.delenv(pto_isa._TREM_PATCH_DISABLE_ENV, raising=False)

    revision = pto_isa.effective_pto_isa_revision("90a37965")

    assert revision.startswith("90a37965+trem-patch:")
    assert len(revision.rsplit(":", 1)[1]) == 12


def test_effective_revision_honors_the_disable_knob(monkeypatch):
    monkeypatch.setenv(pto_isa._TREM_PATCH_DISABLE_ENV, "1")

    assert pto_isa.effective_pto_isa_revision("90a37965") == "90a37965"


def test_effective_revision_tracks_the_patch_contents(tmp_path, monkeypatch):
    monkeypatch.delenv(pto_isa._TREM_PATCH_DISABLE_ENV, raising=False)
    first = tmp_path / "first.patch"
    second = tmp_path / "second.patch"
    first.write_text("patch one\n", encoding="utf-8")
    second.write_text("patch two\n", encoding="utf-8")

    monkeypatch.setattr(pto_isa, "_TREM_PATCH_PATH", first)
    first_revision = pto_isa.effective_pto_isa_revision("pin")
    monkeypatch.setattr(pto_isa, "_TREM_PATCH_PATH", second)
    second_revision = pto_isa.effective_pto_isa_revision("pin")

    assert first_revision != second_revision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
