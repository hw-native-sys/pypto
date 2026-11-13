# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the temporary A2/A3 TREM PTO-ISA fallback."""

import subprocess
from pathlib import Path

import pytest
from pypto.runtime import pto_isa


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_checkout(root: Path, content: str) -> None:
    header = root / pto_isa._TREM_PINNED_HEADER
    header.parent.mkdir(parents=True)
    header.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True, text=True)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "pin",
    )


def test_patch_targets_the_remainder_headers_and_marks_the_fix():
    text = pto_isa._TREM_PATCH_PATH.read_text(encoding="utf-8")
    assert "include/pto/npu/a2a3/TRem.hpp" in text
    assert "include/pto/npu/a2a3/TRemS.hpp" in text
    assert pto_isa._TREM_FIX_MARKER in text


def test_fix_already_present_leaves_the_pin_untouched(tmp_path):
    pin = tmp_path / "pin"
    _init_checkout(pin, f"// {pto_isa._TREM_FIX_MARKER}\n")

    assert pto_isa._ensure_trem_patch(str(pin)) == str(pin)


def test_patch_creates_a_clean_reusable_worktree(tmp_path, monkeypatch):
    pin = tmp_path / "pin"
    _init_checkout(pin, "original\n")
    pin_revision = _git(pin, "rev-parse", "HEAD")

    patch = tmp_path / "fix.patch"
    patch.write_text(
        "diff --git a/include/pto/npu/a2a3/TRem.hpp b/include/pto/npu/a2a3/TRem.hpp\n"
        "--- a/include/pto/npu/a2a3/TRem.hpp\n"
        "+++ b/include/pto/npu/a2a3/TRem.hpp\n"
        "@@ -1 +1,2 @@\n"
        " original\n"
        f"+// {pto_isa._TREM_FIX_MARKER}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pto_isa, "_TREM_PATCH_PATH", patch)

    patched = Path(pto_isa._ensure_trem_patch(str(pin)))

    assert patched != pin
    assert pto_isa._TREM_FIX_MARKER in (patched / pto_isa._TREM_PINNED_HEADER).read_text(encoding="utf-8")
    assert _git(patched, "status", "--porcelain") == ""
    assert _git(patched, "rev-parse", "HEAD") != pin_revision

    # The patched tree is reused rather than rebuilt.
    assert pto_isa._ensure_trem_patch(str(pin)) == str(patched)


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
