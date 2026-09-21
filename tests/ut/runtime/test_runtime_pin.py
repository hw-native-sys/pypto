# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the runtime pin check: installed ``simpler`` versus ``runtime/``.

Both sides are faked. The check exists precisely for environments PyPTO's own test
run does not reproduce -- a wheel-installed simpler in site-packages -- so driving it
through a real checkout would test one arrangement and leave the interesting ones
unexercised.
"""

import shlex
import shutil
import subprocess
import sys
import warnings
from types import ModuleType

import pytest
from pypto.runtime import runtime_pin

PINNED = "a" * 40
STALE = "b" * 40
OFF_PIN = "c" * 40
EXTENSION = "/fake/site-packages/_task_interface.cpython-310-aarch64-linux-gnu.so"


class FakePin:
    """Drives both sides of the pin for one test."""

    def __init__(self, monkeypatch, root):
        self._monkeypatch = monkeypatch
        self.root = root
        self.expected: str | None = PINNED
        self.pinned: str | None = PINNED
        self.porcelain: str | None = None

    def git(self, args: list[str], cwd) -> str | None:
        """Stand in for :func:`runtime_pin._git`."""
        assert cwd in (self.root, self.root / "runtime")
        if args == ["rev-parse", "HEAD"]:
            return self.expected
        if args == ["rev-parse", "HEAD:runtime"]:
            return self.pinned
        if args == ["status", "--porcelain"]:
            return self.porcelain
        raise AssertionError(f"unexpected git command: {args}")

    def install(self, revision: str | None, *, stamped: bool = True) -> None:
        """Put a fake ``_task_interface`` in ``sys.modules``, or none at all.

        ``revision=None`` with ``stamped=True`` means the extension is absent;
        ``stamped=False`` means it is present but predates ``__build_commit__``.
        """
        runtime_pin.runtime_pin_status.cache_clear()
        if revision is None and stamped:
            # importlib raises ImportError for a None entry -- simpler is not installed.
            self._monkeypatch.setitem(sys.modules, "_task_interface", None)
            return
        module = ModuleType("_task_interface")
        module.__file__ = EXTENSION
        if stamped:
            setattr(module, "__build_commit__", revision)  # noqa: B010 -- not a ModuleType attribute
        self._monkeypatch.setitem(sys.modules, "_task_interface", module)


@pytest.fixture
def pin(monkeypatch, tmp_path):
    """A PyPTO checkout and an installed simpler, both fully under test control."""
    runtime_pin.runtime_pin_status.cache_clear()
    runtime_pin._state["warned"] = False
    monkeypatch.delenv(runtime_pin._ENV_SKIP, raising=False)

    root = tmp_path / "pypto"
    (root / "runtime").mkdir(parents=True)
    # An initialized submodule's `.git` is a gitdir file, not a directory.
    (root / "runtime" / ".git").write_text("gitdir: ../.git/modules/runtime\n")
    fake = FakePin(monkeypatch, root)
    monkeypatch.setattr(runtime_pin, "_pypto_root", lambda: root)
    monkeypatch.setattr(runtime_pin, "_git", fake.git)
    fake.install(PINNED)

    yield fake

    runtime_pin.runtime_pin_status.cache_clear()
    runtime_pin._state["warned"] = False


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_matching_revision_passes(pin, recwarn):
    """An install built from runtime/ HEAD is accepted silently."""
    status = runtime_pin.check_runtime_pin()

    assert status.matches
    assert not status.off_pin
    assert not recwarn.list


def test_mismatch_raises(pin):
    """A different revision is refused, with both sides and the fix in the message."""
    pin.install(STALE)

    with pytest.raises(runtime_pin.RuntimePinMismatch) as error:
        runtime_pin.check_runtime_pin()

    message = str(error.value)
    assert PINNED[:12] in message
    assert STALE[:12] in message
    assert f"pip install --no-build-isolation -e {pin.root / 'runtime'}" in message
    assert EXTENSION in message


def test_mismatch_is_an_import_error(pin):
    """Callers that catch ImportError around the simpler imports still see it."""
    pin.install(STALE)

    with pytest.raises(ImportError):
        runtime_pin.check_runtime_pin()


def test_reinstall_command_quotes_a_path_with_spaces(tmp_path):
    """The printed repair must still be one argument when copied into a shell."""
    runtime_dir = tmp_path / "my project" / "pypto" / "runtime"
    status = runtime_pin.RuntimePinStatus(runtime_dir, PINNED, PINNED, STALE, False, None)

    words = shlex.split(status.reinstall_command())

    assert words == ["pip", "install", "--no-build-isolation", "-e", str(runtime_dir)]


def test_missing_build_stamp_raises(pin):
    """An extension predating the stamp is by definition a different revision."""
    pin.install(None, stamped=False)

    with pytest.raises(runtime_pin.RuntimePinMismatch, match="predates the build stamp"):
        runtime_pin.check_runtime_pin()


# ---------------------------------------------------------------------------
# The skips -- each is "cannot tell", never a failure
# ---------------------------------------------------------------------------


def test_empty_build_stamp_skips(pin):
    """A simpler built without git carries an empty stamp and cannot be compared."""
    pin.install("")

    status = runtime_pin.check_runtime_pin()

    assert status.installed is None
    assert not status.comparable


def test_simpler_not_installed_skips(pin):
    """A codegen-only environment has nothing to compare against."""
    pin.install(None)

    status = runtime_pin.check_runtime_pin()

    assert status.installed is None
    assert status.extension_path is None


def test_not_a_checkout_skips(pin, monkeypatch):
    """A wheel-installed PyPTO has no runtime/ and therefore no pin."""
    monkeypatch.setattr(runtime_pin, "_pypto_root", lambda: None)
    pin.install(STALE)

    status = runtime_pin.check_runtime_pin()

    assert status.runtime_dir is None
    assert status.expected is None
    assert not status.comparable


def test_git_unavailable_skips(pin, monkeypatch):
    """No git, no verdict -- a release tarball must still import."""
    monkeypatch.setattr(runtime_pin, "_git", lambda args, cwd: None)
    pin.install(STALE)

    status = runtime_pin.check_runtime_pin()

    assert status.expected is None
    assert not status.comparable


def test_uninitialized_runtime_submodule_skips(pin):
    """An empty runtime/ has no revision of its own -- skip, never mismatch or warn."""
    (pin.root / "runtime" / ".git").unlink()
    pin.install(STALE)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        status = runtime_pin.check_runtime_pin()

    assert status.expected is None
    assert status.pinned == PINNED
    assert not status.off_pin
    assert not runtime_pin.runtime_tree_is_dirty(status)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
def test_uninitialized_runtime_submodule_does_not_read_parent_head(monkeypatch, tmp_path):
    """Real git: `git -C <empty dir>` walks up to the parent repo and reports ITS HEAD.

    Everything else in this file fakes `_git`, which is exactly why this case went
    unnoticed -- the upward discovery only happens in real git.
    """
    runtime_pin.runtime_pin_status.cache_clear()
    root = tmp_path / "pypto"
    (root / "runtime").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"],
        cwd=root,
        check=True,
    )
    assert runtime_pin._git(["rev-parse", "HEAD"], root / "runtime") is not None  # the trap
    monkeypatch.setattr(runtime_pin, "_pypto_root", lambda: root)
    monkeypatch.setitem(sys.modules, "_task_interface", None)

    status = runtime_pin.runtime_pin_status()
    runtime_pin.runtime_pin_status.cache_clear()

    assert status.runtime_dir == root / "runtime"
    assert status.expected is None


def test_env_skip_bypasses_a_real_mismatch(pin, monkeypatch):
    """The escape hatch suppresses the refusal, not the collection."""
    pin.install(STALE)
    monkeypatch.setenv(runtime_pin._ENV_SKIP, "1")

    status = runtime_pin.check_runtime_pin()

    assert not status.matches


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_env_skip_off_values_do_not_bypass(pin, monkeypatch, value):
    """An unset-looking value must not silently disable the check."""
    pin.install(STALE)
    monkeypatch.setenv(runtime_pin._ENV_SKIP, value)

    with pytest.raises(runtime_pin.RuntimePinMismatch):
        runtime_pin.check_runtime_pin()


# ---------------------------------------------------------------------------
# Off-pin: a warning, never a refusal
# ---------------------------------------------------------------------------


def test_off_pin_warns_without_raising(pin):
    """Working on a runtime branch is legitimate; it only earns a warning."""
    pin.expected = OFF_PIN
    pin.install(OFF_PIN)

    with pytest.warns(runtime_pin.RuntimePinWarning) as warned:
        status = runtime_pin.check_runtime_pin()

    assert status.matches
    assert status.off_pin
    assert OFF_PIN[:12] in str(warned[0].message)
    assert PINNED[:12] in str(warned[0].message)


def test_off_pin_warns_once_per_process(pin):
    """Two guarded modules import; the developer sees one warning."""
    pin.expected = OFF_PIN
    pin.install(OFF_PIN)

    with pytest.warns(runtime_pin.RuntimePinWarning) as warned:
        runtime_pin.check_runtime_pin()
        runtime_pin.check_runtime_pin()

    assert len(warned.list) == 1


def test_off_pin_and_mismatch_both_reported(pin):
    """An off-pin runtime does not mask a stale install."""
    pin.expected = OFF_PIN
    pin.install(STALE)

    with pytest.warns(runtime_pin.RuntimePinWarning), pytest.raises(runtime_pin.RuntimePinMismatch):
        runtime_pin.check_runtime_pin()


# ---------------------------------------------------------------------------
# Collection details
# ---------------------------------------------------------------------------


def test_status_is_cached(pin, monkeypatch):
    """The git calls happen once per process, not once per guarded import."""
    calls: list[list[str]] = []

    def counting_git(args, cwd):
        calls.append(args)
        return pin.git(args, cwd)

    monkeypatch.setattr(runtime_pin, "_git", counting_git)
    runtime_pin.check_runtime_pin()
    before = len(calls)
    runtime_pin.check_runtime_pin()

    assert before == 2
    assert len(calls) == before


def test_cleanliness_is_not_collected_eagerly(pin, monkeypatch):
    """`git status` is far costlier than rev-parse, so only the CLI pays for it."""
    seen: list[list[str]] = []

    def recording_git(args, cwd):
        seen.append(args)
        return pin.git(args, cwd)

    monkeypatch.setattr(runtime_pin, "_git", recording_git)
    runtime_pin.runtime_pin_status()

    assert ["status", "--porcelain"] not in seen


def test_runtime_tree_is_dirty(pin):
    """Uncommitted runtime changes are reported on demand."""
    status = runtime_pin.runtime_pin_status()
    assert not runtime_pin.runtime_tree_is_dirty(status)

    pin.porcelain = " M src/common/worker/chip_worker.cpp"
    assert runtime_pin.runtime_tree_is_dirty(status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reports_a_healthy_pin(pin, capsys):
    """A matching install exits 0 and says so."""
    assert runtime_pin.main([]) == 0
    assert "ok" in capsys.readouterr().out


def test_cli_reports_a_mismatch(pin, capsys):
    """A stale install exits 1 and prints both revisions plus the fix."""
    pin.install(STALE)

    assert runtime_pin.main([]) == 1

    out = capsys.readouterr().out
    assert PINNED[:12] in out
    assert STALE[:12] in out
    assert "pip install --no-build-isolation" in out


def test_cli_skips_when_not_a_checkout(pin, monkeypatch, capsys):
    """A wheel install is not an error state, so the CLI exits 0."""
    monkeypatch.setattr(runtime_pin, "_pypto_root", lambda: None)

    assert runtime_pin.main([]) == 0
    assert "skip" in capsys.readouterr().out


def test_cli_notes_a_dirty_runtime(pin, capsys):
    """A matching revision against an edited tree is reported, not trusted blindly."""
    pin.porcelain = " M src/common/worker/chip_worker.cpp"

    assert runtime_pin.main([]) == 0
    assert "uncommitted changes" in capsys.readouterr().out


def test_cli_fix_reinstalls_then_reverifies(pin, monkeypatch, capsys):
    """--fix runs pip, then re-checks in a fresh interpreter rather than in this one."""
    pin.install(STALE)
    commands: list[list[str]] = []

    class Completed:
        returncode = 0

    def fake_run(command, check):  # noqa: FBT002 -- mirrors subprocess.run's keyword
        commands.append(command)
        return Completed()

    monkeypatch.setattr(runtime_pin.subprocess, "run", fake_run)

    assert runtime_pin.main(["--fix"]) == 0

    assert commands[0][1:5] == ["-m", "pip", "install", "--no-build-isolation"]
    assert commands[0][-1] == str(pin.root / "runtime")
    # `-c`, not `-m`: `pypto.runtime` imports this module, so `-m` warns about a
    # double import before printing anything useful.
    assert commands[1][1] == "-c"
    assert "pypto.runtime.runtime_pin" in commands[1][2]
    assert "fresh interpreter" in capsys.readouterr().out


def test_cli_fix_propagates_a_failed_reinstall(pin, monkeypatch):
    """A pip failure is surfaced, and no bogus re-check claims success."""
    pin.install(STALE)

    class Failed:
        returncode = 1

    monkeypatch.setattr(runtime_pin.subprocess, "run", lambda command, check: Failed())

    assert runtime_pin.main(["--fix"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
