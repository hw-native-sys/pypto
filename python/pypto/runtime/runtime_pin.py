# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Consistency check between PyPTO's pinned runtime and the installed ``simpler``.

``runtime/`` *is* the simpler submodule, so PyPTO's pin is its gitlink -- there is no
second pin file to drift, the same reason ``toolchain/versions.env`` declines to
restate the pto-isa revision.

The installed side has no version that tracks it: simpler's ``pyproject.toml`` says
``0.1.0`` and always will. Its only revision identity is
``_task_interface.__build_commit__``, stamped by ``runtime/python/bindings/CMakeLists.txt``
at build time.

Simpler carries its own guard (``simpler.task_interface._assert_bindings_match_source_tree``),
but that one compares the extension against *its own* source tree and returns early when
there is no ``.git`` beside it -- so a wheel, or a plain ``pip install ./runtime`` that
copies into ``site-packages``, is never checked at all. That is exactly the install this
module covers, and the one where staleness is hardest to see: a changed struct layout makes
fields read as 0 with no error, and the failure surfaces much later as a plausible-looking
runtime rejection.

Nothing here imports simpler at module scope. ``pypto.runtime`` must stay importable in a
codegen-only installation, and ``tests/ut/test_optional_runtime_imports.py`` enforces it.
"""

import argparse
import os
import shlex
import subprocess
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

_ENV_SKIP: Final[str] = "PYPTO_SKIP_RUNTIME_PIN_CHECK"
_GIT_TIMEOUT_S: Final[int] = 5
_OFF: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})

# ``warnings`` dedups per (message, category, calling module, line), and this module is
# imported from two call sites, so the off-pin warning would otherwise fire twice.
_state: dict[str, bool] = {"warned": False}


class RuntimePinMismatch(ImportError):
    """The installed ``simpler`` was built from a revision other than ``runtime/``.

    An ``ImportError`` because it is raised while importing the modules that pull in
    simpler, and because that is what simpler's own equivalent guard raises.
    """


class RuntimePinWarning(UserWarning):
    """``runtime/`` is checked out off the revision this PyPTO commit pins."""


@dataclass(frozen=True)
class RuntimePinStatus:
    """What each side of the runtime pin currently says.

    ``None`` anywhere means *cannot tell*, which is always a skip and never a failure:
    a wheel-installed PyPTO has no ``runtime/`` to compare against, a release tarball has
    no git, and a simpler built without git carries an empty stamp. Refusing to run
    because the question is unanswerable would break every legitimate install.

    Attributes:
        runtime_dir: ``<pypto>/runtime``, or None when PyPTO is not a source checkout.
        expected: ``runtime/`` working-tree HEAD -- the revision installing it would give.
        pinned: The gitlink, ``git rev-parse HEAD:runtime`` -- what this PyPTO commit pins.
        installed: ``_task_interface.__build_commit__``, or None when it cannot be read.
        stamp_missing: The extension is importable but predates ``__build_commit__``.
        extension_path: Filesystem path of the ``_task_interface`` module actually loaded.
    """

    runtime_dir: Path | None
    expected: str | None
    pinned: str | None
    installed: str | None
    stamp_missing: bool
    extension_path: str | None

    @property
    def comparable(self) -> bool:
        """True when both sides are known, so a verdict is meaningful."""
        return self.expected is not None and self.installed is not None

    @property
    def matches(self) -> bool:
        """True when the installed simpler was built from ``runtime/``'s HEAD."""
        return self.comparable and self.expected == self.installed

    @property
    def off_pin(self) -> bool:
        """True when ``runtime/`` is checked out somewhere other than the gitlink."""
        return self.expected is not None and self.pinned is not None and self.expected != self.pinned

    def reinstall_command(self) -> str:
        """The command that repairs a mismatch, quoted so it survives a copy-paste."""
        target = self.runtime_dir if self.runtime_dir is not None else Path("./runtime")
        return f"pip install --no-build-isolation -e {shlex.quote(str(target))}"


def check_runtime_pin() -> RuntimePinStatus:
    """Refuse an installed ``simpler`` built from a revision other than ``runtime/``.

    Called at import time from every PyPTO module that imports simpler at module scope.
    Cheap enough to do so: :func:`runtime_pin_status` is cached, so the two ``git
    rev-parse`` calls happen once per process.

    Set ``PYPTO_SKIP_RUNTIME_PIN_CHECK=1`` to bypass it -- at the cost of the silent
    struct corruption the check exists to prevent.

    Returns:
        The collected status, so a caller can inspect it even when the check passes.

    Raises:
        RuntimePinMismatch: The installed simpler and ``runtime/`` are different revisions.
    """
    status = runtime_pin_status()
    if _skip_requested():
        return status
    if status.off_pin and not _state["warned"]:
        _state["warned"] = True
        warnings.warn(off_pin_message(status), RuntimePinWarning, stacklevel=2)
    if _has_verdict(status):
        raise RuntimePinMismatch(mismatch_message(status))
    return status


def _has_verdict(status: RuntimePinStatus) -> bool:
    """True when both sides are known and disagree -- the only failing state."""
    if status.expected is None:
        return False
    return status.stamp_missing or (status.comparable and not status.matches)


@lru_cache(maxsize=1)
def runtime_pin_status() -> RuntimePinStatus:
    """Collect both sides of the runtime pin, once per process.

    Cached because it shells out to git. ``runtime/`` cleanliness is deliberately *not*
    collected here: ``git status`` over the runtime tree costs far more than the two
    ``rev-parse`` calls, and only the CLI reports it (see :func:`runtime_tree_is_dirty`).

    Returns:
        The resolved :class:`RuntimePinStatus`.
    """
    installed, stamp_missing, extension_path = _installed_revision()
    root = _pypto_root()
    if root is None:
        return RuntimePinStatus(None, None, None, installed, stamp_missing, extension_path)
    runtime_dir = root / "runtime"
    return RuntimePinStatus(
        runtime_dir=runtime_dir,
        expected=_git(["rev-parse", "HEAD"], runtime_dir) if _is_own_checkout(runtime_dir) else None,
        pinned=_git(["rev-parse", "HEAD:runtime"], root),
        installed=installed,
        stamp_missing=stamp_missing,
        extension_path=extension_path,
    )


def runtime_tree_is_dirty(status: RuntimePinStatus) -> bool:
    """Report whether ``runtime/`` has uncommitted changes.

    A dirty tree makes a *matching* revision inconclusive -- the install may predate
    edits that never became a commit. Reported, not enforced: warning on every import
    would punish exactly the people editing the runtime on purpose.

    Args:
        status: A status whose ``runtime_dir`` locates the tree to inspect.

    Returns:
        True when ``git status --porcelain`` reports anything.
    """
    if status.runtime_dir is None or not _is_own_checkout(status.runtime_dir):
        return False
    return bool(_git(["status", "--porcelain"], status.runtime_dir))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


_HAZARD: Final[str] = (
    "The compiled extension does not rebuild on import, so its struct layouts may no\n"
    "longer match the Python that drives them -- fields can read as 0 with no error."
)


def mismatch_message(status: RuntimePinStatus) -> str:
    """Render the full failure text, table included.

    Used for the raised exception, which carries no other context.
    """
    return "\n\n".join([_headline(status), "\n".join(_status_table(status)), _HAZARD, _fix_text(status)])


def verdict_message(status: RuntimePinStatus) -> str:
    """Render the failure text without the table, for a CLI that just printed one."""
    return "\n\n".join([_headline(status), _HAZARD, _fix_text(status)])


def _headline(status: RuntimePinStatus) -> str:
    """Name the specific failure: a differing revision, or one too old to have a stamp."""
    if status.stamp_missing:
        return (
            "The installed `_task_interface` predates the build stamp, so it was compiled\n"
            "before the revision you are running."
        )
    return "PyPTO's runtime pin and the installed `simpler` are different revisions."


def _fix_text(status: RuntimePinStatus) -> str:
    """Render the repair options and the escape hatch."""
    return (
        f"Reinstall:  {status.reinstall_command()}\n"
        f"Or:         pypto-runtime-pin --fix\n\n"
        f"Bypass with {_ENV_SKIP}=1, and accept the silent struct corruption."
    )


def off_pin_message(status: RuntimePinStatus) -> str:
    """Render the warning text for a ``runtime/`` checkout that is off the gitlink."""
    expected = _short(status.expected)
    pinned = _short(status.pinned)
    return (
        f"runtime/ is at {expected} but this PyPTO commit pins {pinned}. "
        f"You are running a runtime other than the one this PyPTO was tested against.\n"
        f"  git submodule update --init runtime"
    )


def _status_table(status: RuntimePinStatus) -> list[str]:
    """Render both sides of the pin as aligned, self-describing lines."""
    installed = "<no build stamp>" if status.stamp_missing else _short(status.installed)
    lines = [
        f"  pypto              {status.runtime_dir.parent if status.runtime_dir else '<not a checkout>'}",
        f"  runtime/ HEAD      {_short(status.expected)}   <- installing this tree gives you this",
        f"  pypto pins         {_short(status.pinned)}   <- git rev-parse HEAD:runtime",
        f"  installed simpler  {installed}   <- _task_interface.__build_commit__",
    ]
    if status.extension_path:
        lines.append(f"                     {status.extension_path}")
    return lines


def _short(revision: str | None) -> str:
    """Abbreviate a revision for display, keeping 'unknown' explicit."""
    if not revision:
        return "<unknown>"
    return revision[:12]


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _skip_requested() -> bool:
    """True when the developer has explicitly bypassed the check.

    Read inline rather than through a name parameter so
    ``tests/lint/check_environment_inputs.py`` can resolve which variable this is.
    """
    return os.environ.get(_ENV_SKIP, "").strip().lower() not in _OFF


def _pypto_root() -> Path | None:
    """Return the repo root of the *running* PyPTO, or None when it is not a checkout.

    Anchored on ``__file__``, never on the working directory: an editable install can
    serve PyPTO from a tree other than the one you are standing in, and the pin that
    matters belongs to the code actually imported.
    """
    root = Path(__file__).resolve().parents[3]
    if not (root / "runtime").is_dir():
        return None
    # A worktree's `.git` is a file, not a directory -- `exists` covers both.
    return root if (root / ".git").exists() else None


def _is_own_checkout(runtime_dir: Path) -> bool:
    """True when ``runtime/`` is a repository of its own, not just a directory in PyPTO's.

    An uninitialized submodule -- a clone without ``--recurse-submodules``, or after
    ``git submodule deinit runtime`` -- leaves ``runtime/`` empty. git then discovers the
    repository by walking upward, so ``git -C runtime rev-parse HEAD`` silently answers
    with *PyPTO's* HEAD and every installed simpler would look like a mismatch. A
    submodule's ``.git`` is a file, so ``exists`` covers it.
    """
    return (runtime_dir / ".git").exists()


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git command, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _installed_revision() -> tuple[str | None, bool, str | None]:
    """Return ``(revision, stamp_missing, extension_path)`` for the loaded extension.

    A *missing* ``__build_commit__`` is not the same as an empty one. The attribute only
    disappears on an extension compiled before the stamp existed, which in a tree new
    enough to run this check is by definition a different revision -- the exact case this
    guards. An empty stamp means git was unavailable at build time, which genuinely
    cannot be compared. Simpler's own guard draws the same distinction.
    """
    from importlib import import_module  # noqa: PLC0415 -- keep simpler off module scope

    try:
        module = import_module("_task_interface")
    except ImportError:
        return None, False, None
    path = getattr(module, "__file__", None)
    if not hasattr(module, "__build_commit__"):
        return None, True, path
    revision = module.__build_commit__
    if not isinstance(revision, str) or not revision.strip():
        return None, False, path
    return revision.strip(), False, path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Report the runtime pin, and optionally reinstall ``runtime/`` to repair it.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        Process exit status -- 0 when the pin holds or cannot be checked, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="pypto-runtime-pin",
        description="Check that the installed `simpler` matches PyPTO's pinned runtime.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="reinstall runtime/ into the current environment when it does not match",
    )
    args = parser.parse_args(argv)

    runtime_pin_status.cache_clear()
    status = runtime_pin_status()
    for line in _status_table(status):
        print(line)
    if runtime_tree_is_dirty(status):
        print("  note               runtime/ has uncommitted changes; a matching revision")
        print("                     does not prove the install is current")

    if status.off_pin:
        print(f"\nwarn  {off_pin_message(status)}")

    if not _has_verdict(status):
        print(f"\n{_skip_or_ok(status)}")
        return 0

    print(f"\n{verdict_message(status)}")
    if not args.fix:
        return 1
    return _reinstall(status)


def _skip_or_ok(status: RuntimePinStatus) -> str:
    """Explain a passing run: the pin holds, or the question is unanswerable."""
    if status.runtime_dir is None:
        return "skip  PyPTO is not a source checkout, so there is no pin to compare against."
    if status.expected is None:
        return "skip  no git revision for runtime/, so there is no pin to compare against."
    if status.installed is None:
        return "skip  no comparable `simpler` revision (not installed, or built without git)."
    return "ok    the installed `simpler` was built from runtime/."


def _reinstall(status: RuntimePinStatus) -> int:
    """Reinstall ``runtime/`` into the running interpreter, then re-check in a clean one."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-e",
        str(status.runtime_dir),
    ]
    print(f"\n--fix: {' '.join(command)}\n")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print("\nreinstall failed; the pin is unchanged.")
        return completed.returncode
    # This process already imported the old extension, so its stamp is stale no matter
    # what pip just did. Only a fresh interpreter can answer whether the fix took.
    # `-c` rather than `-m`: `pypto.runtime` imports this module, so running it as
    # `__main__` would warn about a double import before printing anything useful.
    print("\nre-checking in a fresh interpreter:\n")
    verify = subprocess.run(
        [sys.executable, "-c", "from pypto.runtime.runtime_pin import main; raise SystemExit(main())"],
        check=False,
    )
    return verify.returncode


if __name__ == "__main__":
    sys.exit(main())
