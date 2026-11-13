# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resolution of the pinned PTO-ISA checkout.

``runtime/pto_isa.pin`` is the single source of truth for the revision, and
``simpler_setup.pto_isa`` owns the only resolver. PyPTO deliberately keeps no
resolver of its own: a second implementation inevitably drifts, and
:class:`~pypto.runtime.kernel_compiler.KernelCompiler` skips its own revision
re-check *because* the resolver already guaranteed the pin -- handing it an
off-pin tree silently compiles kernels against the wrong ISA headers.

This module is deliberately separate from ``device_runner``. Resolving the pin
is a build-input question, not a device-execution one, so it must stay
importable in a codegen-only installation that has no ``simpler`` and therefore
no ``task_interface`` extension. Nothing here imports ``simpler_setup`` at
module scope.

Temporary A2/A3 TREM fallback
-----------------------------

The pinned PTO-ISA predates ``3d36366a0`` ("trem/trems(a2a3): never write the
input tiles"), whose fix first shipped in pto-isa ``8fb67fa7e``. Until the
managed pin moves, :func:`ensure_pto_isa_root` returns a detached worktree of
the pin with that commit backported. The worktree is committed locally so its
HEAD is a distinct, stable kernel-cache identity; the managed checkout is never
modified. The fallback disables itself as soon as the pinned headers carry the
fix. Set ``PYPTO_DISABLE_PTO_ISA_TREM_PATCH=1`` to resolve the pristine pin.
"""

import fcntl
import hashlib
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Provenance: cann/pto-isa 3d36366a0 "trem/trems(a2a3): never write the input
# tiles" (GitCode !2081), first released in pto-isa 8fb67fa7e. Remove this
# directory and the resolver hook once the managed pin contains it.
_TREM_PATCH_PATH = Path(__file__).resolve().parent / "pto_isa_patches" / "a2a3-trem-never-write-inputs.patch"
_TREM_PINNED_HEADER = Path("include") / "pto" / "npu" / "a2a3" / "TRem.hpp"
_TREM_FIX_MARKER = "read-only-src form"
_TREM_PATCH_DISABLE_ENV = "PYPTO_DISABLE_PTO_ISA_TREM_PATCH"
_TREM_PATCH_AUTHOR = ("PyPTO TREM fallback", "pypto-trem-fallback@localhost")
# Git repository-location variables override even `-C`; a hook or wrapper that
# exports them must not redirect the worktree or commit into another checkout.
_GIT_LOCATION_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    }
)


def ensure_pto_isa_root() -> str:
    """Resolve the PTO-ISA checkout PyPTO must compile against.

    Delegates to :func:`simpler_setup.pto_isa.ensure_pto_isa_root`, for which
    ``pto_isa.pin`` is the single source of truth: it reuses a managed checkout
    only when that checkout is clean and already at the pin, re-clones it
    otherwise, serializes concurrent resolutions with a file lock, and verifies
    ``HEAD`` before returning.

    When the pinned headers predate the A2/A3 TREM/TREMS fix, the returned path
    is a patched detached worktree of that pin (see the module docstring). The
    managed checkout itself is never modified.

    The resolved path is *exported* as ``PTO_ISA_ROOT`` for downstream consumers
    that build extern CCE kernels and need the ISA include directory. It is
    never *read* back: an ambient value is not the pin.

    Returns:
        Absolute path to the pinned (or TREM-patched) PTO-ISA checkout.

    Raises:
        RuntimeError: If ``pto_isa.pin`` is missing or malformed, or the TREM
            fallback patch no longer applies to the pin.
        OSError: If the pinned checkout cannot be obtained.
    """
    pinned = _resolve_pinned_pto_isa_root()
    resolved = _resolve_effective_pto_isa_root(pinned)
    os.environ["PTO_ISA_ROOT"] = resolved
    return resolved


def pto_isa_include_dir() -> Path:
    """Return the include directory of the PTO-ISA checkout PyPTO compiles against.

    Use this instead of reading ``PTO_ISA_ROOT`` when a ``pl.jit.extern`` kernel
    or an out-of-tree build needs the PTO-ISA headers -- it resolves the pin on
    demand rather than trusting whatever the ambient environment happens to say.

    Example::

        from pypto.runtime import pto_isa_include_dir

        _INCLUDE = pto_isa_include_dir()

    Returns:
        Absolute path to ``<pto-isa>/include``.

    Raises:
        RuntimeError: If ``pto_isa.pin`` is missing or malformed.
        OSError: If the pinned checkout cannot be obtained.
    """
    return Path(ensure_pto_isa_root()) / "include"


def effective_pto_isa_revision(pin: str) -> str:
    """Return the persistent identity of the PTO-ISA tree PyPTO compiles against.

    The pin alone is not enough while the temporary TREM backport is active:
    artifacts built against the patched headers must not be reused for the
    pristine pin, or vice versa. The patch id keeps the two trees distinct
    without requiring the checkout, so a cache hit still needs neither a
    checkout nor a git status. Once the fallback is disabled or removed, the
    identity is the pin again.

    Args:
        pin: The revision read from the runtime's ``pto_isa.pin``.

    Returns:
        ``pin`` when the fallback is disabled or its patch is missing, otherwise
        ``<pin>+trem-patch:<sha12>``.
    """
    if os.environ.get(_TREM_PATCH_DISABLE_ENV) == "1":
        return pin
    try:
        patch_id = hashlib.sha256(_TREM_PATCH_PATH.read_bytes()).hexdigest()[:12]
    except OSError:
        return pin
    return f"{pin}+trem-patch:{patch_id}"


@lru_cache(maxsize=1)
def _resolve_pinned_pto_isa_root() -> str:
    """Resolve the pin once per process (the underlying call takes a file lock)."""
    # noqa: PLC0415 -- importing at module scope would make this module, and
    # therefore ``pypto.runtime``, require simpler in a codegen-only install.
    from simpler_setup.pto_isa import (  # noqa: PLC0415
        ensure_pto_isa_root as _resolve,
    )

    return str(Path(_resolve(verbose=True)).resolve())


@lru_cache(maxsize=1)
def _resolve_effective_pto_isa_root(pinned: str) -> str:
    """Return the tree PyPTO must compile against for *pinned*.

    The pinned tree is returned unchanged when it already carries the TREM fix,
    when the fallback is disabled, or when it is not a PTO-ISA checkout.
    """
    if os.environ.get(_TREM_PATCH_DISABLE_ENV) == "1":
        return pinned
    return _ensure_trem_patch(pinned)


def _ensure_trem_patch(pinned: str) -> str:
    """Return a committed worktree of *pinned* with the TREM fix applied.

    The worktree lives next to the managed checkout, is detached at the pin and
    carries one local commit, so ``git status`` is clean and ``HEAD`` is a
    patch-specific revision. Concurrent callers serialize on a sibling lock.
    """
    pinned_path = Path(pinned)
    if not (pinned_path / _TREM_PINNED_HEADER).is_file():
        # Not a PTO-ISA checkout (e.g. a test double); leave it untouched.
        return str(pinned_path)
    if _trem_fix_present(pinned_path):
        return str(pinned_path)
    if not _TREM_PATCH_PATH.is_file():
        raise RuntimeError(f"PyPTO TREM fallback patch is missing: {_TREM_PATCH_PATH}")
    if not (pinned_path / ".git").exists():
        raise RuntimeError(f"PyPTO TREM fallback needs a git PTO-ISA checkout to patch, got {pinned_path}")

    pin_revision = _git_output(pinned_path, "rev-parse", "HEAD")
    patch_id = hashlib.sha256(_TREM_PATCH_PATH.read_bytes()).hexdigest()[:12]
    target = pinned_path.parent / f"pto-isa-trem-patch-{pin_revision[:12]}-{patch_id}"
    lock_path = pinned_path.parent / ".pto-isa-trem-patch.lock"

    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _is_patched_worktree(target):
            return str(target)

        shutil.rmtree(target, ignore_errors=True)
        _git(pinned_path, "worktree", "prune", check=False)
        _git(pinned_path, "worktree", "add", "--detach", str(target), pin_revision)
        try:
            _git(target, "apply", "--check", str(_TREM_PATCH_PATH))
            _git(target, "apply", str(_TREM_PATCH_PATH))
        except subprocess.CalledProcessError as error:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(
                "PyPTO TREM fallback patch no longer applies to the pinned PTO-ISA "
                f"{pin_revision}. Update or remove the fallback "
                f"({_TREM_PATCH_PATH})."
            ) from error

        _git(
            target,
            "-c",
            f"user.name={_TREM_PATCH_AUTHOR[0]}",
            "-c",
            f"user.email={_TREM_PATCH_AUTHOR[1]}",
            "commit",
            # This machine-generated commit must not inherit an ambient
            # commit.gpgSign: signing can fail without a key/agent (blocking
            # every resolution) and would otherwise perturb the cache identity.
            "--no-gpg-sign",
            "-a",
            "-m",
            f"PyPTO TREM fallback {patch_id}\n\nBase pin: {pin_revision}",
        )
        if not _trem_fix_present(target):
            raise RuntimeError(f"PyPTO TREM fallback patch did not produce the expected header in {target}")
        return str(target)


def _trem_fix_present(root: Path) -> bool:
    """Return whether *root*'s A2/A3 TREM header already carries the fix."""
    try:
        return _TREM_FIX_MARKER in (root / _TREM_PINNED_HEADER).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _is_patched_worktree(target: Path) -> bool:
    """Return whether *target* is a clean worktree carrying the TREM fix."""
    if not target.is_dir():
        return False
    try:
        return not _git_output(target, "status", "--porcelain") and _trem_fix_present(target)
    except (OSError, subprocess.CalledProcessError):
        return False


def _git_output(root: Path, *args: str) -> str:
    return _git(root, *args).stdout.strip()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Strip repository-location overrides: they win over `-C`, so a Git hook or
    # wrapper could otherwise redirect this resolver into the caller's checkout.
    environment = {key: value for key, value in os.environ.items() if key not in _GIT_LOCATION_ENV}
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        env=environment,
        check=check,
    )
