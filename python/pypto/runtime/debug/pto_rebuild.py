# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-run ptoas for stale .pto files and splice the new body into kernel cpps.

Invoked from :func:`pypto.runtime.debug.replay.replay` before the
``invalidate_binary_cache`` step, so the freshly-spliced kernel cpp drives
the subsequent cpp -> .so rebuild.

The discriminator between the ``pto -> cpp -> .o`` path and the existing
``cpp -> .o`` path is mtime, evaluated per ``.pto`` independently:

- ``ptoas/<unit>.cpp`` mtime >= ``ptoas/<unit>.pto`` mtime  →  skip ptoas
  rerun for this unit (user only touched the kernel cpp, or nothing).
- otherwise                                                  →  rerun ptoas
  and splice the new body into every matching ``kernels/<core>/<func>.cpp``.

``ptoas/<unit>.cpp`` thus doubles as a per-unit "ptoas build stamp" — no
extra metadata file is persisted.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

__all__ = ["rebuild_kernel_cpp_from_pto", "PTOAS_BODY_BEGIN", "PTOAS_BODY_END"]

# These literals MUST match the strings written by
# ``pypto.backend.pto_backend._generate_kernel_wrapper`` (the BEGIN sentinel
# is emitted at the header/body join, the END sentinel inside the wrapper
# template). ``tests/ut/backend/test_ptoas_sentinels_match.py`` asserts both
# still appear in that wrapper's source as a drift guard.
PTOAS_BODY_BEGIN = "// --- ptoas-generated code ---"
PTOAS_BODY_END = "// --- Kernel entry point ---"

# Mirror ``pto_backend._preprocess_ptoas_output``'s function-definition
# pattern: the same regex that converts the ptoas-emitted
# ``AICORE void <name>(...)`` qualifier into ``static __aicore__ void
# <name>(...)``. Capturing the name here lets us discover which kernel cpps
# each .pto feeds without persisting a map at compile time.
_PTOAS_FUNC_DEF_RE = re.compile(r"(?:__global__\s+)?AICORE\s+void\s+(\w+)\s*\(")


def _ptoas_binary() -> str | None:
    """Locate the ``ptoas`` executable, or return None when unavailable."""
    root = os.environ.get("PTOAS_ROOT")
    if root:
        cand = os.path.join(root, "ptoas")
        return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None
    return shutil.which("ptoas")


def _disabled_via_env() -> bool:
    """Return True when ``PYPTO_REBUILD_FROM_PTO`` opts out of rebuild."""
    return os.environ.get("PYPTO_REBUILD_FROM_PTO", "").strip().lower() in (
        "0",
        "false",
        "no",
    )


def _ptoas_flags() -> list[str]:
    """Base ptoas flags shared with ``pto_backend._get_ptoas_flags``.

    Backend-specific extras (from ``get_handler().get_extra_ptoas_flags()``)
    are intentionally omitted — the rebuild path has no backend handler in
    scope. Edits that rely on backend-specific flags require a fresh
    ``ir.compile()`` instead of a ``.pto`` splice.
    """
    return ["--enable-insert-sync", "--pto-level=level3"]


def _run_ptoas(ptoas_bin: str, pto_path: Path, out_cpp: Path) -> None:
    """Invoke the ``ptoas`` binary on *pto_path*, writing to *out_cpp*."""
    cmd = [ptoas_bin, str(pto_path), "-o", str(out_cpp), *_ptoas_flags()]
    result = subprocess.run(  # noqa: S603 — args are constructed locally, no shell
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ptoas rebuild failed for {pto_path.name}: {result.stderr.strip()}")


def _preprocess_ptoas_body(content: str) -> str:
    """Local copy of ``pto_backend._preprocess_ptoas_output``.

    Must stay in step with the two-pass rewrite there: the first sub makes
    top-level kernels file-local, the second normalises remaining ``AICORE``
    qualifiers on mixed-kernel sub-functions and helpers.
    """
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#include") and ("pto-inst" in s or "cstdint" in s or "tensor.h" in s):
            continue
        if s == "using namespace pto;":
            continue
        filtered.append(line)
    result = "".join(filtered)
    result = re.sub(
        r'(?:extern\s*"C"\s*)?(?:__global__\s+)?AICORE\s+void',
        "static __aicore__ void",
        result,
    )
    return re.sub(r"\bAICORE\b", "__aicore__", result)


def _extract_func_names(ptoas_cpp: str) -> list[str]:
    """Return all function names defined in a raw ptoas-emitted cpp."""
    return _PTOAS_FUNC_DEF_RE.findall(ptoas_cpp)


def _find_kernel_cpp(work_dir: Path, func_name: str) -> Path | None:
    """Locate ``kernels/{aic,aiv}/<func_name>.cpp`` under *work_dir*."""
    for core in ("aic", "aiv"):
        candidate = work_dir / "kernels" / core / f"{func_name}.cpp"
        if candidate.is_file():
            return candidate
    return None


def _splice_body(target: Path, new_body: str) -> None:
    """Replace the section between BEGIN/END sentinels in *target*.

    Header (above BEGIN) and wrapper (END onwards, including ``kernel_entry``)
    are kept byte-for-byte. Only the ptoas-generated body in the middle is
    swapped out.
    """
    text = target.read_text()
    begin = text.find(PTOAS_BODY_BEGIN)
    end = text.find(PTOAS_BODY_END)
    if begin == -1 or end == -1 or end <= begin:
        raise RuntimeError(
            f"{target}: cannot splice — missing or out-of-order ptoas sentinels. "
            f"This build_output predates the .pto rebuild feature; recompile via "
            f"ir.compile() to refresh."
        )
    head = text[: begin + len(PTOAS_BODY_BEGIN)]
    tail = text[end:]
    target.write_text(f"{head}\n{new_body}\n{tail}")


def rebuild_kernel_cpp_from_pto(work_dir: Path | str) -> list[str]:
    """Re-run ptoas for any ``.pto`` newer than its derived kernel cpp(s).

    Per ``.pto``: if its sibling ``ptoas/<unit>.cpp`` is older than the
    ``.pto`` (or missing), rerun ``ptoas`` and splice the new preprocessed
    body into every matching ``kernels/<core>/<func>.cpp``. Other .pto
    files in the same directory are untouched.

    Returns the list of touched cpp paths (relative to *work_dir*) for
    logging. No-op (returns ``[]``) when ``ptoas/`` is missing, when the
    ptoas binary cannot be found, or when ``PYPTO_REBUILD_FROM_PTO=0``.

    Prints stage status to stdout so users running ``debug/run.py`` see
    which mode is taken (``pto -> cpp`` vs ``cpp -> .o``) without needing
    to enable verbose logging.
    """
    work_dir = Path(work_dir)
    if _disabled_via_env():
        print("[pto->cpp] skipped (PYPTO_REBUILD_FROM_PTO=0)")
        return []
    ptoas_dir = work_dir / "ptoas"
    if not ptoas_dir.is_dir():
        return []
    ptoas_bin = _ptoas_binary()
    if ptoas_bin is None:
        print("[pto->cpp] skipped: ptoas binary not found (set PTOAS_ROOT or PATH)")
        return []

    touched: list[str] = []
    for pto_path in sorted(ptoas_dir.glob("*.pto")):
        unit_name = pto_path.stem
        out_cpp = ptoas_dir / f"{unit_name}.cpp"

        if out_cpp.exists() and out_cpp.stat().st_mtime >= pto_path.stat().st_mtime:
            continue

        print(f"[pto->cpp] regenerating from ptoas/{pto_path.name}")
        _run_ptoas(ptoas_bin, pto_path, out_cpp)
        raw_cpp = out_cpp.read_text()
        new_body = _preprocess_ptoas_body(raw_cpp)

        for func_name in _extract_func_names(raw_cpp):
            target = _find_kernel_cpp(work_dir, func_name)
            if target is None:
                continue  # ptoas may emit helpers that are not exported kernels
            _splice_body(target, new_body)
            rel = str(target.relative_to(work_dir))
            print(f"[pto->cpp]   spliced -> {rel}")
            touched.append(rel)

    if not touched:
        print("[pto->cpp] no .pto changes detected")
    else:
        print(f"[pto->cpp] updated {len(touched)} kernel cpp(s)")
    return touched
