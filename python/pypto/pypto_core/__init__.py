# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bootstrap the compiled ``pypto_core`` nanobind extension.

When running from a worktree, prefer the locally built extension under
``build/python/bindings`` so tests exercise the current C++ sources instead of
an older site-packages install. Fall back to the installed extension when no
local build artifact exists.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _iter_extension_candidates() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent.parent
    candidates: list[Path] = []

    search_dirs = [
        pkg_dir.parent,
        repo_root / "build" / "python" / "bindings",
    ]
    for directory in search_dirs:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = directory / f"pypto_core{suffix}"
            if candidate.exists():
                candidates.append(candidate)

    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = base / "pypto" / f"pypto_core{suffix}"
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _load_extension() -> object:
    errors: list[str] = []
    for path in _iter_extension_candidates():
        loader = importlib.machinery.ExtensionFileLoader(__name__, str(path))
        spec = importlib.util.spec_from_file_location(__name__, path, loader=loader)
        if spec is None or spec.loader is None:
            errors.append(f"{path}: failed to create import spec")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[__name__] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # pragma: no cover - exercised only on broken builds
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "no candidate pypto_core extension found"
    raise ImportError(f"Could not load compiled extension for {__name__}:\n{detail}")


_EXTENSION_MODULE = _load_extension()

for _submodule_name in ("arith", "backend", "codegen", "ir", "passes", "testing"):
    _submodule = getattr(_EXTENSION_MODULE, _submodule_name, None)
    if _submodule is not None:
        sys.modules[f"{__name__}.{_submodule_name}"] = _submodule

globals().update(_EXTENSION_MODULE.__dict__)
