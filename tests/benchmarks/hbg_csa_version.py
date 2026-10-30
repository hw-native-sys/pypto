# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run an unchanged CSA harness against one complete, pinned Python checkout.

Native modules stay mapped to the common independently built environment.
Only source mappings change, before any pypto module is imported. Worktrees
retain their real Git identity and runtime pin checks.
"""

import hashlib
import importlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    root = Path(os.environ["CSA_VERSION_ROOT"]).resolve()
    source = root / "python"
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    assert not output.exists(), output
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    finders = [finder for finder in sys.meta_path if "pypto" in getattr(finder, "known_source_files", {})]
    assert len(finders) == 1, finders
    finder: Any = finders[0]
    old = Path(finder.known_source_files["pypto"]).resolve().parents[1]

    def remap(path: str) -> str:
        original = Path(path)
        if original.is_relative_to(old):
            replacement = source / original.relative_to(old)
            assert replacement.exists(), replacement
            return str(replacement)
        return path

    finder.known_source_files = {name: remap(path) for name, path in finder.known_source_files.items()}
    finder.submodule_search_locations = {
        name: {remap(path) for path in paths} for name, paths in finder.submodule_search_locations.items()
    }
    assert not finder.rebuild_flag, "Native artifacts must remain fixed"

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    assert not git("diff", "HEAD", "--", "python"), "Version source differs from its commit"
    manifest = {
        "version": os.environ["CSA_VERSION_NAME"],
        "revision": git("rev-parse", "HEAD"),
        "runtime_pin": git("rev-parse", "HEAD:runtime"),
        "python_sources": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / "pypto").rglob("*.py"))
        },
    }
    try:
        if "--preflight" in sys.argv:
            output.mkdir(parents=True)
            for name in ("torch", "torch_npu", "pypto", "pypto.pypto_core", "pypto._torch_npu"):
                importlib.import_module(name)
            pin = importlib.import_module("pypto.runtime.runtime_pin").check_runtime_pin()
            assert pin.pinned == manifest["runtime_pin"]
            assert pin.installed == manifest["runtime_pin"]
            print("VERSION_PREFLIGHT_PASS", manifest["version"], manifest["revision"])
        else:
            runpy.run_path(str(Path(__file__).with_name("hbg_csa_stability.py")), run_name="__main__")
    finally:
        loaded = {}
        for name, module in tuple(sys.modules.items()):
            if name != "pypto" and not name.startswith("pypto."):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            path = Path(filename).resolve()
            if path.suffix == ".py":
                assert path.is_relative_to(source), (name, path, source)
            loaded[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["loaded_modules"] = loaded
        if output.is_dir():
            (output / "version-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
