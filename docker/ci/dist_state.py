# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Apply the image's runtime patch and attest the resulting installed binding.

The record is written only after a successful build. It is not a blanket dirty
tree exemption: source revisions, the complete diff, untracked files and the
installed extension must still match. Mounted or subsequently edited trees fail.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

PATCH_FILES = {
    "cann92-a2a3-aicpu-class-memaccess": "src/a2a3/platform/onboard/aicpu/CMakeLists.txt",
    "cann92-a5-aicpu-class-memaccess": "src/a5/platform/onboard/aicpu/CMakeLists.txt",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_patch(source: Path, patch: Path) -> None:
    runtime = source / "runtime"
    if git(runtime, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Runtime must be clean before applying an image patch")
    command = ["git", "-C", str(runtime), "apply", "--check", str(patch.resolve())]
    if subprocess.run(command, capture_output=True, check=False).returncode == 0:
        subprocess.run(command[:-2] + [str(patch.resolve())], check=True)
    else:
        # An exact reverse application proves this fix is already upstream.
        # Unknown source layouts fail closed instead of silently losing the fix.
        subprocess.run(command[:-1] + ["--reverse", str(patch.resolve())], check=True)


def snapshot(source: Path) -> dict[str, str]:
    runtime = source / "runtime"
    pinned = git(source, "rev-parse", "HEAD:runtime")
    if git(runtime, "rev-parse", "HEAD") != pinned:
        raise ValueError("Runtime HEAD differs from the PyPTO gitlink")
    for checkout in (source, runtime):
        if git(checkout, "ls-files", "--others", "--exclude-standard"):
            raise ValueError(f"{checkout.name} contains untracked files")
    spec = importlib.util.find_spec("_task_interface")
    if spec is None or spec.origin is None:
        raise ValueError("Cannot locate the installed simpler extension")
    return {
        "pypto": git(source, "rev-parse", "HEAD"),
        "pypto_diff": hashlib.sha256(git(source, "diff", "--binary", "HEAD").encode()).hexdigest(),
        "runtime": pinned,
        "runtime_diff": hashlib.sha256(git(runtime, "diff", "--binary", "HEAD").encode()).hexdigest(),
        "extension": digest(Path(spec.origin)),
    }


def record(source: Path, home: Path, patch_name: str) -> None:
    runtime = source / "runtime"
    patch_file = PATCH_FILES.get(patch_name)
    changed = git(runtime, "diff", "--name-only", "HEAD")
    if changed:
        if patch_file is None or changed != patch_file:
            raise ValueError("Only the configured runtime patch may differ from HEAD")
        original = git(runtime, "show", f"HEAD:{patch_file}")
        anchor = "        -Wno-error=variadic-macros\n"
        flag = "        $<$<CXX_COMPILER_ID:GNU>:-Wno-error=class-memaccess>\n"
        expected = original.replace(anchor, anchor + flag)
        if original.count(anchor) != 1 or (runtime / patch_file).read_text().strip() != expected:
            raise ValueError("Runtime changes differ from the configured patch")
    state = snapshot(source)
    state["patch"] = patch_name
    if patch_name:
        patch = home / "image-tools" / f"{patch_name}.patch"
        state["patch_sha256"] = digest(patch)
    (home / "dist-state.json").write_text(json.dumps(state, indent=2) + "\n")


def verify(source: Path, home: Path) -> None:
    state = json.loads((home / "dist-state.json").read_text())
    if state["patch"]:
        patch = home / "image-tools" / f"{state['patch']}.patch"
        if state["patch"] not in PATCH_FILES or digest(patch) != state["patch_sha256"]:
            raise ValueError("Image patch differs from its build record")
    current = snapshot(source)
    for key, value in current.items():
        if state.get(key) != value:
            raise ValueError(f"Image build record mismatch: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["apply", "record", "verify"])
    parser.add_argument("--source", type=Path, default=Path("/workspace/pypto"))
    parser.add_argument("--home", type=Path, default=Path("/opt/pypto"))
    parser.add_argument("--patch", choices=["", *PATCH_FILES], default="")
    args = parser.parse_args()
    if args.operation == "apply":
        if args.patch:
            apply_patch(args.source, args.home / "image-tools" / f"{args.patch}.patch")
    elif args.operation == "record":
        record(args.source, args.home, args.patch)
    else:
        verify(args.source, args.home)


if __name__ == "__main__":
    main()
