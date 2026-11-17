# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expose this job's installed wheel binaries to standalone source-tree tests."""

import importlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def link_control_binaries(installed: Path, destination: Path) -> None:
    """Never synthesize metadata or replace an existing source build."""
    if not (installed / "pto_isa_build.json").is_file():
        raise RuntimeError(f"Installed runtime metadata is missing: {installed}")
    if destination.resolve() == installed.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to replace existing control runtime: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(installed.resolve(), target_is_directory=True)


def main() -> None:
    # Resolve hardware dependencies only for CI; link validation also runs on CPU.
    interface = importlib.import_module("_task_interface")
    installed_root = importlib.import_module("simpler_setup.environment").PROJECT_ROOT
    pto_isa = importlib.import_module("simpler_setup.pto_isa")

    runtime = ROOT / "runtime"
    pinned = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    if interface.__build_commit__ != pinned:
        raise RuntimeError("Installed Simpler extension differs from this checkout's runtime pin")
    if (installed_root / "pto_isa.pin").read_text().strip() != (runtime / "pto_isa.pin").read_text().strip():
        raise RuntimeError("Installed Simpler PTO-ISA pin differs from the source control pin")
    installed = installed_root / "build/lib"
    pto_isa.validate_runtime_pto_isa_current_pin(installed, runtime_key="a5/onboard/tensormap_and_ringbuffer")
    # Running pytest from runtime/ imports the source simpler_setup, whose
    # PROJECT_ROOT points at that checkout. A wheel-cache hit builds no source
    # build/lib. Use the very same validated installed binaries PyPTO uses.
    source_lib = runtime / "build/lib"
    if source_lib.exists():
        # A wheel-cache miss builds source libraries before packaging. Retain
        # them only when their own per-runtime metadata validates against pin.
        pto_isa.validate_runtime_pto_isa_current_pin(
            source_lib, runtime_key="a5/onboard/tensormap_and_ringbuffer"
        )
    else:
        link_control_binaries(installed, source_lib)
    receipt = {"installed_lib": str(installed), "source_lib": str(runtime / "build/lib"), "runtime": pinned}
    output = ROOT / "outputs/urma-mfe/control-runtime.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Standalone controls use validated runtime libraries: {receipt}", flush=True)


if __name__ == "__main__":
    main()
