# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Warm the three exact PyPTO MFE builds without starting a device worker."""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig
from pypto.runtime.distributed_runner import _assemble_chip_callables

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    options = parser.parse_args()
    devices = [int(value) for value in options.devices.split(",")]
    if len(devices) != 2 or devices[0] == devices[1]:
        parser.error("Pass a distinct device pair")
    path = ROOT / "tests/st/distributed/test_urma_mfe.py"
    spec = importlib.util.spec_from_file_location("test_urma_mfe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    inputs = module._rank_inputs(2)
    config = RunConfig(
        platform="a5", distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0)
    )
    manifest = []
    for name, builder in (
        ("put", module._build_put_only),
        ("get", module._build_get_only),
        ("exchange", module._build_urma_exchange),
    ):
        args = [inputs, torch.zeros_like(inputs)]
        if name == "exchange":
            args = [
                inputs,
                module._rank_inputs(2, round_index=1),
                torch.zeros_like(inputs),
                torch.zeros_like(inputs),
            ]
        started = time.monotonic()
        print(f"devices={options.devices} case={name} codegen START", flush=True)
        compiled = builder(2).compile(*args, config=config)
        print(
            f"case={name} codegen END elapsed={time.monotonic() - started:.1f}s; assembly START", flush=True
        )
        # This is the same assembly used at dispatch, without Worker init or a
        # device allocation. Execution still enters the normal JIT/runtime path.
        _assemble_chip_callables(compiled)
        print(f"case={name} assembly END total={time.monotonic() - started:.1f}s", flush=True)
        source_root = compiled.output_dir
        evidence = ROOT / "outputs/urma-mfe/prepared" / options.devices.replace(",", "-") / name
        for source in source_root.rglob("*"):
            if source.is_file() and source.suffix in (".cpp", ".h", ".pto", ".json", ".py"):
                target = evidence / source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        manifest.append({"case": name, "build_directory": str(source_root)})
    target = ROOT / "outputs/urma-mfe/prepared" / options.devices.replace(",", "-") / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
