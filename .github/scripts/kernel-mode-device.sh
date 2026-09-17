# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Entered inside ONE task-submit allocation, after sourcing activate.sh.
set -euo pipefail
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD/build/torch_npu_tests${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
: "${TASK_DEVICE:?Run through task-submit to reserve a device}"
case "${1:-}" in
  eager)
    cases=(
      tests/st/runtime/kernel/test_jit_eager.py
      tests/st/runtime/kernel/test_kernel_context.py
      tests/st/runtime/kernel/test_kernel_shutdown.py
      tests/st/runtime/kernel/test_torch_interop.py
      tests/st/runtime/kernel/test_torch_launch.py
      tests/st/runtime/kernel/test_torch_ops.py
      tests/st/runtime/kernel/test_hot_path.py
    )
    selection=(-k 'not capture'
      '--deselect=tests/st/runtime/kernel/test_torch_launch.py::test_torch_kernel_launch[delayed-0]')
    ;;
  capture)
    cases=(tests/st/runtime/kernel/test_capture.py tests/st/runtime/kernel/test_torch_ops.py)
    selection=(-k capture)
    ;;
  *) echo "Expected eager or capture suite, got '${1:-}'" >&2; exit 2 ;;
esac
mkdir -p test-results/kernel-mode
# A reused self-hosted checkout must never reuse an earlier JUnit result.
rm -f "test-results/kernel-mode/$1.xml"
python - "$1" <<'PYTHON'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import simpler
import torch
import torch_npu
from simpler.task_interface import ChipWorker

from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
from pypto.torch.launch import _load_native

if torch.__version__.split("+")[0] != "2.6.0" or torch_npu.__version__ != "2.6.0.post2":
    raise RuntimeError(f"Expected Torch 2.6.0 / torch_npu 2.6.0.post2, got {torch.__version__}/{torch_npu.__version__}")
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise RuntimeError("Kernel adapter requires the C++11 ABI")
if not hasattr(ChipWorker, "kernel_init") or not hasattr(ChipWorker, "kernel_prepare_callable"):
    raise RuntimeError(f"Simpler at {simpler.__file__} lacks the pinned kernel API; install this checkout runtime")
native = _load_native()
revision = subprocess.check_output(["git", "-C", "runtime", "rev-parse", "HEAD"], text=True).strip()
if revision != SIMPLER_KERNEL_REVISION:
    raise RuntimeError(f"Runtime checkout {revision} differs from kernel ABI {SIMPLER_KERNEL_REVISION}")
device = int(os.environ["TASK_DEVICE"])
torch_npu.npu.set_device(device)
metadata = {
    "pypto": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "simpler": revision,
    "adapter_simpler": native.simpler_revision,
    "device": device,
    "chip": torch_npu.npu.get_device_name(device),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "nanobind": importlib.metadata.version("nanobind"),
    "runtime": "tensormap_and_ringbuffer",
    "warmup_required": True,
}
Path(f"test-results/kernel-mode/{sys.argv[1]}-environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PYTHON
npu-smi info > "test-results/kernel-mode/$1-npu-smi.txt"
if [ -f "${ASCEND_HOME_PATH:-}/version.cfg" ]; then
  cp "$ASCEND_HOME_PATH/version.cfg" "test-results/kernel-mode/$1-cann-version.txt"
fi
# These tests create isolated child processes and own queue-mode parametrization.
# Do not use xdist: one allocated device runs one test at a time.
python -m pytest "${cases[@]}" "${selection[@]}" --platform=a2a3 \
  --device="$TASK_DEVICE" -v -o junit_family=legacy --junitxml="test-results/kernel-mode/$1.xml" 2>&1 | tee "test-results/kernel-mode/$1-pytest.log"
