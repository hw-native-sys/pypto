# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check the optional adapter's ABI probe without Torch or CANN dependencies."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("cmake") is None, reason="requires cmake")
@pytest.mark.parametrize("optimize", ["0", "1", "2"])
@pytest.mark.parametrize("compatible_abi", [False, True])
def test_torch_abi_probe_under_python_optimization(tmp_path, optimize, compatible_abi):
    prefix = tmp_path / "torch-cmake"
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"_C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI={compatible_abi!r})\n"
        f"utils = SimpleNamespace(cmake_prefix_path={str(prefix)!r})\n"
    )
    (tmp_path / "torch_npu.py").write_text("")
    probe = Path(__file__).resolve().parents[3] / "python/bindings/torch/CMakeLists.txt"
    result = tmp_path / "probe-result.txt"
    script = tmp_path / "probe.cmake"
    # Include the production probe, stopping at find_package after it validates
    # the ABI and parses both paths. No native framework build is required.
    script.write_text(
        "cmake_minimum_required(VERSION 3.15)\n"
        f'set(Python_EXECUTABLE "{sys.executable}")\n'
        "macro(find_package)\n"
        f'  file(WRITE "{result}" "${{_torch_cmake}}\\n${{_torch_npu}}")\n'
        "  return()\n"
        "endmacro()\n"
        f'include("{probe}")\n'
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONOPTIMIZE": optimize}
    completed = subprocess.run(
        ["cmake", "-P", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if compatible_abi:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert result.read_text().splitlines() == [str(prefix), str(tmp_path)]
    else:
        assert completed.returncode != 0
        assert "PyPTO core requires the C++11 libstdc++ ABI" in completed.stderr
        assert not result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
