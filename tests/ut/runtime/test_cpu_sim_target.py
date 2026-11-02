# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Real compiler coverage for the Simpler CPU target admission contract."""

import ctypes
import shutil

import pytest


# pto::NPUArch defines A2A3 = 0 and A5 = 1. Without the SDK's target flags,
# an A5 kernel silently uses A2/A3 memory-model behavior; check both cores
# so a missing architecture selection cannot pass as a successful build.
@pytest.mark.parametrize(
    ("platform", "expected_arch", "family"), [("a2a3sim", 0, b"a2a3sim"), ("a5sim", 1, b"a5sim")]
)
@pytest.mark.parametrize("core_type", ["aic", "aiv"])
def test_pypto_compiler_preserves_simulator_target(tmp_path, platform, expected_arch, family, core_type):
    """Both cores must use the requested ISA family and export admission metadata."""
    pytest.importorskip("simpler_setup")
    if shutil.which("g++-15") is None:
        pytest.skip("GCC 15 is required for the real CPU simulator compiler")
    # Keep SDK imports after the optional-dependency skip for compiler-only CI.
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    source = tmp_path / "target.cpp"
    source.write_text(
        "#include <cstdint>\n#include <pto/pto-inst.hpp>\n"
        'extern "C" int selected_arch() {\n'
        "  return static_cast<int>(pto::NPUMemoryModel::Instance().GetArch());\n}\n"
    )
    output = tmp_path / "target.so"
    # Exercise both explicit roots and the SDK-managed default root.
    pto_isa_root = ensure_pto_isa_root() if core_type == "aic" else None
    output.write_bytes(
        KernelCompiler(platform).compile_incore(str(source), core_type=core_type, pto_isa_root=pto_isa_root)
    )
    library = ctypes.CDLL(str(output))
    assert library.selected_arch() == expected_arch
    library.simpler_cpu_sim_target.restype = ctypes.c_char_p
    assert library.simpler_cpu_sim_target() == family


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
