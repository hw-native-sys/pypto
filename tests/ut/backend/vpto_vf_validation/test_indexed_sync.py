# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTOAS must order indexed accesses without frontend VV_ALL fences (#1730)."""

import re
import subprocess
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("count", [None, 0, 37, 64])
@pytest.mark.parametrize("hazard", ["raw", "war", "waw", "control"])
@pytest.mark.parametrize("separate", [False, True])
def test_indexed_barrier(
    enabled: bool, count: int | None, hazard: str, separate: bool, tmp_path: Path
) -> None:
    source = Path(__file__).with_name("indexed_sync.pto").read_text()
    scatter = next(line for line in source.splitlines() if "pto.vmi.vscatter" in line)
    gather = next(line for line in source.splitlines() if "pto.vmi.vgather" in line)
    pair = f"{scatter}\n{gather}"
    if hazard == "war":
        replacement = f"{gather}\n{scatter}"
        first, second, barrier = "vgather", "vscatter", "vld.vst"
    elif hazard == "waw":
        replacement = f"{scatter}\n{scatter}"
        source = source.replace("vstore %reread", "vstore %x")
        first, second, barrier = "vscatter", "vscatter", "vst.vst"
    elif hazard == "control":
        replacement = (
            "      pto.vmi.vstore %x, %dst[%c0_index] : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>\n"
            "      %reread = pto.vmi.vload %dst[%c0_index] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>"
        )
        first, second, barrier = "vstsx1", "vldsx1", "vst.vld"
    else:
        replacement = pair
        first, second, barrier = "vscatter", "vgather", "vst.vld"
    if separate:
        # Move only the second access by 256 elements. Indexed footprints have
        # unknown upper bounds, so PTOAS may conservatively retain their barrier.
        source = source.replace(
            "    pto.vecscope {",
            "    %other = pto.addptr %dst, %c256_index : <f32, ub> -> <f32, ub>\n    pto.vecscope {",
        )
        lines = replacement.splitlines()
        lines[1] = lines[1].replace("%dst", "%other")
        replacement = "\n".join(lines)
    source = source.replace(pair, replacement)
    if count is not None:
        source = source.replace(
            "    %c0_i32 =", f"    %count = arith.constant {count} : i32\n    %c0_i32 ="
        ).replace("cast %arg1 :", "cast %count :")
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    directory = tmp_path
    name = f"{hazard}-separate{separate}-count{count}-enabled{enabled}"
    input_path, output_path = directory / f"{name}.pto", directory / f"{name}.ll"
    input_path.write_text(source)
    result = subprocess.run(
        [
            compiler,
            str(input_path),
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir",
            "--vmi-fastmath=false",
            f"--enable-vecscope-mem-bar={str(enabled).lower()}",
            "-o",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    llvm = output_path.read_text()
    calls = [line for line in llvm.splitlines() if re.search(r"\bcall\b", line)]
    if hazard == "control" and separate:
        # Independent contiguous loads can be scheduled before the first store.
        assert sum("@llvm.hivm.vldsx1" in line for line in calls) == 2
        assert sum("@llvm.hivm.vstsx1" in line for line in calls) == 2
        assert not any("@llvm.hivm.mem.bar" in line for line in calls)
        return
    begin = next(i for i, line in enumerate(calls) if f"@llvm.hivm.{first}" in line)
    end = next(i for i in range(begin + 1, len(calls)) if f"@llvm.hivm.{second}" in calls[i])
    between = "\n".join(calls[begin + 1 : end])
    expected = enabled
    assert (f"@llvm.hivm.mem.bar.{barrier}" in between) == expected, between
    if not enabled:
        assert "call void @llvm.hivm.mem.bar" not in llvm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
