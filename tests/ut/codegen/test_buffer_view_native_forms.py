# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native contracts required by static Buffer views over one byte allocation."""

import re
import subprocess
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary


def _descriptor(dtype: str, rows: int, cols: int) -> str:
    return (
        f"!pto.tile_buf<loc=vec, dtype={dtype}, rows={rows}, cols={cols}, "
        f"v_row={rows}, v_col={cols}, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )


def _view_program(arch: str, addressed: bool, dtype: str, byte_offset: int) -> str:
    # Full-width byte rows make this subview a contiguous 2048-byte window.
    # Reshape must preserve that byte extent even though element counts differ.
    root = _descriptor("ui8", 128, 32)
    window = _descriptor("ui8", 64, 32)
    element_bytes = 2 if dtype in {"i16", "f16", "bf16"} else 4
    typed = _descriptor(dtype, 64 // element_bytes, 32)
    reshaped = _descriptor(dtype, 32 // element_bytes, 64)
    output = f"!pto.partition_tensor_view<{32 // element_bytes}x64x{dtype}>"
    address = " addr = %address" if addressed else ""
    scalar = "1.0" if dtype in {"f16", "bf16", "f32"} else "1"
    return f"""module attributes {{pto.target_arch = "{arch}"}} {{
  func.func @buffer_view(%output: {output}) {{
    %zero = arith.constant 0 : index
    %row = arith.constant {byte_offset // 32} : index
    %address = arith.constant 4096 : i64
    %one = arith.constant {scalar} : {dtype}
    %root = pto.alloc_tile{address} : {root}
    %window = pto.subview %root[%row, %zero] sizes [64, 32] : {root} -> {window}
    %typed = pto.treshape %window : {window} -> {typed}
    pto.texpands ins(%one : {dtype}) outs(%typed : {typed})
    %reshaped = pto.treshape %typed : {typed} -> {reshaped}
    pto.tstore ins(%reshaped : {reshaped}) outs(%output : {output})
    return
  }}
}}
"""


@pytest.mark.parametrize("arch", ["a2", "a5"])
@pytest.mark.parametrize("addressed", [False, True], ids=["level2", "level3"])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "i32", "i16"])
@pytest.mark.parametrize("byte_offset", [0, 64])
def test_native_byte_storage_typed_views(
    tmp_path: Path, arch: str, addressed: bool, dtype: str, byte_offset: int
) -> None:
    """Subview and bitwise reshape retain alias edges without data movement."""
    ptoas = find_ptoas_binary()
    if ptoas is None:
        pytest.skip("PTOAS is not available")
    source = _view_program(arch, addressed, dtype, byte_offset)
    source_path = tmp_path / "buffer_view.pto"
    output_path = tmp_path / "buffer_view.cpp"
    source_path.write_text(source)
    result = subprocess.run(
        [
            ptoas,
            str(source_path),
            "-o",
            str(output_path),
            f"--pto-arch={arch}",
            f"--pto-level={'level3' if addressed else 'level2'}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cpp = output_path.read_text()
    assert source.count("pto.alloc_tile") == 1
    aliases = re.findall(r"TRESHAPE\((\w+), (\w+)\);", cpp)
    assert len(aliases) == 2, cpp
    assert aliases[1][1] == aliases[0][0], cpp
    # A strided subview may retain its parent's physical dimensions. Reshape
    # needs this contiguous window to have exactly the smaller physical size.
    assert re.search(rf"Tile<TileType::Vec, uint8_t, 64, 32,[^;]+> {aliases[0][1]};", cpp), cpp
    assert re.search(rf"TEXPANDS\({aliases[0][0]},", cpp), cpp
    assert re.search(rf"TSTORE\([^\n]*, {aliases[1][0]}\);", cpp), cpp
    assert "TMOV(" not in cpp and "TEXTRACT(" not in cpp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
