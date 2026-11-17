# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native compiler probes; compilation does not establish numerical correctness."""

import re
import subprocess
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary

ELEMENTS = ("si8", "ui8", "si16", "ui16", "si32", "ui32", "f16", "bf16", "f32")


def _source(op, element, lanes, mask):
    source = Path(__file__).with_name("interleave.pto").read_text()
    active = {"empty": 0, "partial": lanes // 2, "full": lanes}[mask]
    return (
        source.replace("f32", element)
        .replace("8x", f"{lanes}x")
        .replace("vintlv", op)
        .replace("constant 8 : index", f"constant {active} : index")
    )


def _compile(source, tmp_path, llvm=False):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    input_path = tmp_path / "interleave.pto"
    output_path = tmp_path / ("interleave.ll" if llvm else "interleave.vpto")
    input_path.write_text(source)
    result = subprocess.run(
        [
            compiler,
            str(input_path),
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir" if llvm else "--emit-vpto",
            "-o",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_path.read_text()


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("op", ["vintlv", "vdintlv"])
@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("lanes", [8, 64, 128, 256])
@pytest.mark.parametrize("mask", ["empty", "partial", "full"])
def test_native_interleave_compiles(op, element, lanes, mask, tmp_path):
    # Include #1659's VL8: its compilation success must never be confused with
    # correct placement of active lanes. The separate A5 suite checks the bits.
    source = _source(op, element, lanes, mask)
    assert "pto.vmi.vsel" not in source
    llvm = _compile(source, tmp_path, llvm=True)
    assert "define void @interleave" in llvm


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("op", ["vintlv", "vdintlv"])
@pytest.mark.parametrize("element", ELEMENTS)
def test_native_mask_reaches_both_result_stores(op, element, tmp_path):
    bits = 8 if element.endswith("8") else 16 if element.endswith("16") else 32
    source = _source(op, element, 2048 // bits, "partial")
    lowered = _compile(source, tmp_path)
    pair = re.search(rf"(%[\w]+), (%[\w]+) = pto\.{op} ", lowered)
    assert pair, lowered
    selections = []
    for value in pair.groups():
        selected = re.search(rf"(%[\w]+) = pto.vsel {re.escape(value)}, (%[\w]+), (%[\w]+) :", lowered)
        assert selected, lowered
        result, zero, mask = selected.groups()
        assert re.search(rf"pto.vsts {re.escape(result)},", lowered), lowered
        selections.append((zero, mask))
    # Both outputs must select the same zero with the same governing mask.
    assert selections[0] == selections[1]
    full = _compile(_source(op, element, 2048 // bits, "full"), tmp_path)
    assert "pto.vsel " not in full, full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
