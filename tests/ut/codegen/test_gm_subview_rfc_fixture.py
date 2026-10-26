# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Executable PTOAS check for the RFC #3042 lowering fixture."""

import subprocess
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary


@pytest.mark.parametrize("arch", ["a2", "a3"])
@pytest.mark.parametrize("geometry", ["static", "dynamic"])
def test_gm_subview_fixture_is_accepted_by_ptoas(tmp_path: Path, arch: str, geometry: str) -> None:
    """A strict parent-window subview lowers to one result-tile TLOAD/TSTORE path."""
    ptoas = find_ptoas_binary()
    if ptoas is None:
        pytest.skip("PTOAS is not available")

    source = Path(__file__).parents[2] / "fixtures" / f"gm_subview_{geometry}.pto"
    output = tmp_path / "gm_subview_static.cpp"
    result = subprocess.run(
        [
            ptoas,
            str(source),
            f"--pto-arch={arch}",
            "--pto-level=level2",
            "--enable-insert-sync",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generated = output.read_text()
    assert generated.count("TLOAD(") == 1, generated
    assert generated.count("TSTORE(") == 1, generated
    assert generated.count("Tile<TileType::Vec, bfloat16_t, 2, 512,") == 1, generated
    if geometry == "static":
        assert "pto::Stride<3072, 3072, 3072, 1536, 1>" in generated, generated
    assert "TMOV(" not in generated, generated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
