# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep the operator docstring parity check active after implementation splits."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def lint() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "check_op_docstring_parity.py"
    spec = importlib.util.spec_from_file_location("check_op_docstring_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("split", [False, True], ids=["monolithic", "split"])
@pytest.mark.parametrize(
    ("docstring", "problem"),
    [
        (None, "has no docstring at all"),
        ("Add tiles.", "drops the explanation its IR twin carries"),
        ("Add tiles.\n\nScalar operands broadcast to the tile shape.", None),
    ],
    ids=["missing", "summary-only", "complete"],
)
def test_docstring_parity_survives_module_split(
    lint: ModuleType, tmp_path: Path, split: bool, docstring: str | None, problem: str | None
) -> None:
    # The public facade has no definitions after a split. The lint must still
    # detect missing explanations and point at the private implementation.
    source_name = "_tile_elementwise.py" if split else "tile_ops.py"
    for layer in ("ir", "language"):
        directory = tmp_path / "python" / "pypto" / layer / "op"
        directory.mkdir(parents=True)
        if split:
            (directory / "tile_ops.py").write_text("from ._tile_elementwise import add\n")
        doc = "Add tiles.\n\nScalar operands broadcast to the tile shape." if layer == "ir" else docstring
        body = f"    {doc!r}\n" if doc is not None else ""
        (directory / source_name).write_text(f"def add(lhs, rhs):\n{body}    return lhs\n")

    findings = lint.find_violations(tmp_path)
    expected_path = f"python/pypto/language/op/{source_name}"
    assert findings == ([] if problem is None else [("tile", "add", 1, expected_path, problem)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
