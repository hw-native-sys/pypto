# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""允许中文说明，同时保留对代码字符串、英文正文及其他语言的检查。"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def lint() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "check_english_only.py"
    spec = importlib.util.spec_from_file_location("check_english_only", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("suffix", "source", "violation_lines"),
    [
        (".py", '# \u4e2d\u6587\n"""\u8bf4\u660e"""\nvalue = 1\n', []),
        (".pyi", 'def run():\n    """\u8bf4\u660e"""\n    ...\n', []),
        (".py", 'class Example:\n    """\u8bf4\u660e"""\n', []),
        (".py", 'async def run():\n    """\u8bf4\u660e"""\n', []),
        (".py", '# \u4e2d\u6587\nvalue = "\u4e2d\u6587"\n', [2]),
        (".py", 'value = "# \u4e2d\u6587"\n', [1]),
        (".py", 'value = 1\n"""\u975e\u6587\u6863\u5b57\u7b26\u4e32"""\n', [2]),
        (".py", "# \u3042\u3044\u3046\nvalue = 1\n", [1]),
        (".py", "def broken(:\n    # \u4e2d\u6587\n", [2]),
        (".md", "English text\n```python\n# \u4e2d\u6587\nvalue = 1\n```\n", []),
        (".md", "\u4e2d\u6587\n```python\n# \u4e2d\u6587\n```\n", [1]),
        (".md", '```python\nvalue = "\u4e2d\u6587"\n```\n', [2]),
        (".md", "```text\n# \u4e2d\u6587\n```\n", [2]),
        (".cpp", "// \u4e2d\u6587\n", [1]),
    ],
)
def test_annotation_language_boundary(
    lint: ModuleType, tmp_path: Path, suffix: str, source: str, violation_lines: list[int]
) -> None:
    file = tmp_path / ("sample" + suffix)
    file.write_text(source, encoding="utf-8")
    passed, violations = lint.check_file_english_only(file)
    assert passed == (not violation_lines)
    assert [line for line, _ in violations] == violation_lines
