# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for clang-tidy diff filtering."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_clang_tidy() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "clang_tidy.py"
    spec = importlib.util.spec_from_file_location("pypto_clang_tidy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clang_tidy = _load_clang_tidy()


@pytest.mark.parametrize(
    "path",
    [
        "tests/lint/clang_tidy.py",
        ".clang-tidy",
        ".github/workflows/ci.yml",
        ".github/workflows/daily_ci.yml",
        "CMakeLists.txt",
        "python/bindings/CMakeLists.txt",
        "cmake/libbacktrace.cmake",
        "3rdparty/libbacktrace",
        "3rdparty/msgpack-c",
        "runtime",
    ],
)
def test_lint_infrastructure_change_requires_full_check(path: str):
    assert clang_tidy._get_full_check_triggers({path}) == [path]


def test_source_change_does_not_require_full_check():
    assert clang_tidy._get_full_check_triggers({"src/ir/arith/canonical_simplify.cpp"}) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
