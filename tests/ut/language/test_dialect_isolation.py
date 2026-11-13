# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Import isolation between the dialect tier and the parser frontend.

The dialect tier (constructs, diagnostics, typing vocabulary) is shared by
both the tracing and the parsing frontends. Importing any dialect module must
not drag the parser package in; the parser must stay reachable lazily via
``pypto.language`` for existing ``pl.function`` / ``pl.parse`` spellings.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypto.pypto_core import InternalError

_PYTHON_DIR = Path(__file__).resolve().parents[3] / "python"


def _run_python(code: str) -> None:
    # Import isolation requires a fresh interpreter, while dependency and native
    # library paths still come from the caller's configured environment.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(_PYTHON_DIR), env.get("PYTHONPATH", "")) if path
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def _import_without_parser(module: str) -> None:
    code = (
        f"import sys; import {module}; "
        'assert "pypto.language.parser" not in sys.modules, '
        f'"importing {module} pulled in pypto.language.parser"'
    )
    _run_python(code)


def test_constructs_optimizations_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.constructs.optimizations")


def test_diagnostics_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.diagnostics")


def test_typing_arg_direction_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.typing.arg_direction")


def test_lazy_parser_spelling_still_works() -> None:
    code = (
        "import sys; import pypto.language as pl; "
        "assert pl.function is not None; "
        "assert pl.parse is not None; "
        "assert pl.parser.decorator.function is pl.function; "
        "assert 'pypto.language.parser' in sys.modules"
    )
    _run_python(code)


def test_diagnostics_exports_its_bug_class_family() -> None:
    """``BUG_CLASS_EXCEPTIONS`` is the shared classification callers widen on.

    It is defined and re-exported through the compatibility paths, so its
    absence from the canonical module's ``__all__`` made wildcard imports of the
    two spellings disagree.
    """
    from pypto.language import diagnostics  # noqa: PLC0415 - dialect-tier seam

    assert "BUG_CLASS_EXCEPTIONS" in diagnostics.__all__
    assert diagnostics.BUG_CLASS_EXCEPTIONS == (InternalError, AssertionError)


def test_lazy_parser_module_resolves_on_a_direct_first_access() -> None:
    # ``pl.parser`` as the very first parser-family touch must resolve to
    # the submodule itself — not go looking for a ``parser`` attribute
    # inside the parser package (an ordering the other lazy test masks,
    # because importing any parser submodule binds ``parser`` onto this
    # package as an import side effect).
    code = (
        "import sys; import pypto.language as pl; "
        "first = pl.parser; "
        "import pypto.language.parser as parser_module; "
        "assert first is parser_module; "
        "assert first.decorator.function is not None"
    )
    _run_python(code)


def test_import_subprocess_preserves_environment(monkeypatch, tmp_path) -> None:
    """An inherited import path and other environment variables reach the child."""
    (tmp_path / "inherited_dependency.py").write_text("VALUE = 42\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYPTO_TEST_IMPORT_SENTINEL", "preserved")
    _run_python(
        "import inherited_dependency, os; "
        "assert inherited_dependency.VALUE == 42; "
        "assert os.environ['PYPTO_TEST_IMPORT_SENTINEL'] == 'preserved'"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
