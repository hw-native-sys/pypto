# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check that optional-runtime isolation cannot silently miss import attempts."""

import pytest


@pytest.mark.parametrize("optimize", ["0", "1", "2"])
def test_optional_runtime_guard_records_all_caught_imports(
    run_without_optional_runtime, monkeypatch, optimize
):
    """The guard must record every blocked root even when ImportError is swallowed."""
    modules = ["torch_npu", "simpler", "simpler_setup", "_task_interface", "pypto._torch_npu"]
    monkeypatch.setenv("PYTHONOPTIMIZE", optimize)
    # Exercise the complete blocked-root list in each optimization mode without
    # starting a separate interpreter (and importing PyPTO) for every root.
    result = run_without_optional_runtime(f"""
import importlib
for module in {modules!r}:
    try:
        importlib.import_module(module)
    except ImportError:
        pass
""")
    assert result.returncode != 0
    assert f"Optional runtime imports attempted: {modules!r}" in result.stderr


@pytest.mark.parametrize("route", ["statement", "dynamic"])
@pytest.mark.parametrize(
    "module,optimize",
    [
        ("torch_npu", "0"),
        ("torch_npu", "1"),
        ("torch_npu", "2"),
        ("pypto._torch_npu", "0"),
    ],
)
def test_optional_runtime_import_attempt_is_detected(
    run_without_optional_runtime, monkeypatch, module, route, optimize
):
    """Uncaught imports fail through both Python import APIs, for root and dotted names."""
    # The optimization-level behavior is independent of the requested module:
    # the explicit guard checks run for all roots in the caught-import test above.
    monkeypatch.setenv("PYTHONOPTIMIZE", optimize)
    source = (
        f"import {module}"
        if route == "statement"
        else f"import importlib; importlib.import_module({module!r})"
    )
    result = run_without_optional_runtime(source)
    assert result.returncode != 0
    assert "Optional runtime imports attempted:" in result.stderr
    assert module in result.stderr


def test_optional_runtime_guard_allows_program_configuration(run_without_optional_runtime):
    """Ordinary program configuration works while optional runtimes are unavailable."""
    result = run_without_optional_runtime("""
import importlib
from pypto.runtime import RunConfig

assert importlib.import_module('json').loads('{"ok": true}')['ok']
config = RunConfig(platform='a2a3sim')
assert config.compile_kwargs()['backend_type'] == config.backend_type
assert config.compile_kwargs()['execution_mode'] is config.execution_mode
assert 'platform' not in config.compile_kwargs()
""")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
