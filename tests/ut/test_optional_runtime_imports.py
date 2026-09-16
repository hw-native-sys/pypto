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


@pytest.mark.parametrize("module", ["torch_npu", "simpler", "simpler_setup", "pypto._torch_npu"])
@pytest.mark.parametrize("route", ["statement", "dynamic", "caught_dynamic"])
def test_optional_runtime_import_attempt_is_detected(run_without_optional_runtime, module, route):
    """Both import APIs must fail the guard, even when the caller catches ImportError."""
    if route == "statement":
        source = f"import {module}"
    elif route == "dynamic":
        source = f"import importlib; importlib.import_module({module!r})"
    else:
        source = f"""
import importlib
try:
    importlib.import_module({module!r})
except ImportError:
    pass
"""
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
assert config.compile_kwargs()['platform'] == 'a2a3sim'
""")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
