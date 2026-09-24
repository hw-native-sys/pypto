# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pytest configuration for JIT unit tests."""

import sys
from functools import wraps
from pathlib import Path

import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import passes

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure backend before each test."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


@pytest.fixture(autouse=True)
def pass_verification_context(monkeypatch, pass_verification_instruments):
    """Check every executed pass without turning cache lookups into diagnostic requests.

    JIT deliberately bypasses its cache when an outer context has instruments.
    Install the shared checks at pipeline execution instead, after that decision,
    preserving caller instruments and all pass settings.
    """
    run_passes = PassManager.run_passes

    @wraps(run_passes)
    def run_verified(self, *args, **kwargs):
        outer = passes.PassContext.current() or passes.PassContext([])
        with passes.PassContext(
            [*outer.get_instruments(), *pass_verification_instruments],
            outer.get_verification_level(),
            outer.get_diagnostic_phase(),
            outer.get_disabled_diagnostics(),
            outer.get_memory_planner(),
            outer.get_enable_pypto_l0c_double_buffer(),
            outer.get_runtime(),
            outer.get_enable_buffer_ir(),
        ):
            return run_passes(self, *args, **kwargs)

    monkeypatch.setattr(PassManager, "run_passes", run_verified)


@pytest.fixture(autouse=True)
def _redirect_prog_build_dir(tmp_path, monkeypatch):
    """Isolate artifacts without an explicit output request that bypasses JIT caching."""
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
