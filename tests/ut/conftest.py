# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for unit tests."""

import os

import pytest
from pypto import backend as _backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes


@pytest.fixture
def ascend_backend(request):
    """Configure an Ascend backend for the duration of a test, then reset.

    Use either as a plain fixture (defaults to ``Ascend910B``) or via
    ``pytest.mark.parametrize("ascend_backend", [...], indirect=True)`` to
    cycle through multiple backends. Replaces the per-test
    ``backend.reset_for_testing()`` + ``backend.set_backend_type(...)`` pair
    that is otherwise duplicated across pass / codegen tests.
    """
    backend_type = getattr(request, "param", BackendType.Ascend910B)
    _backend.reset_for_testing()
    _backend.set_backend_type(backend_type)
    try:
        yield backend_type
    finally:
        _backend.reset_for_testing()


@pytest.fixture
def default_pass_manager():
    """Return the default-strategy PassManager.

    Use this in tests that want to run the production pipeline without
    constructing the manager inline. Strategy-specific tests (covering
    ``DebugTileOptimization`` etc.) should keep building the manager
    themselves so the choice stays visible at the test site.
    """
    return PassManager.get_strategy(OptimizationStrategy.Default)


@pytest.fixture(autouse=True)
def _reset_backend_singleton():
    """Reset the process-global backend singleton around every unit test.

    The backend type can only be set once per process (``set_backend_type``
    rejects a change once set). Tests that call ``ir.compile`` / codegen without
    explicitly choosing a backend rely on a clean singleton, so a sibling test
    that left e.g. ``Ascend950`` set would make them fail with "Backend type
    already set" — an order-dependent flake under pytest-xdist. Resetting before
    and after each test makes every test start from a clean slate regardless of
    scheduling. Tests that set a backend (inline or via ``ascend_backend``) are
    unaffected: the reset runs first, then their own set wins.
    """
    _backend.reset_for_testing()
    try:
        yield
    finally:
        _backend.reset_for_testing()


@pytest.fixture(autouse=True)
def _redirect_prog_build_dir(tmp_path, monkeypatch):
    """Redirect ir.compile() / @pl.jit artifacts into pytest's per-test tmp dir.

    Without an explicit ``output_dir``, ``ir.compile()`` writes generated
    kernels and pass dumps to ``build_output/<name>_<timestamp>`` relative to
    the working directory. Under pytest that accumulates stale directories in
    the repo / build tree. Pointing ``PYPTO_PROG_BUILD_DIR`` at a
    ``build_output`` dir inside pytest's per-test ``tmp_path`` keeps every
    test's artifacts isolated and auto-cleaned by pytest.
    """
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Enable pass verification and optional roundtrip checking for all pass executions.

    The behavior is controlled by the PYPTO_VERIFY_LEVEL environment variable:

    - ``roundtrip`` (default) — BEFORE_AND_AFTER property verification + print→parse
      roundtrip structural-equality check after every pass.
    - ``basic`` — BEFORE_AND_AFTER property verification only (faster, no roundtrip).
    - ``none`` — no pass verification at all (fastest, for debugging only).
    """
    level_str = os.environ.get("PYPTO_VERIFY_LEVEL", "roundtrip").lower()

    instruments: list[passes.PassInstrument] = []

    if level_str != "none":
        instruments.append(passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER))

    if level_str == "roundtrip":
        from pypto.ir.instruments import make_roundtrip_instrument  # noqa: PLC0415

        instruments.append(make_roundtrip_instrument())

    with passes.PassContext(instruments):
        yield
