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
from pathlib import Path

import pytest
from pypto import backend
from pypto.backend import BackendType

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
def pass_verification_context():
    """Use default pass verification for JIT tests.

    JIT-generated programs use FunctionType.Opaque for entry functions so that
    OutlineIncoreScopes can outline ``with pl.incore():`` scopes and promote the
    function to Orchestration.  This means pass verification now works correctly.
    """
    yield
