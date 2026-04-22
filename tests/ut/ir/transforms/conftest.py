# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for transforms tests."""

import pytest
from pypto import backend as _backend
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def ascend950_backend():
    """Configure Ascend950 backend for every test, then reset.

    Replaces the per-file ``_setup_backend`` autouse fixture that is
    duplicated across expand_mixed_kernel test files.
    """
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend950)
    try:
        yield
    finally:
        _backend.reset_for_testing()
