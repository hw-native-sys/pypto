# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Output isolation for direct runtime and JIT integration tests."""

import pytest


@pytest.fixture(autouse=True)
def _redirect_prog_build_dir(request, tmp_path, monkeypatch):
    """Keep normal JIT calls cacheable while isolating temporary artifacts."""
    # PYPTO_PROG_BUILD_DIR is an explicit output request and bypasses the JIT
    # cache. Use the working directory for test isolation instead, preserving
    # the shared fixture's --save-kernels behavior.
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    if not request.config.getoption("--save-kernels"):
        monkeypatch.chdir(tmp_path)
