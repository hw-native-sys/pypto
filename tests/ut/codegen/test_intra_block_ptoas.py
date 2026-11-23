# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native entry points collected by the existing PTOAS CI job.

Reuse the IR contract assertions, opting into their compiler phase. The ordinary
IR suite still checks its contracts without requiring an installed assembler.
"""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from pypto import backend
from pypto.backend import BackendType

pytestmark = pytest.mark.ptoas_compile


@pytest.fixture(scope="module")
def contracts() -> Iterator[ModuleType]:
    source = Path(__file__).resolve().parents[1] / "ir/test_intra_block_ops.py"
    spec = importlib.util.spec_from_file_location("_intra_block_contracts", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def test_named_event_native_round_trip(contracts, tmp_path):
    contracts.test_named_event_emission(tmp_path, validate_native=True)


def test_named_scalar_pipe_native(contracts, tmp_path):
    contracts.test_scalar_pipe_emission(tmp_path, validate_native=True)


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_named_pipeline_native(contracts, tmp_path, stage, nested, dynamic):
    contracts.test_pipeline_preserves_named_event_fences(
        stage, nested, dynamic, tmp_path, validate_native=True
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
