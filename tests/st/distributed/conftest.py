# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared pytest behavior for distributed system tests."""

from typing import Any

import pytest
from harness import card_lock


def pytest_configure(config: pytest.Config) -> None:
    """Serialize the device phase so xdist workers can overlap their compiles.

    Installed unconditionally: with one worker the lock is uncontended and costs
    a file open per dispatch, and it still keeps two concurrent runs that were
    lent the same cards from driving them at once. See ``card_lock`` for what the
    guarded region deliberately excludes and why.

    ``--codegen-only`` layers its own patches over these per test, so no lock is
    taken on a run that never reaches a card.
    """
    if config.getoption("--codegen-only"):
        return
    device_ids = [int(part) for part in str(config.getoption("--device")).split(",") if part.strip()]
    card_lock.install(device_ids)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the runtime and drop the lock, however deep this worker left it."""
    del config
    card_lock.uninstall()


@pytest.fixture(autouse=True)
def disable_runtime_execution_in_codegen_only(request, monkeypatch) -> None:
    """Let distributed tests compile, then skip at every public execution edge."""
    if not request.config.getoption("--codegen-only"):
        return

    from pypto.ir import DistributedCompiledProgram  # noqa: PLC0415
    from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

    def skip_execution(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pytest.skip("--codegen-only disables distributed runtime execution")

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", skip_execution)
    monkeypatch.setattr("pypto.runtime.distributed_runner._execute_distributed", skip_execution)
    monkeypatch.setattr(DistributedCompiledProgram, "prepare", skip_execution)
    monkeypatch.setattr(DistributedWorker, "__init__", skip_execution)
