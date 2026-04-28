# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.Worker`` reuse logic.

Patches the ``_SimplerWorker`` alias in :mod:`pypto.runtime.worker` so tests
run without a device. The reuse path is observed by counting ``init`` /
``run`` / ``close`` calls on the mock.
"""

from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime import RunConfig, Worker

# ``execute_on_device`` is imported lazily inside individual tests to keep
# this module importable in environments where the underlying ``simpler``
# package is not installed (e.g. unit-tests CI). ``device_runner`` eagerly
# imports ``simpler.task_interface`` at module load.


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so Worker construction does not touch a device."""
    with patch("pypto.runtime.worker._SimplerWorker") as cls:
        instance = MagicMock()
        cls.return_value = instance
        yield instance


class TestLevelGuard:
    def test_level_3_rejected(self, fake_simpler_worker):
        with pytest.raises(ValueError, match="only supports level=2"):
            Worker(config=RunConfig(platform="a2a3sim"), level=3)

    def test_level_2_accepted(self, fake_simpler_worker):
        w = Worker(config=RunConfig(platform="a2a3sim"))
        assert w.level == 2
        w.close()


class TestLifecycleIdempotency:
    def test_auto_init_on_construction(self, fake_simpler_worker):
        Worker(config=RunConfig(platform="a2a3sim"))
        fake_simpler_worker.init.assert_called_once()

    def test_init_idempotent(self, fake_simpler_worker):
        w = Worker(config=RunConfig(platform="a2a3sim"))  # first init
        w.init()  # must not raise, must not double-init
        assert fake_simpler_worker.init.call_count == 1

    def test_close_idempotent(self, fake_simpler_worker):
        w = Worker(config=RunConfig(platform="a2a3sim"))
        w.close()
        w.close()  # second close is a no-op
        assert fake_simpler_worker.close.call_count == 1

    def test_close_then_reinit(self, fake_simpler_worker):
        w = Worker(config=RunConfig(platform="a2a3sim"))
        w.close()
        w.init()  # the wrapper supports re-init after close
        assert fake_simpler_worker.init.call_count == 2
        assert fake_simpler_worker.close.call_count == 1
        w.close()


class TestActiveWorkerLookup:
    def test_no_active_worker_outside_with_block(self, fake_simpler_worker):
        Worker(config=RunConfig(platform="a2a3sim"))  # constructed but not entered
        assert Worker.current(level=2, platform="a2a3sim", device_id=0, runtime="host_build_graph") is None

    def test_with_block_publishes_worker(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim")) as w:
            found = Worker.current(level=2, platform="a2a3sim", device_id=0, runtime="host_build_graph")
            assert found is w

    def test_exit_unpublishes(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim")):
            pass
        assert Worker.current(level=2, platform="a2a3sim", device_id=0, runtime="host_build_graph") is None

    def test_device_mismatch_returns_none(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim", device_id=0)):
            assert (
                Worker.current(level=2, platform="a2a3sim", device_id=1, runtime="host_build_graph") is None
            )

    def test_runtime_mismatch_returns_none(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim"), runtime="host_build_graph"):
            assert (
                Worker.current(
                    level=2,
                    platform="a2a3sim",
                    device_id=0,
                    runtime="tensormap_and_ringbuffer",
                )
                is None
            )

    def test_nested_distinct_binding_picks_topmost(self, fake_simpler_worker):
        # Distinct device_id — both Workers can coexist on the stack.
        with Worker(config=RunConfig(platform="a2a3sim", device_id=0)) as outer:
            with Worker(config=RunConfig(platform="a2a3sim", device_id=1)) as inner:
                # Lookup for device_id=1 finds inner.
                assert (
                    Worker.current(level=2, platform="a2a3sim", device_id=1, runtime="host_build_graph")
                    is inner
                )
                # Lookup for device_id=0 still finds the outer Worker — the
                # filter walks the whole stack, not just the topmost entry.
                assert (
                    Worker.current(level=2, platform="a2a3sim", device_id=0, runtime="host_build_graph")
                    is outer
                )

    def test_nested_same_binding_rejected(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim", device_id=0)):
            with pytest.raises(ValueError, match="already active in an enclosing scope"):
                with Worker(config=RunConfig(platform="a2a3sim", device_id=0)):
                    pass

    def test_with_block_closes_on_exit(self, fake_simpler_worker):
        with Worker(config=RunConfig(platform="a2a3sim")):
            pass
        fake_simpler_worker.close.assert_called_once()


# ``execute_on_device`` lives in ``device_runner`` which eagerly imports the
# ``simpler`` package. The Worker-only tests above mock just the
# ``simpler.Worker`` class via ``_SimplerWorker`` and do not need
# ``device_runner`` loaded — but the tests in this class invoke
# ``execute_on_device`` directly, so they are skipped when ``simpler`` is not
# installed (e.g. unit-tests CI).
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True


@pytest.mark.skipif(not _has_simpler, reason="execute_on_device requires the simpler package")
class TestExecuteOnDeviceReuse:
    """Verify ``execute_on_device`` reuses an active Worker rather than constructing a new one."""

    def test_reuse_skips_init_and_close(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        chip_callable = MagicMock(name="chip_callable")
        orch_args = MagicMock(name="orch_args")

        with Worker(config=RunConfig(platform="a2a3sim")):
            # Reset call counters after the with-block's auto-init.
            fake_simpler_worker.init.reset_mock()
            fake_simpler_worker.close.reset_mock()
            fake_simpler_worker.run.reset_mock()

            with patch("pypto.runtime.device_runner.CallConfig", MagicMock):
                execute_on_device(
                    chip_callable,
                    orch_args,
                    platform="a2a3sim",
                    runtime_name="host_build_graph",
                    device_id=0,
                )

            # Reuse path: the active Worker's run was invoked, no new init/close.
            assert fake_simpler_worker.run.call_count == 1
            assert fake_simpler_worker.init.call_count == 0
            assert fake_simpler_worker.close.call_count == 0

    def test_no_active_worker_uses_one_shot_path(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        chip_callable = MagicMock(name="chip_callable")
        orch_args = MagicMock(name="orch_args")

        # No `with` block — execute_on_device must construct its own Worker
        # (one init + one run + one close on the underlying simpler.Worker).
        # The one-shot path imports simpler.Worker directly into device_runner,
        # so patch that name in addition to the wrapper's _SimplerWorker.
        one_shot = MagicMock()
        with (
            patch("pypto.runtime.device_runner.CallConfig", MagicMock),
            patch("pypto.runtime.device_runner.Worker", return_value=one_shot),
        ):
            execute_on_device(
                chip_callable,
                orch_args,
                platform="a2a3sim",
                runtime_name="host_build_graph",
                device_id=0,
            )
        assert one_shot.init.call_count == 1
        assert one_shot.run.call_count == 1
        assert one_shot.close.call_count == 1

    def test_level_mismatch_rejected(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        with pytest.raises(ValueError, match="only supports level=2"):
            execute_on_device(
                MagicMock(),
                MagicMock(),
                platform="a2a3sim",
                runtime_name="host_build_graph",
                device_id=0,
                level=3,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
