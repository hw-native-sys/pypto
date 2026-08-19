# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.ChipWorker`` reuse logic.

Patches the ``_SimplerWorker`` alias in :mod:`pypto.runtime.worker` so tests
run without a device. The reuse path is observed by counting ``init`` /
``run`` / ``close`` calls on the mock.
"""

import gc
from unittest.mock import MagicMock, patch

import pytest
from pypto.runtime import ChipWorker, RunConfig

# ``execute_on_device`` is imported lazily inside individual tests to keep
# this module importable in environments where the underlying ``simpler``
# package is not installed (e.g. unit-tests CI). ``device_runner`` eagerly
# imports ``simpler.task_interface`` at module load.


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so ChipWorker construction does not touch a device."""
    with (
        patch("pypto.runtime.worker._SimplerWorker") as cls,
        # init() builds a prewarm CallConfig; patch the cache so no simpler import happens.
        patch("pypto.runtime.worker._SimplerCallConfig", MagicMock()),
    ):
        instance = MagicMock()
        cls.return_value = instance
        yield instance


@pytest.fixture
def fake_worker_cls():
    """Patch and expose the simpler worker constructor for capability assertions."""
    with (
        patch("pypto.runtime.worker._SimplerWorker") as cls,
        patch("pypto.runtime.worker._SimplerCallConfig", MagicMock()),
    ):
        yield cls


class TestSdmaCapability:
    def test_constructor_forwards_enabled_sdma(self, fake_worker_cls):
        ChipWorker(config=RunConfig(platform="a2a3"), enable_sdma=True)

        fake_worker_cls.assert_called_once_with(
            level=2,
            device_id=0,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            enable_sdma=True,
        )

    def test_constructor_defaults_sdma_off(self, fake_worker_cls):
        ChipWorker(config=RunConfig(platform="a2a3"))

        fake_worker_cls.assert_called_once_with(
            level=2,
            device_id=0,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            enable_sdma=False,
        )


class TestLevelGuard:
    def test_level_3_rejected(self, fake_simpler_worker):
        with pytest.raises(ValueError, match="only supports level=2"):
            ChipWorker(config=RunConfig(platform="a2a3sim"), level=3)

    def test_level_2_accepted(self, fake_simpler_worker):
        w = ChipWorker(config=RunConfig(platform="a2a3sim"))
        assert w.level == 2
        w.close()


class TestLifecycleIdempotency:
    def test_auto_init_on_construction(self, fake_simpler_worker):
        ChipWorker(config=RunConfig(platform="a2a3sim"))
        fake_simpler_worker.init.assert_called_once()

    def test_init_prewarms_arena_cache(self, fake_simpler_worker):
        ChipWorker(config=RunConfig(platform="a2a3sim"))
        # init() builds the prebuilt runtime-arena so the first dispatch doesn't
        # pay the ~800ms cold build. Sizing is the runtime's own default (a bare
        # CallConfig): dispatch takes ring sizing from the per-call RunConfig, so
        # prewarming this worker's config would build an arena nobody asks for.
        assert fake_simpler_worker.init.call_args.kwargs["prewarm_config"] is not None

    def test_init_idempotent(self, fake_simpler_worker):
        w = ChipWorker(config=RunConfig(platform="a2a3sim"))  # first init
        w.init()  # must not raise, must not double-init
        assert fake_simpler_worker.init.call_count == 1

    def test_close_idempotent(self, fake_simpler_worker):
        w = ChipWorker(config=RunConfig(platform="a2a3sim"))
        w.close()
        w.close()  # second close is a no-op
        assert fake_simpler_worker.close.call_count == 1

    def test_close_then_reinit_constructs_fresh_simpler_worker(self, fake_worker_cls):
        first = MagicMock(name="first_simpler_worker")
        second = MagicMock(name="second_simpler_worker")
        fake_worker_cls.side_effect = [first, second]
        w = ChipWorker(config=RunConfig(platform="a2a3sim"))
        w.close()
        w.init()  # the wrapper supports re-init after close
        first.init.assert_called_once()
        first.close.assert_called_once()
        second.init.assert_called_once()
        assert fake_worker_cls.call_count == 2
        w.close()

    def test_init_failure_closes_terminal_impl_and_rebuilds(self, fake_worker_cls):
        first = MagicMock(name="failed_simpler_worker")
        second = MagicMock(name="replacement_simpler_worker")
        first.init.side_effect = RuntimeError("startup failed")
        fake_worker_cls.side_effect = [first, second]
        w = ChipWorker(config=RunConfig(platform="a2a3sim"), auto_init=False)

        with pytest.raises(RuntimeError, match="startup failed"):
            w.init()
        first.close.assert_called_once()

        w.init()
        second.init.assert_called_once()
        w.close()

    def test_init_cleanup_failure_requires_close_retry_before_rebuild(self, fake_worker_cls):
        first = MagicMock(name="failed_simpler_worker")
        second = MagicMock(name="replacement_simpler_worker")
        first.init.side_effect = RuntimeError("startup failed")
        first.close.side_effect = [RuntimeError("cleanup pending"), None]
        fake_worker_cls.side_effect = [first, second]
        w = ChipWorker(config=RunConfig(platform="a2a3sim"), auto_init=False)

        with pytest.raises(RuntimeError, match="startup failed"):
            w.init()
        with pytest.raises(RuntimeError, match="finish cleanup"):
            w.init()
        w.close()  # retries the same simpler cleanup journal
        assert first.close.call_count == 2

        w.init()
        second.init.assert_called_once()
        w.close()

    def test_auto_init_cleanup_failure_is_retried_when_constructor_is_abandoned(self, fake_worker_cls):
        failed = MagicMock(name="failed_simpler_worker")
        cleanup_attempts = 0

        def fail_init(**_kwargs) -> None:
            raise RuntimeError("startup failed")

        def fail_cleanup_twice() -> None:
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            if cleanup_attempts < 3:
                raise RuntimeError("cleanup pending")

        failed.init.side_effect = fail_init
        failed.close.side_effect = fail_cleanup_twice
        fake_worker_cls.return_value = failed

        def construct_and_abandon() -> None:
            with pytest.raises(RuntimeError, match="startup failed"):
                ChipWorker(config=RunConfig(platform="a2a3sim"))

        construct_and_abandon()
        gc.collect()

        # init() and __init__ each attempted cleanup and both failed. The
        # impl-only finalizer makes the third attempt after construction raises
        # and no caller-visible ChipWorker exists.
        assert failed.close.call_count == 3

    def test_close_failure_retains_impl_until_retry_succeeds(self, fake_worker_cls):
        first = MagicMock(name="first_simpler_worker")
        second = MagicMock(name="replacement_simpler_worker")
        first.close.side_effect = [RuntimeError("cleanup pending"), None]
        fake_worker_cls.side_effect = [first, second]
        w = ChipWorker(config=RunConfig(platform="a2a3sim"))

        with pytest.raises(RuntimeError, match="cleanup pending"):
            w.close()
        with pytest.raises(RuntimeError, match="finish cleanup"):
            w.init()
        w.close()
        assert first.close.call_count == 2

        w.init()
        assert w._impl is second
        w.close()


class TestActiveChipWorkerLookup:
    def test_no_active_worker_outside_with_block(self, fake_simpler_worker):
        ChipWorker(config=RunConfig(platform="a2a3sim"))  # constructed but not entered
        assert (
            ChipWorker.current(level=2, platform="a2a3sim", device_id=0, runtime="tensormap_and_ringbuffer")
            is None
        )

    def test_with_block_publishes_worker(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim")) as w:
            found = ChipWorker.current(
                level=2, platform="a2a3sim", device_id=0, runtime="tensormap_and_ringbuffer"
            )
            assert found is w

    def test_exit_unpublishes(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim")):
            pass
        assert (
            ChipWorker.current(level=2, platform="a2a3sim", device_id=0, runtime="tensormap_and_ringbuffer")
            is None
        )

    def test_device_mismatch_returns_none(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim", device_id=0)):
            assert (
                ChipWorker.current(
                    level=2, platform="a2a3sim", device_id=1, runtime="tensormap_and_ringbuffer"
                )
                is None
            )

    def test_runtime_mismatch_returns_none(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim"), runtime="host_build_graph"):
            assert (
                ChipWorker.current(
                    level=2,
                    platform="a2a3sim",
                    device_id=0,
                    runtime="tensormap_and_ringbuffer",
                )
                is None
            )

    def test_nested_distinct_binding_picks_topmost(self, fake_simpler_worker):
        # Distinct device_id — both ChipWorkers can coexist on the stack.
        with ChipWorker(config=RunConfig(platform="a2a3sim", device_id=0)) as outer:
            with ChipWorker(config=RunConfig(platform="a2a3sim", device_id=1)) as inner:
                # Lookup for device_id=1 finds inner.
                assert (
                    ChipWorker.current(
                        level=2, platform="a2a3sim", device_id=1, runtime="tensormap_and_ringbuffer"
                    )
                    is inner
                )
                # Lookup for device_id=0 still finds the outer ChipWorker — the
                # filter walks the whole stack, not just the topmost entry.
                assert (
                    ChipWorker.current(
                        level=2, platform="a2a3sim", device_id=0, runtime="tensormap_and_ringbuffer"
                    )
                    is outer
                )

    def test_nested_same_binding_rejected(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim", device_id=0)):
            with pytest.raises(ValueError, match="already active in an enclosing scope"):
                with ChipWorker(config=RunConfig(platform="a2a3sim", device_id=0)):
                    pass

    def test_with_block_closes_on_exit(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim")):
            pass
        fake_simpler_worker.close.assert_called_once()

    def test_sdma_required_rejects_ordinary_active_worker(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim")):
            with pytest.raises(
                RuntimeError,
                match="active ChipWorker was created without enable_sdma=True",
            ):
                ChipWorker.current(
                    level=2,
                    platform="a2a3sim",
                    device_id=0,
                    runtime="tensormap_and_ringbuffer",
                    require_sdma=True,
                )

    def test_sdma_enabled_worker_accepts_ordinary_lookup(self, fake_simpler_worker):
        with ChipWorker(config=RunConfig(platform="a2a3sim"), enable_sdma=True) as worker:
            assert (
                ChipWorker.current(
                    level=2,
                    platform="a2a3sim",
                    device_id=0,
                    runtime="tensormap_and_ringbuffer",
                )
                is worker
            )


# ``execute_on_device`` lives in ``device_runner`` which eagerly imports the
# ``simpler`` package. The ChipWorker-only tests above mock just the
# ``simpler.ChipWorker`` class via ``_SimplerWorker`` and do not need
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
    """Verify ``execute_on_device`` reuses an active ChipWorker rather than constructing a new one."""

    def test_reuse_skips_init_and_close(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        chip_callable = MagicMock(name="chip_callable")
        orch_args = [MagicMock(name="host_tensor")]
        wire_args = MagicMock(name="TaskArgs")

        with ChipWorker(config=RunConfig(platform="a2a3sim")):
            # Reset call counters after the with-block's auto-init.
            fake_simpler_worker.init.reset_mock()
            fake_simpler_worker.close.reset_mock()
            fake_simpler_worker.run.reset_mock()

            with (
                patch("pypto.runtime.device_runner.CallConfig", MagicMock),
                patch("pypto.runtime.runner._coerced_to_orch_args", return_value=wire_args) as pack,
            ):
                execute_on_device(
                    chip_callable,
                    orch_args,
                    platform="a2a3sim",
                    runtime_name="tensormap_and_ringbuffer",
                    device_id=0,
                )

            # Reuse path: the active ChipWorker's run was invoked, no new init/close.
            assert fake_simpler_worker.run.call_count == 1
            pack.assert_called_once_with(orch_args, fake_simpler_worker)
            assert fake_simpler_worker.run.call_args.args[1] is wire_args
            assert fake_simpler_worker.init.call_count == 0
            assert fake_simpler_worker.close.call_count == 0

    def test_no_active_worker_uses_one_shot_path(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        chip_callable = MagicMock(name="chip_callable")
        orch_args = [MagicMock(name="host_tensor")]
        wire_args = MagicMock(name="TaskArgs")

        # No `with` block — execute_on_device must construct its own ChipWorker
        # (one init + one run + one close on the underlying simpler.ChipWorker).
        # The one-shot path imports simpler.ChipWorker directly into device_runner,
        # so patch that name in addition to the wrapper's _SimplerWorker.
        one_shot = MagicMock()
        with (
            patch("pypto.runtime.device_runner.CallConfig", MagicMock),
            patch("pypto.runtime.device_runner.Worker", return_value=one_shot) as worker_cls,
            patch("pypto.runtime.runner._coerced_to_orch_args", return_value=wire_args) as pack,
        ):
            execute_on_device(
                chip_callable,
                orch_args,
                platform="a2a3sim",
                runtime_name="host_build_graph",
                device_id=0,
            )
        worker_cls.assert_called_once_with(
            level=2,
            device_id=0,
            platform="a2a3sim",
            runtime="host_build_graph",
            enable_sdma=False,
        )
        assert one_shot.init.call_count == 1
        assert one_shot.run.call_count == 1
        pack.assert_called_once_with(orch_args, one_shot)
        assert one_shot.run.call_args.args[1] is wire_args
        assert one_shot.close.call_count == 1

    def test_no_active_worker_forwards_enabled_sdma(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        with (
            patch("pypto.runtime.device_runner.CallConfig", MagicMock),
            patch("pypto.runtime.device_runner.Worker") as worker_cls,
        ):
            execute_on_device(
                MagicMock(),
                MagicMock(),
                platform="a2a3",
                runtime_name="tensormap_and_ringbuffer",
                device_id=0,
                enable_sdma=True,
            )

        worker_cls.assert_called_once_with(
            level=2,
            device_id=0,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            enable_sdma=True,
        )

    def test_sdma_required_dispatch_rejects_ordinary_active_worker(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        with (
            ChipWorker(config=RunConfig(platform="a2a3sim")),
            patch("pypto.runtime.device_runner.CallConfig", MagicMock),
            patch("pypto.runtime.device_runner.Worker") as worker_cls,
        ):
            with pytest.raises(
                RuntimeError,
                match="active ChipWorker was created without enable_sdma=True",
            ):
                execute_on_device(
                    MagicMock(),
                    MagicMock(),
                    platform="a2a3sim",
                    runtime_name="tensormap_and_ringbuffer",
                    device_id=0,
                    enable_sdma=True,
                )

        worker_cls.assert_not_called()

    def test_sdma_enabled_active_worker_runs_ordinary_dispatch(self, fake_simpler_worker):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        with ChipWorker(config=RunConfig(platform="a2a3sim"), enable_sdma=True):
            fake_simpler_worker.run.reset_mock()
            with patch("pypto.runtime.device_runner.CallConfig", MagicMock):
                execute_on_device(
                    MagicMock(),
                    MagicMock(),
                    platform="a2a3sim",
                    runtime_name="tensormap_and_ringbuffer",
                    device_id=0,
                )

        fake_simpler_worker.run.assert_called_once()

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
