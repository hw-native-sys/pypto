# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.runner.RunConfig`` and DFX plumbing."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from pypto.backend import BackendType
from pypto.runtime.runner import RunConfig, _DfxOpts


class TestRunConfigPlatformResolution:
    """Verify platform/backend synchronization in ``RunConfig``."""

    @pytest.mark.parametrize(
        ("platform", "expected_backend"),
        [
            ("a2a3", BackendType.Ascend910B),
            ("a2a3sim", BackendType.Ascend910B),
            ("a5", BackendType.Ascend950),
            ("a5sim", BackendType.Ascend950),
        ],
    )
    def test_platform_selects_matching_backend(self, platform, expected_backend):
        cfg = RunConfig(platform=platform)

        assert cfg.platform == platform
        assert cfg.backend_type == expected_backend

    def test_enable_l2_swimlane_forces_save_kernels(self):
        cfg = RunConfig(platform="a5", enable_l2_swimlane=True)

        assert cfg.platform == "a5"
        assert cfg.backend_type == BackendType.Ascend950
        assert cfg.save_kernels is True


class TestRunConfigDfxFlags:
    """Verify the four DFX flags are independent and propagate correctly."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_l2_swimlane": True},
            {"enable_dump_tensor": True},
            {"enable_pmu": 2},
            {"enable_dep_gen": True},
        ],
    )
    def test_any_dfx_flag_forces_save_kernels(self, kwargs):
        cfg = RunConfig(platform="a5", **kwargs)
        assert cfg.save_kernels is True, f"save_kernels not auto-enabled for {kwargs}"
        assert cfg.any_dfx_enabled() is True

    def test_no_dfx_leaves_save_kernels_default(self):
        cfg = RunConfig(platform="a5")
        assert cfg.save_kernels is False
        assert cfg.any_dfx_enabled() is False

    def test_pmu_zero_means_disabled(self):
        cfg = RunConfig(platform="a5", enable_pmu=0)
        assert cfg.any_dfx_enabled() is False
        assert cfg.save_kernels is False

    def test_pmu_positive_means_enabled(self):
        # The runtime maps enable_pmu > 0 to "enabled, event type N".
        cfg = RunConfig(platform="a5", enable_pmu=4)
        assert cfg.any_dfx_enabled() is True
        assert cfg.enable_pmu == 4
        assert cfg.save_kernels is True

    def test_dfx_flags_are_independent(self):
        # Enabling one flag must not implicitly enable another.
        cfg = RunConfig(platform="a5", enable_dep_gen=True)
        assert cfg.enable_dep_gen is True
        assert cfg.enable_l2_swimlane is False
        assert cfg.enable_dump_tensor is False
        assert cfg.enable_pmu == 0

    def test_dfx_opts_from_run_config_carries_all_four(self):
        cfg = RunConfig(
            platform="a5",
            enable_l2_swimlane=True,
            enable_dump_tensor=True,
            enable_pmu=2,
            enable_dep_gen=True,
        )
        opts = _DfxOpts.from_run_config(cfg)
        assert opts.enable_l2_swimlane is True
        assert opts.enable_dump_tensor is True
        assert opts.enable_pmu == 2
        assert opts.enable_dep_gen is True
        assert opts.any() is True

    def test_dfx_opts_any_false_when_all_off(self):
        assert _DfxOpts().any() is False


class TestRunConfigRuntimeProfilingDeprecation:
    """Verify the deprecated ``runtime_profiling`` field still works as an alias."""

    def test_runtime_profiling_aliases_enable_l2_swimlane(self):
        with pytest.warns(DeprecationWarning, match="runtime_profiling is deprecated"):
            cfg = RunConfig(platform="a5", runtime_profiling=True)
        assert cfg.enable_l2_swimlane is True
        assert cfg.save_kernels is True  # any-DFX auto-enables save_kernels

    def test_runtime_profiling_false_emits_no_warning(self):
        # An unset/explicit-False deprecated flag must not nag callers that
        # never touched it.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            cfg = RunConfig(platform="a5", runtime_profiling=False)
        assert cfg.enable_l2_swimlane is False

    def test_runtime_profiling_does_not_clobber_explicit_enable_l2_swimlane(self):
        # Both set: result is still True (OR semantics). The new field stays
        # the source of truth — the deprecated alias only ever turns it ON,
        # never OFF.
        with pytest.warns(DeprecationWarning):
            cfg = RunConfig(platform="a5", runtime_profiling=True, enable_l2_swimlane=True)
        assert cfg.enable_l2_swimlane is True


# ``execute_on_device`` lives in ``device_runner`` which eagerly imports the
# ``simpler`` package (via ``task_interface``). Unit-tests CI runs without
# ``simpler`` installed, so the import fails at collection time. Mirror the
# skip pattern from ``test_worker_reuse.py``.
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True


@pytest.mark.skipif(not _has_simpler, reason="execute_on_device requires the simpler package")
class TestExecuteOnDeviceDfxValidation:
    """Verify ``execute_on_device`` rejects DFX flags without ``output_prefix``."""

    def test_dfx_without_output_prefix_raises_value_error(self):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        with pytest.raises(ValueError, match="output_prefix is required"):
            execute_on_device(
                chip_callable=MagicMock(),
                orch_args=MagicMock(),
                platform="a5sim",
                runtime_name="tensormap_and_ringbuffer",
                device_id=0,
                output_prefix=None,
                enable_l2_swimlane=True,
            )

    def test_dfx_without_output_prefix_raises_for_each_flag(self):
        from pypto.runtime.device_runner import execute_on_device  # noqa: PLC0415

        for flag in [
            {"enable_l2_swimlane": True},
            {"enable_dump_tensor": True},
            {"enable_pmu": 2},
            {"enable_dep_gen": True},
        ]:
            with pytest.raises(ValueError, match="output_prefix is required"):
                execute_on_device(
                    chip_callable=MagicMock(),
                    orch_args=MagicMock(),
                    platform="a5sim",
                    runtime_name="tensormap_and_ringbuffer",
                    device_id=0,
                    output_prefix=None,
                    **flag,
                )

    def test_no_dfx_without_output_prefix_is_ok(self):
        # When no DFX flag is set, output_prefix=None must NOT raise.
        # The function would fail later on the actual device call, so we
        # patch the Worker plumbing to short-circuit after CallConfig setup.
        from pypto.runtime import device_runner  # noqa: PLC0415

        with patch.object(device_runner, "Worker") as worker_cls:
            worker = worker_cls.return_value
            # _PyptoWorker.current returns None → falls to the new-Worker path.
            with patch("pypto.runtime.worker.Worker.current", return_value=None):
                device_runner.execute_on_device(
                    chip_callable=MagicMock(),
                    orch_args=MagicMock(),
                    platform="a5sim",
                    runtime_name="tensormap_and_ringbuffer",
                    device_id=0,
                    output_prefix=None,
                )
            assert worker.init.called
            assert worker.run.called
            assert worker.close.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
