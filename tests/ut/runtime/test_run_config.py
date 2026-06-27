# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.runner.RunConfig`` and DFX plumbing."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from pypto.backend import BackendType
from pypto.runtime.runner import RunConfig, _DfxOpts, compile_program, execute_compiled, run


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

    def test_auto_scope_deps_switch_defaults_off(self):
        cfg = RunConfig(platform="a5")

        assert cfg.analyze_auto_scopes_for_deps is False


class TestRunConfigDfxFlags:
    """Verify the five DFX flags are independent and propagate correctly."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_l2_swimlane": True},
            {"enable_dump_tensor": True},
            {"enable_pmu": 2},
            {"enable_dep_gen": True},
            {"enable_scope_stats": True},
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

    def test_dump_tensor_level_enables_dfx(self):
        # enable_dump_tensor is a level: 0=off, 1=partial, 2=full. Any
        # positive level enables DFX and forces save_kernels (artefact dir).
        off = RunConfig(platform="a5", enable_dump_tensor=0)
        assert off.any_dfx_enabled() is False
        for level in (1, 2):
            cfg = RunConfig(platform="a5", enable_dump_tensor=level)
            assert cfg.enable_dump_tensor == level
            assert cfg.any_dfx_enabled() is True
            assert cfg.save_kernels is True

    def test_dump_tensor_bool_maps_to_level(self):
        # Back-compat: True is the partial level (1), False is off (0). bool is
        # an int subtype so `> 0` truthiness and pass-through to CallConfig hold.
        assert RunConfig(platform="a5", enable_dump_tensor=True).enable_dump_tensor == 1
        assert RunConfig(platform="a5", enable_dump_tensor=False).enable_dump_tensor == 0
        assert RunConfig(platform="a5", enable_dump_tensor=True).any_dfx_enabled() is True

    def test_dfx_flags_are_independent(self):
        # Enabling one flag must not implicitly enable another.
        cfg = RunConfig(platform="a5", enable_dep_gen=True)
        assert cfg.enable_dep_gen is True
        assert cfg.enable_l2_swimlane is False
        assert cfg.enable_dump_tensor == 0
        assert cfg.enable_pmu == 0
        assert cfg.enable_scope_stats is False

    def test_scope_stats_forces_save_kernels(self):
        # scope_stats is the fifth DFX flag; like the others it must be
        # independent and auto-force kernel retention.
        cfg = RunConfig(platform="a5", enable_scope_stats=True)
        assert cfg.enable_scope_stats is True
        assert cfg.any_dfx_enabled() is True
        assert cfg.save_kernels is True
        assert cfg.enable_l2_swimlane is False
        assert cfg.enable_dump_tensor == 0
        assert cfg.enable_pmu == 0
        assert cfg.enable_dep_gen is False

    def test_dfx_opts_from_run_config_carries_all_five(self):
        cfg = RunConfig(
            platform="a5",
            enable_l2_swimlane=True,
            enable_dump_tensor=2,
            enable_pmu=2,
            enable_dep_gen=True,
            enable_scope_stats=True,
        )
        opts = _DfxOpts.from_run_config(cfg)
        assert opts.enable_l2_swimlane is True
        assert opts.enable_dump_tensor == 2
        assert opts.enable_pmu == 2
        assert opts.enable_dep_gen is True
        assert opts.enable_scope_stats is True
        assert opts.any() is True

    def test_dfx_opts_any_true_for_scope_stats_only(self):
        # _DfxOpts.any() must report True when scope_stats is the sole flag.
        assert _DfxOpts(enable_scope_stats=True).any() is True

    def test_dfx_opts_any_false_when_all_off(self):
        assert _DfxOpts().any() is False


class TestRunConfigRingSizing:
    """Verify per-task ring-sizing overrides on ``RunConfig``.

    ``None`` (default) means "unset" so the runtime falls back to its
    ``PTO2_RING_*`` env var / compile-time default. Provided values must
    satisfy the same constraints the runtime's ``RuntimeEnv::validate()``
    enforces — ``RunConfig`` checks them early for a clear error message.
    """

    def test_ring_fields_default_none(self):
        cfg = RunConfig(platform="a2a3sim")
        assert cfg.ring_task_window is None
        assert cfg.ring_heap is None
        assert cfg.ring_dep_pool is None

    def test_valid_ring_values_accepted(self):
        cfg = RunConfig(
            platform="a2a3sim",
            ring_task_window=128,
            ring_heap=8 * 1024 * 1024,
            ring_dep_pool=256,
        )
        assert cfg.ring_task_window == 128
        assert cfg.ring_heap == 8 * 1024 * 1024
        assert cfg.ring_dep_pool == 256

    def test_ring_min_boundaries_accepted(self):
        cfg = RunConfig(
            platform="a2a3sim",
            ring_task_window=4,
            ring_heap=1024,
            ring_dep_pool=4,
        )
        assert cfg.ring_task_window == 4
        assert cfg.ring_heap == 1024
        assert cfg.ring_dep_pool == 4

    @pytest.mark.parametrize("bad", [3, 5, 0, 6, 100])
    def test_ring_task_window_must_be_pow2_ge4(self, bad):
        with pytest.raises(ValueError, match="ring_task_window must be a power of 2 >= 4"):
            RunConfig(platform="a2a3sim", ring_task_window=bad)

    @pytest.mark.parametrize("bad", [512, 1000, 1536, 0])
    def test_ring_heap_must_be_pow2_ge1024(self, bad):
        with pytest.raises(ValueError, match="ring_heap must be a power of 2 >= 1024"):
            RunConfig(platform="a2a3sim", ring_heap=bad)

    @pytest.mark.parametrize("bad", [3, 0, -1, 2**31])
    def test_ring_dep_pool_must_be_in_int32_range(self, bad):
        with pytest.raises(ValueError, match=r"ring_dep_pool must be in \[4, INT32_MAX\]"):
            RunConfig(platform="a2a3sim", ring_dep_pool=bad)

    def test_ring_dep_pool_need_not_be_pow2(self):
        # Unlike task_window / heap, the dep pool is a plain int range.
        cfg = RunConfig(platform="a2a3sim", ring_dep_pool=100)
        assert cfg.ring_dep_pool == 100

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("ring_task_window", 16.0),  # float, even when value would be valid as int
            ("ring_heap", 1024.0),
            ("ring_dep_pool", 64.5),
            ("ring_task_window", True),  # bool must not masquerade as a size
            ("ring_dep_pool", False),
        ],
    )
    def test_non_int_ring_values_rejected(self, field, bad):
        # Reject floats / bools with a clear ValueError instead of letting the
        # pow2 bitwise check raise TypeError or a float slip through.
        with pytest.raises(ValueError, match=f"{field} must"):
            RunConfig(platform="a2a3sim", **{field: bad})


class TestRunConfigPerRingList:
    """Verify the per-scope-depth ring list form of the ring overrides.

    A list sizes rings 0..3 independently; a ``0`` entry means "leave that ring
    at its env/compile-time default". A scalar (validated elsewhere) is broadcast
    to every ring by the runtime.
    """

    def test_valid_per_ring_lists_accepted(self):
        cfg = RunConfig(
            platform="a2a3sim",
            ring_task_window=[16, 32, 128, 256],
            ring_heap=[1024, 2048, 4096, 8192],
            ring_dep_pool=[8, 16, 100, 256],
        )
        assert cfg.ring_task_window == [16, 32, 128, 256]
        assert cfg.ring_heap == [1024, 2048, 4096, 8192]
        assert cfg.ring_dep_pool == [8, 16, 100, 256]

    def test_zero_entry_is_per_ring_unset_sentinel(self):
        # 0 = leave that ring at its default; other entries still validated.
        cfg = RunConfig(platform="a2a3sim", ring_task_window=[16, 0, 0, 256])
        assert cfg.ring_task_window == [16, 0, 0, 256]

    def test_tuple_accepted_and_normalized_to_list(self):
        # A tuple is a valid per-ring form and is normalized to a list so
        # downstream transcription always sees a list.
        cfg = RunConfig(platform="a2a3sim", ring_heap=(1024, 2048, 4096, 8192))
        assert cfg.ring_heap == [1024, 2048, 4096, 8192]
        assert isinstance(cfg.ring_heap, list)

    @pytest.mark.parametrize(
        "field",
        ["ring_task_window", "ring_heap", "ring_dep_pool"],
    )
    @pytest.mark.parametrize("length", [0, 1, 3, 5])
    def test_wrong_length_list_rejected(self, field, length):
        with pytest.raises(ValueError, match=f"{field} must have exactly 4 entries"):
            RunConfig(platform="a2a3sim", **{field: [4] * length})

    @pytest.mark.parametrize(
        ("field", "bad_list"),
        [
            ("ring_task_window", [16, 32, 48, 64]),  # 48 not a power of 2
            ("ring_heap", [1024, 2048, 512, 4096]),  # 512 < 1024
            ("ring_dep_pool", [8, 16, 2, 256]),  # 2 < 4
        ],
    )
    def test_invalid_entry_rejected(self, field, bad_list):
        with pytest.raises(ValueError, match=f"{field} entries must"):
            RunConfig(platform="a2a3sim", **{field: bad_list})

    @pytest.mark.parametrize(
        ("field", "bad_list"),
        [
            ("ring_task_window", [16, 32, True, 64]),  # bool must not pass as 0/size
            ("ring_dep_pool", [8, False, 16, 32]),  # False must not pass as the 0 sentinel
            ("ring_heap", [1024, 2048.0, 4096, 8192]),  # float entry
        ],
    )
    def test_non_int_entry_rejected(self, field, bad_list):
        with pytest.raises(ValueError, match=f"{field} entries must"):
            RunConfig(platform="a2a3sim", **{field: bad_list})


class _SpyRuntimeEnv:
    """Records writes to ``ring_*`` fields; defaults mirror the runtime (0)."""

    def __init__(self) -> None:
        self.ring_task_window = 0
        self.ring_heap = 0
        self.ring_dep_pool = 0


class _SpyCallConfig:
    """Stand-in for simpler's ``CallConfig`` with a nested ``runtime_env``.

    Carries the same DFX defaults as the real ``CallConfig`` (all off,
    ``output_prefix`` empty) so tests can assert the builder leaves them
    untouched on the no-DFX path.
    """

    def __init__(self) -> None:
        self.runtime_env = _SpyRuntimeEnv()
        self.enable_l2_swimlane = False
        self.enable_dump_tensor = 0
        self.enable_pmu = 0
        self.enable_dep_gen = False
        self.enable_scope_stats = False
        self.output_prefix = ""


def _build_with_fake_callconfig(run_config, monkeypatch, **kwargs):
    """Invoke ``_build_call_config`` with a spy ``CallConfig`` so the test
    runs without the optional ``simpler`` package installed.
    """
    fake_task_interface = types.SimpleNamespace(CallConfig=_SpyCallConfig)
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", fake_task_interface)
    from pypto.runtime.runner import _build_call_config  # noqa: PLC0415

    return _build_call_config(run_config, runtime_config={}, **kwargs)


class TestBuildCallConfigRing:
    """Verify ``_build_call_config`` transcribes ring sizing into ``runtime_env``."""

    def test_unset_leaves_runtime_env_at_zero(self, monkeypatch):
        cfg = _build_with_fake_callconfig(RunConfig(platform="a2a3sim"), monkeypatch)
        assert cfg.runtime_env.ring_task_window == 0
        assert cfg.runtime_env.ring_heap == 0
        assert cfg.runtime_env.ring_dep_pool == 0

    def test_set_values_transcribed(self, monkeypatch):
        run_config = RunConfig(
            platform="a2a3sim",
            ring_task_window=16,
            ring_heap=1024 * 1024,
            ring_dep_pool=64,
        )
        cfg = _build_with_fake_callconfig(run_config, monkeypatch)
        assert cfg.runtime_env.ring_task_window == 16
        assert cfg.runtime_env.ring_heap == 1024 * 1024
        assert cfg.runtime_env.ring_dep_pool == 64

    def test_partial_set_only_touches_provided_fields(self, monkeypatch):
        run_config = RunConfig(platform="a2a3sim", ring_heap=2 * 1024 * 1024)
        cfg = _build_with_fake_callconfig(run_config, monkeypatch)
        assert cfg.runtime_env.ring_heap == 2 * 1024 * 1024
        # Unset fields stay at the runtime's 0 default.
        assert cfg.runtime_env.ring_task_window == 0
        assert cfg.runtime_env.ring_dep_pool == 0

    def test_per_ring_list_transcribed_unchanged(self, monkeypatch):
        # A per-ring list flows straight through to runtime_env; the runtime's
        # RuntimeEnv setter accepts both a scalar (broadcast) and a 4-list.
        run_config = RunConfig(
            platform="a2a3sim",
            ring_task_window=[16, 32, 128, 256],
            ring_dep_pool=[8, 0, 0, 64],
        )
        cfg = _build_with_fake_callconfig(run_config, monkeypatch)
        assert cfg.runtime_env.ring_task_window == [16, 32, 128, 256]
        assert cfg.runtime_env.ring_dep_pool == [8, 0, 0, 64]
        assert cfg.runtime_env.ring_heap == 0


def _make_dist_call_config_with_fake(dc, run_config, monkeypatch, *, dfx_base=None):
    """Invoke ``distributed_runner._make_call_config`` with a spy ``CallConfig``.

    Injects a fake ``simpler.task_interface`` so the L3 config builder runs
    without the optional ``simpler`` package installed, mirroring
    :func:`_build_with_fake_callconfig` for the L2 path.
    """
    fake_task_interface = types.SimpleNamespace(CallConfig=_SpyCallConfig)
    fake_simpler = types.ModuleType("simpler")
    fake_simpler.task_interface = fake_task_interface  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "simpler", fake_simpler)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", fake_task_interface)
    from pypto.runtime.distributed_runner import _make_call_config  # noqa: PLC0415

    return _make_call_config(dc, run_config, dfx_base=dfx_base)


class TestMakeCallConfigRing:
    """Verify L3 ``_make_call_config`` overlays per-dispatch ring sizing.

    The ``block_dim`` / ``aicpu_thread_num`` baseline always comes from the
    program's :class:`DistributedConfig`; a per-dispatch :class:`RunConfig`
    overlays the ``ring_*`` overrides on top. ``None`` leaves every ring field
    at the runtime's ``0`` default.
    """

    def test_no_run_config_leaves_runtime_env_at_zero(self, monkeypatch):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        cfg = _make_dist_call_config_with_fake(DistributedConfig(), None, monkeypatch)
        assert cfg.runtime_env.ring_task_window == 0
        assert cfg.runtime_env.ring_heap == 0
        assert cfg.runtime_env.ring_dep_pool == 0

    def test_run_config_ring_overrides_transcribed(self, monkeypatch):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        run_config = RunConfig(
            platform="a2a3sim",
            ring_task_window=32,
            ring_heap=2 * 1024 * 1024,
            ring_dep_pool=128,
        )
        cfg = _make_dist_call_config_with_fake(DistributedConfig(), run_config, monkeypatch)
        assert cfg.runtime_env.ring_task_window == 32
        assert cfg.runtime_env.ring_heap == 2 * 1024 * 1024
        assert cfg.runtime_env.ring_dep_pool == 128

    def test_baseline_preserved_and_partial_ring_overlay(self, monkeypatch):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dc = DistributedConfig(block_dim=8, aicpu_thread_num=3)
        run_config = RunConfig(platform="a2a3sim", ring_heap=1024 * 1024)
        cfg = _make_dist_call_config_with_fake(dc, run_config, monkeypatch)
        # DistributedConfig baseline is preserved.
        assert cfg.block_dim == 8
        assert cfg.aicpu_thread_num == 3
        # Only the provided ring field is written; the rest stay at 0.
        assert cfg.runtime_env.ring_heap == 1024 * 1024
        assert cfg.runtime_env.ring_task_window == 0
        assert cfg.runtime_env.ring_dep_pool == 0

    def test_per_ring_list_overlaid_on_l3_dispatch(self, monkeypatch):
        # A per-program L3 dispatch can size each scope-depth ring independently
        # (e.g. a wider task window for prefill than decode).
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        run_config = RunConfig(platform="a2a3sim", ring_task_window=[16, 32, 128, 256])
        cfg = _make_dist_call_config_with_fake(DistributedConfig(), run_config, monkeypatch)
        assert cfg.runtime_env.ring_task_window == [16, 32, 128, 256]
        assert cfg.runtime_env.ring_heap == 0
        assert cfg.runtime_env.ring_dep_pool == 0


class TestMakeCallConfigDfx:
    """Verify L3 ``_make_call_config`` wires the runtime DFX diagnostics.

    The runtime-diagnostic flags (``enable_dump_tensor`` / ``enable_pmu`` /
    ``enable_dep_gen`` / ``enable_scope_stats`` / ``enable_l2_swimlane``) are
    transcribed onto the shared ``CallConfig`` and their artifacts rooted at
    ``dfx_base``. ``enable_l2_swimlane`` additionally co-enables ``dep_gen`` so
    the converter has a task graph (see :class:`test_swimlane_sets_flag...`).
    """

    def test_dfx_flags_transcribed_and_prefix_set(self, monkeypatch, tmp_path):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dfx_base = tmp_path / "dfx_outputs"
        run_config = RunConfig(
            platform="a2a3sim",
            enable_dump_tensor=2,
            enable_pmu=1,
            enable_dep_gen=True,
            enable_scope_stats=True,
        )
        cfg = _make_dist_call_config_with_fake(
            DistributedConfig(), run_config, monkeypatch, dfx_base=dfx_base
        )
        assert cfg.enable_dump_tensor == 2
        assert cfg.enable_pmu == 1
        assert cfg.enable_dep_gen is True
        assert cfg.enable_scope_stats is True
        assert cfg.output_prefix == str(dfx_base)
        # The builder creates the base dir so the runtime's validate() accepts it.
        assert dfx_base.is_dir()

    def test_swimlane_sets_flag_and_co_enables_dep_gen(self, monkeypatch, tmp_path):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dfx_base = tmp_path / "dfx_outputs"
        # User asks for swimlane only; dep_gen is auto-enabled because the
        # converter needs deps.json to resolve task arrows / kernel names.
        run_config = RunConfig(platform="a2a3sim", enable_l2_swimlane=True)
        cfg = _make_dist_call_config_with_fake(
            DistributedConfig(), run_config, monkeypatch, dfx_base=dfx_base
        )
        assert cfg.enable_l2_swimlane is True
        assert cfg.enable_dep_gen is True  # co-enabled
        assert cfg.output_prefix == str(dfx_base)

    def test_dfx_without_base_raises(self, monkeypatch):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        run_config = RunConfig(platform="a2a3sim", enable_pmu=1)
        with pytest.raises(ValueError, match="dfx_base is required"):
            _make_dist_call_config_with_fake(DistributedConfig(), run_config, monkeypatch, dfx_base=None)

    def test_no_run_config_leaves_dfx_off(self, monkeypatch):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        cfg = _make_dist_call_config_with_fake(DistributedConfig(), None, monkeypatch)
        assert cfg.output_prefix == ""
        assert cfg.enable_pmu == 0
        assert cfg.enable_dep_gen is False

    def test_ring_only_run_config_creates_no_dfx_dir(self, monkeypatch, tmp_path):
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        dfx_base = tmp_path / "dfx_outputs"
        run_config = RunConfig(platform="a2a3sim", ring_heap=1024 * 1024)
        cfg = _make_dist_call_config_with_fake(
            DistributedConfig(), run_config, monkeypatch, dfx_base=dfx_base
        )
        # Ring sizing applied, DFX untouched, and no artifact dir materialized.
        assert cfg.runtime_env.ring_heap == 1024 * 1024
        assert cfg.output_prefix == ""
        assert not dfx_base.exists()


class TestRunConfigCompileForwarding:
    """Compile-side RunConfig fields are forwarded into ``ir.compile``."""

    def test_run_forwards_auto_scope_deps_switch(self, monkeypatch):
        captured: dict = {}

        class FakeCompiled:
            def __call__(self, *_args, **_kwargs):
                return None

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return FakeCompiled()

        import pypto.ir as ir_mod  # noqa: PLC0415

        monkeypatch.setattr(ir_mod, "compile", fake_compile)

        run(object(), config=RunConfig(platform="a2a3sim", analyze_auto_scopes_for_deps=True))

        assert captured["analyze_auto_scopes_for_deps"] is True

    def test_execute_compiled_accepts_auto_scope_deps_switch(self, tmp_path, monkeypatch):
        captured: dict = {}

        def fake_compile_and_assemble(_work_dir, platform, pto_isa_commit):
            captured["compile"] = {
                "platform": platform,
                "pto_isa_commit": pto_isa_commit,
            }
            return object(), "fake_runtime", {}

        def fake_execute_on_device(*args, **kwargs):
            captured["execute"] = {"args": args, "kwargs": kwargs}

        class FakeChipStorageTaskArgs:
            def add_tensor(self, _arg):
                return None

            def add_scalar(self, _arg):
                return None

        fake_device_runner = types.SimpleNamespace(
            ChipStorageTaskArgs=FakeChipStorageTaskArgs,
            compile_and_assemble=fake_compile_and_assemble,
            execute_on_device=fake_execute_on_device,
            make_tensor_arg=lambda _arg: object(),
            scalar_to_uint64=lambda _arg: 0,
        )
        fake_task_interface = types.SimpleNamespace(device_tensor_to_tensor=lambda _arg: object())
        monkeypatch.setitem(sys.modules, "pypto.runtime.device_runner", fake_device_runner)
        monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", fake_task_interface)

        execute_compiled(
            tmp_path,
            [],
            platform="a2a3sim",
            device_id=0,
            analyze_auto_scopes_for_deps=True,
        )

        assert captured["compile"]["platform"] == "a2a3sim"
        assert captured["execute"]["args"][3] == "fake_runtime"
        assert captured["execute"]["kwargs"]["block_dim"] is None
        assert captured["execute"]["kwargs"]["aicpu_thread_num"] is None

    def test_compile_program_forwards_auto_scope_deps_switch(self, tmp_path, monkeypatch):
        captured: dict = {}

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return object()

        import pypto.ir as ir_mod  # noqa: PLC0415
        import pypto.runtime.runner as runner_mod  # noqa: PLC0415

        monkeypatch.setattr(ir_mod, "compile", fake_compile)
        monkeypatch.setattr(runner_mod, "_patch_orchestration_headers", lambda _work_dir: None)

        compile_program(
            object(),
            tmp_path,
            strategy=RunConfig().strategy,
            backend_type=BackendType.Ascend910B,
            analyze_auto_scopes_for_deps=True,
        )

        assert captured["analyze_auto_scopes_for_deps"] is True


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
            {"enable_scope_stats": True},
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
            # ``current`` lives on ``ChipWorker``, not the ABC base ``Worker``.
            with patch("pypto.runtime.worker.ChipWorker.current", return_value=None):
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
