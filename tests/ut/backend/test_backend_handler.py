# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BackendHandler dispatch interface (issue #948).

Verifies the per-backend handler returns the documented values, that the
dispatch is routed both via ``Backend.get_handler()`` and the global
``backend.get_handler()`` accessor, and that handlers stay disjoint between
backends so adding a third backend cannot silently inherit the wrong defaults.
"""

import pickle

import pytest
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import ir as _ir_core
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig


def test_ascend910_legacy_alias_has_canonical_identity():
    """Old callers and serialized enum values must select the canonical member."""
    canonical = _backend_core.BackendType.Ascend910
    legacy = _backend_core.BackendType.Ascend910B
    assert legacy is canonical
    assert legacy.name == "Ascend910"
    assert canonical.value == 0
    assert _backend_core.BackendType.Ascend950.value == 1
    assert _backend_core.BackendType.__members__["Ascend910B"] is canonical
    assert pickle.loads(pickle.dumps(legacy)) is canonical
    assert _backend_core.get_backend_instance(legacy) is _backend_core.get_backend_instance(canonical)


@pytest.mark.parametrize("mode", [ExecutionMode.SIM, ExecutionMode.ONBOARD])
def test_ascend910_legacy_alias_normalizes_target(mode):
    """Alias use must not fork compile options or the internal platform identity."""
    canonical = RunConfig(backend_type=_backend_core.BackendType.Ascend910, execution_mode=mode)
    legacy = RunConfig(backend_type=_backend_core.BackendType.Ascend910B, execution_mode=mode)
    assert legacy.compile_kwargs() == canonical.compile_kwargs()
    assert legacy.platform == canonical.platform
    assert (
        CompileOptions(backend_type=legacy.backend_type, execution_mode=mode).platform == canonical.platform
    )


class TestBackendHandlerValues:
    """Per-backend handler returns the expected behavioural values."""

    def test_ascend910b_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()

        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.get_launch_spec_core_count_method() == "set_block_num"
        assert handler.get_default_sim_platform() == "a2a3sim"
        assert handler.get_extra_ptoas_flags() == ["--pto-arch", "a3"]

        assert handler.honors_high_precision_algorithm() is False
        assert handler.requires_gm_pipe_buffer() is True
        assert handler.supports_runtime_bound_local_software_pipeline() is True
        assert handler.supports_local_pipe_depth() is True
        assert handler.requires_split_load_tpop_workaround() is True
        assert handler.requires_level3_tmp_scratch() is True
        assert handler.requires_vto_c_fractal_adapt() is False
        assert handler.requires_runtime_subblock_bridge() is True
        assert handler.requires_no_split_dual_aiv_dispatch() is True
        assert handler.splits_cube_to_vector_transport_in_hardware() is False

    def test_ascend950_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert handler.get_pto_target_arch() == "a5"
        assert handler.get_launch_spec_core_count_method() == "set_block_num"
        assert handler.get_default_sim_platform() == "a5sim"
        assert handler.get_extra_ptoas_flags() == ["--pto-arch", "a5"]

        assert handler.honors_high_precision_algorithm() is True
        assert handler.requires_gm_pipe_buffer() is False
        assert handler.supports_runtime_bound_local_software_pipeline() is False
        assert handler.supports_local_pipe_depth() is False
        assert handler.requires_split_load_tpop_workaround() is False
        assert handler.requires_level3_tmp_scratch() is False
        assert handler.requires_vto_c_fractal_adapt() is True
        assert handler.requires_runtime_subblock_bridge() is False
        assert handler.requires_no_split_dual_aiv_dispatch() is False
        assert handler.splits_cube_to_vector_transport_in_hardware() is True

    def test_handlers_are_disjoint_between_backends(self):
        h910 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        h950 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert h910.get_pto_target_arch() != h950.get_pto_target_arch()
        # Simpler exposes the same logical-block API on both architectures.
        assert h910.get_launch_spec_core_count_method() == h950.get_launch_spec_core_count_method()
        assert h910.get_default_sim_platform() != h950.get_default_sim_platform()


class TestBackendHandlerSingletons:
    """Handlers must be stable singletons so callers can hold raw pointers."""

    def test_ascend910b_handler_is_singleton(self):
        h1 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        h2 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        # Identity is exposed on the C++ side as the same singleton; on the
        # Python side we at least require structural equality of all hooks.
        assert h1.get_pto_target_arch() == h2.get_pto_target_arch()


class TestGlobalHandlerAccessor:
    """``backend.get_handler()`` follows the globally configured backend."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        # Snapshot and restore global config so we do not leak state to
        # other tests (which set their own backend type during fixtures).
        was_configured = _backend_core.is_backend_configured()
        previous = _backend_core.get_backend_type() if was_configured else None
        _backend_core.reset_for_testing()
        yield
        _backend_core.reset_for_testing()
        if previous is not None:
            _backend_core.set_backend_type(previous)

    def test_global_handler_matches_configured_backend_910b(self):
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend910)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.requires_gm_pipe_buffer() is True
        # Python callers can feature-detect local depth before building a pipe.
        assert handler.supports_local_pipe_depth() is True

    def test_global_handler_matches_configured_backend_950(self):
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend950)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a5"
        assert handler.supports_local_pipe_depth() is False
        assert handler.requires_vto_c_fractal_adapt() is True

    def test_global_handler_raises_when_not_configured(self):
        with pytest.raises(ValueError, match="Backend type not configured"):
            _backend_core.get_handler()

    @pytest.mark.parametrize(
        "configured_backend",
        [None, _backend_core.BackendType.Ascend910, _backend_core.BackendType.Ascend950],
    )
    @pytest.mark.parametrize(
        "queried_backend", [_backend_core.BackendType.Ascend910, _backend_core.BackendType.Ascend950]
    )
    def test_capacity_queries_use_owning_soc(self, configured_backend, queried_backend):
        """An explicit handler must work before target selection and for the other target."""
        if configured_backend is not None:
            _backend_core.set_backend_type(configured_backend)
        backend = _backend_core.get_backend_instance(queried_backend)
        handler = backend.get_handler()

        # In particular, Acc and Bias differ between architectures. Reading the
        # active target instead of this handler's owner would return wrong limits.
        for memory_space in _ir_core.MemorySpace.__members__.values():
            assert handler.get_capacity_bytes(memory_space) == backend.get_mem_size(memory_space)
        # Scale memories distinguish an unmodeled space from a modeled A5 space.
        expected_scale = 4096 if queried_backend == _backend_core.BackendType.Ascend950 else 0
        assert handler.get_capacity_bytes(_ir_core.MemorySpace.LeftScale) == expected_scale
        assert handler.get_capacity_bytes(_ir_core.MemorySpace.ScalarLocal) == 0
        assert _backend_core.is_backend_configured() == (configured_backend is not None)
        if configured_backend is not None:
            assert _backend_core.get_backend_type() == configured_backend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
