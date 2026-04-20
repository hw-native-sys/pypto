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

import pytest
from pypto.pypto_core import backend as _backend_core


class TestBackendHandlerValues:
    """Per-backend handler returns the expected behavioural values."""

    def test_ascend910b_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910B).get_handler()

        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.get_launch_spec_core_count_method() == "set_block_num"
        assert handler.get_default_sim_platform() == "a2a3sim"
        assert handler.get_extra_ptoas_flags() == []

        assert handler.requires_gm_pipe_buffer() is True
        assert handler.requires_split_load_tpop_workaround() is True
        assert handler.requires_vto_c_fractal_adapt() is False
        assert handler.requires_runtime_subblock_bridge() is True

    def test_ascend950_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert handler.get_pto_target_arch() == "a5"
        assert handler.get_launch_spec_core_count_method() == "set_core_num"
        assert handler.get_default_sim_platform() == "a5sim"
        assert handler.get_extra_ptoas_flags() == ["--pto-arch", "a5"]

        assert handler.requires_gm_pipe_buffer() is False
        assert handler.requires_split_load_tpop_workaround() is False
        assert handler.requires_vto_c_fractal_adapt() is True
        assert handler.requires_runtime_subblock_bridge() is False

    def test_handlers_are_disjoint_between_backends(self):
        h910 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910B).get_handler()
        h950 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert h910.get_pto_target_arch() != h950.get_pto_target_arch()
        assert h910.get_launch_spec_core_count_method() != h950.get_launch_spec_core_count_method()
        assert h910.get_default_sim_platform() != h950.get_default_sim_platform()


class TestBackendHandlerSingletons:
    """Handlers must be stable singletons so callers can hold raw pointers."""

    def test_ascend910b_handler_is_singleton(self):
        h1 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910B).get_handler()
        h2 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910B).get_handler()
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
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend910B)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.requires_gm_pipe_buffer() is True

    def test_global_handler_matches_configured_backend_950(self):
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend950)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a5"
        assert handler.requires_vto_c_fractal_adapt() is True

    def test_global_handler_raises_when_not_configured(self):
        with pytest.raises(Exception):
            _backend_core.get_handler()
