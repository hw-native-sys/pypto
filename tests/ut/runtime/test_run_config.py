# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.runner.RunConfig``."""

import pytest
from pypto.backend import BackendType
from pypto.runtime.runner import RunConfig


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

    def test_runtime_profiling_forces_save_kernels(self):
        cfg = RunConfig(platform="a5", runtime_profiling=True)

        assert cfg.platform == "a5"
        assert cfg.backend_type == BackendType.Ascend950
        assert cfg.save_kernels is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
