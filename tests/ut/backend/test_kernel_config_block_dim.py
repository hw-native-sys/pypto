# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for ``block_dim`` emission in generated ``kernel_config.py``.

Covers the issue #1173 fix that drops the hard-coded ``"block_dim": 24``
from ``RUNTIME_CONFIG`` and only emits the key when the user explicitly
passes ``compile(block_dim=...)``.

These tests exercise the codegen-side helper ``_generate_config_file``
directly and do not require the optional ``simpler`` runtime package.
"""

import pytest
from pypto.backend.pto_backend import _generate_config_file
from pypto.pypto_core import ir as _ir_core


def _inputs() -> dict:
    return {
        "orch_func_name": "main",
        "func_name_to_id": {"k0": 0},
        "func_name_to_core_type": {"k0": _ir_core.CoreType.VECTOR},
    }


class TestKernelConfigBlockDim:
    def test_block_dim_omitted_when_none(self) -> None:
        text = _generate_config_file(**_inputs(), block_dim=None)
        # Default path: no block_dim baked in.
        assert '"block_dim"' not in text
        # aicpu_thread_num is still 4 — required by the tensormap_and_ringbuffer runtime.
        assert '"aicpu_thread_num": 4,' in text
        assert '"runtime": "tensormap_and_ringbuffer",' in text

    def test_block_dim_default_omitted(self) -> None:
        # Calling without the keyword keeps the same default behavior.
        text = _generate_config_file(**_inputs())
        assert '"block_dim"' not in text

    def test_block_dim_emitted_when_set(self) -> None:
        text = _generate_config_file(**_inputs(), block_dim=8)
        assert '"block_dim": 8,' in text

    def test_block_dim_zero_emits_value(self) -> None:
        # Edge case: ``0`` is a valid user override (only ``None`` opts out).
        text = _generate_config_file(**_inputs(), block_dim=0)
        assert '"block_dim": 0,' in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
