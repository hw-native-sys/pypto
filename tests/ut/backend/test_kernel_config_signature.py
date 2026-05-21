# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for per-kernel ``signature`` emission in ``kernel_config.py``.

Covers the issue #1458 fix: codegen now emits each kernel's runtime
``ArgDirection`` signature so the tensormap_and_ringbuffer tensor dump builds a
non-empty CoreCallable signature and its per-subtask tensor-arg count matches
the task payload ``tensor_count``. Without this, ``--dump-tensor`` captured
nothing for a codegen matmul (signature was empty -> count 0 != payload 3).

These tests exercise the codegen-side helper ``_generate_config_file`` directly
and do not require the optional ``simpler`` runtime package. The generated text
is checked for syntactic validity with ``compile`` (which does not execute the
``simpler.task_interface`` import).
"""

import pytest
from pypto.backend.pto_backend import _generate_config_file
from pypto.pypto_core import ir as _ir_core


def _base_inputs() -> dict:
    return {
        "orch_func_name": "main",
        "func_name_to_id": {"matmul_aic": 0},
        "func_name_to_core_type": {"matmul_aic": _ir_core.CoreType.CUBE},
    }


def _is_valid_python(text: str) -> bool:
    compile(text, "kernel_config.py", "exec")
    return True


class TestKernelConfigSignature:
    def test_signature_emitted_for_matmul(self) -> None:
        # The matmul AIC kernel takes a, b (inputs) and c (write target).
        text = _generate_config_file(
            **_base_inputs(),
            func_name_to_signature={"matmul_aic": ["IN", "IN", "INOUT"]},
        )
        # ArgDirection import is present and the kernel carries the signature.
        assert "from simpler.task_interface import ArgDirection as _D" in text
        assert '"signature": [_D.IN, _D.IN, _D.INOUT]' in text
        # 3 non-SCALAR entries == payload tensor_count for the matmul (a, b, c).
        assert text.count("_D.") == 3
        assert _is_valid_python(text)

    def test_no_import_without_signatures(self) -> None:
        # Omitting signatures keeps the pre-fix behavior: no import, no key.
        text = _generate_config_file(**_base_inputs())
        assert "ArgDirection" not in text
        assert '"signature"' not in text
        assert _is_valid_python(text)

    def test_empty_signature_map_is_noop(self) -> None:
        text = _generate_config_file(**_base_inputs(), func_name_to_signature={})
        assert "ArgDirection" not in text
        assert '"signature"' not in text

    def test_scalar_args_emitted_as_scalar(self) -> None:
        # Scalars stay in the signature so order matches the payload; the
        # runtime skips them when walking tensor args.
        text = _generate_config_file(
            **_base_inputs(),
            func_name_to_signature={"matmul_aic": ["IN", "OUT", "SCALAR"]},
        )
        assert '"signature": [_D.IN, _D.OUT, _D.SCALAR]' in text
        assert _is_valid_python(text)

    def test_partial_signatures_across_kernels(self) -> None:
        # One kernel has a signature, another does not: import is still emitted,
        # only the kernel with directions gets a "signature" field.
        text = _generate_config_file(
            orch_func_name="main",
            func_name_to_id={"k_with": 0, "k_without": 1},
            func_name_to_core_type={
                "k_with": _ir_core.CoreType.VECTOR,
                "k_without": _ir_core.CoreType.VECTOR,
            },
            func_name_to_signature={"k_with": ["IN", "OUT"]},
        )
        assert "from simpler.task_interface import ArgDirection as _D" in text
        assert text.count('"signature"') == 1
        assert '"signature": [_D.IN, _D.OUT]' in text
        assert _is_valid_python(text)

    def test_block_dim_still_works_with_signatures(self) -> None:
        text = _generate_config_file(
            **_base_inputs(),
            func_name_to_signature={"matmul_aic": ["IN", "IN", "INOUT"]},
            block_dim=8,
        )
        assert '"block_dim": 8,' in text
        assert '"signature": [_D.IN, _D.IN, _D.INOUT]' in text
        assert _is_valid_python(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
