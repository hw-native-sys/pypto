# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce #3098's legacy producer bytes on silicon with the correct golden.

Only the emitted epilogue attributes are removed. The current compiler,
protocol, nosplit policy, input data and independent transpose golden stay fixed.
"""

import re
from pathlib import Path

import pytest
import torch
from pypto.backend import pto_backend
from test_c2v_fixpipe import _case, column_view


@pytest.mark.platforms("a5", reason="The legacy physical-byte oracle is validated on A5 hardware only.")
@pytest.mark.inline_case(reason="The replay must compile after its PTOAS monkeypatch is installed.")
def test_legacy_column_major_bytes_are_detected(test_runner, monkeypatch):
    real_ptoas = pto_backend._run_ptoas

    def legacy_producer(pto_path, output_path, ptoas_flags=None):
        source = Path(pto_path)
        text = source.read_text()
        assert "layout = nz2dn" in text
        text, count = re.subn(r", acc_push_epilogue = #pto\.acc_push_epilogue<[^>]+>", "", text)
        assert count == 2
        source.write_text(text)
        return real_ptoas(pto_path, output_path, ptoas_flags)

    def detect_legacy_bytes(actual, expected):
        correct = expected["out"]
        legacy = correct.T.contiguous().reshape(correct.shape)
        assert not torch.equal(legacy, correct)
        assert torch.equal(actual["out"], legacy), "unexpected legacy transport byte pattern"
        raise AssertionError("legacy row-major producer bytes reproduced; transpose golden rejected")

    monkeypatch.setattr(pto_backend, "_run_ptoas", legacy_producer)
    case = _case(column_view, signed=False, buffer_ir=False)
    case.name = "legacy_column_major_byte_oracle"
    case.compare = detect_legacy_bytes
    result = test_runner.run(case)
    assert not result.passed
    assert "legacy row-major producer bytes reproduced" in result.error, result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
