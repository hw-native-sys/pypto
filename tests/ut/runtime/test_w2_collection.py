# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""W2 collection stays safe for fork while preserving its numerical inputs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest
import torch
from pypto.runtime.tensor_spec import TensorSpec


@pytest.fixture
def w2_module(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> ModuleType:
    pytestconfig.addinivalue_line("markers", "platforms(*names): supported system-test platforms")
    st_dir = Path(__file__).resolve().parents[2] / "st"
    monkeypatch.syspath_prepend(str(st_dir))
    path = st_dir / "runtime" / "cross_core" / "test_w2_fp4_nz_mx_scale.py"
    spec = importlib.util.spec_from_file_location("_w2_collection_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    def reject_collection_inputs(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("W2 input generation during collection can deadlock forked tests")

    # Exercise the actual decorators, before any case is executed. This fails
    # deterministically instead of requiring a platform-dependent fork hang.
    with monkeypatch.context() as guard:
        for name in ("Generator", "randint", "arange", "tensor", "full"):
            guard.setattr(torch, name, reject_collection_inputs)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case_index,row_base,valid_rows", [(0, 5, 13), (1, 17, 7), (2, 31, 1)])
def test_w2_collection_defers_inputs_and_preserves_bytes(
    w2_module: ModuleType, case_index: int, row_base: int, valid_rows: int
) -> None:
    marker = next(
        marker
        for marker in w2_module.test_w2_nz_and_compact_scale_loads.pytestmark
        if marker.name == "parametrize"
    )
    cases = marker.args[1]
    assert len(cases) == 3
    specs = cases[case_index].tensor_specs
    assert [spec.name for spec in specs] == ["a", "a_scale", "weight", "b_scale", "counts", "out"]
    expected = w2_module._w2_inputs(row_base, valid_rows)
    for spec, reference in zip(specs, expected, strict=True):
        assert callable(spec.init_value)
        assert spec.is_output == (spec.name == "out")
        actual = TensorSpec(
            spec.name, spec.shape, spec.dtype.torch_dtype, spec.init_value, spec.is_output
        ).create_tensor()
        assert list(actual.shape) == spec.shape == list(reference.shape)
        assert actual.dtype == reference.dtype
        # Compare FP8 and scale carriers byte-for-byte, including the sentinel
        # output rows which the partial-row golden must preserve.
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
