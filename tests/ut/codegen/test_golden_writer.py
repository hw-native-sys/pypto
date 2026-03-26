# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pypto.runtime.golden_writer."""

import pytest
from pypto.runtime.golden_writer import _extract_compute_golden, generate_golden_source
from pypto.runtime.tensor_spec import ScalarSpec, TensorSpec

torch = pytest.importorskip("torch")


def _dummy_golden(tensors, params=None):
    tensors["out"][:] = tensors["a"] * 3


class _BoundGoldenCase:
    def __init__(self, scale: float):
        self._scale = scale

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["a"] * self._scale


class TestGoldenWriterScalar:
    """Tests for scalar OrchArg entries in generated golden.py."""

    def test_scalar_int64_in_generate_inputs(self):
        """Scalar INT64 entry appears after tensors with ctypes.c_int64."""
        specs = [
            TensorSpec("a", [16], torch.float32, init_value=1.0),
            TensorSpec("out", [16], torch.float32, is_output=True),
        ]
        scalars = [ScalarSpec("factor", 3, ctype="int64")]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5, scalar_specs=scalars)

        assert "import ctypes" in src
        assert "factor = ctypes.c_int64(3)" in src
        assert '("factor", factor)' in src
        lines = src.splitlines()
        return_lines = [line.strip() for line in lines if line.strip().startswith('("')]
        assert return_lines[0].startswith('("a"')
        assert return_lines[1].startswith('("out"')
        assert return_lines[2].startswith('("factor"')

    def test_no_scalars_no_ctypes_import(self):
        """When no scalars, ctypes is not imported."""
        specs = [
            TensorSpec("a", [16], torch.float32, init_value=1.0),
            TensorSpec("out", [16], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "import ctypes" not in src
        assert "ctypes" not in src

    def test_no_size_entries(self):
        """Generated output must not contain size_* entries (legacy dict-mode convention)."""
        specs = [
            TensorSpec("a", [16], torch.float32, init_value=1.0),
            TensorSpec("out", [16], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "size_a" not in src
        assert "size_out" not in src

    def test_multiple_scalars_ordering(self):
        """Multiple scalar specs appear in declaration order after tensors."""
        specs = [
            TensorSpec("x", [8], torch.float32, init_value=1.0),
            TensorSpec("y", [8], torch.float32, is_output=True),
        ]
        scalars = [
            ScalarSpec("alpha", 10, ctype="int64"),
            ScalarSpec("beta", 20, ctype="int32"),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5, scalar_specs=scalars)

        assert "alpha = ctypes.c_int64(10)" in src
        assert "beta = ctypes.c_int32(20)" in src
        lines = src.splitlines()
        return_lines = [line.strip() for line in lines if line.strip().startswith('("')]
        names = [line.split('"')[1] for line in return_lines]
        assert names == ["x", "y", "alpha", "beta"]

    def test_invalid_ctype_raises(self):
        """ScalarSpec with unsupported ctype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported ctype"):
            ScalarSpec("bad", 1, ctype="float32")

    def test_compute_golden_src_adds_struct_import(self):
        """Generated golden.py includes struct when compute_golden uses it."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        compute_golden_src = "\n".join(
            [
                "def compute_golden(tensors, params=None):",
                '    scale_value = struct.unpack("f", struct.pack("I", 1065353216))[0]',
                '    tensors["out"][:] = tensors["a"] * scale_value',
            ]
        )

        src = generate_golden_source(specs, None, 1e-5, 1e-5, compute_golden_src=compute_golden_src)

        assert "import struct" in src

        namespace: dict[str, object] = {}
        exec(src, namespace)  # noqa: S102 - Test verifies generated source executes correctly.

        tensors = {
            "a": torch.ones((4,), dtype=torch.float32),
            "out": torch.zeros((4,), dtype=torch.float32),
        }
        namespace["compute_golden"](tensors)

        assert torch.equal(tensors["out"], torch.ones((4,), dtype=torch.float32))

    def test_extract_compute_golden_inlines_bound_self_attributes(self):
        """Bound method extraction should inline simple self attributes."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        compute_golden_src = _extract_compute_golden(_BoundGoldenCase(scale=2.5).compute_expected)

        assert "self._scale" not in compute_golden_src
        assert "2.5" in compute_golden_src

        src = generate_golden_source(specs, None, 1e-5, 1e-5, compute_golden_src=compute_golden_src)
        namespace: dict[str, object] = {}
        exec(src, namespace)  # noqa: S102 - Test verifies generated source executes correctly.

        tensors = {
            "a": torch.ones((4,), dtype=torch.float32),
            "out": torch.zeros((4,), dtype=torch.float32),
        }
        namespace["compute_golden"](tensors)

        assert torch.equal(tensors["out"], torch.full((4,), 2.5, dtype=torch.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
