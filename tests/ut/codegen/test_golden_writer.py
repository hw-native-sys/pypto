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
from pypto.runtime.golden_writer import (
    _extract_callable_expr,
    _extract_closure_constants,
    _extract_compute_golden,
    generate_golden_source,
    write_golden,
)
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
    """Tests for scalar TaskArg entries in generated golden.py."""

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


def _make_arange_tensor():
    """Named helper used as init_value in callable extraction tests."""
    return torch.arange(0, 4, dtype=torch.float32)


class TestCallableInitValue:
    """Tests for callable init_value via source extraction."""

    def test_named_function_emitted_as_preamble(self):
        """Named function init_value is extracted and emitted before generate_inputs."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=_make_arange_tensor),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "def _make_arange_tensor():" in src
        assert "a = torch.as_tensor(_make_arange_tensor(), dtype=torch.float32)" in src
        assert src.index("def _make_arange_tensor") < src.index("def generate_inputs")

    def test_named_function_golden_executes(self):
        """Generated golden.py with named-function init_value is executable."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=_make_arange_tensor),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        namespace: dict[str, object] = {}
        exec(src, namespace)  # noqa: S102 - Test verifies generated source executes correctly.

        result = namespace["generate_inputs"](None)
        a_tensor = result[0][1]
        assert torch.equal(a_tensor, torch.arange(0, 4, dtype=torch.float32))

    def test_lambda_falls_through_to_known_factory(self):
        """Lambda wrapping a known factory falls through to factory path."""
        # Lambda __name__ is '<lambda>' — not a valid identifier, so
        # _extract_callable_expr returns None.  torch.randn itself is the
        # init_value here; a lambda would hit the error path.
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=torch.randn),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "torch.randn" in src

    def test_extract_callable_expr_rejects_lambda(self):
        """_extract_callable_expr returns None for lambdas (invalid __name__)."""
        fn = lambda: torch.zeros(4)  # noqa: E731
        preambles: dict[str, str] = {}
        result = _extract_callable_expr(fn, preambles)

        assert result is None
        assert len(preambles) == 0

    def test_extract_callable_expr_accepts_named_function(self):
        """_extract_callable_expr succeeds for named functions."""
        preambles: dict[str, str] = {}
        result = _extract_callable_expr(_make_arange_tensor, preambles)

        assert result == "_make_arange_tensor()"
        assert len(preambles) == 1
        assert "def _make_arange_tensor" in preambles["_make_arange_tensor"]

    def test_large_tensor_literal_raises(self):
        """Tensor init_value with >100 elements raises ValueError."""
        large_tensor = torch.arange(0, 200, dtype=torch.float32)
        specs = [
            TensorSpec("a", [200], torch.float32, init_value=large_tensor),
            TensorSpec("out", [200], torch.float32, is_output=True),
        ]
        with pytest.raises(ValueError, match="too large to inline"):
            generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

    def test_unsupported_callable_raises(self):
        """Callable that is not extractable and not a known factory raises ValueError."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=abs),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        with pytest.raises(ValueError, match="not supported"):
            generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

    def test_multiple_callable_init_values(self):
        """Multiple tensors with callable init_values each get their own preamble."""

        def make_ones():
            return torch.ones(4, dtype=torch.float32)

        specs = [
            TensorSpec("a", [4], torch.float32, init_value=_make_arange_tensor),
            TensorSpec("b", [4], torch.float32, init_value=make_ones),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "def _make_arange_tensor" in src
        assert "def make_ones" in src
        assert "a = torch.as_tensor(_make_arange_tensor(), dtype=torch.float32)" in src
        assert "b = torch.as_tensor(make_ones(), dtype=torch.float32)" in src


class TestExtractClosureConstants:
    """Tests for _extract_closure_constants."""

    def test_closure_variable_captured(self):
        """Closure variables with simple scalar values are extracted."""
        scale = 42
        offset = 3.14

        def fn(tensors, params=None):
            tensors["out"][:] = tensors["a"] * scale + offset

        lines = _extract_closure_constants(fn)
        assert "scale = 42" in lines
        assert "offset = 3.14" in lines

    def test_closure_string_and_none(self):
        """String and None closure values are captured."""
        tag = "hello"
        sentinel = None

        def fn(tensors, params=None):
            _ = tag, sentinel

        lines = _extract_closure_constants(fn)
        assert "tag = 'hello'" in lines
        assert "sentinel = None" in lines

    def test_closure_non_simple_types_skipped(self):
        """Non-scalar closure values (lists, dicts, objects) are silently skipped."""
        data = [1, 2, 3]

        def fn(tensors, params=None):
            _ = data

        lines = _extract_closure_constants(fn)
        assert len(lines) == 0

    def test_non_finite_floats_skipped(self):
        """nan and inf closure values are skipped (repr produces invalid Python)."""
        nan_val = float("nan")
        inf_val = float("inf")

        def fn(tensors, params=None):
            _ = nan_val, inf_val

        lines = _extract_closure_constants(fn)
        assert len(lines) == 0

    def test_no_closure_returns_empty(self):
        """Functions without closures return an empty list."""

        def fn(tensors, params=None):
            tensors["out"][:] = tensors["a"] * 2

        lines = _extract_closure_constants(fn)
        assert lines == []

    def test_closure_injected_into_extract_compute_golden(self):
        """Closure constants appear above function def in _extract_compute_golden output."""
        factor = 5

        def my_golden(tensors, params=None):
            tensors["out"][:] = tensors["a"] * factor

        src = _extract_compute_golden(my_golden)
        assert "factor = 5" in src
        assert src.index("factor = 5") < src.index("def compute_golden(")


class TestDataFileMode:
    """Tests for data-file persistence mode (data_dir parameter)."""

    def test_golden_source_uses_torch_load(self, tmp_path):
        """Input tensors use torch.load from in/, pure outputs use torch.zeros."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(
            specs,
            _dummy_golden,
            1e-5,
            1e-5,
            data_dir=tmp_path,
        )

        assert 'torch.load(_DATA_DIR / "in" / "a.pt"' in src
        assert "torch.zeros" in src
        assert "torch.full" not in src
        assert 'torch.load(_DATA_DIR / "in" / "out.pt"' not in src

    def test_explicit_data_dir_uses_absolute_path(self, tmp_path):
        """Explicit data_dir generates _DATA_DIR with absolute path."""
        specs = [TensorSpec("x", [2], torch.float32, init_value=1.0)]
        src = generate_golden_source(
            specs,
            _dummy_golden,
            1e-5,
            1e-5,
            data_dir=tmp_path,
        )

        assert f'_DATA_DIR = Path("{tmp_path.resolve()}")' in src
        assert "from pathlib import Path" in src

    def test_use_data_files_generates_relative_path(self):
        """use_data_files=True without data_dir generates portable relative path."""
        specs = [TensorSpec("x", [2], torch.float32, init_value=1.0)]
        src = generate_golden_source(
            specs,
            _dummy_golden,
            1e-5,
            1e-5,
            use_data_files=True,
        )

        assert '_DATA_DIR = Path(__file__).parent / "data"' in src
        assert "from pathlib import Path" in src
        assert 'torch.load(_DATA_DIR / "in" / "x.pt"' in src

    def test_no_data_dir_legacy_mode(self):
        """_DATA_DIR is absent when data_dir is None (legacy mode)."""
        specs = [TensorSpec("x", [2], torch.float32, init_value=1.0)]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "_DATA_DIR" not in src
        assert "from pathlib import Path" not in src

    def test_legacy_mode_preserves_inline(self):
        """Legacy mode (data_dir=None) keeps inline expressions."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(specs, _dummy_golden, 1e-5, 1e-5)

        assert "torch.full" in src
        assert "torch.zeros" in src
        assert "torch.load" not in src

    def test_write_golden_creates_data_dir(self, tmp_path):
        """write_golden creates data/in/ and data/out/ directories with .pt files."""
        specs = [
            TensorSpec("a", [8], torch.float32, init_value=torch.randn),
            TensorSpec("b", [8], torch.float32, init_value=2.0),
            TensorSpec("out", [8], torch.float32, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"
        write_golden(specs, _dummy_golden, golden_path)

        in_dir = tmp_path / "data" / "in"
        out_dir = tmp_path / "data" / "out"
        assert in_dir.is_dir()
        assert out_dir.is_dir()
        assert (in_dir / "a.pt").exists()
        assert (in_dir / "b.pt").exists()
        assert not (in_dir / "out.pt").exists()
        assert (out_dir / "out.pt").exists()

    def test_write_golden_roundtrip(self, tmp_path):
        """Generated golden.py is executable and loads persisted data."""
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=3.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"
        write_golden(specs, _dummy_golden, golden_path)

        src = golden_path.read_text(encoding="utf-8")
        assert '_DATA_DIR = Path(__file__).parent / "data"' in src

        namespace: dict[str, object] = {"__file__": str(golden_path)}
        exec(compile(src, str(golden_path), "exec"), namespace)  # noqa: S102

        result = namespace["generate_inputs"](None)
        a_tensor = result[0][1]
        assert torch.equal(a_tensor, torch.full((4,), 3.0, dtype=torch.float32))

        out_tensor = result[1][1]
        assert torch.equal(out_tensor, torch.zeros(4, dtype=torch.float32))

        out_dir = tmp_path / "data" / "out"
        assert out_dir.is_dir()
        golden_out = torch.load(out_dir / "out.pt", weights_only=True)
        assert torch.equal(golden_out, torch.full((4,), 9.0, dtype=torch.float32))

    def test_write_golden_large_tensor(self, tmp_path):
        """Large tensors (>100 elements) work in data-file mode."""
        large = torch.arange(0, 200, dtype=torch.float32)
        specs = [
            TensorSpec("big", [200], torch.float32, init_value=large),
            TensorSpec("out", [200], torch.float32, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"

        def golden_big(tensors, params=None):
            tensors["out"][:] = tensors["big"] * 2

        write_golden(specs, golden_big, golden_path)

        in_dir = tmp_path / "data" / "in"
        assert (in_dir / "big.pt").exists()

        loaded = torch.load(in_dir / "big.pt", weights_only=True)
        assert torch.equal(loaded, large)

    def test_write_golden_random_factory_persisted(self, tmp_path):
        """torch.randn init_value is materialised once and persisted."""
        specs = [
            TensorSpec("r", [16], torch.float32, init_value=torch.randn),
            TensorSpec("out", [16], torch.float32, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"

        def golden_r(tensors, params=None):
            tensors["out"][:] = tensors["r"] * 2

        write_golden(specs, golden_r, golden_path)

        saved = torch.load(tmp_path / "data" / "in" / "r.pt", weights_only=True)

        src = golden_path.read_text(encoding="utf-8")
        namespace: dict[str, object] = {"__file__": str(golden_path)}
        exec(compile(src, str(golden_path), "exec"), namespace)  # noqa: S102
        result = namespace["generate_inputs"](None)
        loaded = result[0][1]

        assert torch.equal(loaded, saved)

    def test_scalars_still_inline_in_data_file_mode(self, tmp_path):
        """ScalarSpec entries remain inline ctypes even with data_dir."""
        specs = [
            TensorSpec("x", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        scalars = [ScalarSpec("factor", 3, ctype="int64")]
        src = generate_golden_source(
            specs,
            _dummy_golden,
            1e-5,
            1e-5,
            scalar_specs=scalars,
            data_dir=tmp_path,
        )

        assert "factor = ctypes.c_int64(3)" in src
        assert "import ctypes" in src

    def test_external_data_dir_generates_absolute_path(self, tmp_path):
        """When data_dir is specified, generated source uses absolute path."""
        ext_dir = tmp_path / "external_data"
        ext_dir.mkdir()
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        src = generate_golden_source(
            specs,
            _dummy_golden,
            1e-5,
            1e-5,
            data_dir=ext_dir,
        )

        assert f'_DATA_DIR = Path("{ext_dir.resolve()}")' in src
        assert 'torch.load(_DATA_DIR / "in" / "a.pt"' in src
        assert "from pathlib import Path" in src

    def test_write_golden_reuses_existing_data_dir(self, tmp_path):
        """write_golden with existing data_dir reuses files without regeneration."""
        ext_dir = tmp_path / "ext"
        in_dir = ext_dir / "in"
        out_dir = ext_dir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        expected = torch.full((4,), 7.0, dtype=torch.float32)
        torch.save(expected, in_dir / "a.pt")
        torch.save(torch.zeros(4, dtype=torch.float32), out_dir / "out.pt")

        golden_path = tmp_path / "out" / "golden.py"
        golden_path.parent.mkdir()
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=1.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        write_golden(specs, _dummy_golden, golden_path, data_dir=ext_dir)

        assert not (golden_path.parent / "data").exists()

        src = golden_path.read_text(encoding="utf-8")
        namespace: dict[str, object] = {"__file__": str(golden_path)}
        exec(compile(src, str(golden_path), "exec"), namespace)  # noqa: S102

        result = namespace["generate_inputs"](None)
        a_tensor = result[0][1]
        assert torch.equal(a_tensor, expected)

    def test_write_golden_creates_missing_data_dir(self, tmp_path):
        """write_golden creates data_dir and generates data when dir is absent."""
        ext_dir = tmp_path / "new_data"
        golden_path = tmp_path / "golden.py"
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=3.0),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]
        write_golden(specs, _dummy_golden, golden_path, data_dir=ext_dir)

        assert (ext_dir / "in").is_dir()
        assert (ext_dir / "out").is_dir()
        assert (ext_dir / "in" / "a.pt").exists()
        assert not (ext_dir / "in" / "out.pt").exists()
        assert (ext_dir / "out" / "out.pt").exists()
        assert not (tmp_path / "data").exists()

        src = golden_path.read_text(encoding="utf-8")
        assert f'_DATA_DIR = Path("{ext_dir.resolve()}")' in src

        namespace: dict[str, object] = {"__file__": str(golden_path)}
        exec(compile(src, str(golden_path), "exec"), namespace)  # noqa: S102
        result = namespace["generate_inputs"](None)
        a_tensor = result[0][1]
        assert torch.equal(a_tensor, torch.full((4,), 3.0, dtype=torch.float32))

    def test_inout_tensor_in_both_dirs(self, tmp_path):
        """Inout tensor (is_output=True, init_value set) appears in both in/ and out/."""

        def golden_with_inout(tensors, params=None):
            tensors["acc"][:] = tensors["a"] + tensors["acc"]

        specs = [
            TensorSpec("a", [4], torch.float32, init_value=2.0),
            TensorSpec("acc", [4], torch.float32, init_value=1.0, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"
        write_golden(specs, golden_with_inout, golden_path)

        in_dir = tmp_path / "data" / "in"
        out_dir = tmp_path / "data" / "out"
        assert (in_dir / "a.pt").exists()
        assert (in_dir / "acc.pt").exists()
        assert (out_dir / "acc.pt").exists()

        in_acc = torch.load(in_dir / "acc.pt", weights_only=True)
        assert torch.equal(in_acc, torch.full((4,), 1.0, dtype=torch.float32))

        out_acc = torch.load(out_dir / "acc.pt", weights_only=True)
        assert torch.equal(out_acc, torch.full((4,), 3.0, dtype=torch.float32))

        src = golden_path.read_text(encoding="utf-8")
        assert 'torch.load(_DATA_DIR / "in" / "a.pt"' in src
        assert 'torch.load(_DATA_DIR / "in" / "acc.pt"' in src

    def test_pure_output_not_in_input_dir(self, tmp_path):
        """Pure output tensors must not have files in data/in/."""

        def golden_xy(tensors, params=None):
            tensors["y"][:] = tensors["x"] * 5

        specs = [
            TensorSpec("x", [4], torch.float32, init_value=1.0),
            TensorSpec("y", [4], torch.float32, is_output=True),
        ]
        golden_path = tmp_path / "golden.py"
        write_golden(specs, golden_xy, golden_path)

        in_dir = tmp_path / "data" / "in"
        out_dir = tmp_path / "data" / "out"
        assert (in_dir / "x.pt").exists()
        assert not (in_dir / "y.pt").exists()
        assert (out_dir / "y.pt").exists()

    def test_repeated_runs_reuse_data_dir(self, tmp_path):
        """Multiple write_golden calls with same data_dir reuse data, skipping golden_fn."""
        data_dir = tmp_path / "shared_data"
        specs = [
            TensorSpec("a", [4], torch.float32, init_value=torch.randn),
            TensorSpec("out", [4], torch.float32, is_output=True),
        ]

        call_log: list[int] = []

        def counting_golden(tensors, params=None):
            call_log.append(1)
            tensors["out"][:] = tensors["a"] * 3

        golden_path_1 = tmp_path / "run1" / "golden.py"
        golden_path_1.parent.mkdir()
        write_golden(specs, counting_golden, golden_path_1, data_dir=data_dir)
        assert len(call_log) == 1

        saved_a = torch.load(data_dir / "in" / "a.pt", weights_only=True)
        saved_out = torch.load(data_dir / "out" / "out.pt", weights_only=True)

        golden_path_2 = tmp_path / "run2" / "golden.py"
        golden_path_2.parent.mkdir()
        write_golden(specs, counting_golden, golden_path_2, data_dir=data_dir)
        assert len(call_log) == 1

        assert torch.equal(torch.load(data_dir / "in" / "a.pt", weights_only=True), saved_a)
        assert torch.equal(torch.load(data_dir / "out" / "out.pt", weights_only=True), saved_out)

        for gp in (golden_path_1, golden_path_2):
            src = gp.read_text(encoding="utf-8")
            ns: dict[str, object] = {"__file__": str(gp)}
            exec(compile(src, str(gp), "exec"), ns)  # noqa: S102
            result = ns["generate_inputs"](None)
            assert torch.equal(result[0][1], saved_a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
