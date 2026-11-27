# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Input-layout contracts through the pass and both runtime lowering pipelines."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType


def _compile_in_runtime(program, output_dir: Path, runtime):
    outer = passes.PassContext.current()
    assert outer is not None
    # Keep the repository's property and roundtrip instruments in both modes.
    with passes.PassContext(outer.get_instruments(), outer.get_verification_level(), runtime=runtime):
        return ir.compile(
            program,
            output_dir=str(output_dir),
            skip_ptoas=True,
            backend_type=BackendType.Ascend950,
            dump_passes=True,
        )


def _assert_runtime_pitch(output_dir: Path, factor: int):
    paths = list(output_dir.rglob("kernel.pto"))
    assert len(paths) == 1
    source = paths[0].read_text()
    view = re.search(
        r"pto\.make_tensor_view %arg0, shape = \[([^\]]+)\], strides = \[([^\]]+)\]",
        source,
    )
    assert view is not None, source
    shape, strides = (group.split(", ") for group in view.groups())
    width, pitch = shape[1], strides[0]
    if factor == 1:
        assert pitch == width, source
    else:
        constant = re.search(rf"(%\w+) = arith\.constant {factor} : index", source)
        assert constant is not None, source
        scale = constant.group(1)
        assert (
            f"{pitch} = arith.muli {scale}, {width}" in source
            or f"{pitch} = arith.muli {width}, {scale}" in source
        ), source


@pytest.mark.parametrize(
    "runtime",
    [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH],
    ids=["inlined_graph", "retained_graph"],
)
@pytest.mark.parametrize(
    ("width", "source_pitch", "kernel_factor", "matches"),
    [
        ("W", None, 1, True),
        ("W", "2 * W", 2, True),
        ("W", None, 2, False),
        ("W", "3 * W", 2, False),
        ("2 * W", None, 2, False),
        ("2 * W", "4 * W", 2, True),
    ],
    ids=["packed", "pitched", "packed_mismatch", "pitched_mismatch", "rebound_mismatch", "rebound_match"],
)
def test_symbolic_input_layout_uses_actual_shape(
    tmp_path, runtime, width, source_pitch, kernel_factor, matches
):
    source_view = "" if source_pitch is None else f", pl.TensorView(stride=[{source_pitch}, 1])"
    kernel_pitch = "W" if kernel_factor == 1 else f"{kernel_factor} * W"
    before = pl.parse_program(f"""
W = pl.dynamic("W")
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[8, W], pl.FP32, pl.TensorView(stride=[{kernel_pitch}, 1])],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        value = pl.tile.load(x, [1, 0], [1, 16])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Graph)
    def forward(
        self, x: pl.Tensor[[8, W], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        return self.kernel(x, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, basis: pl.Tensor[[W], pl.FP32], x: pl.Tensor[[8, {width}], pl.FP32{source_view}],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        return self.forward(x, out)
""")
    # The same symbol is rebound from each function's own argument descriptor.
    # A packed [8, 2*W] caller must not match the kernel's 2*W pitch by spelling.
    if not matches:
        with pytest.raises(ValueError, match="physical strides"):
            passes.optimize_orch_tensors()(before)
        with pytest.raises(ValueError, match="physical strides"):
            _compile_in_runtime(before, tmp_path, runtime)
    else:
        _compile_in_runtime(before, tmp_path, runtime)
        _assert_runtime_pitch(tmp_path, kernel_factor)


@pytest.mark.parametrize("order", [("narrow", "wide"), ("wide", "narrow")])
@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH]
)
def test_equal_symbolic_pitch_requires_compatible_shapes(tmp_path, order, runtime):
    before = pl.parse_program(f"""
W = pl.dynamic("W")
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[8, W], pl.FP32, pl.TensorView(stride=[2 * W, 1])],
        out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        value = pl.tile.load(x, [1, 0], [1, 16])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Graph)
    def forward(
        self, x: pl.Tensor[[8, W], pl.FP32], out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        return self.kernel(x, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, basis: pl.Tensor[[W], pl.FP32],
        narrow: pl.Tensor[[8, W], pl.FP32, pl.TensorView(stride=[2 * W, 1])],
        wide: pl.Tensor[[8, 2 * W], pl.FP32],
        out_a: pl.Out[pl.Tensor[[8, 16], pl.FP32]], out_b: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
    ) -> tuple[pl.Tensor[[8, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        a = self.forward({order[0]}, out_a)
        b = self.forward({order[1]}, out_b)
        return a, b
""")
    # Both physical pitches spell 2*W, but only narrow has twice its own width.
    with pytest.raises(ValueError, match="physical strides"):
        passes.optimize_orch_tensors()(before)
    with pytest.raises(ValueError, match="physical strides"):
        _compile_in_runtime(before, tmp_path, runtime)


@pytest.mark.parametrize("pitch", ["4 * W", "2 * W"], ids=["matching", "mismatching"])
@pytest.mark.parametrize("path", ["transpose", "layout_view", "valid_shape", "full_slice", "slab"])
def test_metadata_views_keep_shape_and_pitch_in_same_coordinates(path, pitch):
    operand = {
        "transpose": "pl.tensor.transpose(x, -2, -1)",
        "layout_view": "pl.tensor.view(x, layout=pl.TensorLayout.DN)",
        "valid_shape": "pl.tensor.set_validshape(x, 8, W)",
        "full_slice": "pl.tensor.slice(x, [8, W], [0, 0])",
        "slab": "pl.tensor.slice(x, [4, W], [0, 0])",
    }[path]
    transposed = path in ("transpose", "layout_view")
    kernel_shape = "[W, H]" if transposed else "[H, W]"
    kernel_pitch = "(H // 2) * W" if path == "slab" else "2 * W"
    view = (
        f"pl.TensorView(stride=[1, {kernel_pitch}], layout=pl.TensorLayout.DN)"
        if transposed
        else f"pl.TensorView(stride=[{kernel_pitch}, 1])"
    )
    offset = "[0, 1]" if transposed else "[1, 0]"
    before = pl.parse_program(f"""
W = pl.dynamic("W")
H = pl.dynamic("H")
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[{kernel_shape}, pl.FP32, {view}],
        out: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
    ) -> pl.Tensor[[1, 1], pl.FP32]:
        value = pl.tile.load(x, {offset}, [1, 1])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Graph)
    def forward(
        self, x: pl.Tensor[[8, W], pl.FP32], out: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
    ) -> pl.Tensor[[1, 1], pl.FP32]:
        view = {operand}
        return self.kernel(view, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, basis: pl.Tensor[[W], pl.FP32],
        x: pl.Tensor[[8, 2 * W], pl.FP32, pl.TensorView(stride=[{pitch}, 1])],
        out: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
    ) -> pl.Tensor[[1, 1], pl.FP32]:
        return self.forward(x, out)
""")
    # The slab's H is 4, so (H//2)*W is twice its width, not four times.
    if pitch == "4 * W":
        ir.assert_structural_equal(passes.optimize_orch_tensors()(before), before)
    else:
        with pytest.raises(ValueError, match="physical strides"):
            passes.optimize_orch_tensors()(before)


@pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
@pytest.mark.parametrize("order", [("left", "other"), ("other", "left")])
def test_static_layout_variants_compile_with_loop_carried_outputs(tmp_path, submit, order):
    calls = f"a = self.kernel({order[0]}, out_a)\n        b = self.kernel({order[1]}, out_b)"
    if submit:
        calls = (
            "with pl.manual_scope():\n"
            f"            a, first = pl.submit(self.kernel, {order[0]}, out_a)\n"
            f"            b, second = pl.submit(self.kernel, {order[1]}, out_b, deps=[first])"
        )
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[8, 256], pl.FP32], out: pl.Out[pl.Tensor[[8, 256], pl.FP32]],
    ) -> pl.Tensor[[8, 256], pl.FP32]:
        for row, (carry,) in pl.range(0, 8, init_values=(out,)):
            value = pl.tile.load(x, [row, 0], [1, 256])
            next_out = pl.tile.store(value, [row, 0], carry)
            result = pl.yield_(next_out)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, parent: pl.Tensor[[16, 288], pl.FP32],
        other: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[320, 1])],
        out_a: pl.Out[pl.Tensor[[8, 256], pl.FP32]], out_b: pl.Out[pl.Tensor[[8, 256], pl.FP32]],
    ) -> tuple[pl.Tensor[[8, 256], pl.FP32], pl.Tensor[[8, 256], pl.FP32]]:
        left = pl.tensor.slice(parent, [8, 256], [0, 0])
        {calls}
        return a, b
""")
    after = passes.optimize_orch_tensors()(before)
    kernels = [fn for fn in after.functions.values() if fn.func_type == ir.FunctionType.InCore]
    assert len(kernels) == 2
    pitches = []
    for kernel in kernels:
        tensor = kernel.params[0].type
        assert isinstance(tensor, ir.TensorType)
        view = tensor.tensor_view
        assert view is not None
        pitch = view.stride[0]
        assert isinstance(pitch, ir.ConstInt)
        pitches.append(pitch.value)
    assert sorted(pitches) == [288, 320]
    ir.assert_structural_equal(after, pl.parse_program(ir.python_print(after)))
    ir.compile(before, output_dir=str(tmp_path), skip_ptoas=True, backend_type=BackendType.Ascend950)
    assert len(list(tmp_path.rglob("*.pto"))) == 2


@pytest.mark.parametrize(
    "runtime",
    [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH],
    ids=["default_tmr", "host_graph"],
)
def test_unknown_slice_extents_preserve_static_pitch(tmp_path, runtime):
    before = pl.parse_program("""
W = pl.dynamic("UnknownSliceWidth")
@pl.program
class UnknownExtentProbe:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[4, 128], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        value = pl.tile.load(x, [1, 0], [1, 16])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Graph)
    def forward(
        self, x: pl.Tensor[[8, W], pl.FP32, pl.TensorView(stride=[320, 1])],
        out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        window = pl.tensor.slice(x, [4, 128], [0, 0])
        return self.kernel(window, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[8, W], pl.FP32, pl.TensorView(stride=[320, 1])],
        out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        return self.forward(x, out)
""")
    (tmp_path / "before.py").write_text(ir.python_print(before))
    after = passes.optimize_orch_tensors()(before)
    kernel = after.get_function("kernel")
    assert kernel is not None
    tensor = kernel.params[0].type
    assert isinstance(tensor, ir.TensorType)
    view = tensor.tensor_view
    assert view is not None
    pitch = view.stride[0]
    assert isinstance(pitch, ir.ConstInt) and pitch.value == 320
    assert "slice_extent" not in ir.python_print(after)
    _compile_in_runtime(before, tmp_path / "compiled", runtime)
    paths = list((tmp_path / "compiled").rglob("*.pto"))
    assert len(paths) == 1
    source = paths[0].read_text()
    assert "slice_extent" not in source and "view_source" not in source
    pitch = re.search(r"(%\w+) = arith\.constant 320 : index", source)
    assert pitch is not None, source
    assert f"strides = [{pitch.group(1)}," in source, source
    for snapshot in (tmp_path / "compiled" / "passes_dump").glob("*.py"):
        assert "slice_extent" not in snapshot.read_text(), snapshot


@pytest.mark.parametrize("control", ["branch", "loop"])
def test_scalar_shape_facts_do_not_cross_control_flow(control):
    if control == "branch":
        body = """
        width = pl.const(32, pl.INDEX)
        if flag:
            width = pl.const(64, pl.INDEX)
        view = pl.tensor.reshape(src, [256 // width, width])
        self.kernel(view, out)
"""
    else:
        body = """
        initial_width = pl.const(32, pl.INDEX)
        for i, (width, out_iter) in pl.range(0, 2, init_values=(initial_width, out)):
            view = pl.tensor.reshape(src, [256 // width, width])
            next_out = self.kernel(view, out_iter)
            next_width = width * 2
            final_width, out_rv = pl.yield_(next_width, next_out)
"""
    result_name = "out" if control == "branch" else "out_rv"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        value = pl.tile.load(x, [1, 0], [1, 16])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, src: pl.Tensor[[4, 64], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        flag: pl.Scalar[pl.BOOL],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
{body}
        return {result_name}
""")
    # Width can be 32 or 64. Neither the last visited arm nor a loop's
    # initial scalar value proves a fixed 32-element pitch for every call.
    with pytest.raises(ValueError, match="physical strides"):
        passes.optimize_orch_tensors()(before)


@pytest.mark.parametrize("axis", [1, -1], ids=["positive_axis", "negative_axis"])
def test_descriptor_dim_aliases_prove_static_pitch(tmp_path, axis):
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        value = pl.tile.load(x, [1, 0], [1, 16])
        return pl.tile.store(value, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, src: pl.Tensor[[4, 64], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        width = pl.tensor.dim(src, {axis})
        half = width // 2
        view = pl.tensor.reshape(src, [256 // half, half])
        return self.kernel(view, out)
""")
    # Pure local descriptor facts prove half=32 without rewriting the program
    # or treating a first caller's dynamic dimension as a global constant.
    ir.assert_structural_equal(passes.optimize_orch_tensors()(before), before)
    ir.compile(before, output_dir=str(tmp_path), skip_ptoas=True, backend_type=BackendType.Ascend950)
    _assert_runtime_pitch(tmp_path, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
