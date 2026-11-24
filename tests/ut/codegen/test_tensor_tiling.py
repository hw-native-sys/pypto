# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile tensor regions whose working sets exceed on-chip memory."""

import random
import re
import textwrap
from collections import Counter
from contextlib import nullcontext

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.compile_profiling import CompileProfiler
from pypto.language.parser.diagnostics import InvalidOperationError

pytestmark = pytest.mark.ptoas_compile


def _compile_program(tmp_path, program: ir.Program) -> None:
    ir.compile(
        program,
        output_dir=str(tmp_path),
        dump_passes=False,
        skip_ptoas=find_ptoas_binary() is None,
    )


def _compile_region(tmp_path, a, b, out, expr, *, dtype, out_dtype, scoped=True):
    inputs = f"a: pl.Tensor[{a}, pl.{dtype}]"
    if b is not None:
        inputs += f", b: pl.Tensor[{b}, pl.{dtype}]"
    result = f"pl.Tensor[{out}, pl.{out_dtype}]"
    kind = "Orchestration" if scoped else "InCore"
    scope = '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):\n' if scoped else ""
    body = textwrap.indent(f"out[:] = {expr}", "            " if scoped else "        ")
    source = (
        f"@pl.program\nclass Compute:\n"
        f"    @pl.function(type=pl.FunctionType.{kind})\n"
        f"    def main(self, {inputs}, out: pl.Out[{result}]) -> {result}:\n"
        f"{scope}{body}\n        return out\n"
    )
    _compile_program(tmp_path, pl.parse_program(source))


def _pto_sources(tmp_path):
    return {path.stem: path.read_text() for path in tmp_path.rglob("*.pto")}


def test_whole_vector_region_can_reuse_an_intermediate_buffer(tmp_path, ascend_backend):
    # Each logical tile is 64 KiB. The original allocation reuses the same
    # physical buffer for the load and both arithmetic results.
    shape = [128, 128]
    _compile_region(
        tmp_path, shape, [1], shape, "pl.mul(pl.add(a, 1.0), 2.0)", dtype="FP32", out_dtype="FP32"
    )

    sources = _pto_sources(tmp_path)
    assert set(sources) == {"compute"}
    compute = sources["compute"]
    assert "scf.for" not in compute
    assert compute.count("pto.alloc_tile ") == 1
    assert "rows=128, cols=128" in compute


def test_vector_region_still_splits_when_three_tiles_are_live(tmp_path, ascend_backend):
    source = textwrap.dedent(
        """\
        @pl.program
        class Fanout:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[128, 128], pl.FP32], out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    first = pl.add(a, 1.0)
                    second = pl.mul(a, 2.0)
                    third = pl.mul(a, 3.0)
                    out[:] = pl.add(pl.add(first, third), second)
                return out
        """
    )
    _compile_program(tmp_path, pl.parse_program(source))

    # Unlike a chain, first, second, and third all survive until the final
    # expression. Three live 64 KiB tiles exceed the 184 KiB Vec capacity.
    sources = _pto_sources(tmp_path)
    assert "scf.for" in sources["compute"]
    assert "compute_writeback" in sources


@pytest.mark.parametrize(
    "a,b,out,expr,instruction,split_reduction",
    [
        pytest.param([196613], [196613], [196613], "pl.add(a, b)", "tadd", False, id="rank-one-tail"),
        pytest.param(
            [1537, 1027],
            [1537, 1027],
            [1537, 1027],
            "pl.mul(pl.add(a, b), a)",
            "tmul",
            False,
            id="pointwise-chain",
        ),
        pytest.param(
            [768, 2048],
            [1, 2048],
            [768, 2048],
            "pl.mul(a, b)",
            "tcolexpand",
            False,
            id="row-broadcast",
        ),
        pytest.param(
            [3, 129, 1031],
            [1, 1, 1031],
            [3, 129, 1031],
            "pl.mul(pl.add(a, b), a)",
            "tcolexpand",
            False,
            id="batch-broadcast",
        ),
        pytest.param(
            [769, 1027],
            [769, 1],
            [769, 1027],
            "pl.sub(b, a)",
            "trowexpand",
            False,
            id="column-broadcast",
        ),
        pytest.param(
            [1027, 1024],
            [1, 1],
            [1027, 1],
            "pl.row_sum(a)",
            "trowsum",
            False,
            id="row-sum",
        ),
        pytest.param(
            [1024, 1027],
            [1, 1],
            [1, 1027],
            "pl.col_sum(a)",
            "tcolsum",
            False,
            id="col-sum",
        ),
        pytest.param(
            [9, 131089],
            [1, 1],
            [9, 1],
            "pl.row_sum(a)",
            "trowsum",
            True,
            id="chunked-row-sum",
        ),
        pytest.param(
            [131089, 9],
            [1, 1],
            [1, 9],
            "pl.col_sum(a)",
            "tcolsum",
            True,
            id="chunked-col-sum",
        ),
    ],
)
def test_oversized_vector_region_compiles(
    tmp_path, ascend_backend, a, b, out, expr, instruction, split_reduction
):
    _compile_region(tmp_path, a, b, out, expr, dtype="FP32", out_dtype="FP32")

    sources = _pto_sources(tmp_path)
    pto = sources["compute"]
    assert "scf.for" in pto
    assert f"pto.{instruction}" in pto
    if split_reduction and instruction == "tcolsum":
        # Default col_sum carries its previous sum into the next native
        # reduction. Adding independent partials would change row order.
        assert "pto.tadd " not in pto
        assert pto.count("pto.tcolsum ins(") >= 2
    elif split_reduction:
        assert "pto.tadd" in pto
    assert "compute_writeback" in sources
    if find_ptoas_binary() is not None:
        assert len(list((tmp_path / "kernels").rglob("*.cpp"))) == 2


@pytest.mark.parametrize(
    "a,b,out,dtype,expr,epilogue",
    [
        pytest.param(
            [1024, 384],
            [640, 1024],
            [384, 640],
            "BF16",
            "pl.matmul(a, b, a_trans=True, b_trans=True)",
            False,
            id="transposed-bf16",
        ),
        pytest.param(
            [385, 1024],
            [1024, 577],
            [385, 577],
            "BF16",
            "pl.matmul(a, b)",
            False,
            id="row-and-column-tails",
        ),
        pytest.param(
            [3, 384, 512],
            [1, 512, 640],
            [3, 384, 640],
            "BF16",
            "pl.matmul(a, b)",
            False,
            id="batch-broadcast",
        ),
        pytest.param(
            [384, 1024],
            [1024, 640],
            [384, 640],
            "FP32",
            "pl.mul(pl.matmul(a, b, out_dtype=pl.FP32), 2.0)",
            True,
            id="vector-epilogue",
        ),
    ],
)
def test_oversized_matmul_region_compiles(tmp_path, ascend_backend, a, b, out, dtype, expr, epilogue):
    _compile_region(tmp_path, a, b, out, expr, dtype="BF16", out_dtype=dtype)

    sources = _pto_sources(tmp_path)
    pto = sources["compute"]
    assert "scf.for" in pto
    assert "pto.tmatmul.acc" in pto
    if epilogue:
        assert "pto.tmuls" in sources["compute_epilogue"]
    name = "compute_epilogue" if epilogue else "compute"
    assert f"{name}_writeback" in sources
    if find_ptoas_binary() is not None:
        assert len(list((tmp_path / "kernels").rglob("*.cpp"))) == (3 if epilogue else 2)


@pytest.mark.parametrize("transpose", [False, True])
def test_unaligned_contraction_stages_zero_padded_panels(tmp_path, ascend_backend, transpose):
    m, n, k = 641, 577, 513
    a = [k, m] if transpose else [m, k]
    b = [n, k] if transpose else [k, n]
    expr = f"pl.matmul(a, b, a_trans={transpose}, b_trans={transpose}, out_dtype=pl.FP32)"
    _compile_region(tmp_path, a, b, [m, n], expr, dtype="BF16", out_dtype="FP32")

    sources = _pto_sources(tmp_path)
    for name in ("lhs_k_tail", "rhs_k_tail"):
        pad = sources[f"compute_{name}"]
        assert "pto.tfillpad" in pad
        assert "pto.tstore" in pad
    column_pad = sources["compute_rhs_k_tail" if transpose else "compute_lhs_k_tail"]
    assert "scf.for" in column_pad
    assert "pto.tmatmul.acc" in sources["compute"]
    if find_ptoas_binary() is not None:
        assert len(list((tmp_path / "kernels").rglob("*.cpp"))) == 4


def test_wide_softmax_uses_independent_windows(tmp_path, ascend_backend):
    source = textwrap.dedent(
        """\
        @pl.program
        class Compute:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[513, 4097], pl.FP32]) -> pl.Tensor[[513, 4097], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    top = pl.row_max(a)
                    weight = pl.exp(pl.sub(a, top))
                    total = pl.row_sum(weight)
                    result = pl.div(weight, total)
                return result
        """
    )
    _compile_program(tmp_path, pl.parse_program(source))

    sources = _pto_sources(tmp_path)
    maximum = next(pto for pto in sources.values() if "pto.trowmax" in pto)
    assert "scf.for" in maximum
    assert "pto.tmax" in maximum
    assert any("pto.trowsum" in pto for pto in sources.values())
    assert any("pto.tdiv" in pto for pto in sources.values())


@pytest.mark.parametrize("width, tiled", [(128, False), (196613, True)])
def test_tensor_region_return_value(tmp_path, ascend_backend, width, tiled):
    source = textwrap.dedent(
        f"""\
        @pl.program
        class Compute:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[1, {width}], pl.FP32]) -> pl.Tensor[[1, {width}], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    result = pl.add(a, 2.0)
                return result
        """
    )
    _compile_program(tmp_path, pl.parse_program(source))

    sources = _pto_sources(tmp_path)
    pto = sources["compute"]
    assert ("scf.for" in pto) == tiled
    assert "pto.tadds" in pto
    assert "compute_writeback" not in sources


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param([92, 513], "FP32", id="matrix"),
        pytest.param([1, 47196], "FP32", id="row-vector"),
        pytest.param([124, 512], "BF16", id="legalized-two-hop"),
        pytest.param([125, 513], "BF16", id="legalized-two-hop-with-tail"),
    ],
)
def test_oversized_cast_budgets_backend_scratch(tmp_path, ascend_backend, shape, dtype):
    _compile_region(
        tmp_path,
        shape,
        [1],
        shape,
        'pl.cast(a, pl.INT16, saturation_mode="off")',
        dtype=dtype,
        out_dtype="INT16",
    )

    compute = _pto_sources(tmp_path)["compute"]
    assert "scf.for" in compute
    assert any("pto.tcvt ins(" in line and ", %tcvt_tmp_view" in line for line in compute.splitlines())


@pytest.mark.parametrize("explicit_shapes", [False, True])
def test_rank_one_store_rejects_multiple_valid_rows(ascend_backend, explicit_shapes):
    source = textwrap.dedent(
        """\
        @pl.program
        class Store:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, a: pl.Tensor[[8, 1], pl.FP32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
            ) -> pl.Tensor[[8], pl.FP32]:
                value = pl.tile.load(a, [0, 0], [8, 1])
                return pl.tile.store(value, [0], out)
        """
    )
    if explicit_shapes:
        source = source.replace("pl.tile.store(value, [0], out)", "pl.tile.store(value, [0], out, [8])")
    with pytest.raises(
        InvalidOperationError, match="rank-one tensor requires a tile with at most one valid row"
    ):
        pl.parse(source)


def test_assemble_into_region_local_tensor_does_not_split(tmp_path, ascend_backend):
    source = textwrap.dedent(
        """\
        @pl.program
        class Assemble:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, a: pl.Tensor[[64, 256], pl.FP32]) -> pl.Tensor[[64, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    tmp = pl.add(a, 1.0)
                    value = pl.add(tmp, 2.0)
                    result = pl.tensor.assemble(tmp, value, [0, 0])
                return result
        """
    )
    program = pl.parse(source)
    assert isinstance(program, ir.Program)
    _compile_program(tmp_path, program)

    sources = _pto_sources(tmp_path)
    assert "pto.tstore" in sources["compute"]
    assert "compute_writeback" not in sources


@pytest.mark.parametrize(
    "large,small,op",
    [
        ([1, 65536], [1, 1], "row_sum"),
        ([32768, 8], [1, 8], "col_sum"),
    ],
    ids=["row-sum", "col-sum"],
)
def test_split_reduction_uses_its_own_input_extent(tmp_path, ascend_backend, large, small, op):
    source = textwrap.dedent(
        f"""\
        @pl.program
        class Reduction:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, big: pl.Tensor[{large}, pl.FP32], small: pl.Tensor[{small}, pl.FP32]
            ) -> pl.Tensor[{small}, pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    unused = pl.add(big, 1.0)
                    result = pl.{op}(small)
                return result
        """
    )
    program = pl.parse(source)
    assert isinstance(program, ir.Program)
    _compile_program(tmp_path, program)

    reduction = _pto_sources(tmp_path)["compute_vec_1"]
    assert "pto.tstore" in reduction
    assert "%reduce_axis" not in reduction
    instruction = "pto.trowsum " if op == "row_sum" else "pto.tcolsum "
    assert reduction.count(instruction) == 1


def _generated_pointwise_graph(shape, dtype, graph, depth, *, scoped):
    # The same graph is compiled as an authored whole kernel and through automatic
    # outlining. The ordinary allocator is the oracle, independent of tiling's
    # capacity prediction. Vary operations without changing their tensor shapes.
    generator = random.Random(depth)
    tensor = f"pl.Tensor[{shape}, pl.{dtype}]"
    inputs = f"a: {tensor}"
    previous = "a"
    statements = []
    if graph in ("two-inputs", "dag"):
        inputs += f", b: {tensor}"
        statements.append("value = pl.add(a, b)")
        previous = "value"
    if graph == "fanout":
        statements.extend(
            [
                f"left = pl.add(a, pl.const(0.25, pl.{dtype}))",
                f"right = pl.mul(a, pl.const(0.5, pl.{dtype}))",
            ]
        )
    available = ["a", previous]
    for index in range(depth):
        value = f"v{index}"
        if graph == "dag":
            left, right = generator.choices(available, k=2)
            op = generator.choice(("add", "mul", "sub"))
            statements.append(f"{value} = pl.{op}({left}, {right})")
        else:
            op = generator.choice(("add", "mul", "sub"))
            statements.append(f"{value} = pl.{op}({previous}, pl.const(0.5, pl.{dtype}))")
        available.append(value)
        previous = value
    if graph == "fanout":
        statements.extend([f"combined = pl.add({previous}, left)", "result = pl.add(combined, right)"])
        previous = "result"
    statements.append(f"out[:] = {previous}")
    kind = "Orchestration" if scoped else "InCore"
    body = textwrap.indent("\n".join(statements), "            " if scoped else "        ")
    scope = '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):\n' if scoped else ""
    return (
        f"@pl.program\nclass Compute:\n"
        f"    @pl.function(type=pl.FunctionType.{kind})\n"
        f"    def main(self, {inputs}, out: pl.Out[{tensor}]) -> {tensor}:\n"
        f"{scope}{body}\n        return out\n"
    )


@pytest.mark.parametrize("dtype,element_bytes", [("FP16", 2), ("FP32", 4)])
@pytest.mark.parametrize("graph,live_tiles", [("chain", 1), ("two-inputs", 2), ("fanout", 3), ("dag", 6)])
@pytest.mark.parametrize("depth", [3, 64, 129])
@pytest.mark.parametrize("boundary", [-1, 0, 1])
def test_generated_pointwise_graphs_follow_ordinary_allocation(
    tmp_path, ascend_backend, dtype, element_bytes, graph, live_tiles, depth, boundary
):
    # Probe both sides of a physical capacity boundary for different graph
    # topologies, dtypes and lengths. Exact fits come from an independently
    # compiled whole kernel, not from a list of expected tiling decisions.
    capacity = backend.get_backend_instance(ascend_backend).get_mem_size(ir.MemorySpace.Vec)
    columns = (32 // element_bytes) * random.Random(depth + live_tiles).choice((8, 16, 32))
    rows = max(1, capacity // (live_tiles * columns * element_bytes) + boundary)
    shape = [rows, columns]
    ordinary_fits = True
    try:
        _compile_program(
            tmp_path / "ordinary",
            pl.parse_program(_generated_pointwise_graph(shape, dtype, graph, depth, scoped=False)),
        )
    except (ValueError, RuntimeError) as error:
        # Every other failure must fail this test, including toolchain and
        # lowering failures. An ordinary capacity rejection is an expected oracle.
        if "buffer usage (" not in str(error) or "exceeds platform limit" not in str(error):
            raise
        ordinary_fits = False

    automatic = tmp_path / "automatic"
    _compile_program(
        automatic, pl.parse_program(_generated_pointwise_graph(shape, dtype, graph, depth, scoped=True))
    )
    actual = _pto_sources(automatic)
    if ordinary_fits:
        expected = _pto_sources(tmp_path / "ordinary")
        assert len(actual) == len(expected) == 1
        compute = actual["compute"]
        assert "scf.for" not in compute
        assert compute.count("pto.alloc_tile ") == next(iter(expected.values())).count("pto.alloc_tile ")
    else:
        assert any("scf.for" in source for source in actual.values())


@pytest.mark.parametrize("columns", [1, 2, 7, 8, 9])
def test_streamed_column_sum_pads_its_row_result(tmp_path, ascend_backend, columns):
    capacity = backend.get_backend_instance(ascend_backend).get_mem_size(ir.MemorySpace.Vec)
    rows = capacity // 4 + 1
    _compile_region(
        tmp_path, [rows, columns], [1], [1, columns], "pl.col_sum(a)", dtype="FP32", out_dtype="FP32"
    )
    compute = _pto_sources(tmp_path)["compute"]
    assert compute.count("pto.tcolsum ins(") >= 2
    assert "pto.tadd " not in compute
    assert "scf.for" in compute


@pytest.mark.parametrize("enabled", [False, True])
def test_window_plan_reports_only_selected_execution_costs(tmp_path, ascend_backend, enabled):
    phase = passes.DiagnosticPhase.PRE_PIPELINE if enabled else passes.DiagnosticPhase.NONE
    # compile() already routes performance hints into its report directory.
    # Disabled diagnostics must also suppress a planner's direct hints.
    with passes.PassContext([], diagnostic_phase=phase):
        _compile_region(
            tmp_path,
            [768, 2048],
            [1, 2048],
            [768, 2048],
            "pl.mul(a, b)",
            dtype="FP32",
            out_dtype="FP32",
        )
    path = tmp_path / "report" / "perf_hints.log"
    hints = path.read_text() if path.exists() else ""
    chosen = [line for line in hints.splitlines() if "PH-TENSOR-001" in line]
    assert len(chosen) == int(enabled)
    if enabled:
        assert "Scope 'compute'" in chosen[0]
        assert "Vector output windows" in chosen[0]
        assert "compiler-owned GM output of 6291456 bytes" in chosen[0]
        assert "separate writeback task" in chosen[0]


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_streamed_reduction_reports_input_chunks(tmp_path, ascend_backend, axis):
    capacity = backend.get_backend_instance(ascend_backend).get_mem_size(ir.MemorySpace.Vec)
    extent = capacity // 4 + 17
    shape, output, expr = (
        ([extent, 8], [1, 8], "pl.col_sum(a)") if axis == "rows" else ([1, extent], [1, 1], "pl.row_sum(a)")
    )
    _compile_region(tmp_path, shape, [1], output, expr, dtype="FP32", out_dtype="FP32")
    hints = (tmp_path / "report" / "perf_hints.log").read_text()
    matches = re.findall(
        rf"Each output window streams (\d+) input {axis} in (\d+) chunks of up to (\d+)\.", hints
    )
    assert len(matches) == 1
    reported_extent, count, step = map(int, matches[0])
    # The input-axis plan is independent of the much smaller output window.
    # Do not pin a chosen chunk size: check complete coverage instead.
    assert reported_extent == extent
    assert 0 < step < extent
    assert count == (extent + step - 1) // step
    compute = _pto_sources(tmp_path)["compute"]
    # Check the reported length against the emitted input loop, independent of
    # SSA names and any outer loop over output windows.
    constants = {
        name: int(value) for name, value in re.findall(r"(%[\w.$-]+) = arith.constant (\d+) : index", compute)
    }
    loops = re.findall(r"scf.for %[\w.$-]+ = (%[\w.$-]+) to (%[\w.$-]+) step (%[\w.$-]+)", compute)
    bounds = [tuple(constants.get(name) for name in loop) for loop in loops]
    assert (step, extent - extent % step, step) in bounds


@pytest.mark.parametrize("rows, attempts", [(128, 1), (1024, 2)])
@pytest.mark.parametrize("dump_passes", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_allocation_feedback_preserves_success_and_retries_failure(
    tmp_path, ascend_backend, rows, attempts, dump_passes, profile
):
    source = textwrap.dedent(
        f"""\
        @pl.program
        class Compute:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[{rows}, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[{rows}, 128], pl.FP32]]
            ) -> pl.Tensor[[{rows}, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="compute"):
                    out[:] = pl.add(a, 1.0)
                return out
        """
    )
    program = pl.parse_program(source)
    before, after = Counter(), Counter()

    def record_before(pass_obj, _program):
        before[pass_obj.get_name()] += 1

    def record_after(pass_obj, _program):
        after[pass_obj.get_name()] += 1

    instrument = passes.CallbackInstrument(before_pass=record_before, after_pass=record_after)
    with passes.PassContext([instrument]), CompileProfiler() if profile else nullcontext():
        ir.compile(
            program,
            output_dir=str(tmp_path),
            dump_passes=dump_passes,
            skip_ptoas=find_ptoas_binary() is None,
        )

    # Failed allocation has a before hook but no successful after hook. A
    # normal fit traverses every pass once and keeps its single native kernel.
    assert before["AllocateMemoryAddr"] == attempts
    assert after["AllocateMemoryAddr"] == 1
    assert before["OutlineIncoreScopes"] == attempts
    assert before["ConvertToSSA"] == 1
    sources = _pto_sources(tmp_path)
    assert ("compute_writeback" in sources) == (attempts > 1)
    assert ("scf.for" in sources["compute"]) == (attempts > 1)
    if attempts == 1:
        assert set(sources) == {"compute"}
        assert sources["compute"].count("pto.alloc_tile ") == 1
    if dump_passes:
        dumped = list(tmp_path.rglob("*_after_AllocateMemoryAddr.py"))
        assert len(dumped) == 1


@pytest.mark.parametrize("failures", [1, 2, 16])
@pytest.mark.parametrize("kind", ["vector", "matrix"])
@pytest.mark.parametrize("identical", [False, True], ids=["distinct", "deduplicated"])
def test_capacity_retry_changes_only_the_failed_scope(tmp_path, ascend_backend, failures, kind, identical):
    shape = [1024, 128] if kind == "vector" else [384, 640]
    expressions = [
        f"pl.add(b, {1 if identical else index + 1}.0)"
        if kind == "vector"
        else f"pl.mul(pl.matmul(lhs, rhs, out_dtype=pl.FP32), {1 if identical else index + 1}.0)"
        for index in range(failures)
    ]
    names = (["failed", "second_failed"] + [f"failed_{index}" for index in range(2, failures)])[:failures]
    scopes = "\n".join(
        f'        with pl.at(level=pl.Level.CORE_GROUP, name_hint="{name}"):\n'
        f"            out[:] = {expression}\n"
        for name, expression in zip(names, expressions, strict=True)
    )
    source = textwrap.dedent(
        f"""\
        @pl.program
        class Compute:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[1024, 128], pl.FP32],
                lhs: pl.Tensor[[384, 1024], pl.BF16], rhs: pl.Tensor[[1024, 640], pl.BF16],
                small: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
                out: pl.Out[pl.Tensor[{shape}, pl.FP32]]
            ) -> pl.Tensor[{shape}, pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="healthy"):
                    small[:] = pl.add(a, 1.0)
                return out
        """
    )
    source = source.replace("        return out", scopes + "        return out")
    healthy = []
    outlined = []

    def record(pass_obj, program):
        if pass_obj.get_name() == "OutlineIncoreScopes":
            healthy.append(program.get_function("healthy"))
            outlined.append(
                sum(func.func_type == ir.FunctionType.InCore for func in program.functions.values())
            )

    with passes.PassContext([passes.CallbackInstrument(after_pass=record)]):
        _compile_program(tmp_path, pl.parse_program(source))

    # All failed kernels in one capacity stage are reported together. One,
    # two or sixteen failures therefore need exactly one common replay.
    assert len(healthy) == 2
    assert outlined[0] == failures + 1
    ir.assert_structural_equal(healthy[0], healthy[1], enable_auto_mapping=True)
    sources = _pto_sources(tmp_path)
    assert "scf.for" not in sources["healthy"]
    assert "healthy_writeback" not in sources
    assert any(name.endswith("_writeback") for name in sources)
    if kind == "vector" and not identical:
        for name in names:
            assert "scf.for" in sources[name]
    else:
        # Deduplication may share the entire computation or only its Cube stage.
        instruction = "pto.tadds" if kind == "vector" else "pto.tmatmul"
        assert any("scf.for" in source and instruction in source for source in sources.values())


def test_allocation_retry_does_not_catch_unrelated_errors(tmp_path, ascend_backend):
    attempts = Counter()

    def fail_before_allocation(pass_obj, _program):
        name = pass_obj.get_name()
        attempts[name] += 1
        if name == "AllocateMemoryAddr":
            raise ValueError("unrelated compilation failure")

    with passes.PassContext([passes.CallbackInstrument(before_pass=fail_before_allocation)]):
        with pytest.raises(ValueError, match="unrelated compilation failure"):
            _compile_region(
                tmp_path, [1024, 128], None, [1024, 128], "pl.add(a, 1.0)", dtype="FP32", out_dtype="FP32"
            )
    assert attempts["OutlineIncoreScopes"] == 1
    assert attempts["AllocateMemoryAddr"] == 1


def test_authored_kernel_capacity_failure_remains_an_error(tmp_path, ascend_backend):
    # No dispatchable Tensor scope exists to replan. Do not recursively run
    # the same failed allocation or silently rewrite an authored kernel.
    with pytest.raises(ValueError, match="buffer usage .*exceeds platform limit"):
        _compile_region(
            tmp_path,
            [1024, 128],
            None,
            [1024, 128],
            "pl.add(a, 1.0)",
            dtype="FP32",
            out_dtype="FP32",
            scoped=False,
        )


@pytest.mark.parametrize("columns", [1025, 2049])
@pytest.mark.parametrize(
    "expr",
    [
        "pl.matmul(pl.matmul(a, b, out_dtype=pl.BF16), b, out_dtype=pl.FP32)",
        "pl.matmul_acc(pl.cast(a, pl.FP32), a, b)",
        "pl.matmul_acc(pl.matmul(a, b, out_dtype=pl.FP32), a, b)",
    ],
    ids=["chained-products", "gm-accumulator", "product-accumulator"],
)
def test_matrix_capacity_feedback_precedes_consumer_shape_deduction(tmp_path, ascend_backend, columns, expr):
    # Ordinary conversion otherwise reaches a mismatched padded K/N or a GM
    # accumulator before AllocateMemoryAddr. One Mat operand alone already
    # exceeds capacity, so the same typed failure must reach the upper planner.
    attempts = Counter()

    def record(pass_obj, _program):
        attempts[pass_obj.get_name()] += 1

    with passes.PassContext([passes.CallbackInstrument(before_pass=record)]):
        _compile_region(
            tmp_path, [771, columns], [columns, columns], [771, columns], expr, dtype="BF16", out_dtype="FP32"
        )

    assert attempts["OutlineIncoreScopes"] == 2
    assert attempts["ConvertToSSA"] == 1
    sources = _pto_sources(tmp_path)
    assert any("scf.for" in source and "pto.tmatmul" in source for source in sources.values())


def test_small_matrix_product_keeps_ordinary_allocation(tmp_path, ascend_backend):
    attempts = Counter()

    def record(pass_obj, _program):
        attempts[pass_obj.get_name()] += 1

    with passes.PassContext([passes.CallbackInstrument(before_pass=record)]):
        _compile_region(
            tmp_path,
            [64, 64],
            [64, 64],
            [64, 64],
            "pl.matmul(a, b, out_dtype=pl.FP32)",
            dtype="BF16",
            out_dtype="FP32",
        )

    assert attempts["OutlineIncoreScopes"] == 1
    assert attempts["AllocateMemoryAddr"] == 1
    sources = _pto_sources(tmp_path)
    assert set(sources) == {"compute"}
    assert "scf.for" not in sources["compute"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
