# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for GM-to-Vec and GM-to-Mat ``pto.mgather``."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

ROWS = 64
COLS = 32
INDEX_ROWS = 16
ELEM_M = 8
ELEM_N = 32
MAT_M = 32
MAT_N = 32
_PL_DTYPE = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.BF16: pl.BF16,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_TORCH_DTYPE = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.int16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.int32,
    DataType.BF16: torch.bfloat16,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}
# Mat results need a legal readback consumer: TMatmul supports signed i8 and
# floating inputs. The complete MGATHER dtype set is covered by Vec ST and Mat UT.
_MAT_NUMERIC_DTYPES = [
    DataType.INT8,
    DataType.BF16,
    DataType.FP16,
    DataType.FP32,
]


class MgatherRowTestCase(PTOTestCase):
    """Row coalescing: each index selects one complete GM row."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        valid_rows: int = INDEX_ROWS,
        column_indices: bool = False,
        gather_oob: str = "undefined",
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._valid_rows = valid_rows
        self._column_indices = column_indices
        self._gather_oob = gather_oob

    def get_name(self) -> str:
        orientation = "col" if self._column_indices else "row"
        return f"mgather_row_{self._dtype.value}_{orientation}_{self._gather_oob}_v{self._valid_rows}"

    def define_tensors(self) -> list[TensorSpec]:
        dtype = _TORCH_DTYPE[self._dtype]
        table = torch.arange(ROWS * COLS).reshape(ROWS, COLS).remainder(97).to(dtype)
        index_values = (
            [-1, 64, 65, 7, 80, 0, 18, 55, 4, 70, 13, 47, 8, 38, 22, 60]
            if self._gather_oob != "undefined"
            else [63, 2, 41, 7, 31, 0, 18, 55, 4, 29, 13, 47, 8, 38, 22, 60]
        )
        index_shape = [INDEX_ROWS, 1] if self._column_indices else [1, INDEX_ROWS]
        indices = torch.tensor(index_values, dtype=torch.int32).reshape(index_shape)
        return [
            TensorSpec("mem", [ROWS, COLS], self._dtype, init_value=table),
            TensorSpec("idx", index_shape, DataType.INT32, init_value=indices),
            TensorSpec(
                "out",
                [INDEX_ROWS, COLS],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        valid_rows = self._valid_rows
        column_indices = self._column_indices
        gather_oob = self._gather_oob
        index_shape = [INDEX_ROWS, 1] if column_indices else [1, INDEX_ROWS]
        index_valid_shape = [valid_rows, 1] if column_indices else [1, valid_rows]

        @pl.program
        class MgatherRowProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[ROWS, COLS], dtype],
                idx: pl.Tensor[index_shape, pl.INT32],
                out: pl.InOut[pl.Tensor[[INDEX_ROWS, COLS], dtype]],
            ) -> pl.Tensor[[INDEX_ROWS, COLS], dtype]:
                idx_tile: pl.Tile[index_shape, pl.INT32] = pl.load(
                    idx, [0, 0], index_shape, valid_shape=index_valid_shape
                )
                result: pl.Tile[[INDEX_ROWS, COLS], dtype] = pl.tile.mgather(
                    mem, idx_tile, coalesce="row", gather_oob=gather_oob
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mem: pl.Tensor[[ROWS, COLS], dtype],
                idx: pl.Tensor[index_shape, pl.INT32],
                out: pl.InOut[pl.Tensor[[INDEX_ROWS, COLS], dtype]],
            ) -> pl.Tensor[[INDEX_ROWS, COLS], dtype]:
                return self.kernel(mem, idx, out)

        return MgatherRowProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        expected = torch.zeros_like(tensors["out"])
        indices = tensors["idx"].reshape(-1).to(torch.int64).bitwise_and(0xFFFFFFFF)
        active = indices[: self._valid_rows]
        if self._gather_oob == "clamp":
            expected[: self._valid_rows] = tensors["mem"][active.clamp(max=ROWS - 1)]
        elif self._gather_oob == "wrap":
            expected[: self._valid_rows] = tensors["mem"][active.remainder(ROWS)]
        elif self._gather_oob == "zero":
            in_bounds = active < ROWS
            expected_rows = expected[: self._valid_rows]
            expected_rows[in_bounds] = tensors["mem"][active[in_bounds]]
        else:
            expected[: self._valid_rows] = tensors["mem"][active]
        tensors["out"][:] = expected


class MgatherElemTestCase(PTOTestCase):
    """Element coalescing: every index selects one flat GM element."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        valid_shape: tuple[int, int] = (ELEM_M, ELEM_N),
        gather_oob: str = "undefined",
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._valid_shape = valid_shape
        self._gather_oob = gather_oob

    def get_name(self) -> str:
        rows, cols = self._valid_shape
        return f"mgather_elem_{self._dtype.value}_{self._gather_oob}_v{rows}x{cols}"

    def define_tensors(self) -> list[TensorSpec]:
        table_size = ELEM_M * ELEM_N
        table = torch.arange(table_size).remainder(53).to(_TORCH_DTYPE[self._dtype])
        if self._gather_oob == "undefined":
            indices = torch.arange(table_size - 1, -1, -1, dtype=torch.int32)
        else:
            indices = torch.arange(table_size, dtype=torch.int32) - 4
            indices[0:4] = torch.tensor([-1, table_size, table_size + 1, table_size * 2 + 3])
        indices = indices.reshape(ELEM_M, ELEM_N)
        return [
            TensorSpec("mem", [table_size], self._dtype, init_value=table),
            TensorSpec("idx", [ELEM_M, ELEM_N], DataType.INT32, init_value=indices),
            TensorSpec(
                "out",
                [ELEM_M, ELEM_N],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        valid_shape = list(self._valid_shape)
        gather_oob = self._gather_oob

        @pl.program
        class MgatherElemProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[ELEM_M * ELEM_N], dtype],
                idx: pl.Tensor[[ELEM_M, ELEM_N], pl.INT32],
                out: pl.InOut[pl.Tensor[[ELEM_M, ELEM_N], dtype]],
            ) -> pl.Tensor[[ELEM_M, ELEM_N], dtype]:
                idx_tile: pl.Tile[[ELEM_M, ELEM_N], pl.INT32] = pl.load(
                    idx, [0, 0], [ELEM_M, ELEM_N], valid_shape=valid_shape
                )
                result: pl.Tile[[ELEM_M, ELEM_N], dtype] = pl.tile.mgather(
                    mem, idx_tile, coalesce="elem", gather_oob=gather_oob
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mem: pl.Tensor[[ELEM_M * ELEM_N], dtype],
                idx: pl.Tensor[[ELEM_M, ELEM_N], pl.INT32],
                out: pl.InOut[pl.Tensor[[ELEM_M, ELEM_N], dtype]],
            ) -> pl.Tensor[[ELEM_M, ELEM_N], dtype]:
                return self.kernel(mem, idx, out)

        return MgatherElemProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self._valid_shape
        expected = torch.zeros_like(tensors["out"])
        indices = tensors["idx"][:rows, :cols].to(torch.int64).bitwise_and(0xFFFFFFFF)
        table_size = tensors["mem"].numel()
        if self._gather_oob == "clamp":
            expected[:rows, :cols] = tensors["mem"][indices.clamp(max=table_size - 1)]
        elif self._gather_oob == "wrap":
            expected[:rows, :cols] = tensors["mem"][indices.remainder(table_size)]
        elif self._gather_oob == "zero":
            in_bounds = indices < table_size
            gathered = torch.zeros_like(indices, dtype=tensors["mem"].dtype)
            gathered[in_bounds] = tensors["mem"][indices[in_bounds]]
            expected[:rows, :cols] = gathered
        else:
            expected[:rows, :cols] = tensors["mem"][indices]
        tensors["out"][:] = expected


class MgatherMatTestCase(PTOTestCase):
    """GM-to-L1 overload consumed through the legal ``eye @ gathered`` path."""

    __test__ = False

    def __init__(
        self,
        *,
        coalesce: str,
        dtype: DataType = DataType.FP16,
        gather_oob: str = "undefined",
        scratch_elements: int = MAT_M * MAT_N,
        valid_shape: tuple[int, int] = (MAT_M, MAT_N),
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._coalesce = coalesce
        self._dtype = dtype
        self._gather_oob = gather_oob
        self._scratch_elements = scratch_elements
        self._valid_shape = valid_shape

    def get_name(self) -> str:
        valid_rows, valid_cols = self._valid_shape
        return (
            f"mgather_mat_{self._dtype.value}_{self._coalesce}_{self._gather_oob}"
            f"_scratch{self._scratch_elements}_v{valid_rows}x{valid_cols}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        dtype = _TORCH_DTYPE[self._dtype]
        output_dtype = DataType.INT32 if self._dtype in (DataType.INT8, DataType.UINT8) else DataType.FP32
        if self._coalesce == "row":
            mem = torch.arange(ROWS * MAT_N).reshape(ROWS, MAT_N).remainder(53).to(dtype)
            index_values = (
                [-1, ROWS, ROWS + 1, 7, ROWS + 16, 0] + [(i * 29 + 11) % ROWS for i in range(MAT_M - 6)]
                if self._gather_oob != "undefined"
                else [(i * 29 + 7) % ROWS for i in range(MAT_M)]
            )
            idx = torch.tensor(
                [index_values],
                dtype=torch.int32,
            )
            return [
                TensorSpec("mem", [ROWS, MAT_N], self._dtype, init_value=mem),
                TensorSpec("idx", [1, MAT_M], DataType.INT32, init_value=idx),
                TensorSpec(
                    "eye",
                    [MAT_M, MAT_M],
                    self._dtype,
                    init_value=torch.eye(MAT_M, dtype=dtype),
                ),
                TensorSpec(
                    "out",
                    [self._valid_shape[0], MAT_N],
                    output_dtype,
                    is_output=True,
                    init_value=torch.zeros,
                ),
            ]

        table_size = MAT_M * MAT_N
        mem = torch.arange(table_size).remainder(53).to(dtype)
        if self._gather_oob == "undefined":
            idx = torch.arange(table_size - 1, -1, -1, dtype=torch.int32)
        else:
            idx = torch.arange(table_size, dtype=torch.int32) - 4
            idx[0:4] = torch.tensor([-1, table_size, table_size + 1, table_size * 2 + 3])
        idx = idx.reshape(MAT_M, MAT_N)
        return [
            TensorSpec("mem", [table_size], self._dtype, init_value=mem),
            TensorSpec("idx", [MAT_M, MAT_N], DataType.INT32, init_value=idx),
            TensorSpec("scratch", [self._scratch_elements], self._dtype, init_value=torch.zeros),
            TensorSpec(
                "eye",
                [MAT_M, MAT_M],
                self._dtype,
                init_value=torch.eye(MAT_M, dtype=dtype),
            ),
            TensorSpec(
                "out",
                [self._valid_shape[0], MAT_N],
                output_dtype,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        output_dtype = pl.INT32 if self._dtype in (DataType.INT8, DataType.UINT8) else pl.FP32
        gather_oob = self._gather_oob
        scratch_elements = self._scratch_elements
        valid_shape = list(self._valid_shape)
        output_rows = self._valid_shape[0]
        if self._coalesce == "row":

            @pl.program
            class MgatherMatRowProgram:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    mem: pl.Tensor[[ROWS, MAT_N], dtype],
                    idx: pl.Tensor[[1, MAT_M], pl.INT32],
                    eye: pl.Tensor[[MAT_M, MAT_M], dtype],
                    out: pl.InOut[pl.Tensor[[output_rows, MAT_N], output_dtype]],
                ) -> pl.Tensor[[output_rows, MAT_N], output_dtype]:
                    result: pl.Tile[[MAT_M, MAT_N], dtype] = pl.tile.mgather(
                        mem,
                        idx,
                        coalesce="row",
                        gather_oob=gather_oob,
                        target_memory=pl.MemorySpace.Mat,
                        valid_shape=valid_shape,
                    )
                    eye_tile: pl.Tile[[MAT_M, MAT_M], dtype] = pl.load(
                        eye,
                        [0, 0],
                        [MAT_M, MAT_M],
                        valid_shape=[MAT_M, valid_shape[0]],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    product: pl.Tile[[MAT_M, MAT_N], output_dtype] = pl.matmul(eye_tile, result)
                    visible: pl.Tile[[output_rows, MAT_N], output_dtype] = pl.tile.slice(
                        product, [output_rows, MAT_N], [0, 0]
                    )
                    return pl.store(visible, [0, 0], out)

                @pl.function(type=pl.FunctionType.Orchestration)
                def orchestrator(
                    self,
                    mem: pl.Tensor[[ROWS, MAT_N], dtype],
                    idx: pl.Tensor[[1, MAT_M], pl.INT32],
                    eye: pl.Tensor[[MAT_M, MAT_M], dtype],
                    out: pl.InOut[pl.Tensor[[output_rows, MAT_N], output_dtype]],
                ) -> pl.Tensor[[output_rows, MAT_N], output_dtype]:
                    return self.kernel(mem, idx, eye, out)

            return MgatherMatRowProgram

        @pl.program
        class MgatherMatElemProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[MAT_M * MAT_N], dtype],
                idx: pl.Tensor[[MAT_M, MAT_N], pl.INT32],
                scratch: pl.Tensor[[scratch_elements], dtype],
                eye: pl.Tensor[[MAT_M, MAT_M], dtype],
                out: pl.InOut[pl.Tensor[[output_rows, MAT_N], output_dtype]],
            ) -> pl.Tensor[[output_rows, MAT_N], output_dtype]:
                result: pl.Tile[[MAT_M, MAT_N], dtype] = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    gather_oob=gather_oob,
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                    valid_shape=valid_shape,
                )
                eye_tile: pl.Tile[[MAT_M, MAT_M], dtype] = pl.load(
                    eye,
                    [0, 0],
                    [MAT_M, MAT_M],
                    valid_shape=[MAT_M, valid_shape[0]],
                    target_memory=pl.MemorySpace.Mat,
                )
                product: pl.Tile[[MAT_M, MAT_N], output_dtype] = pl.matmul(eye_tile, result)
                visible: pl.Tile[[output_rows, MAT_N], output_dtype] = pl.tile.slice(
                    product, [output_rows, MAT_N], [0, 0]
                )
                return pl.store(visible, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mem: pl.Tensor[[MAT_M * MAT_N], dtype],
                idx: pl.Tensor[[MAT_M, MAT_N], pl.INT32],
                scratch: pl.Tensor[[scratch_elements], dtype],
                eye: pl.Tensor[[MAT_M, MAT_M], dtype],
                out: pl.InOut[pl.Tensor[[output_rows, MAT_N], output_dtype]],
            ) -> pl.Tensor[[output_rows, MAT_N], output_dtype]:
                return self.kernel(mem, idx, scratch, eye, out)

        return MgatherMatElemProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        indices = tensors["idx"].to(torch.int64).bitwise_and(0xFFFFFFFF)
        bound = ROWS if self._coalesce == "row" else tensors["mem"].numel()
        if self._gather_oob == "clamp":
            indices = indices.clamp(max=bound - 1)
            gathered = tensors["mem"][indices.reshape(-1) if self._coalesce == "row" else indices]
        elif self._gather_oob == "wrap":
            indices = indices.remainder(bound)
            gathered = tensors["mem"][indices.reshape(-1) if self._coalesce == "row" else indices]
        elif self._gather_oob == "zero":
            gathered = torch.zeros((MAT_M, MAT_N), dtype=tensors["mem"].dtype)
            in_bounds = indices < bound
            if self._coalesce == "row":
                flat_indices = indices.reshape(-1)
                gathered[in_bounds.reshape(-1)] = tensors["mem"][flat_indices[in_bounds.reshape(-1)]]
            else:
                gathered[in_bounds] = tensors["mem"][indices[in_bounds]]
        else:
            gathered = tensors["mem"][indices.reshape(-1) if self._coalesce == "row" else indices]
        gathered = gathered.to(tensors["out"].dtype)
        valid_rows, valid_cols = self._valid_shape
        expected = torch.zeros_like(tensors["out"])
        expected[:valid_rows, :valid_cols] = gathered[:valid_rows, :valid_cols]
        tensors["out"][:] = expected


class TestMgather:
    """MGATHER row/elem, dtype, index order, and valid-shape coverage."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", list(_PL_DTYPE))
    def test_row_dtypes_and_nonmonotonic_indices(self, test_runner, platform, dtype):
        result = test_runner.run(MgatherRowTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("valid_rows", [1, 9, INDEX_ROWS], ids=("single-row", "partial", "full"))
    def test_row_valid_shape(self, test_runner, platform, valid_rows):
        result = test_runner.run(MgatherRowTestCase(valid_rows=valid_rows, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("platform", ["a5"])
    def test_a5_column_vector_row_indices(self, test_runner, platform):
        result = test_runner.run(MgatherRowTestCase(column_indices=True, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("gather_oob", ["clamp", "wrap", "zero"])
    def test_row_out_of_bounds_modes(self, test_runner, platform, gather_oob):
        result = test_runner.run(MgatherRowTestCase(gather_oob=gather_oob, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [(ELEM_M, ELEM_N), (5, ELEM_N), (ELEM_M, 19), (5, 19), (1, 1)],
        ids=("full", "row-tail", "col-tail", "row-col-tail", "scalar-boundary"),
    )
    def test_elem_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(MgatherElemTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", list(_PL_DTYPE))
    def test_elem_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(MgatherElemTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("gather_oob", ["clamp", "wrap", "zero"])
    def test_elem_out_of_bounds_modes(self, test_runner, platform, gather_oob):
        result = test_runner.run(MgatherElemTestCase(gather_oob=gather_oob, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("coalesce", ["row", "elem"])
    @pytest.mark.parametrize("dtype", _MAT_NUMERIC_DTYPES)
    def test_gm_to_mat_dtypes(self, test_runner, platform, coalesce, dtype):
        result = test_runner.run(MgatherMatTestCase(coalesce=coalesce, dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("coalesce", ["row", "elem"])
    @pytest.mark.parametrize("gather_oob", ["clamp", "wrap", "zero"])
    def test_gm_to_mat_out_of_bounds_modes(self, test_runner, platform, coalesce, gather_oob):
        result = test_runner.run(
            MgatherMatTestCase(coalesce=coalesce, gather_oob=gather_oob, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"

    # TMatmul observes Mat results with a C0-aligned K; arbitrary valid rows are
    # covered directly by the MGATHER IR and codegen tests.
    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("coalesce", ["row", "elem"])
    @pytest.mark.parametrize(
        "valid_shape",
        [(MAT_M, MAT_N), (MAT_M // 2, MAT_N), (MAT_M, 21), (MAT_M // 2, 21)],
        ids=("full", "row-tail", "col-tail", "row-col-tail"),
    )
    def test_gm_to_mat_valid_shape(self, test_runner, platform, coalesce, valid_shape):
        result = test_runner.run(
            MgatherMatTestCase(coalesce=coalesce, valid_shape=valid_shape, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"

    # Declared rather than built in the body: ``scratch_elements`` is an
    # arithmetic expression, which collection's source-parsing route cannot
    # evaluate, so this case compiled on its own instead of in the pool. The
    # platform moves from an explicit parametrize to the harness matrix --
    # ``MgatherMatTestCase`` only forwards it to ``PTOTestCase`` and never reads
    # it -- held to the on-board ids by the marker so the coverage is the
    # ``ONBOARD_PLATFORMS`` list it replaces.
    @pytest.mark.platforms(
        "a2a3",
        "a5",
        reason="oversized-scratch acceptance is an on-board property; the simulator does not model it",
    )
    @st.cases(st.from_legacy(MgatherMatTestCase(coalesce="elem", scratch_elements=MAT_M * MAT_N + MAT_N)))
    def test_gm_to_mat_elem_accepts_oversized_scratch(self, case_run):
        case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
