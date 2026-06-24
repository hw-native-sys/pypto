# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile partial-combine binary operations.

Covers four tile-level ops (and their tensor-level mirrors):
- ``tile.part_add`` -> ``pto.tpartadd``
- ``tile.part_mul`` -> ``pto.tpartmul``
- ``tile.part_max`` -> ``pto.tpartmax``
- ``tile.part_min`` -> ``pto.tpartmin``

The distinguishing semantics: the op runs over the destination valid region, and
where only ONE source is valid at an element the result copies that source. The
output valid region follows the first source (``src0``), so every case keeps
``src0`` fully valid (output is fully written) and narrows ``src1``'s valid
region to drive the copy path.

Per op there are three tile cases plus one tensor-level case:
- ``aligned``     - both sources fully valid -> reduces to the plain op.
- ``valid_cols``  - src1 valid columns narrowed (keeps valid rows = 16) -> the
                    trailing columns copy src0 (the "with valid_shape" case).
- ``valid_rows``  - src1 valid rows narrowed (valid rows < 16) -> the trailing
                    rows copy src0; also stresses the narrow-valid-row tail path.
- ``tensor``      - tensor-level op lowered by ConvertTensorToTileOps (fully
                    valid; DDR tensors cannot express a partial valid region).

Golden (one formula): ``out = op(src0, src1)`` then overwrite the region where
src1 is invalid (rows >= vr or cols >= vc) with ``src0``.

Note: the DSL source parser requires literal ``pl.tile.part_*`` / ``pl.part_*``
calls, so each op has its own program factory (the op name cannot be an alias).
"""

from collections.abc import Callable
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16

# op name -> torch golden for the both-valid region
_GOLDEN: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "part_add": torch.add,
    "part_mul": torch.mul,
    "part_max": torch.maximum,
    "part_min": torch.minimum,
}


def _src0() -> torch.Tensor:
    """First operand covering negatives, zero, and positives."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(13) - 6).contiguous()


def _src1() -> torch.Tensor:
    """Second operand, distinct from the first so max/min/mul/add all differ."""
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(7) - 3).contiguous()


# ---------------------------------------------------------------------------
# Tile-level program factories. src0 is fully valid; src1 valid = [v_rows, v_cols].
# The op call must be a literal `pl.tile.part_*` (the DSL parser rejects aliases),
# so there is one factory per op. Each @pl.program uses a distinct class name.
# ---------------------------------------------------------------------------


def _tile_part_add(v_rows: int, v_cols: int):
    @pl.program
    class TilePartAdd:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            t0: pl.Tile[[M, N], pl.FP32] = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            t1: pl.Tile[[M, N], pl.FP32] = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.part_add(t0, t1)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(src0, src1, out)
            return out

    return TilePartAdd


def _tile_part_mul(v_rows: int, v_cols: int):
    @pl.program
    class TilePartMul:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            t0: pl.Tile[[M, N], pl.FP32] = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            t1: pl.Tile[[M, N], pl.FP32] = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.part_mul(t0, t1)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(src0, src1, out)
            return out

    return TilePartMul


def _tile_part_max(v_rows: int, v_cols: int):
    @pl.program
    class TilePartMax:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            t0: pl.Tile[[M, N], pl.FP32] = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            t1: pl.Tile[[M, N], pl.FP32] = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.part_max(t0, t1)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(src0, src1, out)
            return out

    return TilePartMax


def _tile_part_min(v_rows: int, v_cols: int):
    @pl.program
    class TilePartMin:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            t0: pl.Tile[[M, N], pl.FP32] = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            t1: pl.Tile[[M, N], pl.FP32] = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.part_min(t0, t1)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(src0, src1, out)
            return out

    return TilePartMin


_TILE_FACTORY = {
    "part_add": _tile_part_add,
    "part_mul": _tile_part_mul,
    "part_max": _tile_part_max,
    "part_min": _tile_part_min,
}


# ---------------------------------------------------------------------------
# Tensor-level program factories (aligned only; DDR tensors are fully valid).
# Lowered to tile.part_* by ConvertTensorToTileOps. One factory per op (literal).
# ---------------------------------------------------------------------------


def _tensor_part_add():
    @pl.program
    class TensorPartAdd:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.part_add(src0, src1), [0, 0])
            return out

    return TensorPartAdd


def _tensor_part_mul():
    @pl.program
    class TensorPartMul:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.part_mul(src0, src1), [0, 0])
            return out

    return TensorPartMul


def _tensor_part_max():
    @pl.program
    class TensorPartMax:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.part_max(src0, src1), [0, 0])
            return out

    return TensorPartMax


def _tensor_part_min():
    @pl.program
    class TensorPartMin:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.assemble(out, pl.part_min(src0, src1), [0, 0])
            return out

    return TensorPartMin


_TENSOR_FACTORY = {
    "part_add": _tensor_part_add,
    "part_mul": _tensor_part_mul,
    "part_max": _tensor_part_max,
    "part_min": _tensor_part_min,
}


def _part_golden(
    op_name: str, src0: torch.Tensor, src1: torch.Tensor, v_rows: int, v_cols: int
) -> torch.Tensor:
    """op over the both-valid region; copy src0 where src1 is invalid (rows>=vr or cols>=vc)."""
    out = _GOLDEN[op_name](src0, src1).clone()
    out[v_rows:, :] = src0[v_rows:, :]
    out[:, v_cols:] = src0[:, v_cols:]
    return out


class TilePartTestCase(PTOTestCase):
    """tile.part_*: src0 fully valid, src1 valid = [v_rows, v_cols]."""

    __test__ = False

    def __init__(self, op_name: str, v_rows: int, v_cols: int, label: str, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._op_name = op_name
        self._v_rows = v_rows
        self._v_cols = v_cols
        self._label = label

    def get_name(self) -> str:
        return f"tile_{self._op_name}_{self._label}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src0", [M, N], DataType.FP32, init_value=_src0()),
            TensorSpec("src1", [M, N], DataType.FP32, init_value=_src1()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _TILE_FACTORY[self._op_name](self._v_rows, self._v_cols)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = _part_golden(
            self._op_name, tensors["src0"], tensors["src1"], self._v_rows, self._v_cols
        )


class TensorPartTestCase(PTOTestCase):
    """tensor.part_*: lowers to tile.part_* via ConvertTensorToTileOps (fully valid)."""

    __test__ = False

    def __init__(self, op_name: str, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._op_name = op_name

    def get_name(self) -> str:
        return f"tensor_{self._op_name}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src0", [M, N], DataType.FP32, init_value=_src0()),
            TensorSpec("src1", [M, N], DataType.FP32, init_value=_src1()),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _TENSOR_FACTORY[self._op_name]()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = _GOLDEN[self._op_name](tensors["src0"], tensors["src1"])


_PART_OPS = ["part_add", "part_mul", "part_max", "part_min"]

# (label, src1 valid rows, src1 valid cols)
_TILE_CASES = [
    ("aligned", M, N),
    ("valid_cols", M, 8),
    ("valid_rows", 8, N),
]


class TestTilePartOperations:
    """Test tile partial-combine ops across supported platforms."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("op_name", _PART_OPS)
    @pytest.mark.parametrize(("label", "v_rows", "v_cols"), _TILE_CASES)
    def test_tile_part(self, test_runner, platform, op_name, label, v_rows, v_cols):
        result = test_runner.run(TilePartTestCase(op_name, v_rows, v_cols, label, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorPartOperations:
    """Test tensor-level partial-combine ops (lowered by ConvertTensorToTileOps)."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("op_name", _PART_OPS)
    def test_tensor_part(self, test_runner, platform, op_name):
        result = test_runner.run(TensorPartTestCase(op_name, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
