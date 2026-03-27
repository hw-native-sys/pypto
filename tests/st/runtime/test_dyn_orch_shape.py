# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Runtime tests for dynamic orchestration using the PyPTO frontend with PTO backend.

Four scenarios are covered, mirroring the UT codegen tests in
tests/ut/codegen/test_orchestration_codegen.py (TestDynShapeOrchestration).
The key feature under test is that the **orchestration function itself** uses
``pl.dynamic()`` dims (``M``, ``N``) in its tensor type annotations, exercising
the ``OrchArg::to_tensor()`` path introduced by the dynamic-shape commit.

Scenarios:
- Scenario 1 — fully dynamic M×N orch: both InCore and orchestration use
  ``pl.Tensor[[M, N], pl.FP32]``; validates ``OrchArg::to_tensor()`` for
  external tensors with no static shape information in the orch signature.
- Scenario 2 — dynamic orch + valid_shapes scalars: orchestration uses M×N
  dims; m, n scalars are read from an INT64 tensor via ``pl.tensor.read`` and
  forwarded to the InCore kernel as valid_shapes.
- Scenario 3 — mixed dynamic M / static cols: orchestration uses
  ``pl.Tensor[[M, cols], pl.FP32]`` (M dynamic, cols=16 static); InCore reads
  M via ``pl.tensor.dim`` and iterates in pairs.
- Scenario 4 — ``tensor.dim`` on dynamic orch param: orchestration computes
  ``m_val``, ``n_val`` via ``pl.tensor.dim`` but does not use them in the
  dispatch; validates that unused dim reads do not break execution.

Shapes: (16, 16) for scenarios 1 and 4; (32, 32) full / (16, 16) valid for
scenario 2; (128, 16) for scenario 3 (rows divisible by 2).

All tests use OptimizationStrategy.Default and BackendType.Ascend910B_PTO.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level DynVar names.
# pyright: reportUndefinedVariable=false

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

from examples.ir_parser.paged_attention_example import (
    kernel_init_inplace,
    kernel_online_update,
    kernel_pv_matmul,
    kernel_qk_matmul,
    kernel_softmax_prepare,
)

M = pl.dynamic("M")
N = pl.dynamic("N")

# Dynamic dims for paged attention orchestration
QR = pl.dynamic("QR")  # query_rows  = batch * num_heads
KCR = pl.dynamic("KCR")  # key_cache_rows = batch * max_blocks * block_size
HD = pl.dynamic("HD")  # head_dim
BT = pl.dynamic("BT")  # block_table_flat_size = batch * max_blocks
B = pl.dynamic("B")  # batch

_DYN_SHAPES = [(16, 16)]
_MIXED_SHAPES = [(128, 16)]

# (batch, num_heads, head_dim, block_size, context_len, max_model_len)
_PA_CONFIGS = [(2, 16, 128, 128, 256, 1024)]


class DynOrchAddTestCase(PTOTestCase):
    """Test add kernel where both InCore and orchestration use dynamic M×N shapes.

    Key difference from DynShapeAddTestCase in test_dynamic_shape.py: the
    orchestration signature uses ``pl.Tensor[[M, N], pl.FP32]`` (dynamic dims)
    instead of closure-variable static dims.  Validates that OrchArg::to_tensor()
    correctly delivers runtime tensor metadata for fully dynamic external params.
    Expected result: c = a + b over the full rows×cols tile.
    """

    __test__ = False

    def __init__(self, shape: tuple[int, int], config: RunConfig | None = None):
        super().__init__(config)
        self._rows, self._cols = shape

    def get_name(self) -> str:
        return f"dyn_orch_add_{self._rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Captured as closure variables for tile load/store sizes.
        rows = self._rows
        cols = self._cols

        @pl.program
        class DynOrchAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                """Add two dynamic-shape tensors element-wise."""
                a_tile = pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)
                b_tile = pl.load(b, [0, 0], [rows, cols])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c_out = self.add_kernel(a, b, c)
                return c_out

        return DynOrchAddProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class DynOrchValidShapeAddTestCase(PTOTestCase):
    """Test add with dynamic M×N orchestration and valid_shapes from a scalar tensor.

    Orchestration params a, b, c use dynamic M×N dims.  The scalars m, n are
    read at runtime from the INT64 tensor ``vs`` via ``pl.tensor.read``, which
    generates ``orch[idx].data<void>()`` scalar extraction in the C++ code.
    Expected result: c[:valid_rows, :valid_cols] = a + b, c elsewhere = 0.
    """

    __test__ = False

    def __init__(
        self,
        shape: tuple[int, int],
        valid_shape: tuple[int, int],
        config: RunConfig | None = None,
    ):
        super().__init__(config)
        self._rows, self._cols = shape
        self._valid_rows, self._valid_cols = valid_shape

    def get_name(self) -> str:
        return (
            f"dyn_orch_valid_shape_add_{self._rows}x{self._cols}_valid_{self._valid_rows}x{self._valid_cols}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec(
                "vs",
                [2],
                DataType.INT64,
                init_value=torch.tensor([self._valid_rows, self._valid_cols], dtype=torch.int64),
            ),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Tile size baked in as closure variables.
        rows = self._rows
        cols = self._cols

        @pl.program
        class DynOrchValidShapeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                m: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                """Add two tiles with dynamic valid_shapes [m, n]."""
                a_tile = pl.load(a, [0, 0], [rows, cols], valid_shapes=[m, n])
                b_tile = pl.load(b, [0, 0], [rows, cols], valid_shapes=[m, n])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                vs: pl.Tensor[[2], pl.INDEX],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                m: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [0])
                n: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [1])
                c_out = self.add_kernel(a, b, c, m, n)
                return c_out

        return DynOrchValidShapeProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        vr = int(tensors["vs"][0])
        vc = int(tensors["vs"][1])
        tensors["c"][:vr, :vc] = tensors["a"][:vr, :vc] + tensors["b"][:vr, :vc]


class DynOrchLoopMixedDimsAddTestCase(PTOTestCase):
    """Test add with dynamic M and static cols=16 in orchestration, loop in InCore.

    Shape (rows, cols) is provided at construction time; cols must equal 16 to
    match the static dim in the orchestration type annotation.  rows must be
    divisible by 2 — the loop processes pairs of rows per iteration.  The InCore
    function reads M from the first tensor dimension via pl.tensor.dim.
    Validates OrchArg::to_tensor() for mixed dynamic/static dim tensors.
    Expected result: c = a + b over the full rows×cols tile.
    """

    __test__ = False

    def __init__(self, shape: tuple[int, int], config: RunConfig | None = None):
        super().__init__(config)
        self._rows, self._cols = shape

    def get_name(self) -> str:
        return f"dyn_orch_loop_mixed_dims_add_{self._rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # cols is baked as a static integer in the type annotation [M, cols].
        cols = self._cols

        @pl.program
        class DynOrchLoopProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, cols], pl.FP32],
                b: pl.Tensor[[M, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[M, cols], pl.FP32]],
            ) -> pl.Tensor[[M, cols], pl.FP32]:
                """Iterate over M rows in pairs and add tiles element-wise."""
                M_dim = pl.tensor.dim(a, 0)

                for i in pl.range(0, M_dim, 2):
                    offset = i
                    a_tile = pl.load(a, [offset, 0], [2, cols], target_memory=pl.MemorySpace.Vec)
                    b_tile = pl.load(b, [offset, 0], [2, cols], target_memory=pl.MemorySpace.Vec)
                    result = pl.add(a_tile, b_tile)
                    out = pl.store(result, [offset, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, cols], pl.FP32],
                b: pl.Tensor[[M, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[M, cols], pl.FP32]],
            ) -> pl.Tensor[[M, cols], pl.FP32]:
                c_out = self.add_kernel(a, b, c)
                return c_out

        return DynOrchLoopProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class DynOrchDimOnDynParamAddTestCase(PTOTestCase):
    """Test where orchestration reads tensor dims via pl.tensor.dim on dynamic params.

    The orchestration computes m_val and n_val from the first two dimensions of
    the dynamic M×N input tensor but does not use them in the task dispatch.
    Validates that unused dim reads in orchestration compile and execute without
    affecting the kernel result.
    Expected result: c = a + b over the full rows×cols tile.
    """

    __test__ = False

    def __init__(self, shape: tuple[int, int], config: RunConfig | None = None):
        super().__init__(config)
        self._rows, self._cols = shape

    def get_name(self) -> str:
        return f"dyn_orch_dim_on_dyn_param_add_{self._rows}x{self._cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Captured as closure variables for tile load/store sizes.
        rows = self._rows
        cols = self._cols

        @pl.program
        class DynOrchDimProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                """Add two dynamic-shape tensors element-wise."""
                a_tile = pl.load(a, [0, 0], [rows, cols])
                b_tile = pl.load(b, [0, 0], [rows, cols])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                m_val: pl.Scalar[pl.INT64] = pl.tensor.dim(a, 0)  # noqa: F841
                n_val: pl.Scalar[pl.INT64] = pl.tensor.dim(a, 1)  # noqa: F841
                c_out = self.add_kernel(a, b, c)
                return c_out

        return DynOrchDimProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class DynOrchPagedAttentionTestCase(PTOTestCase):
    """Paged attention where the orchestration uses fully dynamic dims (QR, KCR, HD, BT, B).

    Exercises OrchArg::to_tensor() for all external tensors in the paged attention
    pipeline.  All runtime configuration values (batch, num_heads, head_dim,
    block_size, max_blocks) are derived from tensor shapes via pl.tensor.dim()
    and scalar arithmetic — no config_t tensor is needed.
    Expected result: standard online-softmax paged attention output.
    """

    __test__ = False

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 128,
        context_len: int = 256,
        max_model_len: int = 1024,
        scale: float = 1.0,
        config: RunConfig | None = None,
    ):
        super().__init__(config)
        self.config.atol = 2e-2
        self.config.rtol = 2e-2
        self._batch = batch
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._context_len = context_len
        self._max_model_len = max_model_len
        self._scale = scale
        self._max_num_blocks = max_model_len // block_size

    def get_name(self) -> str:
        return f"dyn_orch_paged_attn_{self._batch}b_{self._num_heads}h_{self._head_dim}d_{self._block_size}bs"

    def define_tensors(self) -> list[TensorSpec]:
        batch = self._batch
        num_heads = self._num_heads
        head_dim = self._head_dim
        block_size = self._block_size
        max_blocks = self._max_num_blocks
        total_pool_rows = batch * max_blocks * block_size

        block_table = torch.randint(
            0, max(batch * max_blocks, 1), size=(batch, max_blocks), dtype=torch.int32
        ).flatten()
        context_lens = torch.full((batch,), self._context_len, dtype=torch.int32)

        return [
            TensorSpec("query", [batch * num_heads, head_dim], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [total_pool_rows, head_dim], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [total_pool_rows, head_dim], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [batch * max_blocks], DataType.INT32, init_value=block_table),
            TensorSpec("context_lens", [batch], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [batch * num_heads, head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        q_tile = 16

        @pl.program
        class DynOrchPagedAttentionProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def paged_attention(
                self,
                query: pl.Tensor[[QR, HD], pl.BF16],
                key_cache: pl.Tensor[[KCR, HD], pl.BF16],
                value_cache: pl.Tensor[[KCR, HD], pl.BF16],
                block_table: pl.Tensor[[BT], pl.INT32],
                context_lens: pl.Tensor[[B], pl.INT32],
                out: pl.Out[pl.Tensor[[QR, HD], pl.FP32]],
            ) -> pl.Tensor[[QR, HD], pl.FP32]:
                """Paged attention orchestration with tensor.dim-derived runtime values."""
                # Derive all runtime config from tensor shapes
                batch_cfg: pl.Scalar[pl.INT64] = pl.tensor.dim(context_lens, 0)
                query_rows: pl.Scalar[pl.INT64] = pl.tensor.dim(query, 0)
                head_dim_cfg: pl.Scalar[pl.INT64] = pl.tensor.dim(query, 1)
                value_cache_rows: pl.Scalar[pl.INT64] = pl.tensor.dim(value_cache, 0)
                block_table_size: pl.Scalar[pl.INT64] = pl.tensor.dim(block_table, 0)
                num_heads_cfg = query_rows // batch_cfg
                block_num_cfg = block_table_size // batch_cfg
                block_size_cfg = value_cache_rows // block_table_size

                q_head_num = num_heads_cfg
                q_loop_cfg = (q_head_num + q_tile - 1) // q_tile

                for b_idx in pl.range(batch_cfg):
                    cur_seq = pl.tensor.read(context_lens, [b_idx])
                    bn_this_batch = (cur_seq + block_size_cfg - 1) // block_size_cfg
                    for q_idx in pl.range(q_loop_cfg):
                        cur_offset = b_idx * q_head_num + q_idx * q_tile

                        oi_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                            [q_tile, head_dim_cfg], dtype=pl.FP32
                        )
                        li_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                        mi_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor([q_tile, 1], dtype=pl.FP32)
                        oi, li_update, mi_update = kernel_init_inplace(oi_buf, li_buf, mi_buf)

                        for bn in pl.range(bn_this_batch):
                            qi: pl.Tensor[[q_tile, head_dim_cfg], pl.BF16] = pl.slice(
                                query, [q_tile, head_dim_cfg], [cur_offset, 0]
                            )
                            cur_block_idx = pl.tensor.read(block_table, [b_idx * block_num_cfg + bn])
                            valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)
                            kv_block_row = cur_block_idx * block_size_cfg
                            kj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                                key_cache, [block_size_cfg, head_dim_cfg], [kv_block_row, 0]
                            )
                            vj: pl.Tensor[[block_size_cfg, head_dim_cfg], pl.BF16] = pl.slice(
                                value_cache, [block_size_cfg, head_dim_cfg], [kv_block_row, 0]
                            )
                            sij_buf: pl.Tensor[[q_tile, block_size_cfg], pl.FP32] = pl.create_tensor(
                                [q_tile, block_size_cfg], dtype=pl.FP32
                            )
                            sij = kernel_qk_matmul(qi, kj, sij_buf)
                            sij_valid: pl.Tensor[[q_tile, valid_len], pl.FP32] = pl.slice(
                                sij, [q_tile, valid_len], [0, 0]
                            )
                            pij_f16_buf: pl.Tensor[[q_tile, block_size_cfg], pl.BF16] = pl.create_tensor(
                                [q_tile, block_size_cfg], dtype=pl.BF16
                            )
                            mi_sm_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                                [q_tile, 1], dtype=pl.FP32
                            )
                            li_sm_buf: pl.Tensor[[q_tile, 1], pl.FP32] = pl.create_tensor(
                                [q_tile, 1], dtype=pl.FP32
                            )
                            pij_f16, mi, li = kernel_softmax_prepare(
                                sij_valid,
                                1.0,  # type: ignore[reportArgumentType]
                                pij_f16_buf,
                                mi_sm_buf,
                                li_sm_buf,
                            )
                            oi_tmp_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.create_tensor(
                                [q_tile, head_dim_cfg], dtype=pl.FP32
                            )
                            oi_tmp = kernel_pv_matmul(pij_f16, vj, oi_tmp_buf)

                            if bn == 0:
                                is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                            else:
                                is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                            if bn == bn_this_batch - 1:
                                is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                            else:
                                is_last: pl.Scalar[pl.INT64] = pl.yield_(0)

                            out_view_buf: pl.Tensor[[q_tile, head_dim_cfg], pl.FP32] = pl.slice(
                                out, [q_tile, head_dim_cfg], [cur_offset, 0]
                            )
                            mi_update, li_update, oi, out_view = kernel_online_update(
                                mi, li, oi_tmp, mi_update, li_update, oi, out_view_buf, is_first, is_last
                            )

                return out

        return DynOrchPagedAttentionProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def compute_expected(self, tensors, params=None):
        # Derive all config from tensor shapes (mirrors orchestration logic)
        batch = tensors["context_lens"].shape[0]
        query_rows = tensors["query"].shape[0]
        head_dim = tensors["query"].shape[1]
        num_heads = query_rows // batch
        block_table_size = tensors["block_table"].shape[0]
        max_num_blocks_per_req = block_table_size // batch
        value_cache_rows = tensors["value_cache"].shape[0]
        block_size = value_cache_rows // block_table_size
        scale_value = self._scale

        query = tensors["query"].float().reshape(batch, num_heads, head_dim)
        total_pool_blocks = batch * max_num_blocks_per_req
        key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
        context_lens = tensors["context_lens"]

        out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
        q_tile = 16
        max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

        for q_offset in range(0, num_heads, q_tile):
            q_tile_size = min(q_tile, num_heads - q_offset)
            qi = query[:, q_offset : q_offset + q_tile_size, :]
            oi, li, mi = None, None, None

            for bn in range(max_bn):
                valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
                if not (valid_lens > 0).any():
                    break
                block_indices = block_table[:, bn]
                kj_all = key_cache[block_indices].float()
                vj_all = value_cache[block_indices].float()
                sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale_value
                pos = torch.arange(block_size).unsqueeze(0)
                valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)
                sij = sij.masked_fill(~valid_mask, float("-inf"))
                mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)
                pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
                pij = pij.to(torch.bfloat16).to(torch.float32)
                lij = pij.sum(dim=-1, keepdim=True)
                oi_new = torch.bmm(pij, vj_all)
                if bn == 0:
                    oi, li, mi = oi_new, lij, mij
                else:
                    mi_new = torch.maximum(mi, mij)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(mij - mi_new)
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new

            out[:, q_offset : q_offset + q_tile_size, :] = oi / li

        tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# =============================================================================
# pytest test suite
# =============================================================================


class TestDynOrchShapeOperations:
    """Test suite for dynamic orchestration shape operations."""

    @pytest.mark.parametrize("shape", _DYN_SHAPES)
    def test_dyn_orch_add(self, test_runner, shape):
        """Test add where both InCore and orchestration use dynamic M×N dims."""
        test_case = DynOrchAddTestCase(shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}: {result.error}"

    @pytest.mark.parametrize("shape,valid_shape", [((32, 32), (16, 16))])
    def test_dyn_orch_valid_shape_add(self, test_runner, shape, valid_shape):
        """Test add with dynamic M×N orchestration and valid_shapes from INT64 tensor."""
        test_case = DynOrchValidShapeAddTestCase(shape, valid_shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}, valid_shape {valid_shape}: {result.error}"

    @pytest.mark.parametrize("shape", _MIXED_SHAPES)
    def test_dyn_orch_loop_mixed_dims_add(self, test_runner, shape):
        """Test add with dynamic M / static cols=16 in orchestration, loop in InCore."""
        test_case = DynOrchLoopMixedDimsAddTestCase(shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}: {result.error}"

    @pytest.mark.parametrize("shape", _DYN_SHAPES)
    def test_dyn_orch_dim_on_dyn_param_add(self, test_runner, shape):
        """Test add where orchestration reads tensor dims via pl.tensor.dim."""
        test_case = DynOrchDimOnDynParamAddTestCase(shape)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for shape {shape}: {result.error}"

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        _PA_CONFIGS,
    )
    def test_dyn_orch_paged_attention(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        """Test paged attention with fully dynamic dims in the orchestration signature."""
        test_case = DynOrchPagedAttentionTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Dyn orch paged attention test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
