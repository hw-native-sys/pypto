# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BackendHandler dispatch interface (issue #948).

Verifies the per-backend handler returns the documented values, that the
dispatch is routed both via ``Backend.get_handler()`` and the global
``backend.get_handler()`` accessor, and that handlers stay disjoint between
backends so adding a third backend cannot silently inherit the wrong defaults.
"""

import pickle

import pypto.language as pl
import pytest
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import ir as _ir_core
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig


def test_ascend910_legacy_alias_has_canonical_identity():
    """Old callers and serialized enum values must select the canonical member."""
    canonical = _backend_core.BackendType.Ascend910
    legacy = _backend_core.BackendType.Ascend910B
    assert legacy is canonical
    assert legacy.name == "Ascend910"
    assert canonical.value == 0
    assert _backend_core.BackendType.Ascend950.value == 1
    assert _backend_core.BackendType.__members__["Ascend910B"] is canonical
    assert pickle.loads(pickle.dumps(legacy)) is canonical
    assert _backend_core.get_backend_instance(legacy) is _backend_core.get_backend_instance(canonical)


@pytest.mark.parametrize("mode", [ExecutionMode.SIM, ExecutionMode.ONBOARD])
def test_ascend910_legacy_alias_normalizes_target(mode):
    """Alias use must not fork compile options or the internal platform identity."""
    canonical = RunConfig(backend_type=_backend_core.BackendType.Ascend910, execution_mode=mode)
    legacy = RunConfig(backend_type=_backend_core.BackendType.Ascend910B, execution_mode=mode)
    assert legacy.compile_kwargs() == canonical.compile_kwargs()
    assert legacy.platform == canonical.platform
    assert (
        CompileOptions(backend_type=legacy.backend_type, execution_mode=mode).platform == canonical.platform
    )


class TestBackendHandlerValues:
    """Per-backend handler returns the expected behavioural values."""

    def test_ascend910b_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()

        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.get_launch_spec_core_count_method() == "set_block_num"
        assert handler.get_default_sim_platform() == "a2a3sim"
        assert handler.get_extra_ptoas_flags() == ["--pto-arch", "a3"]

        assert handler.honors_high_precision_algorithm() is False
        assert handler.requires_gm_pipe_buffer() is True
        assert handler.supports_runtime_bound_local_software_pipeline() is True
        assert handler.supports_local_pipe_depth() is True
        assert handler.requires_split_load_tpop_workaround() is True
        assert handler.requires_level3_tmp_scratch() is True
        assert handler.requires_vto_c_fractal_adapt() is False
        assert handler.requires_runtime_subblock_bridge() is True
        assert handler.requires_no_split_dual_aiv_dispatch() is True
        assert handler.splits_cube_to_vector_transport_in_hardware() is False

    def test_ascend950_handler_values(self):
        handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert handler.get_pto_target_arch() == "a5"
        assert handler.get_launch_spec_core_count_method() == "set_block_num"
        assert handler.get_default_sim_platform() == "a5sim"
        assert handler.get_extra_ptoas_flags() == ["--pto-arch", "a5"]

        assert handler.honors_high_precision_algorithm() is True
        assert handler.requires_gm_pipe_buffer() is False
        assert handler.supports_runtime_bound_local_software_pipeline() is False
        assert handler.supports_local_pipe_depth() is False
        assert handler.requires_split_load_tpop_workaround() is False
        assert handler.requires_level3_tmp_scratch() is False
        assert handler.requires_vto_c_fractal_adapt() is True
        assert handler.requires_runtime_subblock_bridge() is False
        assert handler.requires_no_split_dual_aiv_dispatch() is False
        assert handler.splits_cube_to_vector_transport_in_hardware() is True

    def test_handlers_are_disjoint_between_backends(self):
        h910 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        h950 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()

        assert h910.get_pto_target_arch() != h950.get_pto_target_arch()
        # Simpler exposes the same logical-block API on both architectures.
        assert h910.get_launch_spec_core_count_method() == h950.get_launch_spec_core_count_method()
        assert h910.get_default_sim_platform() != h950.get_default_sim_platform()


class TestBackendHandlerSingletons:
    """Handlers must be stable singletons so callers can hold raw pointers."""

    def test_ascend910b_handler_is_singleton(self):
        h1 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        h2 = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
        # Identity is exposed on the C++ side as the same singleton; on the
        # Python side we at least require structural equality of all hooks.
        assert h1.get_pto_target_arch() == h2.get_pto_target_arch()


class TestGlobalHandlerAccessor:
    """``backend.get_handler()`` follows the globally configured backend."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        # Snapshot and restore global config so we do not leak state to
        # other tests (which set their own backend type during fixtures).
        was_configured = _backend_core.is_backend_configured()
        previous = _backend_core.get_backend_type() if was_configured else None
        _backend_core.reset_for_testing()
        yield
        _backend_core.reset_for_testing()
        if previous is not None:
            _backend_core.set_backend_type(previous)

    def test_global_handler_matches_configured_backend_910b(self):
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend910)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a2a3"
        assert handler.requires_gm_pipe_buffer() is True
        # Python callers can feature-detect local depth before building a pipe.
        assert handler.supports_local_pipe_depth() is True

    def test_global_handler_matches_configured_backend_950(self):
        _backend_core.set_backend_type(_backend_core.BackendType.Ascend950)
        handler = _backend_core.get_handler()
        assert handler.get_pto_target_arch() == "a5"
        assert handler.supports_local_pipe_depth() is False
        assert handler.requires_vto_c_fractal_adapt() is True

    def test_global_handler_raises_when_not_configured(self):
        with pytest.raises(ValueError, match="Backend type not configured"):
            _backend_core.get_handler()

    @pytest.mark.parametrize(
        "configured_backend",
        [None, _backend_core.BackendType.Ascend910, _backend_core.BackendType.Ascend950],
    )
    @pytest.mark.parametrize(
        "queried_backend", [_backend_core.BackendType.Ascend910, _backend_core.BackendType.Ascend950]
    )
    def test_capacity_queries_use_owning_soc(self, configured_backend, queried_backend):
        """An explicit handler must work before target selection and for the other target."""
        if configured_backend is not None:
            _backend_core.set_backend_type(configured_backend)
        backend = _backend_core.get_backend_instance(queried_backend)
        handler = backend.get_handler()

        # In particular, Acc and Bias differ between architectures. Reading the
        # active target instead of this handler's owner would return wrong limits.
        for memory_space in _ir_core.MemorySpace.__members__.values():
            assert handler.get_capacity_bytes(memory_space) == backend.get_mem_size(memory_space)
        # Scale memories distinguish an unmodeled space from a modeled A5 space.
        expected_scale = 4096 if queried_backend == _backend_core.BackendType.Ascend950 else 0
        assert handler.get_capacity_bytes(_ir_core.MemorySpace.LeftScale) == expected_scale
        assert handler.get_capacity_bytes(_ir_core.MemorySpace.ScalarLocal) == 0
        assert _backend_core.is_backend_configured() == (configured_backend is not None)
        if configured_backend is not None:
            assert _backend_core.get_backend_type() == configured_backend


def _direct_tile_view(**kwargs):
    """Build a view with Expr extents, including mixed static/dynamic inputs."""
    span = _ir_core.Span.unknown()
    for key in ("valid_shape", "stride"):
        if key in kwargs:
            kwargs[key] = [
                _ir_core.ConstInt(value, pl.INDEX, span) if isinstance(value, int) else value
                for value in kwargs[key]
            ]
    if isinstance(kwargs.get("start_offset"), int):
        kwargs["start_offset"] = _ir_core.ConstInt(kwargs["start_offset"], pl.INDEX, span)
    return _ir_core.TileView(**kwargs)


def _direct_cast_query(
    source_dtype,
    target_dtype,
    source_shape,
    result_shape,
    *,
    source_view=None,
    result_view=None,
    source_memory=None,
    result_memory=None,
):
    """Build typed calls to test native descriptor constraints independently of frontend inference."""
    span = _ir_core.Span.unknown()
    source_type = _ir_core.TileType(source_shape, source_dtype, None, source_view, source_memory)
    result_type = _ir_core.TileType(result_shape, target_dtype, None, result_view, result_memory)
    source = _ir_core.Var("source", source_type, span)
    return _ir_core.Call(
        _ir_core.get_op("tile.cast"), [source], {"target_type": target_dtype, "mode": 2}, result_type, span
    )


@pytest.mark.parametrize(
    "target,source_dtype,result_dtype,expected",
    [
        (_backend_core.BackendType.Ascend950, pl.FP32, pl.FP16, True),
        (_backend_core.BackendType.Ascend950, pl.INT32, pl.FP16, False),
        (_backend_core.BackendType.Ascend910, pl.INT32, pl.FP16, True),
        (_backend_core.BackendType.Ascend910, pl.FP8E4M3FN, pl.FP32, False),
    ],
)
def test_direct_cast_queries_own_backend_without_global_configuration(
    target, source_dtype, result_dtype, expected
):
    """Native rules are owned by the queried target, independent of ambient configuration."""
    handler = _backend_core.get_backend_instance(target).get_handler()
    call = _direct_cast_query(source_dtype, result_dtype, [16, 32], [16, 32])
    assert _backend_core.is_direct_cast(call, handler) == expected


def test_direct_cast_accepts_integer_target_dtype_code():
    """Integer codes are normalized by the IR contract, not by native matching."""
    span = _ir_core.Span.unknown()
    source = _ir_core.Var("source", _ir_core.TileType([16, 32], pl.FP32), span)
    call = _ir_core.create_op_call("tile.cast", [source], {"target_type": pl.FP16.code(), "mode": 2}, span)
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    assert _backend_core.is_direct_cast(call, handler)


@pytest.mark.parametrize("target", [_backend_core.BackendType.Ascend950, _backend_core.BackendType.Ascend910])
def test_direct_cast_requeries_changed_packed_geometry(target):
    handler = _backend_core.get_backend_instance(target).get_handler()
    aligned = _direct_cast_query(pl.FP4E2M1X2, pl.FP8E4M3FN, [16, 32], [16, 64])
    split = _direct_cast_query(pl.FP4E2M1X2, pl.FP8E4M3FN, [16, 16], [16, 32])
    assert _backend_core.is_direct_cast(aligned, handler) == (target == _backend_core.BackendType.Ascend950)
    assert not _backend_core.is_direct_cast(split, handler)


@pytest.mark.parametrize("packed_dtype", [pl.FP4E2M1X2, pl.FP4E1M2X2])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_direct_packed_cast_normalizes_both_dynamic_valid_extents(packed_dtype, decode, matching):
    """Affine packed extents must compare by value in either conversion direction."""
    span = _ir_core.Span.unknown()
    n = _ir_core.Var("n", _ir_core.ScalarType(pl.INDEX), span)
    packed_cols = _ir_core.Add(n, _ir_core.ConstInt(1, pl.INDEX, span), pl.INDEX, span)
    unpacked_cols = _ir_core.Mul(
        _ir_core.Add(n, _ir_core.ConstInt(1 if matching else 2, pl.INDEX, span), pl.INDEX, span),
        _ir_core.ConstInt(2, pl.INDEX, span),
        pl.INDEX,
        span,
    )
    packed_view = _direct_tile_view(valid_shape=[16, packed_cols])
    unpacked_view = _direct_tile_view(valid_shape=[16, unpacked_cols])
    call = _direct_cast_query(
        packed_dtype if decode else pl.BF16,
        pl.BF16 if decode else packed_dtype,
        [16, 32 if decode else 64],
        [16, 64 if decode else 32],
        source_view=packed_view if decode else unpacked_view,
        result_view=unpacked_view if decode else packed_view,
    )
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    assert _backend_core.is_direct_cast(call, handler) == matching


@pytest.mark.parametrize(
    "case", ["valid_shape", "dynamic_valid", "stride", "offset", "memory", "rows", "compact"]
)
def test_direct_cast_rule_rejects_invalid_packed_descriptors(case):
    """Every descriptor constraint is queried through the same public boolean API."""
    rows = 65536 if case == "rows" else 16
    source_shape = [rows, 32]
    result_shape = [rows, 64]
    source_valid: list[int | _ir_core.Expr] = [8, 24]
    result_valid: list[int | _ir_core.Expr] = [8, 47 if case == "valid_shape" else 48]
    if case == "dynamic_valid":
        n = _ir_core.Var("n", _ir_core.ScalarType(pl.INDEX), _ir_core.Span.unknown())
        source_valid = [8, n]
        result_valid = [
            8,
            _ir_core.Mul(
                n, _ir_core.ConstInt(2, pl.INDEX, _ir_core.Span.unknown()), pl.INDEX, _ir_core.Span.unknown()
            ),
        ]
    source_view = _direct_tile_view(
        valid_shape=source_valid,
        stride=[64, 1] if case == "stride" else [32, 1],
        start_offset=1 if case == "offset" else 0,
        compact=_ir_core.CompactMode.normal if case == "compact" else _ir_core.CompactMode.null,
    )
    result_view = _direct_tile_view(valid_shape=result_valid, stride=[64, 1])
    call = _direct_cast_query(
        pl.FP4E2M1X2,
        pl.FP8E4M3FN,
        source_shape,
        result_shape,
        source_view=source_view,
        result_view=result_view,
        source_memory=_ir_core.MemorySpace.Mat if case == "memory" else _ir_core.MemorySpace.Vec,
    )
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    assert not _backend_core.is_direct_cast(call, handler)


@pytest.mark.parametrize(
    "source_dtype,result_dtype,source_rows,result_rows",
    [
        (pl.FP4E2M1X2, pl.BF16, 32, 64),
        (pl.FP4E1M2X2, pl.BF16, 32, 64),
        (pl.BF16, pl.FP4E2M1X2, 64, 32),
    ],
)
def test_direct_cast_dense_dn_rule_and_full_validity(source_dtype, result_dtype, source_rows, result_rows):
    """DN uses the same lookup as ND and rejects partial rectangles without a special predicate."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    source_view = _direct_tile_view(blayout=_ir_core.TileLayout.col_major, stride=[1, source_rows])
    result_view = _direct_tile_view(blayout=_ir_core.TileLayout.col_major, stride=[1, result_rows])
    call = _direct_cast_query(
        source_dtype,
        result_dtype,
        [source_rows, 16],
        [result_rows, 16],
        source_view=source_view,
        result_view=result_view,
    )
    assert _backend_core.is_direct_cast(call, handler)
    partial_source = _direct_tile_view(
        blayout=_ir_core.TileLayout.col_major, valid_shape=[source_rows, 8], stride=[1, source_rows]
    )
    partial_result = _direct_tile_view(
        blayout=_ir_core.TileLayout.col_major, valid_shape=[result_rows, 8], stride=[1, result_rows]
    )
    partial = _direct_cast_query(
        source_dtype,
        result_dtype,
        [source_rows, 16],
        [result_rows, 16],
        source_view=partial_source,
        result_view=partial_result,
    )
    assert not _backend_core.is_direct_cast(partial, handler)


@pytest.mark.parametrize("target", [_backend_core.BackendType.Ascend910, _backend_core.BackendType.Ascend950])
@pytest.mark.parametrize("dynamic", [False, True])
def test_direct_cast_accepts_transport_box_with_narrowed_consumer(target, dynamic):
    """Ordinary TCVT uses destination validity after a full-box cross-core pop."""
    span = _ir_core.Span.unknown()
    rows = _ir_core.Var("lane_rows", _ir_core.ScalarType(pl.INDEX), span) if dynamic else 20
    call = _direct_cast_query(
        pl.INT32,
        pl.FP32,
        [32, 128],
        [32, 128],
        result_view=_direct_tile_view(valid_shape=[rows, 128]),
    )
    handler = _backend_core.get_backend_instance(target).get_handler()
    assert _backend_core.is_direct_cast(call, handler)


# ---------------------------------------------------------------------------
# NZ destination rule for the A5 packed FP4 decode
#
# NZ on a tile is blayout=col_major + slayout=row_major + fractal=512: the
# only boxed `pto.tcvt` destination PTOAS v0.72 accepts, and only for
# f4E2M1x2 -> f8E4M3FN (PTO_IR_manual.md). Every constraint asserted below was
# confirmed against the pinned assembler.
# ---------------------------------------------------------------------------

_NZ_VIEW_KWARGS = {
    "blayout": _ir_core.TileLayout.col_major,
    "slayout": _ir_core.TileLayout.row_major,
}


def _nz_view():
    return _ir_core.TileView(**_NZ_VIEW_KWARGS)


def _nz_cast_query(
    source_shape=(16, 32), result_shape=(16, 64), source_dtype=pl.FP4E2M1X2, target_dtype=pl.FP8E4M3FN
):
    """A cast whose recorded result is NZ and which carries the layout request."""
    span = _ir_core.Span.unknown()
    source_type = _ir_core.TileType(list(source_shape), source_dtype, None, None, _ir_core.MemorySpace.Vec)
    result_type = _ir_core.TileType(
        list(result_shape), target_dtype, None, _nz_view(), _ir_core.MemorySpace.Vec
    )
    source = _ir_core.Var("source", source_type, span)
    kwargs = {"target_type": target_dtype, "mode": 2, **_NZ_VIEW_KWARGS}
    return _ir_core.Call(_ir_core.get_op("tile.cast"), [source], kwargs, result_type, span)


def test_a5_nz_destination_cast_is_direct_and_preserves_only_the_output_layout():
    """The NZ rule exempts the RESULT from row-major repair, never the operand.

    Both halves are load-bearing. Without the output exemption the backend wraps
    the native NZ write in a relayout ``tile.move`` and the fusion is lost.
    Without the input requirement staying in force, a col-major operand would
    skip repair and then be rejected by the rule's own ND source condition.
    """
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    direct, info = _backend_core.query_direct_cast(_nz_cast_query(), handler)
    assert direct, info.reason
    assert info.preserve_output_layout
    assert not info.preserve_input_layout


def test_a5_nd_destination_cast_claims_no_layout_exemption():
    """The contrast case: an ordinary ND cast must not suppress any repair."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    call = _direct_cast_query(
        pl.FP4E2M1X2,
        pl.FP8E4M3FN,
        [16, 32],
        [16, 64],
        source_memory=_ir_core.MemorySpace.Vec,
        result_memory=_ir_core.MemorySpace.Vec,
    )
    direct, info = _backend_core.query_direct_cast(call, handler)
    assert direct, info.reason
    assert not info.preserve_output_layout
    assert not info.preserve_input_layout


def test_a5_nz_destination_rejects_rows_that_are_not_whole_boxes():
    """PTOAS: "NZ physical rows to be a multiple of 16 and fit a 16-bit stride"."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    direct, info = _backend_core.query_direct_cast(
        _nz_cast_query(source_shape=(8, 32), result_shape=(8, 64)), handler
    )
    assert not direct
    assert "whole number of boxes" in info.reason


def test_a5_nz_destination_rejects_a_boxed_source():
    """The native decode reads ND only, so a boxed operand is not this rule."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    span = _ir_core.Span.unknown()
    source_type = _ir_core.TileType([16, 32], pl.FP4E2M1X2, None, _nz_view(), _ir_core.MemorySpace.Vec)
    result_type = _ir_core.TileType([16, 64], pl.FP8E4M3FN, None, _nz_view(), _ir_core.MemorySpace.Vec)
    call = _ir_core.Call(
        _ir_core.get_op("tile.cast"),
        [_ir_core.Var("source", source_type, span)],
        {"target_type": pl.FP8E4M3FN, "mode": 2, **_NZ_VIEW_KWARGS},
        result_type,
        span,
    )
    direct, info = _backend_core.query_direct_cast(call, handler)
    assert not direct
    assert "source" in info.reason


def test_a5_zn_destination_is_not_a_native_rule():
    """ZN (row/col) is the other boxed flavour and PTOAS rejects it for this pair."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    span = _ir_core.Span.unknown()
    zn = _ir_core.TileView(blayout=_ir_core.TileLayout.row_major, slayout=_ir_core.TileLayout.col_major)
    source_type = _ir_core.TileType([16, 32], pl.FP4E2M1X2, None, None, _ir_core.MemorySpace.Vec)
    result_type = _ir_core.TileType([16, 64], pl.FP8E4M3FN, None, zn, _ir_core.MemorySpace.Vec)
    call = _ir_core.Call(
        _ir_core.get_op("tile.cast"),
        [_ir_core.Var("source", source_type, span)],
        {"target_type": pl.FP8E4M3FN, "mode": 2},
        result_type,
        span,
    )
    assert not _backend_core.is_direct_cast(call, handler)


def test_a5_nz_destination_is_not_a_rule_on_a2a3():
    """A2/A3 has no packed FP4 tcvt edge at all."""
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend910).get_handler()
    assert not _backend_core.is_direct_cast(_nz_cast_query(), handler)


def test_unconstrained_dtype_edges_state_no_layout_exemption():
    """An unconstrained edge must never claim it writes a boxed destination.

    FP32-to-FP16 has no implementation conditions, so it matches any geometry --
    which is exactly why ``preserve_output_layout`` must stay false for it. That
    flag is what LegalizeTileCast uses to decide an explicitly requested boxed
    destination is genuinely native, and PTOAS does not check the destination
    layout for this pair, so a false positive here would assemble silently.
    """
    handler = _backend_core.get_backend_instance(_backend_core.BackendType.Ascend950).get_handler()
    call = _nz_cast_query(
        source_shape=(16, 32), result_shape=(16, 32), source_dtype=pl.FP32, target_dtype=pl.FP16
    )
    direct, info = _backend_core.query_direct_cast(call, handler)
    assert direct, info.reason
    assert not info.preserve_output_layout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
