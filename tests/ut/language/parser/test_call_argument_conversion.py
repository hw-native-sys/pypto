# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A declared scalar parameter converts an admitted argument to its own dtype.

The two frontends ask :mod:`pypto.language.scalar_conversion` the same question,
so they must also *emit* the same thing: one temporary bound to the conversion,
named from the parameter and the target dtype, placed before the call. These
tests pin that on the parsing lane and check that the tracing lane prints the
identical statement.

The lattice is narrow on purpose — one end ``pl.INDEX``, the other a fixed-width
integer — because the language states no signedness or width policy between two
fixed-width dtypes. A pair outside it keeps whatever check its layer already
applies; this lane gains no new refusal.
"""

import pypto.language as pl
from pypto.pypto_core import DataType, ir


@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def callee(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        rank: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        tile = pl.load(x, [0, 0], [8, 8])
        out = pl.store(pl.add(tile, rank), [0, 0], out)
        return out

    @pl.function
    def entry(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        """``r`` is INDEX; the declaration is INT32."""
        for r in pl.range(2):
            out = self.callee(x, r, out)
        return out

    @pl.function
    def explicit(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        """The conversion written out: legal, and not doubled."""
        for r in pl.range(2):
            out = self.callee(x, pl.cast(r, pl.INT32), out)
        return out


@pl.program
class IndexSlot:
    @pl.function(type=pl.FunctionType.InCore)
    def callee(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        block: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        tile = pl.load(x, [0, 0], [8, 8])
        out = pl.store(pl.add(tile, pl.cast(block, pl.FP32)), [0, 0], out)
        return out

    @pl.function
    def entry(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        count: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        """The lattice is symmetric: a fixed-width argument converts to INDEX."""
        out = self.callee(x, count, out)
        return out


@pl.program
class Int64Slot:
    @pl.function(type=pl.FunctionType.InCore)
    def callee(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        offset: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        tile = pl.load(x, [0, 0], [8, 8])
        out = pl.store(pl.add(tile, pl.cast(offset, pl.FP32)), [0, 0], out)
        return out

    @pl.function
    def entry(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        offset: pl.Scalar[pl.INT64],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ):
        """Equal width is still a type change: the conversion is recorded."""
        out = self.callee(x, offset, out)
        return out


def _printed(program) -> str:
    return program.as_python()


class TestTheParsingLaneConverts:
    def test_an_index_loop_variable_reaches_an_int32_parameter_as_int32(self):
        printed = _printed(Program)
        assert "_rank_int32: pl.Scalar[pl.INT32] = pl.cast(r, pl.INT32)" in printed, printed
        assert "self.callee(x, _rank_int32, out)" in printed, printed

    def test_the_temporary_binds_before_the_call(self):
        printed = _printed(Program).splitlines()
        cast_at = next(i for i, line in enumerate(printed) if "_rank_int32: pl.Scalar" in line)
        call_at = next(i for i, line in enumerate(printed) if "self.callee(x, _rank_int32, out)" in line)
        assert cast_at < call_at

    def test_an_author_written_conversion_is_not_doubled(self):
        printed = _printed(Program)
        explicit = printed.split("def explicit", 1)[1]
        assert explicit.count("pl.cast(r, pl.INT32)") == 1, explicit

    def test_a_fixed_width_argument_converts_to_an_index_slot(self):
        printed = _printed(IndexSlot)
        assert "_block_index: pl.Scalar[pl.INDEX] = pl.cast(count, pl.INDEX)" in printed, printed

    def test_an_int64_argument_records_the_type_change(self):
        """``INDEX`` and ``INT64`` share a width but are separate types."""
        printed = _printed(Int64Slot)
        assert "_offset_index: pl.Scalar[pl.INDEX] = pl.cast(offset, pl.INDEX)" in printed, printed

    def test_the_printed_conversion_round_trips_to_the_same_program(self):
        program = Program
        reparsed = pl.parse_program(program.as_python())
        ir.assert_structural_equal(program, reparsed, enable_auto_mapping=True)


class TestAdmissionStaysNarrow:
    def test_the_shared_rule_admits_only_the_index_integer_lattice(self):
        from pypto.language.scalar_conversion import conversion_target  # noqa: PLC0415 — the rule under test

        def scalar(dtype: DataType) -> ir.ScalarType:
            return ir.ScalarType(dtype)

        assert conversion_target(scalar(DataType.INT32), scalar(DataType.INDEX)) == DataType.INT32
        assert conversion_target(scalar(DataType.INDEX), scalar(DataType.UINT8)) == DataType.INDEX
        # The declared end is always the target: here the slot is INDEX.
        assert conversion_target(scalar(DataType.INDEX), scalar(DataType.INT64)) == DataType.INDEX
        # Two fixed-width dtypes: no signedness or width policy is stated.
        assert conversion_target(scalar(DataType.UINT32), scalar(DataType.INT32)) is None
        assert conversion_target(scalar(DataType.INT32), scalar(DataType.INT64)) is None
        # Equal dtypes are an identity, not a conversion.
        assert conversion_target(scalar(DataType.INT32), scalar(DataType.INT32)) is None
        # Not a scalar slot type: a logical value, an opaque handle, a 4-bit carrier.
        assert conversion_target(scalar(DataType.INT32), scalar(DataType.BOOL)) is None
        assert conversion_target(scalar(DataType.TASK_ID), scalar(DataType.INDEX)) is None
        assert conversion_target(scalar(DataType.INT4), scalar(DataType.INDEX)) is None
        # Kinds never convert into each other.
        assert conversion_target(ir.TensorType([8, 8], DataType.FP32), scalar(DataType.INDEX)) is None

    def test_the_temporary_name_is_derived_from_the_parameter_and_target(self):
        from pypto.language.scalar_conversion import conversion_temp_name  # noqa: PLC0415

        assert conversion_temp_name("rank", DataType.INT32) == "_rank_int32"
        assert conversion_temp_name("offset", DataType.INDEX) == "_offset_index"
