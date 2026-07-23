# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for :class:`pypto.runtime.DeviceTensor` construction and metadata."""

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor


class TestDeviceTensorConstruction:
    def test_basic_fp32(self):
        t = DeviceTensor(0x1000, [4, 8], torch.float32)
        assert t.data_ptr == 0x1000
        assert t.shape == (4, 8)
        assert t.dtype is torch.float32
        assert t.nbytes == 4 * 8 * 4

    def test_shape_accepts_tuple_and_list(self):
        a = DeviceTensor(0x100, (2, 3), torch.float16)
        b = DeviceTensor(0x100, [2, 3], torch.float16)
        assert a.shape == b.shape == (2, 3)
        assert a.nbytes == 2 * 3 * 2

    def test_int8_nbytes(self):
        t = DeviceTensor(0x100, [16], torch.int8)
        assert t.nbytes == 16

    def test_zero_ptr_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            DeviceTensor(0, [4], torch.float32)

    def test_negative_ptr_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            DeviceTensor(-1, [4], torch.float32)

    def test_non_int_ptr_raises(self):
        with pytest.raises(ValueError, match="positive int"):
            DeviceTensor("0x100", [4], torch.float32)  # type: ignore[arg-type]

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="all positive"):
            DeviceTensor(0x100, [4, 0], torch.float32)

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError, match="all positive"):
            DeviceTensor(0x100, [-1, 4], torch.float32)

    def test_empty_shape_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            DeviceTensor(0x100, [], torch.float32)

    def test_wrong_dtype_type_raises(self):
        with pytest.raises(TypeError, match=r"torch\.dtype"):
            DeviceTensor(0x100, [4], "fp32")  # type: ignore[arg-type]

    def test_bool_ptr_rejected(self):
        # ``True`` is an int subclass; reject it explicitly so it can't pose as a pointer.
        with pytest.raises(ValueError, match="positive int"):
            DeviceTensor(True, [4], torch.float32)  # type: ignore[arg-type]

    def test_bool_dim_rejected(self):
        with pytest.raises(TypeError, match="must contain ints"):
            DeviceTensor(0x100, [True, 4], torch.float32)  # type: ignore[list-item]

    def test_non_int_dim_rejected_no_silent_truncation(self):
        # ``int(d)`` would silently truncate ``3.7`` to ``3``; the constructor must reject it.
        with pytest.raises(TypeError, match="must contain ints"):
            DeviceTensor(0x100, [3.7, 4], torch.float32)  # type: ignore[list-item]


class TestDeviceTensorImmutability:
    def test_frozen_data_ptr(self):
        t = DeviceTensor(0x100, [4], torch.float32)
        with pytest.raises((AttributeError, TypeError)):
            t.data_ptr = 0x200  # type: ignore[misc]

    def test_frozen_shape(self):
        t = DeviceTensor(0x100, [4], torch.float32)
        with pytest.raises((AttributeError, TypeError)):
            t.shape = (8,)  # type: ignore[misc]

    def test_hashable(self):
        # frozen dataclass with hashable fields → usable as dict key / set element
        t1 = DeviceTensor(0x100, [4], torch.float32)
        t2 = DeviceTensor(0x100, [4], torch.float32)
        assert hash(t1) == hash(t2)
        assert {t1, t2} == {t1}


class TestDeviceTensorRepr:
    def test_repr_hex(self):
        t = DeviceTensor(0xABCD, [2, 3], torch.int8)
        r = repr(t)
        assert "0xabcd" in r.lower()
        assert "(2, 3)" in r
        assert "int8" in r


def _shards(n, shape=(4, 5), dtype=torch.float32):
    return [DeviceTensor(0x1000 + i * 0x100, shape, dtype) for i in range(n)]


class TestStackedDeviceTensorConstruction:
    def test_basic(self):
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (0, 1, 2))
        assert s.shards == tuple(sh)
        assert s.full_shape == (3, 4, 5)
        assert s.shape == (3, 4, 5)
        assert s.worker_ids == (0, 1, 2)
        assert s.dtype is torch.float32

    def test_non_identity_worker_ids(self):
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (2, 0, 1))
        assert s.worker_ids == (2, 0, 1)
        # __getitem__ is keyed by shard index, independent of worker placement.
        assert s[0] is sh[0]

    def test_shard_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="2 shards"):
            StackedDeviceTensor(_shards(3), (2, 4, 5), (0, 1))

    def test_worker_ids_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="worker_ids"):
            StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1))

    def test_duplicate_worker_ids_raises(self):
        with pytest.raises(ValueError, match="distinct"):
            StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1, 1))

    def test_shard_shape_mismatch_raises(self):
        bad = [DeviceTensor(0x1000 + i, (9, 9), torch.float32) for i in range(3)]
        with pytest.raises(ValueError, match="per-shard shape"):
            StackedDeviceTensor(bad, (3, 4, 5), (0, 1, 2))

    def test_shard_dtype_mismatch_raises(self):
        sh = [
            DeviceTensor(0x1000, (4, 5), torch.float32),
            DeviceTensor(0x2000, (4, 5), torch.float16),
        ]
        with pytest.raises(ValueError, match="dtype"):
            StackedDeviceTensor(sh, (2, 4, 5), (0, 1))

    def test_rank_one_full_shape_raises(self):
        with pytest.raises(ValueError, match="rank >= 2"):
            StackedDeviceTensor([DeviceTensor(0x1000, (4,), torch.float32)], (1,), (0,))

    def test_empty_leading_dim_raises(self):
        # B == 0 would leave shards empty; .dtype/.__repr__ would IndexError.
        with pytest.raises(ValueError, match="at least one shard"):
            StackedDeviceTensor([], (0, 4, 5), ())


class TestStackedDeviceTensorIndexing:
    def test_int_index_returns_shard(self):
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (0, 1, 2))
        assert s[0] is sh[0]
        assert s[2] is sh[2]

    def test_tuple_index_full_slices(self):
        # The form the generated host_orch emits: x[r, 0:N, 0:M].
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (0, 1, 2))
        assert s[1, 0:4, 0:5] is sh[1]
        assert s[1, :, :] is sh[1]

    def test_ellipsis_index_returns_shard(self):
        # The documented whole-shard form x[i, ...] must behave like x[i].
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (0, 1, 2))
        assert s[1, ...] is sh[1]
        assert s[2, ...] is sh[2]

    def test_ellipsis_with_leading_full_slice(self):
        sh = _shards(3)
        s = StackedDeviceTensor(sh, (3, 4, 5), (0, 1, 2))
        assert s[1, 0:4, ...] is sh[1]

    def test_multiple_ellipsis_rejected(self):
        s = StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1, 2))
        with pytest.raises(IndexError, match="at most one Ellipsis"):
            _ = s[0, ..., ...]

    def test_partial_tail_slice_rejected(self):
        s = StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1, 2))
        with pytest.raises(ValueError, match="whole-shard"):
            _ = s[0, 0:2, 0:5]

    def test_out_of_range_index_raises(self):
        s = StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1, 2))
        with pytest.raises(IndexError, match="out of range"):
            _ = s[3]

    def test_non_int_leading_index_raises(self):
        s = StackedDeviceTensor(_shards(3), (3, 4, 5), (0, 1, 2))
        with pytest.raises(TypeError, match="leading index"):
            _ = s[0:1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
