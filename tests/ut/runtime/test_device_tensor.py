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
from pypto.runtime import DeviceTensor


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
