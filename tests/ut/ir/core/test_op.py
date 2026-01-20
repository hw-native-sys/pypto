# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Op class."""

import pytest
from pypto import ir


class TestOp:
    """Tests for Op class."""

    def test_op_creation(self):
        """Test creating an Op."""
        op = ir.Op("add")
        assert op.name == "add"

    def test_op_name_immutability(self):
        """Test that Op name is immutable."""
        op = ir.Op("multiply")
        with pytest.raises(AttributeError):
            op.name = "divide"  # type: ignore


if __name__ == "__main__":
    pytest.main(["-v", __file__])
