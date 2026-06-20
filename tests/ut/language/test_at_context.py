# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests that ``pl.at(...)`` forwards its keyword arguments into ``AtContext``.

Regression: ``pl.at(..., allow_early_resolve=True)`` was accepted by ``at()`` but
silently dropped because the flag was not forwarded into the returned
``AtContext`` (PR #1819 review). Every accepted keyword must reach the context.
"""

import pypto.language as pl
import pytest
from pypto import ir


def test_at_forwards_allow_early_resolve():
    """pl.at(..., allow_early_resolve=True) carries the flag onto AtContext."""
    ctx = pl.at(ir.Level.CORE_GROUP, allow_early_resolve=True)
    assert ctx.allow_early_resolve is True


def test_at_allow_early_resolve_defaults_false():
    """allow_early_resolve defaults to False on AtContext when omitted."""
    ctx = pl.at(ir.Level.CORE_GROUP)
    assert ctx.allow_early_resolve is False


def test_at_forwards_name_hint():
    """pl.at(..., name_hint=...) reaches AtContext (sibling-kwarg forwarding guard)."""
    ctx = pl.at(ir.Level.CORE_GROUP, name_hint="fused_scope")
    assert ctx.name_hint == "fused_scope"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
