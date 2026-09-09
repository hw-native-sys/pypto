# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Backend inventory queries must reflect the selected target's actual registry."""

import pytest
from pypto import backend, ir


@pytest.mark.parametrize("target", [backend.BackendType.Ascend910B, backend.BackendType.Ascend950])
def test_backend_inventory_is_a_snapshot_including_legacy_entries(target):
    selected = backend.get_backend_instance(target)
    names = selected.get_registered_op_names()
    assert names == sorted(set(names))
    # The backend still contains historical spellings with no current IR
    # registration. The query must report these too, so an audit can distinguish
    # unreachable legacy emitters from operations that need conversion recipes.
    assert {name for name in names if not ir.is_op_registered(name)} == {
        "tile.assign",
        "tile.move_fp",
        "tile.partadd",
        "tile.partmax",
        "tile.partmin",
        "tile.print",
        "tile.selc",
        "tile.store_fp",
    }
    assert {"tile.load", "tile.store", "tile.matmul", "prefetch.wait", "pld.tile.remote_load"} <= set(names)

    names.clear()
    assert selected.get_registered_op_names()


def test_backend_inventory_preserves_architecture_exclusions():
    a2a3 = set(backend.get_backend_instance(backend.BackendType.Ascend910B).get_registered_op_names())
    a5 = set(backend.get_backend_instance(backend.BackendType.Ascend950).get_registered_op_names())
    mx_ops = {
        ir.get_op(name).name
        for name in (
            "tile.matmul_mx",
            "tile.matmul_mx_acc",
            "tile.matmul_mx_bias",
            "tile.tget_scale_addr",
            "tile.tquant_mx_raw",
            "tile.tmov_x2zz",
        )
    }
    assert mx_ops <= a5
    assert not mx_ops & a2a3
    assert a5 - a2a3 == mx_ops
    assert a2a3 <= a5
