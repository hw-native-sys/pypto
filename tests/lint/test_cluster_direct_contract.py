# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The L4 adapter must not silently activate the retired group transport."""

import ast
from pathlib import Path


def test_l4_adapter_never_activates_group_transport():
    source = Path(__file__).resolve().parents[2] / "python/pypto/runtime/cluster_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {"MpiL3GroupSpec", "add_mpirun_worker_group", "MpiGroupMailboxChannel"}
    used = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert not used & forbidden, f"L4 still activates legacy group transport: {sorted(used & forbidden)}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
