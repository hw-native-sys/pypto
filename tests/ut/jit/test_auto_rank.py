# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Compile the same lib-style HOST callable for local and remote ranks."""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.runtime import RunConfig


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_jit_host_selects_hierarchy_from_global_rank_map(remote, dynamic):
    rows = pl.dynamic("auto_rank_rows") if dynamic else 4

    @pl.jit
    def chip(x: pl.Tensor[[rows, 4], pl.FP32], out: pl.Out[pl.Tensor[[rows, 4], pl.FP32]]):
        return out

    @pl.jit.host
    def host(x: pl.Tensor[[4, rows, 4], pl.FP32], out: pl.Out[pl.Tensor[[4, rows, 4], pl.FP32]]):
        for rank in pl.range(pld.world_size()):
            chip(x[rank], out[rank], worker=rank)
        return out

    placements = [
        ir.RankPlacement("node-a" if not remote or r < 2 else "node-b", r % 2 if remote else r)
        for r in range(4)
    ]
    options = RunConfig(rank_config=ir.RankConfig(4, placements))
    compiled = host.compile(config=options)
    assert compiled.entry_kind == ("cluster" if remote else "host")
    assert host.compile(config=options) is compiled
    assert ir.DistributedCompiledProgram.from_dir(compiled.output_dir).entry_kind == compiled.entry_kind
    reordered = host.compile(config=RunConfig(rank_config=ir.RankConfig(4, list(reversed(placements)))))
    assert reordered is not compiled
    assert reordered.entry_kind == compiled.entry_kind


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
