# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Automatic topology selection and hierarchy generation for an unchanged HOST entry."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.distributed_compiled_program import ClusterConfig
from pypto.ir.rank_placement import RankConfig, RankPlacement, lower_rank_placement
from pypto.runtime import RunConfig


def config(*, remote=False):
    return RankConfig(
        4,
        [
            RankPlacement("node-a" if not remote or rank < 2 else "node-b", rank % 2 if remote else rank)
            for rank in range(4)
        ],
        python_executable="/runtime/python",
    )


def program():
    return pl.parse_program("""
@pl.program
class Distributed:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(
        self, x: pl.Tensor[[4], pl.FP32], out: pl.Out[pl.Tensor[[4], pl.FP32]]
    ) -> pl.Tensor[[4], pl.FP32]:
        return out

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(
        self, x: pl.Tensor[[4, 4], pl.FP32], out: pl.Out[pl.Tensor[[4, 4], pl.FP32]]
    ) -> pl.Tensor[[4, 4], pl.FP32]:
        for rank in pl.range(pld.world_size()):
            self.chip(x[rank], out[rank], worker=rank)
        return out
""")


def test_rank_config_selects_local_and_remote_execution():
    local, remote = config().resolve(), config(remote=True).resolve()
    assert isinstance(local, ir.DistributedConfig)
    assert local.device_ids == [0, 1, 2, 3]
    assert isinstance(remote, ClusterConfig)
    assert [(node.host, node.devices) for node in remote.nodes] == [("node-a", [0, 1]), ("node-b", [0, 1])]


@pytest.mark.parametrize(
    "world_size,placements",
    [
        (True, [RankPlacement("node-a", 0)]),
        (2, [RankPlacement("node-a", 0)]),
        (2, [RankPlacement("node-a", 0), RankPlacement("node-a", 0)]),
    ],
)
def test_invalid_rank_maps_fail_before_compilation(world_size, placements):
    with pytest.raises(ValueError):
        RankConfig(world_size, placements)


def test_auto_options_snapshot_mapping_and_separate_cache_identity():
    source = config(remote=True)
    options = RunConfig(rank_config=source).compile_kwargs()
    source.placements[0] = RankPlacement("different", 7)
    assert options["rank_config"].placements[0].host == "node-a"
    assert options["cluster_config"].nodes[0].host == "node-a"
    with pytest.raises(ValueError):
        RunConfig(rank_config=config(), distributed_config=ir.DistributedConfig())


def test_lowering_keeps_local_program_and_routes_remote_chip_calls():
    original = program()
    assert lower_rank_placement(original, config()) is original
    lowered = lower_rank_placement(original, config(remote=True))
    root = lowered.get_function("host")
    assert root is not None and root.level == ir.Level.CLUSTER_0
    relays = [fn for fn in lowered.functions.values() if fn.level == ir.Level.HOST]
    assert len(relays) == 2  # One reusable relay per local device.
    original_root = original.get_function("host")
    assert original_root is not None and original_root.level == ir.Level.HOST
    assert "world_size()" not in ir.python_print(lowered, format=False)


@pytest.mark.parametrize("remote", [False, True])
def test_same_host_program_compiles_and_reloads_for_both_topologies(tmp_path, remote):
    compiled = ir.compile(
        program(), output_dir=str(tmp_path), rank_config=config(remote=remote), skip_ptoas=True
    )
    assert compiled.entry_kind == ("cluster" if remote else "host")
    restored = ir.DistributedCompiledProgram.from_dir(tmp_path)
    assert restored.entry_kind == compiled.entry_kind


def test_unequal_or_interleaved_groups_fail_explicitly():
    for hosts in (["a", "a", "a", "b"], ["a", "b", "a", "b"]):
        mapping = [RankPlacement(host, rank) for rank, host in enumerate(hosts)]
        with pytest.raises(ValueError):
            RankConfig(4, mapping, python_executable="/runtime/python").resolve()


def window_program():
    return pl.parse_program("""
@pl.program
class DistributedWindow:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, x: pl.Tensor[[4], pl.FP32],
             data: pld.DistributedTensor[[4], pl.FP32],
             out: pl.Out[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
        return out

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, x: pl.Tensor[[4, 4], pl.FP32],
             out: pl.Out[pl.Tensor[[4, 4], pl.FP32]]) -> pl.Tensor[[4, 4], pl.FP32]:
        world = pld.world_size()
        shared = pld.alloc_window_buffer(world * 4)
        for epoch in pl.range(2):
            for rank in pl.range(world):
                data = pld.window(shared, [4], dtype=pl.FP32)
                self.chip(x[rank], data, out[rank], worker=rank)
        return out
""")


def test_rank_mapping_changes_cache_identity():
    from pypto.jit.cache import make_cache_key  # noqa: PLC0415

    def key(mapping):
        return make_cache_key(
            source_hash="same-source",
            param_names=[],
            tensor_shapes={},
            tensor_dtypes={},
            dynamic_dims=set(),
            rank_config=mapping,
        )

    original = config()
    reordered = config()
    reordered.placements.reverse()
    assert key(original) == key(config())
    assert key(original) != key(reordered)
    assert key(original) != key(config(remote=True))


def test_global_rank_views_and_local_dispatch_are_preserved():
    lowered = lower_rank_placement(program(), config(remote=True))
    calls = []

    class Collect(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name.startswith("_rank_"):
                calls.append(op)
            super().visit_call(op)

    root = lowered.get_function("host")
    assert root is not None
    Collect().visit_stmt(root.body)
    assert [call.op.name for call in calls] == ["_rank_chip_0", "_rank_chip_1"] * 2
    assert [call.attrs["worker"].value for call in calls] == [0, 0, 1, 1]
    for rank, call in enumerate(calls):
        for argument in call.args:
            assert argument.op.name == "tensor.slice"
            assert argument.args[2].elements[0].value == rank
    for local in range(2):
        relay = lowered.get_function(f"_rank_chip_{local}")
        assert relay is not None and isinstance(relay.body, ir.SeqStmts)
        assignment = relay.body.stmts[0]
        assert isinstance(assignment, ir.AssignStmt)
        dispatch = assignment.value
        assert isinstance(dispatch, ir.Call)
        assert dispatch.attrs["worker"].value == local


@pytest.mark.parametrize("remote", [False, True])
def test_shared_window_and_nested_epoch_loops_compile(tmp_path, remote):
    compiled = ir.compile(
        window_program(), output_dir=str(tmp_path), rank_config=config(remote=remote), skip_ptoas=True
    )
    restored = ir.DistributedCompiledProgram.from_dir(tmp_path)
    assert restored.entry_kind == ("cluster" if remote else "host")
    assert compiled.entry_kind == restored.entry_kind


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
