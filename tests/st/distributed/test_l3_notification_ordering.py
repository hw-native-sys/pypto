# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""A5 HBG schedules a later self-notifier ahead of its standalone waiter."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.ir import DistributedConfig

_WIDTH = 8
_CONSUMERS = 4
_DELAY_STEPS = 2048


_PROGRAM = r"""
import pypto.language as pl
import pypto.language.distributed as pld
_WIDTH = 8
_CONSUMERS = 4
_DELAY_STEPS = 2048
@pl.program
class NotificationOrder:
    @pl.function(type=pl.FunctionType.BODY_TYPE)
    def body(
        self,
        inp: pl.Tensor[[1, _WIDTH], pl.INT32],
        out: pl.Out[pl.Tensor[[_CONSUMERS, _WIDTH], pl.INT32]],
        payload: pl.InOut[pld.DistributedTensor[[1, _WIDTH], pl.INT32]],
        signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[_CONSUMERS, _WIDTH], pl.INT32]:
        with pl.manual_scope():
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="ordered_waiter") as wait_tid:
                pld.system.defer_wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            for slot in pl.range(_CONSUMERS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ordered_consumer", deps=[wait_tid]):
                    value = pl.load(payload, [0, 0], [1, _WIDTH])
                    out = pl.store(value, [slot, 0], out)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="ordered_publisher"):
                value = pl.load(inp, [0, 0], [1, _WIDTH])
                for _ in pl.range(_DELAY_STEPS):
                    value = pl.add(value, 1)
                pld.tile.remote_store(value, target=payload, peer=0, offsets=[0, 0])
                pld.system.notify(signal, peer=0, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[1, _WIDTH], pl.INT32],
        out: pl.Out[pl.Tensor[[_CONSUMERS, _WIDTH], pl.INT32]],
        payload: pl.InOut[pld.DistributedTensor[[1, _WIDTH], pl.INT32]],
        signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[_CONSUMERS, _WIDTH], pl.INT32]:
        out = self.body(inp, out, payload, signal)
        return out

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[1, 1, _WIDTH], pl.INT32],
        outputs: pl.Out[pl.Tensor[[1, _CONSUMERS, _WIDTH], pl.INT32]],
    ) -> pl.Tensor[[1, _CONSUMERS, _WIDTH], pl.INT32]:
        payload_buffer = pld.alloc_window_buffer(_WIDTH * pl.INT32.get_byte())
        signal_buffer = pld.alloc_window_buffer(pl.INT32.get_byte())
        payload = pld.window(payload_buffer, [1, _WIDTH], dtype=pl.INT32)
        signal = pld.window(signal_buffer, [1, 1], dtype=pl.INT32)
        self.chip_orch(inputs[0], outputs[0], payload, signal, device=0)
        return outputs

"""


def _program(native_graph: bool):
    kind = "Graph" if native_graph else "Orchestration"
    source = _PROGRAM.replace("BODY_TYPE", kind)
    if not native_graph:
        start = source.index("    @pl.function(type=pl.FunctionType.Orchestration)\n    def chip_orch")
        end = source.index("    @pl.function(level=pl.Level.HOST", start)
        source = (source[:start] + source[end:]).replace("def body(", "def chip_orch(")
    return pl.parse(source)


def _assert_notification_order(output_dir: Path) -> dict:
    configurations = list(output_dir.rglob("kernel_config.py"))
    publishers: set[int] = set()
    waiters: set[int] = set()
    for path in configurations:
        for kernel in runpy.run_path(str(path)).get("KERNELS", []):
            if "ordered_publisher" in kernel["name"]:
                publishers.add(kernel["func_id"])
            if "ordered_waiter" in kernel["name"]:
                waiters.add(kernel["func_id"])
    assert publishers and waiters, "missing generated publisher/waiter registrations"
    captures = list(output_dir.rglob("chip_swimlane_records.json"))
    assert captures, "missing resident execution capture"
    evidence = []
    for path in captures:
        raw = json.loads(path.read_text())
        protocol = raw.get("aicore_protocol", {})
        assert protocol.get("producer") == "aicore", path
        tasks = protocol["tasks"]
        produced = [task for task in tasks if publishers.intersection(task["kernel_ids"])]
        waited = [task for task in tasks if waiters.intersection(task["kernel_ids"])]
        assert len(produced) == len(waited) == 1, (path, produced, waited)
        producer, waiter = produced[0], waited[0]
        assert producer["task_ref"] > waiter["task_ref"], "test did not submit notifier later"
        assert producer["schedule_rank"] < waiter["schedule_rank"], "notification ordering was not applied"
        assert producer["task_ref"] not in waiter["dependencies"], "ordering became a data DONE dependency"
        rows = {row[2]: row for row in raw["aicore_tasks"]}
        p = rows[producer["trace_base"]]
        w = rows[waiter["trace_base"]]
        evidence.append(
            {
                "capture": str(path),
                "producer": producer,
                "waiter": waiter,
                "publisher_kernel": p,
                "waiter_kernel": w,
                "waiter_returns_before_publisher": w[4] < p[4],
            }
        )
    return {"captures": evidence}


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("native_graph", [False, True], ids=["ordinary", "graph"])
def test_hbg_notification_source_precedes_standalone_wait(test_config, device_ids, tmp_path, native_graph):
    config = replace(test_config, enable_chip_swimlane=4, save_kernels=True)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = ir.compile(
            _program(native_graph),
            output_dir=str(tmp_path / "compiled"),
            platform=config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:1], num_sub_workers=0),
        )
    inputs = torch.arange(_WIDTH, dtype=torch.int32).reshape(1, 1, _WIDTH).share_memory_()
    outputs = torch.full((1, _CONSUMERS, _WIDTH), -1, dtype=torch.int32).share_memory_()
    with compiled.prepare(config=config) as worker:
        worker(inputs, outputs, config=config)
    expected = (inputs + _DELAY_STEPS).expand(1, _CONSUMERS, _WIDTH)
    assert torch.equal(outputs, expected)
    evidence = _assert_notification_order(compiled.output_dir)
    evidence.update({"devices": device_ids[:1], "native_graph": native_graph, "numeric_passed": True})
    (tmp_path / "notification-order-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
