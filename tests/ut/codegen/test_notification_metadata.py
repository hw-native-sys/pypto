# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Notification planning uses selector IR without executing device kernels."""

import shutil
import subprocess

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, ir, passes
from pypto.backend._notification_metadata import generate_notification_metadata


def test_provider_preserves_peer_loop_offsets_and_epoch():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def publish(
            self,
            signal: pld.DistributedTensor[[4, 8], pl.INT32],
            rank: pl.Scalar[pl.INT32],
            epoch: pl.Scalar[pl.INT64],
        ):
            for peer in pl.range(4):
                if peer != rank:
                    pld.system.notify(
                        signal, peer=peer, offsets=[rank, 2], value=epoch + 1, op=pld.NotifyOp.Set
                    )

    program = passes.convert_to_ssa()(Program)
    reg = (7, "publish")
    source = generate_notification_metadata(program, [reg])
    assert "simpler_hbg_notification_plan_v1" in source
    assert "NotificationEventKind::SET" in source
    assert "event.peer =" in source
    assert "event.offsets[0] =" in source
    assert "event.offsets[1] =" in source
    assert "input->scalars[1]" in source
    assert "for (" in source
    assert "kernel_entry" not in source


def test_bound_metadata_callback_executes_without_device_reads(tmp_path):
    runtime = pytest.importorskip("simpler_setup.environment").PROJECT_ROOT
    header = runtime / "src/common/host_build_graph/host/notification_metadata.h"
    if not header.is_file():
        pytest.skip("installed Simpler has no notification planning ABI")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def publish(
            self,
            signal: pld.DistributedTensor[[4, 8], pl.INT32],
            rank: pl.Scalar[pl.INT32],
            epoch: pl.Scalar[pl.INT64],
        ):
            for peer in pl.range(4):
                if peer != rank:
                    pld.system.notify(
                        signal, peer=peer, offsets=[rank, 2], value=epoch + 1, op=pld.NotifyOp.Set
                    )

        @pl.function(type=pl.FunctionType.InCore)
        def waiter(
            self,
            signal: pld.DistributedTensor[[4, 8], pl.INT32],
            row: pl.Scalar[pl.INT32],
            expected: pl.Scalar[pl.INT64],
        ):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[row, 2], expected=expected, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.InCore)
        def context_waiter(self, signal: pld.DistributedTensor[[4, 8], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            context = pld.system.get_comm_ctx(signal)
            rank = pld.system.rank(context)
            pld.system.defer_wait(signal, offsets=[rank, 2], expected=1, cmp=pld.WaitCmp.Ge)

    program = passes.materialize_dist_tensor_ctx()(passes.convert_to_ssa()(Program))
    source = generate_notification_metadata(
        program,
        [(7, "publish"), (9, "waiter"), (11, "context_waiter")],
    )
    source += r"""
#include <cassert>
#include <vector>
using namespace simpler::hbg;
bool collect(void* context, const NotificationPlanEvent* event) {
    static_cast<std::vector<NotificationPlanEvent>*>(context)->push_back(*event);
    return true;
}
bool resolve_context(void*, uint64_t context, int32_t* rank, uint32_t* size) {
    assert(context == 0x99990000);
    *rank = 2;
    *size = 4;
    return true;
}
int main() {
    TensorData tensor{};
    tensor.buffer.addr = 0x12340000; // Deliberately not dereferenceable on Host.
    uint64_t scalars[3]{1, 10, 0x99990000};
    NotificationPlanInput input{};
    input.tensor_count = 1;
    input.scalar_count = 3;
    input.tensors = &tensor;
    input.scalars = scalars;
    std::vector<NotificationPlanEvent> events;
    assert(simpler_hbg_notification_plan_v1(7, &input, collect, &events) == NotificationPlanStatus::COMPLETE);
    assert(events.size() == 3);
    assert(simpler_hbg_notification_kernel_flags_v1(7) == NOTIFICATION_KERNEL_NOTIFY);
    assert(simpler_hbg_notification_kernel_flags_v1(9) == NOTIFICATION_KERNEL_WAIT);
    assert(simpler_hbg_notification_kernel_flags_v1(123) == UINT32_MAX);
    assert(events[0].peer == 0 && events[1].peer == 2 && events[2].peer == 3);
    for (const auto& event : events) {
        assert(event.kind == NotificationEventKind::SET);
        assert(event.comm_context == 0x99990000);
        assert(event.tensor_index == 0 && event.ndims == 2);
        assert(event.offsets[0] == 1 && event.offsets[1] == 2 && event.value == 11);
    }
    // Rebinding a Graph invocation must evaluate its actual values again.
    scalars[0] = 2;
    scalars[1] = 21;
    events.clear();
    assert(simpler_hbg_notification_plan_v1(9, &input, collect, &events) == NotificationPlanStatus::COMPLETE);
    assert(events.size() == 1);
    assert(events[0].kind == NotificationEventKind::WAIT_GE && events[0].peer == -1);
    assert(events[0].offsets[0] == 2 && events[0].value == 21);
    input.scalar_count = 1;
    input.scalars = &scalars[2];
    events.clear();
    assert(simpler_hbg_notification_plan_v1(11, &input, collect, &events) ==
           NotificationPlanStatus::UNSUPPORTED);
    assert(events.empty());
    input.query_context = resolve_context;
    assert(simpler_hbg_notification_plan_v1(11, &input, collect, &events) ==
           NotificationPlanStatus::COMPLETE);
    assert(events.size() == 1 && events[0].offsets[0] == 2);
    assert(events[0].comm_context == 0x99990000 && events[0].value == 1);
    input.version = 0;
    assert(simpler_hbg_notification_plan_v1(9, &input, collect, &events) == NotificationPlanStatus::INVALID);
}
"""
    cpp = tmp_path / "notification.cpp"
    cpp.write_text(source)
    binary = tmp_path / "notification"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{runtime / 'src/common'}",
            f"-I{runtime / 'src/common/task_interface'}",
            f"-I{runtime / 'src/common/platform/include'}",
            str(cpp),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True)


def test_unsupported_wait_selector_remains_identified_as_waiter():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(
            self,
            signal: pld.DistributedTensor[[8], pl.INT32],
            row: pl.Scalar[pl.INT64],
        ):
            pl.func_attr({"deferred_completion_waiter": True})
            dynamic_row = signal[0]
            pld.system.defer_wait(signal, offsets=[dynamic_row], expected=1, cmp=pld.WaitCmp.Ge)

    source = generate_notification_metadata(passes.convert_to_ssa()(Program), [(3, "waiter")])
    assert "return NotificationPlanStatus::UNSUPPORTED;" in source
    assert "case 3: return 2;" in source
    assert "NotificationPlanEvent event" not in source


def test_helper_wait_is_not_silently_classified_as_ordinary():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def waiter(self, signal: pld.DistributedTensor[[8], pl.INT32]):
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[0], expected=1, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.InCore)
        def wrapper(self, signal: pld.DistributedTensor[[8], pl.INT32]):
            self.waiter(signal)

        @pl.function(type=pl.FunctionType.InCore)
        def ordinary_first(self, signal: pld.DistributedTensor[[8], pl.INT32]):
            self.ordinary_second(signal)

        @pl.function(type=pl.FunctionType.InCore)
        def ordinary_second(self, signal: pld.DistributedTensor[[8], pl.INT32]):
            self.ordinary_first(signal)

    source = generate_notification_metadata(
        passes.convert_to_ssa()(Program), [(5, "wrapper"), (6, "ordinary_first"), (7, "ordinary_second")]
    )
    assert "case 5: return 6;" in source
    assert "case 6: return 4;" in source
    assert "case 7: return 4;" in source
    assert "return NotificationPlanStatus::OPAQUE;" in source


def test_hbg_compilation_appends_metadata_to_orchestration(tmp_path):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(self, signal: pld.DistributedTensor[[4, 8], pl.INT32], epoch: pl.Scalar[pl.INT32]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                pld.system.defer_wait(signal, offsets=[1, 2], expected=epoch, cmp=pld.WaitCmp.Ge)

    backend.reset_for_testing()
    try:
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            ir.compile(
                Program,
                output_dir=str(tmp_path),
                skip_ptoas=True,
                dump_passes=False,
            )
    finally:
        backend.reset_for_testing()
    source = (tmp_path / "orchestration/main.cpp").read_text()
    assert "simpler_hbg_notification_plan_v1" in source
    assert "simpler_hbg_notification_kernel_flags_v1" in source
    assert "NotificationEventKind::WAIT_GE" in source


def test_graph_keeps_materialized_communication_context_as_bound_scalar(tmp_path):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Graph)
        def graph(self, signal: pld.DistributedTensor[[4, 8], pl.INT32], epoch: pl.Scalar[pl.INT32]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="waiter"):
                pld.system.defer_wait(signal, offsets=[1, 2], expected=epoch, cmp=pld.WaitCmp.Ge)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, signal: pld.DistributedTensor[[4, 8], pl.INT32], epoch: pl.Scalar[pl.INT32]):
            self.graph(signal, epoch)

    backend.reset_for_testing()
    try:
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            ir.compile(Program, output_dir=str(tmp_path), skip_ptoas=True, dump_passes=False)
    finally:
        backend.reset_for_testing()
    source = (tmp_path / "orchestration/main.cpp").read_text()
    assert "simpler_hbg_notification_plan_v1" in source
    assert "NotificationEventKind::WAIT_GE" in source
    assert "event.comm_context =" in source
    assert "input->scalars[1]" in source
    assert "args.scalar(1)" in source
