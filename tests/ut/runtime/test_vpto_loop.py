# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Loop admission, cyclic synchronization, and final machine-code evidence."""

from itertools import product

import pytest
from pypto.runtime._vpto import hardware_loop_requirement, validate_hardware_loop_evidence
from pypto.runtime._vpto_memory import Access, validate_loop_history
from pypto.runtime._vpto_sync import validate_sync_contract
from test_vpto_sync import _trace_is_ordered_by_graph

# Canonical LLVM emitted from backend/vpto_vf_validation/hardware_loop.pto.
LLVM = """define void @run_vf_loop(ptr addrspace(6) %0, i32 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 1024
  %4 = getelementptr float, ptr addrspace(6) %0, i64 2048
  br label %5

5:                                                ; preds = %aivscope.latch, %2
  %6 = phi i64 [ %36, %aivscope.latch ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %37

8:                                                ; preds = %5
  %9 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %10 = getelementptr float, ptr addrspace(6) %0, i64 0
  %11 = getelementptr float, ptr addrspace(6) %3, i64 0
  %12 = getelementptr float, ptr addrspace(6) %4, i64 0
  %13 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %1, <256 x i1> %9, i32 1)
  br label %14

14:                                               ; preds = %20, %8
  %15 = phi i16 [ %34, %20 ], [ 0, %8 ]
  %16 = phi ptr addrspace(6) [ %27, %20 ], [ %10, %8 ]
  %17 = phi ptr addrspace(6) [ %30, %20 ], [ %11, %8 ]
  %18 = phi ptr addrspace(6) [ %33, %20 ], [ %12, %8 ]
  %19 = icmp slt i16 %15, 16
  br i1 %19, label %20, label %35

20:                                               ; preds = %14
  %21 = sext i16 %15 to i64
  %22 = mul i64 %21, 64
  %23 = trunc i64 %22 to i32
  %24 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 %23, i32 0)
  %25 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(\
ptr addrspace(6) %16, i32 256, i32 0, i32 1)
  %26 = extractvalue { <64 x float>, ptr addrspace(6) } %25, 0
  %27 = extractvalue { <64 x float>, ptr addrspace(6) } %25, 1
  %28 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(\
ptr addrspace(6) %17, i32 256, i32 0, i32 1)
  %29 = extractvalue { <64 x float>, ptr addrspace(6) } %28, 0
  %30 = extractvalue { <64 x float>, ptr addrspace(6) } %28, 1
  %31 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %26, <64 x float> %29, <256 x i1> %9)
  %32 = call <256 x i1> @llvm.hivm.vcmp.lt.s.z.v64i32(<64 x i32> %24, <64 x i32> %13, <256 x i1> %9)
  %33 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(\
<64 x float> %31, ptr addrspace(6) %18, i32 256, i32 2, i32 1, <256 x i1> %32)
  %34 = add i16 %15, 1
  br label %14

35:                                               ; preds = %14
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %35
  %36 = add i64 %6, 1
  br label %5, !llvm.loop !1

37:                                               ; preds = %5
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
"""


def test_canonical_loop_admission():
    validate_sync_contract(LLVM, "loop.pto", hardware_loop_trip_count=16)
    with pytest.raises(ValueError, match="straight-line"):
        validate_sync_contract(LLVM, "undeclared.pto")


@pytest.mark.parametrize(
    "old,new",
    [
        ("icmp slt i16 %15, 16", "icmp slt i16 %15, 15"),
        ("add i16 %15, 1", "add i16 %15, 2"),
        ("i32 256, i32 0, i32 1", "i32 128, i32 0, i32 1"),
        ("[ %27, %20 ]", "[ %30, %20 ]"),
        ("[ %10, %8 ]", "[ %unknown, %8 ]"),
        ("br label %14", "br label %20"),
        ("i64 2048", "i64 2049"),
        ("llvm.loop.aivector_scope", "llvm.loop.unroll.disable"),
        ("i64 2048", "i64 %1"),
    ],
)
def test_unproved_loop_is_rejected(old, new):
    assert old in LLVM
    with pytest.raises(ValueError, match="unsupported VF synchronization contract"):
        validate_sync_contract(LLVM.replace(old, new), "loop.pto", hardware_loop_trip_count=16)


def test_overlapping_loop_needs_backedge_barrier():
    # Alias the destination with input. The read/write barrier alone does not
    # order the store before a later iteration's read of the overlapping union.
    alias = LLVM.replace("i64 2048", "i64 0")
    with pytest.raises(ValueError, match="missing vld.vst"):
        validate_sync_contract(alias, "loop.pto", hardware_loop_trip_count=16)
    alias = alias.replace("  %33 =", "  call void @llvm.hivm.mem.bar.vld.vst()\n  %33 =")
    with pytest.raises(ValueError, match="missing vst.vld"):
        validate_sync_contract(alias, "loop.pto", hardware_loop_trip_count=16)
    alias = alias.replace("  %34 =", "  call void @llvm.hivm.mem.bar.vst.vld()\n  %34 =")
    validate_sync_contract(alias, "loop.pto", hardware_loop_trip_count=16)


def test_summary_acceptance_implies_concrete_ordering():
    # Independent dependency-graph oracle expands small loops. Admission may
    # conservatively reject, but must never accept an unordered concrete trace.
    choices = (
        ("vld", 0, True),
        ("vst", 0, True),
        ("vst", 256, True),
        ("vst", 256, False),
        "vld.vst",
        "vst.vld",
        "vst.vst",
    )
    checked = 0
    for body in product(choices, repeat=4):
        for trips in (2, 3, 5):
            summary = [
                action
                if isinstance(action, str)
                else Access(
                    action[1],
                    action[1] + 256 * (trips if action[2] else 1),
                    action[0] == "vld",
                    index,
                    action[2],
                )
                for index, action in enumerate(body)
            ]
            try:
                validate_loop_history([], summary, [])
            except ValueError:
                continue
            trace = [
                action
                if isinstance(action, str)
                else (action[0], action[1] + (iteration * 256 if action[2] else 0))
                for iteration in range(trips)
                for action in body
            ]
            assert _trace_is_ordered_by_graph(trace), (body, trips)
            checked += 1
    assert checked > 1000


def test_summary_checks_entry_exit_and_distant_iteration_alias():
    moving = Access(0, 4096, False, 1, True)
    read = Access(3840, 4096, True, 2)
    with pytest.raises(ValueError, match="missing vst.vld"):
        validate_loop_history([], [moving], [read])
    validate_loop_history([], [moving], ["vst.vld", read])
    with pytest.raises(ValueError, match="missing vst.vst"):
        validate_loop_history([Access(3840, 4096, False, 0)], [moving], [])
    validate_loop_history([Access(3840, 4096, False, 0), "vst.vst"], [moving], [])


def test_source_requirement_is_explicit_and_validated():
    assert hardware_loop_requirement("module {}") is None
    assert hardware_loop_requirement("// pypto.hardware_loop: helper 16\nmodule {}") == ("helper", 16)
    for marker in (
        "helper 0",
        "helper 32768",
        "helper dynamic",
        "helper 16\n// pypto.hardware_loop: other 2",
    ):
        with pytest.raises(ValueError, match="Invalid VF"):
            hardware_loop_requirement("// pypto.hardware_loop: " + marker)


@pytest.mark.parametrize(
    "thread,opcode,extra",
    [
        ("helper", "VLOOPV2_V310", ""),
        ("other.vector.thread", "VLOOPV2_V310", ""),
        ("helper.vector.thread", "VADD", ""),
        ("helper.vector.thread", "VLOOP", ""),
        ("helper.vector.thread", "VLOOPV2_V310", "  11c: c1001f1e VLOOPV2_V310 S2, #7, #1, #1"),
    ],
)
def test_missing_or_ambiguous_final_evidence_is_rejected(thread, opcode, extra):
    text = f"00000100 <{thread}>:\n  118: c1001f1e {opcode} S2, #7, #1, #1\n{extra}"
    with pytest.raises(ValueError, match="backend evidence missing"):
        validate_hardware_loop_evidence(text, [("helper", 16)])


def test_final_evidence_is_attached_to_each_required_symbol():
    text = "00000100 <helper.vector.thread>:\n  118: c1001f1e VLOOPV2_V310 S2, #7, #1, #1"
    validate_hardware_loop_evidence(text, [("helper", 16)])
    with pytest.raises(ValueError, match="other"):
        validate_hardware_loop_evidence(text, [("helper", 16), ("other", 16)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
