# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Admission rejects unknown memory relationships instead of adding barriers."""

from itertools import product

import pytest
from pypto.runtime._vpto_sync import _MemoryHistory, validate_sync_contract


def llvm_helper(pointer="%base", offset="128", extra="", setup=""):
    store_args = "<64 x float> %value, ptr addrspace(6) %base, i32 0, i32 2, i32 0, <256 x i1> %mask"
    return f"""define void @helper(ptr addrspace(6) %base, ptr addrspace(6) %other, i64 %dynamic) #0 {{
  {setup}
  br label %head
head:
  %iv = phi i64 [ %next, %latch ], [ 0, %entry ]
  %cmp = icmp slt i64 %iv, 1
  br i1 %cmp, label %body, label %end
body:
  %mask = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %value = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.0, <256 x i1> %mask, i32 1)
  call void @llvm.hivm.vstsx1.v64f32({store_args})
  call void @llvm.hivm.mem.bar.vst.vld()
  %loaded = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) {pointer}, i32 {offset}, i32 0, i32 0)
  {extra}
  br label %latch
latch:
  %next = add i64 %iv, 1
  br label %head, !llvm.loop !1
end:
  ret void
}}
!1 = distinct !{{!1, !2}}
!2 = !{{!"llvm.loop.aivector_scope"}}
"""


@pytest.mark.parametrize("offset", ["0", "128", "256", "512"])
def test_static_common_base_is_admitted_without_rewriting(offset):
    source = llvm_helper(offset=offset)
    validate_sync_contract(source, "helper.pto")
    validate_sync_contract(
        llvm_helper(
            pointer="%view",
            offset="0",
            setup=f"%view = getelementptr float, ptr addrspace(6) %base, i64 {int(offset) // 4}",
        ),
        "helper.pto",
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"pointer": "%other"}, "unrelated pointer"),
        ({"offset": "%dynamic"}, "static offset"),
        ({"offset": "4"}, "32-byte alignment"),
        (
            {"setup": "%view = getelementptr float, ptr addrspace(6) %base, i64 %dynamic"},
            "constant element offset",
        ),
        ({"setup": "%view = getelementptr float, ptr addrspace(6) %base, i64 -1"}, "out of range"),
        ({"setup": "%view = inttoptr i64 %dynamic to ptr addrspace(6)"}, "unmodeled instruction"),
        ({"extra": "%x = load float, ptr addrspace(6) %base"}, "unmodeled instruction"),
        ({"extra": "store float 1.0, ptr addrspace(6) %base"}, "unmodeled instruction"),
        (
            {"extra": "call void @llvm.memcpy(ptr addrspace(6) %base, ptr addrspace(6) %other)"},
            "unmodeled intrinsic",
        ),
        ({"extra": "call void @llvm.hivm.unknown()"}, "unmodeled intrinsic"),
        ({"extra": "call void %indirect()"}, "unmodeled instruction"),
        ({"extra": 'call void asm "", ""()'}, "unmodeled instruction"),
        (
            {"setup": "%p = select i1 true, ptr addrspace(6) %base, ptr addrspace(6) %other"},
            "unmodeled instruction",
        ),
    ],
)
def test_unproven_access_is_rejected_with_source(kwargs, message):
    with pytest.raises(ValueError, match=message) as error:
        validate_sync_contract(llvm_helper(**kwargs), "unsafe.pto")
    assert "unsafe.pto" in str(error.value)


@pytest.mark.parametrize(
    "old,new",
    [
        ("icmp slt i64 %iv, 1", "icmp slt i64 %iv, %dynamic"),
        ("add i64 %iv, 1", "add i64 %iv, 0"),
        ("br label %latch", "br label %body"),
        ("i32 0, i32 2, i32 0", "i32 0, i32 1, i32 0"),
        ("vldsx1.v64f32", "vldsx2.v64f32"),
        ("ptr addrspace(6) %base, ptr", "ptr addrspace(1) %base, ptr"),
    ],
)
def test_unknown_cfg_or_memory_mode_is_rejected(old, new):
    with pytest.raises(ValueError, match="unsupported VF synchronization contract"):
        validate_sync_contract(llvm_helper().replace(old, new), "unsupported.pto")


def test_missing_definition_and_extra_definition_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        validate_sync_contract("declare void @helper()", "empty.pto")
    with pytest.raises(ValueError, match="exactly one"):
        validate_sync_contract(llvm_helper() + llvm_helper(), "multiple.pto")


@pytest.mark.parametrize("replacement", ["", "call void @llvm.hivm.mem.bar.vld.vst()"])
def test_missing_or_wrong_barrier_rejects_overlap(replacement):
    source = llvm_helper().replace("call void @llvm.hivm.mem.bar.vst.vld()", replacement)
    with pytest.raises(ValueError, match="missing vst.vld barrier"):
        validate_sync_contract(source, "missing.pto")
    # No extra synchronization is required for disjoint footprints.
    validate_sync_contract(source.replace("i32 128,", "i32 256,"), "disjoint.pto")


def test_barrier_before_first_access_does_not_cover_later_hazard():
    source = llvm_helper().replace("  call void @llvm.hivm.mem.bar.vst.vld()\n", "")
    source = source.replace("body:\n", "body:\n  call void @llvm.hivm.mem.bar.vst.vld()\n")
    with pytest.raises(ValueError, match="missing vst.vld barrier"):
        validate_sync_contract(source, "early.pto")


def test_missing_aiv_marker_is_rejected():
    with pytest.raises(ValueError, match="missing AIV scope metadata"):
        validate_sync_contract(
            llvm_helper().replace("llvm.loop.aivector_scope", "llvm.loop.unroll.disable"), "scope.pto"
        )


def _trace_is_ordered_by_graph(trace):
    """Independent bounded oracle: barriers create edges, then take closure."""
    memory = [(i, action[0], action[1]) for i, action in enumerate(trace) if isinstance(action, tuple)]
    edges = {i: set() for i, _, _ in memory}
    for position, action in enumerate(trace):
        if isinstance(action, str):
            source, target = action.split(".")
            for i, pipe, _ in memory:
                if i < position and pipe == source:
                    for j, next_pipe, _ in memory:
                        if j > position and next_pipe == target:
                            edges[j].add(i)
    ancestors = {}
    for i, _, _ in memory:
        ancestors[i] = set(edges[i])
        for predecessor in edges[i]:
            ancestors[i].update(ancestors[predecessor])
    for i, pipe, address in memory:
        for j, prior_pipe, prior_address in memory:
            if (
                j < i
                and (address is None or prior_address is None or address == prior_address)
                and "vst" in (pipe, prior_pipe)
            ):
                if j not in ancestors[i]:
                    return False
    return True


def test_transitive_barrier_frontiers_match_dependency_graph():
    actions = (("vld", 0), ("vst", 0), ("vld", 256), ("vst", 256), "vst.vld", "vld.vst", "vst.vst")
    checked = 0
    for trace in product(actions, repeat=5):
        expected = _trace_is_ordered_by_graph(trace)
        history = _MemoryHistory()
        accepted = True
        try:
            for position, action in enumerate(trace):
                if isinstance(action, str):
                    history.barrier(action)
                else:
                    pipe, address = action
                    history.access(address, pipe == "vld", position)
        except ValueError:
            accepted = False
        assert accepted == expected, trace
        checked += 1
    assert checked == 16807


def test_write_read_write_uses_existing_transitive_barriers():
    store = (
        "call void @llvm.hivm.vstsx1.v64f32(<64 x float> %value, ptr addrspace(6) %base, "
        "i32 0, i32 2, i32 0, <256 x i1> %mask)"
    )
    source = llvm_helper(offset="0", extra="call void @llvm.hivm.mem.bar.vld.vst()\n  " + store)
    validate_sync_contract(source, "transitive.pto")
    # Without the bridging read, WR followed by RW cannot prove a WW edge.
    missing_read = "\n".join(line for line in source.splitlines() if "%loaded =" not in line)
    with pytest.raises(ValueError, match="missing vst.vst"):
        validate_sync_contract(missing_read, "no-bridge.pto")


@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("unknown_first", [True, False])
def test_unknown_indexed_access_requires_order_against_every_interval(load, unknown_first):
    history = _MemoryHistory()
    history.access(None if unknown_first else 8192, not load, 0)
    with pytest.raises(ValueError, match="missing"):
        history.access(8192 if unknown_first else None, load, 1)
    history.barrier("vv.all")
    history.access(8192 if unknown_first else None, load, 2)


def test_unknown_footprints_match_independent_dependency_graph():
    actions = (
        ("vld", None),
        ("vst", None),
        ("vld", 0),
        ("vst", 0),
        ("vst", 512),
        "vst.vld",
        "vld.vst",
        "vst.vst",
    )
    for trace in product(actions, repeat=4):
        history = _MemoryHistory()
        accepted = True
        try:
            for position, action in enumerate(trace):
                if isinstance(action, str):
                    history.barrier(action)
                else:
                    history.access(action[1], action[0] == "vld", position)
        except ValueError:
            accepted = False
        assert accepted == _trace_is_ordered_by_graph(trace), trace


@pytest.mark.parametrize("suffix,element", [("v64i32", "i32"), ("v128f16", "half")])
def test_typed_contiguous_memory_keeps_physical_256_byte_footprint(suffix, element):
    lanes = 64 if element == "i32" else 128
    source = llvm_helper().replace("v64f32", suffix).replace("<64 x float>", f"<{lanes} x {element}>")
    # Replace the pure broadcast too, using a scalar of its exact element type.
    source = source.replace("float 5.0", f"{element} 0")
    if element == "half":
        source = source.replace("i32 0, i32 2, i32 0", "i32 0, i32 1, i32 0")
    validate_sync_contract(source, "typed.pto")
    with pytest.raises(ValueError, match="missing vst.vld"):
        validate_sync_contract(source.replace("call void @llvm.hivm.mem.bar.vst.vld()", ""), "typed.pto")


@pytest.mark.parametrize("suffix,indices", [("v64f32", "<64 x i32>"), ("v128f16", "<128 x i16>")])
def test_scatter_and_gather_require_barriers_against_other_accesses(suffix, indices):
    vector = "<64 x float>" if suffix == "v64f32" else "<128 x half>"
    scatter = (
        f"call void @llvm.hivm.vscatter.{suffix}.v300({vector} %value, "
        f"ptr addrspace(6) %base, {indices} %idx, <256 x i1> %mask)"
    )
    gather_name = "vgather2.bc.v64f32" if suffix == "v64f32" else "vgather2.v300.v128f16"
    gather = (
        f"%indexed = call {vector} @llvm.hivm.{gather_name}("
        "ptr addrspace(6) %base, <64 x i32> %packed, <256 x i1> %mask)"
    )
    barrier = "call void @llvm.hivm.mem.bar.vv.all()"
    source = llvm_helper(extra=f"{barrier}\n{scatter}\n{barrier}\n{gather}")
    validate_sync_contract(source, "indexed.pto")
    with pytest.raises(ValueError, match="missing vst.vld"):
        validate_sync_contract(source.replace(f"{scatter}\n{barrier}", scatter), "indexed.pto")
    with pytest.raises(ValueError, match="missing vld.vst"):
        validate_sync_contract(source.replace(f"{barrier}\n{scatter}", scatter), "indexed.pto")


@pytest.mark.parametrize(
    "extra",
    [
        "%x = call <64 x float> @llvm.hivm.vmula.s.m.v64f32("
        "<64 x float> %value, <64 x float> %value, <64 x float> %value, <256 x i1> %mask)",
        "%pair = call { <64 x i32>, <256 x i1> } @llvm.hivm.vaddc.v64u32("
        "<64 x i32> %a, <64 x i32> %b, <256 x i1> %mask)\n"
        "%x = extractvalue { <64 x i32>, <256 x i1> } %pair, 0",
    ],
)
def test_pure_extended_intrinsics_do_not_obscure_memory_dependencies(extra):
    source = llvm_helper(extra=extra)
    validate_sync_contract(source, "arithmetic.pto")
    with pytest.raises(ValueError, match="missing vst.vld"):
        validate_sync_contract(source.replace("call void @llvm.hivm.mem.bar.vst.vld()", ""), "arithmetic.pto")


@pytest.mark.parametrize("stride,update", [(65536, 0), (131072, 0), (65536, 1)])
def test_byte_load_requires_contiguous_non_updating_stride(stride, update):
    source = llvm_helper(
        extra=(
            "%bytes = call <256 x i8> @llvm.hivm.vsldb.v256i8("
            f"ptr addrspace(6) %base, i32 {stride}, i32 {update}, <256 x i1> %mask)"
        )
    )
    if (stride, update) != (65536, 0):
        with pytest.raises(ValueError, match="contiguous non-updating stride"):
            validate_sync_contract(source, "byte.pto")
    else:
        validate_sync_contract(source, "byte.pto")
        with pytest.raises(ValueError, match="missing"):
            validate_sync_contract(source.replace("call void @llvm.hivm.mem.bar.vst.vld()", ""), "byte.pto")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
