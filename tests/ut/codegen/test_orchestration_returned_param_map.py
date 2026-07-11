# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration codegen must bind a consumer task to the tensor its producer actually wrote.

A Group/Spmd wrapper around a MULTI-RESULT inner kernel forwards the inner call's whole tuple
as ONE return expression while declaring N return positions. ``return_lineage::TraceVar``
(``src/ir/transforms/utils/return_lineage_utils.cpp``) cannot describe that shape -- it resolves
a var to a SINGLE param root and bails on a multi-result callee::

    if (ret_map.size() != 1 || !ret_map[0]) return nullptr;

so ``ReturnedParamIndices(fa_fused)`` comes back ``[nullopt]``, ``IsReturnedParamMapPrecise()``
is false, and ``GenerateSubmitReturnAliases`` (``src/codegen/orchestration/
orchestration_codegen.cpp``) falls back to a legacy TAIL-ALIGNMENT heuristic::

    tuple_out_base = tuple_arity >= out_indices.size() ? (tuple_arity - out_indices.size()) : 0;
    param_idx      = out_indices[elem_pos - tuple_out_base];

That shifts every returned element onto the wrong Out/InOut param as soon as the callee writes
Out/InOut params it does not return (an in-place KV cache, a ``__gm_pipe_buffer``, ...), so the
downstream task is handed the WRONG TENSOR. Tensor args bind positionally and the buffers share
a dtype, so nothing downstream catches the swap.

The kernels below are the minimal shape that triggers it -- the conjunction of

  (a) a FUNCTION-PARAM tensor rebound by ``pl.assemble`` inside the task,
  (b) under a RUNTIME CONDITIONAL in that task (a lane guard), and
  (c) inside a ``pl.range`` loop of >= 2 trips (1 trip collapses the loop carry).

Drop any one of the three and the binding comes out correct -- the two control tests pin exactly
that, so a regression here cannot be papered over by an unrelated change to the trigger shape.
"""

import re

import pypto.language as pl
import pytest
import torch
from pypto import codegen, passes
from pypto.pypto_core import ir

NUM_CORES = 24
ROWS_PER_CORE = 16
ROWS = NUM_CORES * ROWS_PER_CORE
HIDDEN = 128
MAX_LAYERS = 8
CACHE_ROWS = MAX_LAYERS * ROWS
LANES = NUM_CORES  # of the 2*NUM_CORES AIV lanes, only these carry the guarded writes
HALF_ROWS = ROWS_PER_CORE // 2


# ---------------------------------------------------------------------------------------------
# The layer body, in two arms. They are written out in full rather than selected by a Python
# ``if`` INSIDE the spmd body on purpose: a Python ``if`` in a device region is not folded, it
# becomes IR control flow, and an if-body whose tail is a loop with non-empty return_vars is
# rejected ("control-flow body tail must be YieldStmt").
# ---------------------------------------------------------------------------------------------


@pl.jit.inline
def _layer_guarded(
    cur: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    wid: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    k_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    v_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    nxt: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    layer_idx: pl.Scalar[pl.INT32],
):
    """Producer writes the k/v FUNCTION PARAMS under a runtime lane guard, then attn_out."""
    cache_base = layer_idx * ROWS
    q_pad = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)

    with pl.spmd(NUM_CORES, name_hint="fa_fused", sync_start=True):
        c = pl.get_block_idx()
        r0 = c * ROWS_PER_CORE
        pr0 = ((c + 1) % NUM_CORES) * ROWS_PER_CORE  # cross-core partner read
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = c * 2 + aiv_id
            if lane < LANES:  # (b) the runtime conditional
                for it in pl.pipeline(ROWS_PER_CORE // HALF_ROWS, stage=2):
                    lr0 = lane * ROWS_PER_CORE + it * HALF_ROWS
                    cur_l = pl.slice(cur, [HALF_ROWS, HIDDEN], [lr0, 0])
                    q_pad = pl.assemble(q_pad, pl.cast(pl.mul(cur_l, 2.0), target_type=pl.BF16), [lr0, 0])
                    # (a) FUNCTION PARAMS rebound inside the task
                    k_cache = pl.assemble(
                        k_cache, pl.cast(pl.mul(cur_l, 4.0), target_type=pl.BF16), [cache_base + lr0, 0]
                    )
                    v_cache = pl.assemble(
                        v_cache, pl.cast(pl.mul(cur_l, 8.0), target_type=pl.BF16), [cache_base + lr0, 0]
                    )

        pl.system.syncall(core_type="mix")

        mm = pl.matmul(pl.slice(q_pad, [ROWS_PER_CORE, HIDDEN], [pr0, 0]), wid, out_dtype=pl.FP32)
        k_p = pl.cast(pl.slice(k_cache, [ROWS_PER_CORE, HIDDEN], [cache_base + pr0, 0]), target_type=pl.FP32)
        v_p = pl.cast(pl.slice(v_cache, [ROWS_PER_CORE, HIDDEN], [cache_base + pr0, 0]), target_type=pl.FP32)
        oi = pl.sub(pl.add(mm, k_p), pl.mul(v_p, 0.5))
        attn_out = pl.assemble(attn_out, pl.cast(oi, target_type=pl.BF16), [r0, 0])

    # The downstream consumer. Its one producer-written input is attn_out.
    for c in pl.spmd(NUM_CORES, name_hint="out_proj"):
        r0 = c * ROWS_PER_CORE
        p = pl.matmul(pl.slice(attn_out, [ROWS_PER_CORE, HIDDEN], [r0, 0]), wo, out_dtype=pl.FP32)
        nxt = pl.assemble(nxt, p, [r0, 0])
    return nxt


@pl.jit.inline
def _layer_unconditional(
    cur: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    wid: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    k_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    v_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    nxt: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    layer_idx: pl.Scalar[pl.INT32],
):
    """Control: the same params written by the same task, but WITHOUT the runtime guard."""
    cache_base = layer_idx * ROWS
    q_pad = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)

    with pl.spmd(NUM_CORES, name_hint="fa_fused", sync_start=True):
        c = pl.get_block_idx()
        r0 = c * ROWS_PER_CORE
        pr0 = ((c + 1) % NUM_CORES) * ROWS_PER_CORE
        cur_t = pl.slice(cur, [ROWS_PER_CORE, HIDDEN], [r0, 0])
        q_pad = pl.assemble(q_pad, pl.cast(pl.mul(cur_t, 2.0), target_type=pl.BF16), [r0, 0])
        k_cache = pl.assemble(k_cache, pl.cast(pl.mul(cur_t, 4.0), target_type=pl.BF16), [cache_base + r0, 0])
        v_cache = pl.assemble(v_cache, pl.cast(pl.mul(cur_t, 8.0), target_type=pl.BF16), [cache_base + r0, 0])

        pl.system.syncall(core_type="mix")

        mm = pl.matmul(pl.slice(q_pad, [ROWS_PER_CORE, HIDDEN], [pr0, 0]), wid, out_dtype=pl.FP32)
        k_p = pl.cast(pl.slice(k_cache, [ROWS_PER_CORE, HIDDEN], [cache_base + pr0, 0]), target_type=pl.FP32)
        v_p = pl.cast(pl.slice(v_cache, [ROWS_PER_CORE, HIDDEN], [cache_base + pr0, 0]), target_type=pl.FP32)
        oi = pl.sub(pl.add(mm, k_p), pl.mul(v_p, 0.5))
        attn_out = pl.assemble(attn_out, pl.cast(oi, target_type=pl.BF16), [r0, 0])

    for c in pl.spmd(NUM_CORES, name_hint="out_proj"):
        r0 = c * ROWS_PER_CORE
        p = pl.matmul(pl.slice(attn_out, [ROWS_PER_CORE, HIDDEN], [r0, 0]), wo, out_dtype=pl.FP32)
        nxt = pl.assemble(nxt, p, [r0, 0])
    return nxt


@pl.jit
def guarded_two_trip(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    wid: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    k_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    v_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    """(a) + (b) + (c): the full trigger."""
    cur = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
    for c in pl.spmd(NUM_CORES, name_hint="copy_in"):
        r0 = c * ROWS_PER_CORE
        cur = pl.assemble(cur, pl.slice(x, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])

    for i in pl.range(2):  # (c) >= 2 trips
        nxt = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
        cur = _layer_guarded(cur, wid, wo, k_cache, v_cache, nxt, i)

    for c in pl.spmd(NUM_CORES, name_hint="copy_out"):
        r0 = c * ROWS_PER_CORE
        out = pl.assemble(out, pl.slice(cur, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])
    return out


@pl.jit
def unconditional_two_trip(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    wid: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    k_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    v_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    """Control: (a) + (c), no runtime conditional."""
    cur = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
    for c in pl.spmd(NUM_CORES, name_hint="copy_in"):
        r0 = c * ROWS_PER_CORE
        cur = pl.assemble(cur, pl.slice(x, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])

    for i in pl.range(2):
        nxt = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
        cur = _layer_unconditional(cur, wid, wo, k_cache, v_cache, nxt, i)

    for c in pl.spmd(NUM_CORES, name_hint="copy_out"):
        r0 = c * ROWS_PER_CORE
        out = pl.assemble(out, pl.slice(cur, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])
    return out


@pl.jit
def guarded_one_trip(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    wid: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    k_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    v_cache: pl.Tensor[[CACHE_ROWS, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    """Control: (a) + (b), a 1-trip loop -- no loop carry."""
    cur = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
    for c in pl.spmd(NUM_CORES, name_hint="copy_in"):
        r0 = c * ROWS_PER_CORE
        cur = pl.assemble(cur, pl.slice(x, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])

    for i in pl.range(1):
        nxt = pl.create_tensor([ROWS, HIDDEN], dtype=pl.FP32)
        cur = _layer_guarded(cur, wid, wo, k_cache, v_cache, nxt, i)

    for c in pl.spmd(NUM_CORES, name_hint="copy_out"):
        r0 = c * ROWS_PER_CORE
        out = pl.assemble(out, pl.slice(cur, [ROWS_PER_CORE, HIDDEN], [r0, 0]), [r0, 0])
    return out


# ---------------------------------------------------------------------------------------------
# Helpers: compile (no device) and read the emitted orchestration back as a string.
# ---------------------------------------------------------------------------------------------

# "// Group out_proj: ..." introduces a task; "L0TaskArgs params_tN;" declares its arg list and
# the add_input/add_inout/add_output calls fill that list in order.
_TASK_RE = re.compile(
    r"^\s*//\s*(?:Group|Spmd)\s+(?P<hdr>[^\n]*?)\s*:\s*(?P<tail>[^\n]*)$\n"
    r"\s*L0TaskArgs\s+(?P<params>\w+);\n"
    r"(?P<ops>(?:\s*(?P=params)\.add_\w+\([^\n]*\n)+)",
    re.MULTILINE,
)
_OP_RE = re.compile(r"\.add_(input|inout|output)\((\w+)\)")


def _orch_code(kernel) -> str:
    """Specialize + run the Default pipeline, then emit the orchestration C++ in-process.

    ``compile_for_test`` returns the post-pass ``ir.Program`` without dispatching to a device;
    the torch tensors are read for shape/dtype only. ``codegen.generate_orchestration`` is the
    same emitter that writes ``<build>/orchestration/<name>.cpp``.

    Scoped to BASIC (property) verification, like
    ``_orchestration_codegen_common._generate_orch_full_pipeline(allow_relaxed_verification=True)``:
    a MIX ``pl.spmd`` whose body nests ``pl.split_aiv`` / ``pl.pipeline`` hits a print->parse
    roundtrip gap that is unrelated to the binding under test (all three kernels below trip it,
    controls included). Property verification still runs, so real IR invariant violations are
    still caught.
    """
    sample = (
        torch.empty(ROWS, HIDDEN, dtype=torch.float32),
        torch.empty(HIDDEN, HIDDEN, dtype=torch.bfloat16),
        torch.empty(HIDDEN, HIDDEN, dtype=torch.bfloat16),
        torch.empty(CACHE_ROWS, HIDDEN, dtype=torch.bfloat16),
        torch.empty(CACHE_ROWS, HIDDEN, dtype=torch.bfloat16),
        torch.empty(ROWS, HIDDEN, dtype=torch.float32),
    )
    kernel._cache.clear()
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        program = kernel.compile_for_test(*sample)
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(program, func).code
    raise AssertionError("no Orchestration function in the compiled program")


def _task_operands(code: str, task_name: str) -> list[tuple[str, str]]:
    """[(direction, tensor), ...] in submit order for the task whose comment names *task_name*."""
    for m in _TASK_RE.finditer(code):
        if task_name in m.group("hdr") or task_name in m.group("tail"):
            return _OP_RE.findall(m.group("ops"))
    raise AssertionError(f"task {task_name!r} not found in generated orchestration:\n{code}")


def _assert_consumer_reads_producer_buffer(code: str) -> None:
    """The 'out_proj' task must read the attn_out buffer that 'fa_fused' wrote."""
    producer = _task_operands(code, "fa_fused")
    consumer = _task_operands(code, "out_proj")

    written = [n for d, n in producer if d in ("inout", "output")]
    attn_bufs = [n for n in written if n.startswith("attn_out")]
    assert attn_bufs, f"the producer should have written attn_out, it wrote {written}"
    attn_out = attn_bufs[0]

    reads = [n for d, n in consumer if d == "input"]
    assert attn_out in reads, (
        f"WRONG OPERAND: 'out_proj' must read the buffer its producer 'fa_fused' wrote "
        f"({attn_out!r}), but it reads {reads}. Tensor args bind positionally and both buffers "
        f"are BF16, so nothing downstream catches the swap.\n"
        f"  producer operands: {producer}\n"
        f"  consumer operands: {consumer}"
    )
    assert re.search(rf"add_input\({re.escape(attn_out)}\)", code), (
        f"DEAD WRITE: {attn_out!r} is written by 'fa_fused' but add_input()'d by no task."
    )


class TestReturnedParamMapBinding:
    """A task's operands must be the tensors its producer actually wrote (pypto #2003)."""

    def test_guarded_param_write_in_loop_binds_consumer_correctly(self):
        """The regression: (a) param write, (b) under a runtime guard, (c) in a 2-trip loop.

        On the buggy codegen 'out_proj' is handed ``v_cache__rv_v2`` instead of ``attn_out``,
        and the loop tail yields ``v_cache__rv_v2 = q_pad_inlineNN;``.
        """
        code = _orch_code(guarded_two_trip)
        _assert_consumer_reads_producer_buffer(code)

        # The poisoned map also yields garbage into the loop carry: a k/v cache carry must
        # never be re-bound from an unrelated scratch buffer.
        for m in re.finditer(r"^\s*(\w+)__rv_v\d+\s*=\s*(\w+);\s*$", code, re.MULTILINE):
            carry, src = m.group(1), m.group(2)
            if carry in ("k_cache", "v_cache"):
                assert src.startswith(("k_cache", "v_cache")), (
                    f"GARBAGE LOOP CARRY: {carry!r} is yielded from {src!r} (line: {m.group(0).strip()})"
                )

    def test_unconditional_param_write_binds_consumer_correctly(self):
        """Control (b) removed: same param writes without the runtime guard bind correctly."""
        _assert_consumer_reads_producer_buffer(_orch_code(unconditional_two_trip))

    def test_single_trip_loop_binds_consumer_correctly(self):
        """Control (c) removed: a 1-trip loop has no carry, so the binding is correct."""
        _assert_consumer_reads_producer_buffer(_orch_code(guarded_one_trip))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
