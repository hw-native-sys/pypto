# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MoE routing in a recorded Graph: dispatch predicates instead of a branch.

A Mixture-of-Experts block poses a problem a recorded graph cannot solve on its
own. `host_build_graph` records a Graph's task topology on the first call and
replays it afterwards, so the topology may not vary between calls — but which
experts receive tokens is decided at run time, and under HBG the whole graph is
built *before* the device runs anything, so the orchestration cannot even read a
count that some task has yet to produce.

The answer is not a branch:

  1. **Enumerate every expert at build time** — a static topology sized to the
     worst case.
  2. **Gate each expert's dispatch with a predicate** — `counts[e, 0] > 0`. The
     scheduler evaluates it at the *dispatch point*, on device, after the task's
     dependencies are satisfied, so the value is current. A task whose predicate
     is false is retired inline and never reaches a core, while its fanin and
     fanout still settle so consumers unlock normally.

That is the shape simpler's own DeepSeek-V4 HBG case uses
(`runtime/examples/a2a3/host_build_graph/deepseek_v4_flash_decode`), where the
per-expert tile tasks are predicated on `recv_count_out[expert][0] > t0`.

**The counts here are host-provided.** When they are produced by an earlier
device task — the usual MoE arrangement, where a router kernel writes them — that
producer must appear in the consuming task's `deps=`, so the dispatch-point read
cannot observe a stale value. The parser enforces that where it can prove it
statically; beyond that it is yours to honour. This example keeps the counts an
input so the routing is visible in the assertions, and does not model the router.

Concepts introduced:
  - `@pl.jit.graph`: a recordable orchestration fragment, recorded once and
    replayed per layer
  - `predicate=` on `pl.spmd`: run-time gating without a branch
  - `pl.unroll`: the expert loop expands at compile time, so each expert's weight
    offset is a per-node constant — a `pl.range` induction variable is not
    something the recording can reproduce

Run:  python examples/advanced/04_moe_graph_predicate.py -p a2a3sim
"""

import argparse

import pypto.language as pl
import torch
from pypto.pypto_core import passes
from pypto.runtime import RunConfig

D = 64
EXPERTS = 4
LAYERS = 2


@pl.jit.incore
def expert_ffn(
    x: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    base: pl.Scalar[pl.INDEX],
):
    """One expert's SwiGLU FFN: two cube matmuls, a vector activation, a third matmul.

        gate = x @ w_gate[e]          (cube)
        up   = x @ w_up[e]            (cube)
        h    = gate * sigmoid(gate) * up   (vector)
        out[e] = h @ w_down[e]        (cube)

    The intermediates stay on chip — ``gate``, ``up`` and ``h`` never round-trip
    through GM, so the whole expert is one task.

    ``base`` selects the expert's band in each weight and in the output.
    Addressing them *inside* the kernel rather than slicing in the orchestration
    is deliberate: a view taken in a recorded region is re-derived from whatever
    the recording froze, so `LegalizeGraphBoundary` rejects one built from a
    region-local loop variable.

    Writing a **disjoint** band per expert also removes a hazard: the per-expert
    dispatches are independent and nothing orders them, so experts sharing one
    accumulator would be a write-after-write race — which looks exactly like a
    predicate misfiring.
    """
    tx = pl.load(x, [0, 0], [D, D], target_memory=pl.MemorySpace.Mat)
    tg = pl.load(w_gate, [base, 0], [D, D], target_memory=pl.MemorySpace.Mat)
    tu = pl.load(w_up, [base, 0], [D, D], target_memory=pl.MemorySpace.Mat)
    td = pl.load(w_down, [base, 0], [D, D], target_memory=pl.MemorySpace.Mat)

    lx = pl.move(tx, target_memory=pl.MemorySpace.Left)
    gate = pl.matmul(lx, pl.move(tg, target_memory=pl.MemorySpace.Right))
    up = pl.matmul(lx, pl.move(tu, target_memory=pl.MemorySpace.Right))

    sigmoid = pl.recip(pl.add(pl.exp(pl.mul(gate, -1.0)), 1.0))
    h = pl.mul(pl.mul(gate, sigmoid), up)

    hm = pl.move(h, target_memory=pl.MemorySpace.Mat)
    result = pl.matmul(
        pl.move(hm, target_memory=pl.MemorySpace.Left),
        pl.move(td, target_memory=pl.MemorySpace.Right),
    )
    out = pl.store(result, [base, 0], out)
    return out


@pl.jit.graph
def moe_block(
    x: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    counts: pl.Tensor,
    out: pl.InOut[pl.Tensor],
):
    """One recorded block: every expert enumerated, each one gated.

    ``pl.unroll`` rather than ``pl.range``: the loop expands at compile time, so
    each expert's ``e * D`` offset is a literal in the IR — a per-node constant,
    identical on every replay. A ``pl.range`` induction variable is local to the
    region and the call site cannot reproduce it, which `LegalizeGraphBoundary`
    rejects.

    ``pl.spmd`` is the form that carries a predicate in a ``@pl.jit`` function,
    and one dispatch per expert is what gives each expert its own gate. Its body
    dispatches a kernel — a bare ``pl.spmd(1)`` whose body neither reads
    ``pl.tile.get_block_idx()`` nor dispatches one is rejected, since every block
    would then run identical work.
    """
    for e in pl.unroll(EXPERTS):
        with pl.spmd(1, predicate=(counts[e, 0] > 0)):
            out = expert_ffn(x, w_gate, w_up, w_down, out, e * D)
    return out


@pl.jit
def moe_decode(
    x: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    counts: pl.Tensor,
    out: pl.InOut[pl.Tensor],
):
    """The graph is recorded on the first layer and replayed on the rest."""
    for _ in pl.range(LAYERS):
        out = moe_block(x, w_gate, w_up, w_down, counts, out)
    return out


def expected_bands(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Torch reference: a routed expert's band is its SwiGLU FFN, else zero."""
    out = torch.zeros(EXPERTS * D, D, dtype=torch.float32)
    for e in range(EXPERTS):
        if int(counts[e, 0]) <= 0:
            continue
        band = slice(e * D, (e + 1) * D)
        gate = x @ w_gate[band]
        up = x @ w_up[band]
        out[band] = (gate * torch.sigmoid(gate) * up) @ w_down[band]
    return out


def _run_and_check(x, w_gate, w_up, w_down, counts, cfg) -> None:
    out = torch.zeros((EXPERTS * D, D), dtype=torch.float32)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        moe_decode(x, w_gate, w_up, w_down, counts, out, config=cfg)

    want = expected_bands(x, w_gate, w_up, w_down, counts)
    for e in range(EXPERTS):
        band = slice(e * D, (e + 1) * D)
        routed = int(counts[e, 0]) > 0
        assert torch.allclose(out[band], want[band], rtol=3e-2, atol=3e-2), (
            f"expert {e} (count={int(counts[e, 0])}) "
            f"{'should have run' if routed else 'should have been retired inline'}; "
            f"max diff = {(out[band] - want[band]).abs().max().item()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3sim")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)

    torch.manual_seed(0)
    x = torch.randn(D, D, dtype=torch.float32) * 0.1
    w_gate = torch.randn(EXPERTS * D, D, dtype=torch.float32) * 0.1
    w_up = torch.randn(EXPERTS * D, D, dtype=torch.float32) * 0.1
    w_down = torch.randn(EXPERTS * D, D, dtype=torch.float32) * 0.1

    # Experts 0 and 2 are routed; 1 and 3 are not, so their tasks are retired
    # inline at the dispatch point and their bands stay zero.
    _run_and_check(x, w_gate, w_up, w_down, torch.tensor([[1], [0], [1], [0]], dtype=torch.int32), cfg)

    # The same shapes hit the JIT cache, so this reuses the compiled program and
    # the recorded graph. Inverting the routing is what shows the predicate
    # operand is re-read on replay: a recording that froze it would still run
    # experts 0 and 2 here.
    _run_and_check(x, w_gate, w_up, w_down, torch.tensor([[0], [1], [0], [1]], dtype=torch.int32), cfg)

    print("OK")


if __name__ == "__main__":
    main()
