# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test for issue #1580: orchestration phi-node naming swap when a
scalar carry and tensor carries are mixed in a ``pl.parallel`` loop carry.

When a ``pl.Scalar`` defined inside a ``pl.spmd`` body is reused as a name in a
later ``pl.parallel`` + ``pl.at`` + ``pl.range`` scope writing multiple output
tensors, ``ConvertToSSA`` promotes the scalar into the parallel loop's
``init_values`` tuple, *mixed with* the tensor carries — and a passed-through
tile carry trails the real outputs. The orchestration codegen used to map each
return-tuple element to the callee's Out params by positional tail-alignment,
which assumed all non-output elements were a leading prefix. The interleaved
scalar (leading) + passthrough tile (trailing) rotated the output names by one,
emitting a phi block that referenced an undeclared ``out_b__rv_v*`` and aliased
``out_c``/``tile`` to the wrong values.

The fix traces each return-tuple element back to the callee arg it aliases, so
the name<->value pairing is position-independent. This test asserts the
generated orchestration C++ is self-consistent: every ``__rv_v*`` rvalue is
declared, and each tensor loop-carry phi aliases its own carry (same base name
on both sides), not a neighbour's.
"""

import re
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402

B = 64
DIM = 512
TILE = 8

DUMP_DIR = Path(__file__).resolve().parents[4] / "build_output" / "mixed_scalar_tensor_carry_repro"


@pl.jit
def mixed_scalar_tensor_carry_repro(
    x: pl.Tensor[[B, DIM], pl.BF16],
    out_a: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
    out_b: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
    out_c: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
):
    # Scope A: a pl.spmd body defines "global_c_idx".
    for idx in pl.spmd(B, name_hint="scope_a"):
        global_c_idx = idx
        tile = pl.cast(x[global_c_idx : global_c_idx + 1, :], target_type=pl.FP32)
        out_a = pl.assemble(out_a, tile, [global_c_idx, 0])

    # Scope B: reuses "global_c_idx" + "tile" with multiple tensor carries, so
    # the parallel loop carry becomes (scalar, out_b, out_c, tile).
    for batch_base_idx in pl.parallel(0, B // TILE):
        batch_base = batch_base_idx * TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scope_b"):
            for inner in pl.range(TILE):
                global_c_idx = batch_base + inner
                tile = pl.cast(x[global_c_idx : global_c_idx + 1, :], target_type=pl.FP32)
                out_b[global_c_idx : global_c_idx + 1, :] = tile
                out_c[global_c_idx : global_c_idx + 1, :] = tile
    return out_a, out_b, out_c


# Matches a phi assignment / alias whose rvalue is an SSA-versioned variable,
# e.g. ``const Tensor& out_b__rv_v4 = out_b__rv_v2;`` or ``out_c__rv_v2 = out_c__rv_v4;``.
_ALIAS = re.compile(
    r"^\s*(?:const\s+Tensor&\s+|Tensor\s+|int64_t\s+)?"
    r"([A-Za-z_]\w*?)__rv_v\d+\s*=\s*([A-Za-z_]\w*?)__rv_v\d+\s*;\s*$"
)


def test_mixed_scalar_tensor_carry_phi_naming():
    """The orchestration C++ for a mixed scalar/tensor loop carry must have a
    self-consistent phi block: no undeclared ``__rv_v*`` rvalue, and no tensor
    carry aliased to a different carry's value (issue #1580)."""
    mixed_scalar_tensor_carry_repro._cache.clear()
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

    x = torch.zeros((B, DIM), dtype=torch.bfloat16)
    out_a = torch.zeros((B, DIM), dtype=torch.float32)
    out_b = torch.zeros((B, DIM), dtype=torch.float32)
    out_c = torch.zeros((B, DIM), dtype=torch.float32)

    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        dump_passes=False,
        save_kernels=True,
        save_kernels_dir=str(DUMP_DIR),
    )
    # codegen_only stops before runtime config assembly; that FileNotFoundError
    # is expected and unrelated to the orchestration C++ we assert on.
    try:
        mixed_scalar_tensor_carry_repro(x, out_a, out_b, out_c, config=cfg)
    except FileNotFoundError:
        pass

    orch = DUMP_DIR / "orchestration" / "mixed_scalar_tensor_carry_repro.cpp"
    if not orch.exists():
        # Fall back to whatever single orchestration .cpp was emitted.
        candidates = list((DUMP_DIR / "orchestration").glob("*.cpp"))
        assert candidates, f"no orchestration C++ emitted under {DUMP_DIR}/orchestration"
        orch = candidates[0]
    text = orch.read_text()

    # 1. Every ``__rv_v*`` rvalue must be declared somewhere as an lvalue.
    #    (The pre-fix bug emitted ``out_b__rv_v2 = out_b__rv_v4;`` where
    #    ``out_b__rv_v4`` was never declared.)
    declared = set()
    used = set()
    for line in text.splitlines():
        m = _ALIAS.match(line)
        if not m:
            continue
        # The full versioned token on each side.
        lhs, rhs = re.findall(r"[A-Za-z_]\w*?__rv_v\d+", line)[:2]
        declared.add(lhs)
        used.add(rhs)
    # Also treat any ``<decl> X__rv_vN = <literal/expr>`` as declaring X__rv_vN.
    for m in re.finditer(r"\b([A-Za-z_]\w*?__rv_v\d+)\s*=", text):
        declared.add(m.group(1))
    undeclared = sorted(u for u in used if u not in declared)
    assert not undeclared, (
        f"orchestration C++ references undeclared SSA carries {undeclared} in {orch}\n\n{text}"
    )

    # 2. No tensor carry phi may alias a *different* carry's value. The fixed
    #    output pairs each base name with itself (out_b<-out_b, out_c<-out_c,
    #    tile<-tile); the bug rotated them (out_c<-out_b, tile<-out_c).
    crossed = []
    for line in text.splitlines():
        m = _ALIAS.match(line)
        if not m:
            continue
        lhs_base, rhs_base = m.group(1), m.group(2)
        if lhs_base != rhs_base:
            crossed.append(line.strip())
    assert not crossed, (
        f"orchestration C++ aliases a loop carry to a different carry's value (issue #1580): "
        f"{crossed} in {orch}\n\n{text}"
    )

    # 3. A Tensor binding must not reference a scalar-typed variable. The scalar
    #    carry global_c_idx is an int64_t; emitting it as a tensor alias
    #    (``const Tensor& global_c_idx__rv_v4 = global_c_idx__rv_v2;``, where the
    #    rhs is int64_t) does not compile. This is the scalar half of #1580 — the
    #    tensor-rotation fix alone left the scalar carry mis-typed.
    scalar_decl = re.compile(
        r"\b(?:int64_t|int32_t|int16_t|int8_t|uint64_t|uint32_t|int|float|double|bool)\s+"
        r"([A-Za-z_]\w*?__rv_v\d+)\s*="
    )
    scalars = set(scalar_decl.findall(text))
    tensor_bind = re.compile(
        r"\b(?:const\s+Tensor&|Tensor)\s+[A-Za-z_]\w*?__rv_v\d+\s*=\s*([A-Za-z_]\w*?__rv_v\d+)"
    )
    mistyped = [rhs for rhs in tensor_bind.findall(text) if rhs in scalars]
    assert not mistyped, (
        f"orchestration C++ binds a Tensor to a scalar-typed carry {sorted(set(mistyped))} "
        f"(issue #1580 scalar carry) in {orch}\n\n{text}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
