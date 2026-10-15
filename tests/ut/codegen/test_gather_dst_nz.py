# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Tests for the KN cast NZ direct-write path (``tile.gather`` dst_nz).

The NZ direct-write path has three pypto pieces:

* ``tile.gather`` ``dst_nz`` kwarg — the deduce relabels the gather output
  view as NZ (``col_major`` + ``row_major`` fractal) and clears
  stride/start_offset. This is a CONTRACT with the device-side gather
  implementation: it must store the gathered lanes directly at NZ-fractal
  addresses.
* ``PYPTO_FP4_CAST_NZ_DIRECT=1`` — LegalizeTileCast emits the FP4->FP8 LUT
  cast gather with ``dst_nz`` so the matmul_mx rhs V2C boundary receives an
  already-NZ tile, and the UINT16 -> FP8E4M3FN pair-view reinterpret
  doubles the last axis on the NZ tile (the LUT lanes are byte pairs).
* The ExpandMixedKernel identity skip — when the push source view already
  equals the required fractal view on every field (blayout, slayout,
  compact, pad, fractal, valid_shape, stride, start_offset), the adapt move
  is a true identity and is elided (TGATHER -> TPUSH with no vec->vec
  TMOV). The stride/start_offset/valid_shape/fractal comparisons are the R2
  hardening: for a tile.move-driven boundary the required view is derived
  from the move destination, and a source matching only the layout fields
  could differ in the remaining ones — skipping the adapt would then
  silently corrupt the push. BuildCrossCoreTransferView copies all of those
  fields from the source, which is why the intended dst_nz producer still
  compares equal and still skips.

The emission knob is read once per process (a C++ static in the pass), so
the on/off A/B below runs each side in a SUBPROCESS with the environment
variable set or removed there.

Everything is device-free: ``RunConfig(codegen_only=True)`` lowers through
PTO codegen and the emitted ``.pto`` text carries the tile views.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import os
import re
import subprocess
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import backend
from pypto.pypto_core import passes as _core_passes
from pypto.runtime import RunConfig

M, K, N = 64, 64, 64  # matmul_mx requires physical K divisible by 64

# A vec->vec adapt move: pto.tmov whose ins AND outs tiles both live in vec.
# With the knob off this includes the V2C layout-adapt TMOV the NZ
# direct-write path elides; with the knob on the count must drop to zero.
_VEC2VEC_TMOV = re.compile(r"pto\.tmov ins\([^)]*loc=vec[^)]*\) outs\([^)]*loc=vec")


@pytest.fixture(autouse=True)
def _clean_backend_and_knob():
    """No backend pinned behind us; the knob is never read in-process.

    Reading the knob here would cache it for the whole pytest run (a C++
    static in the pass), so the on/off A/B lives in child processes.
    """
    os.environ.pop("PYPTO_FP4_CAST_NZ_DIRECT", None)
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


@pl.jit
def fp4_rhs_cast_push(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N // 2], pl.FP4E2M1X2],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """FP4 rhs cast crossing to the cube — the KN cast push shape.

    ``pl.cast`` runs on the vector side and its result crosses to the cube
    as a V2C boundary; the FP4->FP8 cast lowers to the kernel-resident LUT
    gather, so this kernel exercises gather emission, the pair-view
    reinterpret and the boundary adapt-move decision in one shot. The FP4
    tensor declares CARRIER extents (one byte = two nibbles), so its type
    and load use ``N // 2`` along the packed axis while the cast output and
    matmul consume the logical ``[K, N]``.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_rhs_cast_push_tile"):
        at = pl.load(a, [0, 0], [M, K])
        ats = pl.load(a_scale, [0, 0], [M, K // 32])
        bt8 = pl.cast(pl.load(b, [0, 0], [K, N // 2]), pl.FP8E4M3FN)
        bts = pl.load(b_scale, [0, 0], [K // 32, N])
        acc = pl.matmul_mx(at, ats, bt8, bts)
        c = pl.store(acc, [0, 0], c)
    return c


@pl.jit
def fp4_cast_store_only(
    b: pl.Tensor[[K, N // 2], pl.FP4E2M1X2],
    c: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    """FP4->FP8 cast whose consumer is a plain store (NOT a matmul rhs)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_cast_store_only_tile"):
        bt8 = pl.cast(pl.load(b, [0, 0], [K, N // 2]), pl.FP8E4M3FN)
        c = pl.store(bt8, [0, 0], c)
    return c


def _child_compile(out_dir: str, mode: str = "rhs") -> int:
    """Child-process mode: compile the kernel to .pto under *out_dir*.

    Modes: "rhs" (default, the matmul-rhs kernel), "nonrhs" (the
    cast-to-store kernel), "ptoas" (the rhs kernel under the PTOAS memory
    planner).
    """
    kw = {}
    if mode == "ptoas":
        kw["memory_planner"] = _core_passes.MemoryPlanner.PTOAS
    cfg = RunConfig(platform="a5", codegen_only=True, save_kernels=True, save_kernels_dir=out_dir, **kw)
    if mode == "nonrhs":
        fp4_cast_store_only(
            torch.zeros(K, N // 2, dtype=torch.float4_e2m1fn_x2),
            torch.zeros(K, N, dtype=torch.float8_e4m3fn),
            config=cfg,
        )
        return 0
    fp4_rhs_cast_push(
        torch.zeros(M, K, dtype=torch.float8_e4m3fn),
        torch.zeros(M, K // 32, dtype=torch.float8_e8m0fnu),
        torch.zeros(K, N // 2, dtype=torch.float4_e2m1fn_x2),
        torch.zeros(K // 32, N, dtype=torch.float8_e8m0fnu),
        torch.zeros(M, N, dtype=torch.float32),
        config=cfg,
    )
    return 0


def _compile_in_child(knob: str, tmp_path: Path, mode: str = "rhs") -> str:
    """Compile *fp4_rhs_cast_push* in a subprocess with the knob on/off.

    Returns the concatenated ``.pto`` text the child emitted. The child is
    this file itself run as a script, so it imports the same lab/CI pypto
    the parent tests use (PYTHONPATH is inherited, including whatever
    sitecustomize-based tree pinning the harness set up).
    """
    out_dir = tmp_path / f"pto-{knob}-{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if knob == "on":
        env["PYPTO_FP4_CAST_NZ_DIRECT"] = "1"
    else:
        env.pop("PYPTO_FP4_CAST_NZ_DIRECT", None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", str(out_dir), mode],
        check=False,
        env=env,
        cwd=str(out_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"child compile failed (knob={knob}):\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    ptos = sorted(out_dir.rglob("*.pto"))
    assert ptos, f"child emitted no .pto under {out_dir}"
    return "\n".join(p.read_text() for p in ptos)


def test_knob_off_emits_row_major_gather_and_keeps_the_adapt_move(tmp_path):
    """Default path: the LUT gather out stays row-major and the boundary
    keeps its vec->vec layout-adapt TMOV (TGATHER -> TMOV -> TPUSH)."""
    pto = _compile_in_child("off", tmp_path)

    m = re.search(r"pto\.tgather ins\(.*\) outs\([^)]*\)", pto)
    assert m, f"no pto.tgather in emitted .pto:\n{pto[:2000]}"
    assert "blayout=col_major" not in m.group(0), m.group(0)
    assert "blayout=row_major, slayout=none_box" in m.group(0), m.group(0)
    # The cast result is NOT pre-arranged as NZ, so the boundary must adapt.
    assert _VEC2VEC_TMOV.search(pto), (
        "knob off must keep the vec->vec adapt move between the cast gather "
        "and the push (no NZ producer exists to justify skipping it)"
    )
    assert "pto.tpush_to_aic" in pto


def test_knob_on_emits_nz_gather_and_skips_the_adapt_move(tmp_path):
    """PYPTO_FP4_CAST_NZ_DIRECT=1: the gather out view is NZ (col_major +
    row_major fractal, stride/start_offset cleared by the deduce) and the
    identity-qualified boundary push consumes it directly — no vec->vec
    adapt TMOV. This also pins the R2 hardening: the skip must still fire
    with the extended field comparison (fractal/valid_shape/stride/
    start_offset) in place."""
    pto = _compile_in_child("on", tmp_path)

    m = re.search(r"pto\.tgather ins\(.*\) outs\([^)]*\)", pto)
    assert m, f"no pto.tgather in emitted .pto:\n{pto[:2000]}"
    assert "blayout=col_major, slayout=row_major" in m.group(0), m.group(0)
    # R2: no false negative — the intended dst_nz producer still skips.
    assert not _VEC2VEC_TMOV.search(pto), (
        "knob on must elide every vec->vec adapt move (the gather out is already the required fractal view)"
    )
    assert "pto.tpush_to_aic" in pto
    push = re.search(r"pto\.tpush_to_aic\([^)]*blayout=col_major[^)]*\)", pto)
    assert push, "the pushed rhs tile must carry the NZ view"


def test_knob_on_pair_view_reinterpret_doubles_the_last_axis(tmp_path):
    """The UINT16 -> FP8E4M3FN pair-view reinterpret on the NZ gather out
    doubles the last axis: the gather out is a [K, N/2] u16 tile (one lane
    per packed carrier byte = two FP4 codes), and the f8 view it relabels
    is the logical [K, N] FP8 tile (each u16 lane = two FP8 bytes) — both
    keeping the NZ fractal view."""
    pto = _compile_in_child("on", tmp_path)

    g = re.search(r"pto\.tgather ins\(.*\) outs\([^)]*\)", pto)
    assert g, f"no pto.tgather in emitted .pto:\n{pto[:2000]}"
    assert f"dtype=ui16, rows={K}, cols={N // 2}" in g.group(0), g.group(0)
    assert "blayout=col_major, slayout=row_major" in g.group(0), g.group(0)
    f8 = re.search(
        rf"dtype=f8E4M3FN, rows={K}, cols={N}, [^)]*blayout=col_major, slayout=row_major",
        pto,
    )
    assert f8, (
        f"expected an NZ f8 tile with cols={N} (pair-view doubling of the "
        f"{N // 2}-carrier axis) in:\n{pto[:2000]}"
    )


if __name__ == "__main__" and "--child" in sys.argv:
    raise SystemExit(
        _child_compile(
            sys.argv[sys.argv.index("--child") + 1],
            sys.argv[sys.argv.index("--child") + 2]
            if len(sys.argv) > sys.argv.index("--child") + 2
            else "rhs",
        )
    )


def test_knob_on_non_rhs_consumer_stays_row_major(tmp_path):
    """Codex P2 (PR#2987 round 2): PYPTO_FP4_CAST_NZ_DIRECT only relabels a
    gather NZ when the cast result provably feeds a matmul(_mx) RHS (the KN
    contract). A cast-to-store consumer keeps the default row-major gather."""
    pto = _compile_in_child("on", tmp_path, mode="nonrhs")

    m = re.search(r"pto\.tgather ins\(.*\) outs\([^)]*\)", pto)
    assert m, f"no pto.tgather in emitted .pto:\n{pto[:2000]}"
    assert "blayout=col_major" not in m.group(0), m.group(0)


def test_ptoas_planner_falls_back_off_the_lut_route(tmp_path):
    """Codex P1 (PR#2987 round 2): under MemoryPlanner.PtoAS the LUT route's
    zero-copy reinterprets would materialize as treshape, which PTOAS rejects
    for the FP4 source dtype — the cast must fall back to the bridged
    native-hop chain there (compiles, no tgather)."""
    pto = _compile_in_child("off", tmp_path, mode="ptoas")

    assert "pto.tgather" not in pto, (
        "the LUT gather route must be disabled under MemoryPlanner.PtoAS "
        "(its reinterpret chain cannot assemble):\n" + pto[:2000]
    )

    ctl = _compile_in_child("off", tmp_path, mode="rhs")
    assert "pto.tgather" in ctl, "control: the default planner keeps the LUT route"
