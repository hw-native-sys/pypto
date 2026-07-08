# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a5 cost-model VALIDATION harness — "does the model pick the simulator's best tile?".

For each shape it enumerates EVERY legal (m, n, k, stationarity) tile candidate, scores each
with the (a5-calibrated) model, ranks them, and marks the chooser's actual pick. The device
agent then measures each candidate's FULL-matmul a5-sim wall (needs the multi-tile wall
tooling fixed) and checks that the model's #1 is the simulator's #1 -- the argmin-match
validation (stronger than the single-tile ranking correlation). It also prints the pick so
the agent can measure the EMITTED matmul's wall vs the model prediction and the best tile.

Reuses the validated model oracle (`_wall_key` etc.) from the chooser UT, so scoring is
exactly the shipped C++ model. Run: PYTHONPATH=<pypto>/python python device_scripts/a5_validate_picks.py
"""

import os
import sys

TESTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "ut", "ir", "transforms")
sys.path.insert(0, TESTDIR)

import test_l0_tile_chooser as O  # noqa: E402  -- reuse the validated model oracle
from pypto.pypto_core.passes import l0_tile_chooser as L  # noqa: E402

# a5 (Ascend950) calibrated constants (mirror of Backend950::GetL0CostModel()).
A5 = dict(
    bw_a=206.3,
    bw_b=223.8,
    bw_drain=30.0,
    drain_fixed=343.4,
    drain_row=4.59,
    drain_penalty=0.26,
    mad_head=25,
    mad_fp32=8,
)
TOPK = 8


def a5_cfg(M, K, N, bytes_ab):
    c = O._default_config(M, N, K)
    c.l0a_bytes = c.l0b_bytes = 64 * 1024
    c.l0c_bytes = 256 * 1024  # a5
    c.bytes_a = c.bytes_b = bytes_ab
    c.bytes_c = 4
    c.allow_a_stationary = c.allow_b_stationary = True
    c.allow_padding = c.allow_k_boundary = True
    c.bw_a, c.bw_b, c.bw_drain = A5["bw_a"], A5["bw_b"], A5["bw_drain"]
    c.drain_fixed_cycles, c.drain_row_cycles, c.drain_penalty_cycles = (
        A5["drain_fixed"],
        A5["drain_row"],
        A5["drain_penalty"],
    )
    c.drain_c0_bytes = c.mad_k_fractal_bytes = 32
    c.mad_head, c.mad_fp32_passes = A5["mad_head"], A5["mad_fp32"]
    return c


def candidates(cfg):
    """Every legal (m, n, k, stat) with its model wall, ascending. dbc=0 (default planner)."""
    out = []
    c0 = cfg.l0c_bytes // cfg.bytes_c
    for stat in (O._OS, O._AS, O._BS):
        dba, dbb = O._derive_db(stat)
        a0 = cfg.l0a_bytes // (cfg.bytes_a * (2 if dba else 1))
        b0 = cfg.l0b_bytes // (cfg.bytes_b * (2 if dbb else 1))
        m = cfg.min_m
        while m <= cfg.M and m * cfg.min_n <= c0:
            n = cfg.min_n
            while n <= min(cfg.N, c0 // m):
                # A/B-stationary requires full-K (held operand pinned); OS allows split-K.
                for k in O._legal_ks(m, n, cfg, a0, b0):
                    if stat != O._OS and k != cfg.K:
                        continue
                    out.append((O._wall_key(m, n, k, cfg, stat, False)[0], m, n, k, stat))
                n += cfg.align_n
            m += cfg.align_m
    out.sort()
    return out


shapes = [
    ("bf16", 2, (512, 512, 512)),
    ("bf16", 2, (768, 512, 320)),
    ("fp32", 4, (512, 512, 256)),
    ("fp32", 4, (320, 512, 320)),
    ("fp32", 4, (544, 512, 512)),
]
for dn, ba, (M, K, N) in shapes:
    cfg = a5_cfg(M, K, N, ba)
    cands = candidates(cfg)
    r = L.choose_l0_tile(cfg)
    pick = (r.m, r.n, r.k)
    print(f"\n=== {dn} {M}x{K}x{N} | {len(cands)} legal candidates | chooser pick ({r.m},{r.n},{r.k}) ===")
    print(f"  {'rank':>4} {'m,n,k':>14} {'stat':>4} {'model_wall':>11}   (agent: measure a5-sim wall)")
    for i, (w, m, n, k, st) in enumerate(cands[:TOPK]):
        mark = "  <-- CHOOSER PICK" if (m, n, k) == pick else ""
        print(f"  {i + 1:>4} {f'{m},{n},{k}':>14} {st:>4} {w:>11}{mark}")
