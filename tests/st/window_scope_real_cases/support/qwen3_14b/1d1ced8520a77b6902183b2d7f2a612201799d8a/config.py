# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""Qwen3-14B model and decode-kernel configuration."""

import pypto.language as pl

# Dynamic dimensions used by the JIT/program signatures.
USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")
LAYER_DYN = pl.dynamic("LAYER_DYN")
LAYER_HIDDEN_ROWS_DYN = pl.dynamic("LAYER_HIDDEN_ROWS_DYN")
LAYER_INTER_ROWS_DYN = pl.dynamic("LAYER_INTER_ROWS_DYN")

# Model shape.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 17408
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
VOCAB = 152064
NUM_LAYERS = 40

# Numeric constants.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = 1.0 / (HEAD_DIM**0.5)
HALF_DIM = HEAD_DIM // 2

# Scope 1 tiling constants.
INPUT_PROJ_K_CHUNK = 256
KV_PROJ_K_CHUNK = 256
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 256
BATCH_TILE = 16

# Scope 2 tiling constants.
# Q_HEAD_BATCH = q_per_kv = 40/8 = 5 for the official Qwen3-14B config.
# The Q_HEAD_BATCH-row trim runs inside the fa_fused mixed cube+vec root.
# The lane-1 dual-AIV no-op replay rewrites the [0:5] subview to
# valid_row=0; this is accepted by ptoas >= 0.43 (hw-native-sys/PTOAS#708)
# and lowered as a no-op via pto-isa's GetValidRow/GetValidCol valid==0
# support (hw-native-sys/pto-isa#151).
Q_HEAD_BATCH = 5
Q_HEAD_PAD = 16
# SEQ_TILE = 128 keeps each K/V tile at 32 KB (BLOCK_SIZE * HEAD_DIM * BF16),
# letting the cube L0B fit two tiles simultaneously (64 KB platform limit).
# This is required by the fa_fused mixed root in decode_layer.py; raising
# SEQ_TILE to 256 makes a single tile fill L0B and breaks the cube/vec fuse.
SEQ_TILE = 128
SB_BATCH = 128
BLOCK_SIZE = SEQ_TILE

# Scope 3 tiling constants.
K_CHUNK = 256
OUT_PROJ_K_CHUNK = 256
OUT_PROJ_N_CHUNK = 256
MLP_OUT_CHUNK = 256
# SPMD grouping for the MLP gate/up/silu stages: MLP_SPMD_INNER output
# blocks are bundled per parallel chunk and dispatched across SPMD lanes.
MLP_SPMD_INNER = 2
MLP_GROUP_CHUNK = MLP_SPMD_INNER * MLP_OUT_CHUNK
DOWN_MLP_CHUNK = 256
DOWN_OUT_CHUNK = 256
FINAL_RMS_K_CHUNK = 128
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 64

# Decode grouping.
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
# fa_fused groups attention work by (KV head, Q-head batch). qk_norm and
# rope_kv_cache currently loop over NUM_KV_HEADS only (one Q-head batch per
# KV head), so the Q heads per KV head must equal Q_HEAD_BATCH exactly --
# supporting Q_GROUPS > 1 would require also iterating the inner Q groups
# in those two regions.
assert Q_PER_KV == Q_HEAD_BATCH, (
    f"Q_PER_KV ({Q_PER_KV}) must equal Q_HEAD_BATCH ({Q_HEAD_BATCH}) "
    f"(qk_norm / rope_kv_cache assume one Q group per KV head)"
)
# Q_HEAD_PAD is the padded Q row count fa_fused operates on. fa_fused does
# set_validshape(scores, Q_HEAD_PAD // 2, ...) on the vec-side scores tile
# and then trims oi/li to Q_HEAD_BATCH rows, so the *half* must (a) be even
# (an odd valid_row without an explicit operand hits pypto#1031) and
# (b) be >= Q_HEAD_BATCH so the trim is fully covered. Both reduce to
# Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH. (Q_HEAD_PAD = 16
# here -> //2 = 8 >= 5; fa_fused runs SplitMode=None / dual-AIV no-op
# replay, not row halving — see the module docstring.)
assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH, (
    f"Q_HEAD_PAD ({Q_HEAD_PAD}) must be a multiple of 4 with "
    f"Q_HEAD_PAD // 2 ({Q_HEAD_PAD // 2}) >= Q_HEAD_BATCH ({Q_HEAD_BATCH})"
)
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
# fa_fused dispatches via pl.spmd(TOTAL_Q_GROUPS // 2) with an inner
# pl.pipeline(2, stage=2) over the Q-group pair; that requires an even count.
assert TOTAL_Q_GROUPS % 2 == 0, f"TOTAL_Q_GROUPS ({TOTAL_Q_GROUPS}) must be even (fa_fused pairs Q groups)"
MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
