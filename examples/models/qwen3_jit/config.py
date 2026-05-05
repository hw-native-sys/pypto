# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shape and tiling constants for Qwen3-32B single-layer decode.

Centralised so every kernel file imports the same values. Mirrors the
constants from the upstream ``qwen3_32b_decode.py`` reference."""

# ─── Model dimensions ───
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
INTERMEDIATE = 25600
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
CACHE_ROWS = BATCH * NUM_KV_HEADS * MAX_SEQ
HALF_DIM = HEAD_DIM // 2
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
ATTN_SCALE = 1.0 / (HEAD_DIM**0.5)
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# ─── Scope 1 tiles ───
RMSNORM_K_CHUNK = 512
Q_OUT_CHUNK = 256
Q_PROJ_K_CHUNK = 128
KV_OUT_CHUNK = 256
KV_PROJ_K_CHUNK = 128

# ─── Scope 2 tiles ───
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE

# ─── Scope 3 tiles ───
K_CHUNK = 128
OUT_PROJ_K_CHUNK = 128
MLP_OUT_CHUNK = 256
DOWN_N_CHUNK = 256
DOWN_K_CHUNK = 128
