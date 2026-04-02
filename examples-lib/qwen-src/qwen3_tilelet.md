# Qwen3 Decode — TILELET / TILE Size Reference (Dual-Maximised)

This document catalogues every computation tile used in
[`qwen3_32b_decode_tilelet.py`](qwen3_32b_decode_tilelet.py) and verifies each
against the hardware constraints.  Constants are chosen to **simultaneously
maximise both vector TILELET and cube TILE utilisation**: reduction-direction
vector tiles are `[4, 128] FP32 = 2 KB MAX`, and all matmul weight tiles are
`[128, 64] BF16 = 16 KB MAX`.

| Unit | Budget | Applies to |
|---|---|---|
| **TILELET** | **2 KB** (2048 B) | Vector operations: `add`, `mul`, `sub`, `exp`, `rsqrt`, `cast`, `row_sum`, `row_max`, `row_expand_mul`, `col_expand_mul`, `fillpad`, `recip`, `neg`, … |
| **TILE** | **16 KB** (16384 B) | CUBE operations: `matmul` |

Element sizes: **BF16 = 2 B**, **FP32 = 4 B**, **INT32 = 4 B**.

---

## Constants

| Constant | Value | Rationale |
|---|---|---|
| `BATCH` | 16 | Model config |
| `HIDDEN` | 5120 | Qwen3-32B |
| `NUM_HEADS` | 64 | Qwen3-32B |
| `NUM_KV_HEADS` | 8 | GQA group count |
| `HEAD_DIM` | 128 | `HIDDEN / NUM_HEADS` |
| `KV_HIDDEN` | 1024 | `NUM_KV_HEADS × HEAD_DIM` |
| `INTERMEDIATE` | 25600 | MLP hidden |
| `K_CHUNK` | **128** | `[BATCH_TILE, K_CHUNK] FP32 = [4,128]×4 = 2 KB MAX`; `[K_CHUNK, out] BF16 = [128,64]×2 = 16 KB MAX` |
| `Q_OUT_CHUNK` | **64** | `[K_CHUNK, Q_OUT_CHUNK] BF16 = [128,64]×2 = 16 KB MAX` |
| `KV_OUT_CHUNK` | **64** | `[K_CHUNK, KV_OUT_CHUNK] BF16 = [128,64]×2 = 16 KB MAX` |
| `SEQ_TILE` | **64** | `[SEQ_TILE, HEAD_DIM] BF16 = [64,128]×2 = 16 KB MAX` |
| `MLP_OUT_CHUNK` | **64** | `[K_CHUNK, MLP_OUT_CHUNK] BF16 = [128,64]×2 = 16 KB MAX` |
| `BATCH_TILE` | **4** | On-chip buffer constraint: `[BATCH_TILE, HIDDEN] FP32` must fit with double-buffering in 248 KB Vec |
| `Q_HEAD_BATCH` | **4** | `[4, HEAD_DIM] FP32 = [4,128]×4 = 2 KB MAX` for attention |
| `HIDDEN_BLOCKS` | 40 | `HIDDEN / K_CHUNK` |
| `Q_OUT_BLOCKS` | 80 | `HIDDEN / Q_OUT_CHUNK` |
| `KV_OUT_BLOCKS` | 16 | `KV_HIDDEN / KV_OUT_CHUNK` |
| `MLP_OUT_BLOCKS` | 400 | `INTERMEDIATE / MLP_OUT_CHUNK` |
| `Q_GROUPS` | 2 | `Q_PER_KV / Q_HEAD_BATCH = 8 / 4` |
| `TOTAL_Q_GROUPS` | 16 | `NUM_KV_HEADS × Q_GROUPS` |
| `ATTN_INIT_CHUNK` | 512 | `Q_HEAD_BATCH × HEAD_DIM` → `[1,512] FP32 = 2 KB MAX` |

### Binding-Constraint Analysis

The dual-maximisation hinges on `BATCH_TILE=4` and `K_CHUNK=128`:

| Constraint | Formula | Result |
|---|---|---|
| `BATCH_TILE × K_CHUNK × 4 ≤ 2048` (vector) | 4 × 128 × 4 = **2048** | = 2 KB **MAX** |
| `K_CHUNK × Q_OUT_CHUNK × 2 ≤ 16384` (cube) | 128 × 64 × 2 = **16384** | = 16 KB **MAX** |
| `K_CHUNK × KV_OUT_CHUNK × 2 ≤ 16384` (cube) | 128 × 64 × 2 = **16384** | = 16 KB **MAX** |
| `K_CHUNK × MLP_OUT_CHUNK × 2 ≤ 16384` (cube) | 128 × 64 × 2 = **16384** | = 16 KB **MAX** |
| `MLP_OUT_CHUNK × K_CHUNK × 2 ≤ 16384` (down) | 64 × 128 × 2 = **16384** | = 16 KB **MAX** |
| `SEQ_TILE × HEAD_DIM × 2 ≤ 16384` (attn) | 64 × 128 × 2 = **16384** | = 16 KB **MAX** |
| `Q_HEAD_BATCH × HEAD_DIM × 4 ≤ 2048` (attn vec) | 4 × 128 × 4 = **2048** | = 2 KB **MAX** |
| `NUM_KV_HEADS × (HEAD_DIM/2) × 4 ≤ 2048` (K RoPE) | 8 × 64 × 4 = **2048** | = 2 KB **MAX** |
| `BATCH_TILE × HIDDEN × 4 ≤ 248 KB / 2` (on-chip) | 4 × 5120 × 4 = 81920 | = 80 KB ✓ |

Output-direction accumulators `[BATCH_TILE, OUT_CHUNK] = [4, 64] FP32 = 1 KB` (50%).
This is the maximum achievable: `[4, 128] FP32` would require `[128, 128] BF16 = 32 KB` cube tiles (exceeds 16 KB).

---

## Scope 1 — Input RMSNorm + Q/K/V Projection

### 1.1 RMSNorm: Squared-Sum Accumulation

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.cast(…, FP32)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.mul(x_chunk, x_chunk)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.row_sum(…)` | [4, 128]→[4, 1] | FP32 | **2048** | vector | **100%** |
| `pl.add(partial_sq, …)` | [4, 1] | FP32 | 16 | vector | 1% |

### 1.2 Q Projection

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(q_acc, 0.0)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(x_chunk, FP32)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.row_expand_mul(x, inv_rms)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.col_expand_mul(…, gamma)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.cast(normed, BF16)` | [4, 128] | BF16 | 1024 | vector | 50% |
| **`pl.matmul(normed_bf16, wq_chunk)`** | **A=[4,128] B=[128,64]** | **BF16** | **A=1024 B=16384** | **cube** | **100%** |
| `pl.add(q_acc, …)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(q_acc, BF16)` | [4, 64] | BF16 | 512 | vector | 25% |

### 1.3 K/V Projection

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(k_acc/v_acc, 0.0)` | [4, 64] | FP32 | 1024 | vector | 50% |
| (RMSNorm tiles — same as §1.2) | [4, 128] | FP32 | **2048** | vector | **100%** |
| **`pl.matmul(normed_bf16, wk/wv_chunk)`** | **A=[4,128] B=[128,64]** | **BF16** | **A=1024 B=16384** | **cube** | **100%** |
| `pl.add(k_acc/v_acc, …)` | [4, 64] | FP32 | 1024 | vector | 50% |

---

## Scope 2 — RoPE + Cache Update + Decode Attention

### 2.1 K RoPE — Batched (all 8 KV heads)

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.cast(…, FP32)` (per-head gather) | [1, 128] | FP32 | 512 | vector | 25% |
| `pl.col_expand_mul(k_lo, cos_lo)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.col_expand_mul(k_hi, sin_lo)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.sub(…)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.col_expand_mul(k_hi, cos_hi)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.col_expand_mul(k_lo, sin_hi)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.add(…)` | [8, 64] | FP32 | **2048** | vector | **100%** |
| `pl.cast(k_rot_row, BF16)` (per-head write) | [1, 128] | BF16 | 256 | vector | 13% |

### 2.2 Attention Row Zero-Init

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(z, 0.0)` | [1, 512] | FP32 | **2048** | vector | **100%** |

### 2.3 Q RoPE — Batched (Q_HEAD_BATCH=4 heads)

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.cast(…, FP32)` (per-head gather) | [1, 128] | FP32 | 512 | vector | 25% |
| `pl.col_expand_mul(q_lo, cos_lo)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.sub(…)` / `pl.add(…)` (RoPE) | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(q_rot, BF16)` | [4, 128] | FP32→BF16 | **2048**→1024 | vector | **100%** |

### 2.4 Q × K^T Attention Scores

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| **`pl.matmul(q_rot_bf16, k_tile, b_trans=True)`** | **A=[4,128] B=[64,128]^T** | **BF16** | **A=1024 B=16384** | **cube** | **100%** |
| `pl.mul(scores, ATTN_SCALE)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.fillpad(…)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.row_max(…)` | [4, 64]→[4, 1] | FP32 | 1024 | vector | 50% |
| `pl.exp(pl.row_expand_sub(…))` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.row_sum(…)` | [4, 64]→[4, 1] | FP32 | 1024 | vector | 50% |
| `pl.cast(exp_scores, BF16)` | [4, 64] | BF16 | 512 | vector | 25% |
| **`pl.matmul(exp_bf16, v_tile, out_dtype=FP32)`** | **A=[4,64] B=[64,128]** | **BF16** | **A=512 B=16384** | **cube** | **100%** |

### 2.5 Online Softmax Update

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.maximum/exp/sub(mi, …)` | [4, 1] | FP32 | 16 | vector | 1% |
| `pl.mul(alpha/beta, li/cur_li)` | [4, 1] | FP32 | 16 | vector | 1% |
| `pl.row_expand_mul(oi, alpha)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.row_expand_mul(oi_tmp, beta)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.add(…)` (oi update) | [4, 128] | FP32 | **2048** | vector | **100%** |

### 2.6 Context Writeback

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.row_expand_div(oi, li)` | [4, 128] | FP32 | **2048** | vector | **100%** |

---

## Scope 3 — Output Projection + Residual + Post-RMSNorm + MLP + Residual

### 3.1 Output Projection

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(o_acc, 0.0)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(attn_out_chunk, BF16)` | [4, 128] | BF16 | 1024 | vector | 50% |
| **`pl.matmul(a_chunk, w_chunk)`** | **A=[4,128] B=[128,64]** | **BF16** | **A=1024 B=16384** | **cube** | **100%** |
| `pl.add(o_acc, …)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(resid, FP32)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.add(o_acc, resid)` | [4, 64] | FP32 | 1024 | vector | 50% |

### 3.2 Post-RMSNorm Squared-Sum

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(x_chunk, x_chunk)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.row_sum(…)` | [4, 128]→[4, 1] | FP32 | **2048** | vector | **100%** |

### 3.3 Down-Projection Tile Zero-Init

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(z, 0.0)` | [4, 128] | FP32 | **2048** | vector | **100%** |

### 3.4 Post-RMSNorm Application

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.row_expand_mul(x, inv_rms)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.col_expand_mul(…, gamma)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.cast(normed, BF16)` | [4, 128] | BF16 | 1024 | vector | 50% |

### 3.5 MLP Gate/Up Projection

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.mul(gate_acc/up_acc, 0.0)` | [4, 64] | FP32 | 1024 | vector | 50% |
| **`pl.matmul(post_chunk, wg/wu)`** | **A=[4,128] B=[128,64]** | **BF16** | **A=1024 B=16384** | **cube** | **100%** |
| `pl.add(gate_acc/up_acc, …)` | [4, 64] | FP32 | 1024 | vector | 50% |

### 3.6 SiLU Activation

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.neg(gate_acc)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.exp(…)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.add(…, 1.0)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.recip(…)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.mul(gate, sigmoid)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.mul(…, up_acc)` | [4, 64] | FP32 | 1024 | vector | 50% |
| `pl.cast(mlp_chunk, BF16)` | [4, 64] | BF16 | 512 | vector | 25% |

### 3.7 Down Projection

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| **`pl.matmul(mlp_bf16, w_down_chunk)`** | **A=[4,64] B=[64,128]** | **BF16** | **A=512 B=16384** | **cube** | **100%** |
| `pl.add(down_prev, …)` | [4, 128] | FP32 | **2048** | vector | **100%** |

### 3.8 Final Residual Add

| Operation | Tile Shape | DType | Bytes | Type | Util |
|---|---|---|---|---|---|
| `pl.add(down, resid)` | [4, 128] | FP32 | **2048** | vector | **100%** |
| `pl.cast(…, BF16)` | [4, 128] | BF16 | 1024 | vector | 50% |

---

## Summary of All CUBE (matmul) Operand Sizes

| Location | A shape | A bytes | B shape | B bytes | Cube util | ✓ |
|---|---|---|---|---|---|---|
| Q projection | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| K projection | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| V projection | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| Attn Q×K^T | [4, 128] BF16 | 1 KB | [64, 128] BF16 | **16 KB** | **100%** | ✓ |
| Attn score×V | [4, 64] BF16 | 512 B | [64, 128] BF16 | **16 KB** | **100%** | ✓ |
| O projection | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| MLP gate | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| MLP up | [4, 128] BF16 | 1 KB | [128, 64] BF16 | **16 KB** | **100%** | ✓ |
| MLP down | [4, 64] BF16 | 512 B | [64, 128] BF16 | **16 KB** | **100%** | ✓ |

**All 9 matmul weight tiles are at 16 KB = 100% cube utilisation.**

## Summary of Vector TILELET Utilisation

| Tile Shape | DType | Bytes | Util | Used In |
|---|---|---|---|---|
| **[4, 128]** | **FP32** | **2048 = 2 KB** | **100%** | RMSNorm (cast, row/col_expand_mul, sq_sum), down proj add, final residual, attn oi/ctx |
| [4, 64] | FP32 | 1024 = 1 KB | 50% | Projection accumulators, SiLU, O proj residual, attn scores/softmax |
| **[8, 64]** | **FP32** | **2048 = 2 KB** | **100%** | K RoPE halves (batched 8 KV heads) |
| **[1, 512]** | **FP32** | **2048 = 2 KB** | **100%** | attn_row zero-init |
| [4, 128] / [4, 64] | BF16 | 1024 / 512 | 50% / 25% | Cast-to-BF16 results |
| [1, 128] | FP32/BF16 | 512 / 256 | 25% / 13% | Per-head gather/write in K/Q assembly |
| [4, 1] / [8, 1] | FP32 | 16 / 32 | 1% / 2% | Per-row scalars (inherent) |

---

## Design Rationale: Why BATCH_TILE=4, K_CHUNK=128

The choice of `BATCH_TILE=4` is driven by the **on-chip Vec buffer limit** (248 KB):

- Scope 3 requires `[BATCH_TILE, HIDDEN]` FP32 buffers (`resid1_tile`, `down_proj_tile`) that
  need double-buffering during assembly loops.
- With `BATCH_TILE=4`: each buffer is 80 KB, double-buffered = 160 KB, leaving 88 KB for
  `post_norm_tile` (40 KB BF16) and temporaries. Total ≈ 245.6 KB (99.0%).
- With `BATCH_TILE=8`: each buffer is 160 KB, double-buffered = 320 KB — **exceeds 248 KB**.

Given `BATCH_TILE=4`, setting `K_CHUNK=128` simultaneously maximises:
- Vector: `[4, 128] × 4 = 2048 = 2 KB MAX`
- Cube: `[128, 64] × 2 = 16384 = 16 KB MAX`

Output-direction accumulators `[4, 64] FP32 = 1 KB` (50%) are the theoretical ceiling —
increasing them to `[4, 128]` would require `[128, 128] BF16 = 32 KB` cube weights (exceeds 16 KB).

All vector operands ≤ **2 KB**. All cube operands ≤ **16 KB**. ✓
