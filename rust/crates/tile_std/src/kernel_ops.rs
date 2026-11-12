// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Composite vector operations for NPU kernels.
//
// These functions compose the low-level ascend_* intrinsics with proper
// pipe_barrier(PIPE_ALL) synchronization between each step.
//
// All buffer parameters (UbBuf) are buffer IDs returned by __tile_buf_alloc().
// Unless stated otherwise, dst and src must NOT alias.
//
// IMPORTANT: For functions with 3+ buffer parameters, ALL buffers must be
// distinct (different UbBuf IDs). Aliased binary ops (Add, Mul, Sub where
// output == src1 or output == src2) cause non-deterministic crashes or wrong
// values on 310P/910B hardware. See each function's "Buffer constraint" doc
// for specifics. Scalar ops (Muls, Adds, Maxs, Mins) are safe in-place.

use crate::UbBuf;

/// ReLU activation: relu(x) = max(x, 0)
///
/// Maps to: AscendC::Maxs(dst, src, 0.0f, n)
#[inline(always)]
pub unsafe fn relu_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_maxs_f32(dst, src, 0.0f32, n);
    }
}

/// Leaky ReLU: leaky_relu(x) = max(x, 0) + alpha * min(x, 0)
///
/// Requires three buffers: dst, src, neg (workspace).
/// **src is destroyed** (used as workspace for intermediate Add).
///
/// **Buffer constraint:** dst, src, neg must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn leaky_relu_f32(
    dst: &mut UbBuf,
    src: &mut UbBuf,
    neg: &mut UbBuf,
    alpha: f32,
    n: u32,
) {
    unsafe {
        // dst = max(x, 0)
        crate::__tile_maxs_f32(*dst, *src, 0.0f32, n);
        crate::__tile_pipe_barrier();
        // neg = min(x, 0)
        crate::__tile_mins_f32(*neg, *src, 0.0f32, n);
        crate::__tile_pipe_barrier();
        // neg = alpha * min(x, 0)
        crate::__tile_muls_f32(*neg, *neg, alpha, n);
        crate::__tile_pipe_barrier();
        // src = max(x,0) + alpha*min(x,0) — ALL SEPARATE (src != dst != neg)
        crate::__tile_v_add_f32(*src, *dst, *neg, n);
        crate::__tile_pipe_barrier();
        // dst = result
        crate::__tile_muls_f32(*dst, *src, 1.0f32, n);
    }
}

/// Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
///
/// Result is written to dst. src is preserved.
#[inline(always)]
pub unsafe fn sigmoid_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        // dst = -x
        crate::__tile_muls_f32(dst, src, -1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = exp(-x)
        crate::__tile_v_exp_f32(dst, dst, n);
        crate::__tile_pipe_barrier();
        // dst = 1 + exp(-x)
        crate::__tile_adds_f32(dst, dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = 1 / (1 + exp(-x))
        crate::__tile_reciprocal_f32(dst, dst, n);
    }
}

/// Tanh activation: tanh(x) = 2 * sigmoid(2x) - 1
///
/// Result is written to dst. src is preserved.
#[inline(always)]
pub unsafe fn tanh_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        // dst = 2x
        crate::__tile_muls_f32(dst, src, 2.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = -2x
        crate::__tile_muls_f32(dst, dst, -1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = exp(-2x)
        crate::__tile_v_exp_f32(dst, dst, n);
        crate::__tile_pipe_barrier();
        // dst = 1 + exp(-2x)
        crate::__tile_adds_f32(dst, dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = sigmoid(2x)
        crate::__tile_reciprocal_f32(dst, dst, n);
        crate::__tile_pipe_barrier();
        // dst = 2 * sigmoid(2x)
        crate::__tile_muls_f32(dst, dst, 2.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = 2 * sigmoid(2x) - 1 = tanh(x)
        crate::__tile_adds_f32(dst, dst, -1.0f32, n);
    }
}

/// GELU activation (sigmoid approximation): gelu(x) = x * sigmoid(1.702 * x)
///
/// Requires three buffers: dst, src (input x, preserved), and tmp (workspace).
/// Uses direct computation instead of calling sigmoid_f32 to produce a 5-op
/// sequence that works reliably with bisheng's auto_sync on 310P.
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn gelu_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // tmp = -1.702 * x  (combined negate+scale in one Muls)
        crate::__tile_muls_f32(*tmp, *src, -1.702f32, n);
        crate::__tile_pipe_barrier();
        // tmp = exp(-1.702 * x)
        crate::__tile_v_exp_f32(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // tmp = 1 + exp(-1.702 * x)
        crate::__tile_adds_f32(*tmp, *tmp, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // tmp = 1 / (1 + exp(-1.702 * x)) = sigmoid(1.702 * x)
        crate::__tile_reciprocal_f32(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = x * sigmoid(1.702 * x)
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
    }
}

/// GELU activation (f16): gelu(x) = x * sigmoid(1.702 * x)
///
/// **Buffer constraint:** dst, src, tmp must all be distinct.
#[inline(always)]
pub unsafe fn gelu_f16(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        crate::__tile_muls_f16(*tmp, *src, -1.702f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_v_exp_f16(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f16(*tmp, *tmp, 1.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_reciprocal_f16(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        crate::__tile_v_mul_f16(*dst, *src, *tmp, n);
    }
}

/// Softplus activation: softplus(x) = ln(1 + exp(x))
///
/// Result is written to dst. src is preserved.
#[inline(always)]
pub unsafe fn softplus_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        // dst = exp(x)
        crate::__tile_v_exp_f32(dst, src, n);
        crate::__tile_pipe_barrier();
        // dst = 1 + exp(x)
        crate::__tile_adds_f32(dst, dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = ln(1 + exp(x))
        crate::__tile_ln_f32(dst, dst, n);
    }
}

/// Swish activation: swish(x) = x * sigmoid(x)
///
/// Requires three buffers: dst, src (input x, preserved), and tmp (workspace).
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn swish_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // tmp = sigmoid(x)
        sigmoid_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // dst = x * sigmoid(x)
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
    }
}

/// HardTanh (clamp): hardtanh(x) = clamp(x, min_val, max_val)
///
/// Result is written to dst. src is preserved.
#[inline(always)]
pub unsafe fn hardtanh_f32(dst: UbBuf, src: UbBuf, min_val: f32, max_val: f32, n: u32) {
    unsafe {
        // dst = max(x, min_val)
        crate::__tile_maxs_f32(dst, src, min_val, n);
        crate::__tile_pipe_barrier();
        // dst = min(dst, max_val) = clamp(x, min_val, max_val)
        crate::__tile_mins_f32(dst, dst, max_val, n);
    }
}

/// SELU activation: selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
/// where scale = 1.0507, alpha = 1.6733
///
/// Requires three buffers: dst, src, tmp (workspace).
/// **src is destroyed** (used as workspace for intermediate Add).
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn selu_f32(dst: &mut UbBuf, src: &mut UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        let alpha: f32 = 1.6733f32;
        let scale: f32 = 1.0507f32;

        // tmp = exp(x)
        crate::__tile_v_exp_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // tmp = exp(x) - 1
        crate::__tile_adds_f32(*tmp, *tmp, -1.0f32, n);
        crate::__tile_pipe_barrier();
        // tmp = alpha * (exp(x) - 1)
        crate::__tile_muls_f32(*tmp, *tmp, alpha, n);
        crate::__tile_pipe_barrier();
        // tmp = min(0, alpha*(exp(x)-1))
        crate::__tile_mins_f32(*tmp, *tmp, 0.0f32, n);
        crate::__tile_pipe_barrier();

        // dst = max(0, x)  — last read of src
        crate::__tile_maxs_f32(*dst, *src, 0.0f32, n);
        crate::__tile_pipe_barrier();

        // src = max(0,x) + min(0, alpha*(exp(x)-1)) — ALL SEPARATE (src != dst != tmp)
        crate::__tile_v_add_f32(*src, *dst, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = scale * result
        crate::__tile_muls_f32(*dst, *src, scale, n);
    }
}

/// Softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
///
/// Numerically stable implementation using max subtraction.
/// Requires three buffers: dst, src (used as workspace, NOT preserved), and work.
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn softmax_f32(dst: &mut UbBuf, src: &mut UbBuf, work: &mut UbBuf, n: u32) {
    unsafe {
        // Step 1: max_val = max(x)
        let max_val = crate::__tile_v_reduce_max_f32(*work, *src, *dst, n);
        crate::__tile_pipe_barrier();

        // Step 2: dst = x - max(x)
        crate::__tile_adds_f32(*dst, *src, -max_val, n);
        crate::__tile_pipe_barrier();

        // Step 3: dst = exp(x - max(x))
        crate::__tile_v_exp_f32(*dst, *dst, n);
        crate::__tile_pipe_barrier();

        // Save exp values into src (no longer needed) before reduce corrupts dst
        crate::__tile_muls_f32(*src, *dst, 1.0f32, n);
        crate::__tile_pipe_barrier();

        // Step 4: sum = reduce_sum(exp(x - max(x))) — dst may be corrupted, src is safe
        let sum = crate::__tile_v_reduce_sum_f32(*work, *src, *dst, n);
        crate::__tile_pipe_barrier();

        // Step 5: normalize from saved copy
        let inv_sum = 1.0f32 / sum;
        crate::__tile_muls_f32(*dst, *src, inv_sum, n);
    }
}

/// Layer normalization (without learnable parameters):
///   layernorm(x) = (x - mean) / sqrt(var + eps)
///
/// To apply learnable gamma/beta, use muls_f32/adds_f32 or mul_f32/add_f32
/// on the result afterward.
///
/// Requires three buffers: dst, src (preserved), and work (workspace for reductions).
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn layernorm_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32, eps: f32) {
    unsafe {
        // Step 1: mean = sum(x) / n
        let sum = crate::__tile_v_reduce_sum_f32(*work, *src, *dst, n);
        let mean = sum / (n as f32);
        crate::__tile_pipe_barrier();

        // Step 2: dst = x - mean
        crate::__tile_adds_f32(*dst, *src, -mean, n);
        crate::__tile_pipe_barrier();

        // Step 3: work = (x - mean)^2
        crate::__tile_v_mul_f32(*work, *dst, *dst, n);
        crate::__tile_pipe_barrier();

        // Step 4: var = sum((x - mean)^2) / n
        // In-place ReduceSum (dst==src): work is both destination and source.
        // This works because ReduceSum writes partial sums to scratch (src param)
        // and only writes the final result to dst[0] after reading all source data.
        let var_sum = crate::__tile_v_reduce_sum_f32(*work, *work, *src, n);
        let var = var_sum / (n as f32);
        crate::__tile_pipe_barrier();

        // Step 5: dst = (x - mean) / sqrt(var + eps)
        let inv_std = 1.0f32 / crate::core::builtins::sqrtf(var + eps);
        crate::__tile_muls_f32(*dst, *dst, inv_std, n);
    }
}

/// Mean reduction: mean = sum(x) / n
///
/// Returns the scalar mean value.
/// Requires three buffers: dst, src, and work (workspace for reduction).
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn reduce_mean_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32) -> f32 {
    unsafe {
        let sum = crate::__tile_v_reduce_sum_f32(*dst, *src, *work, n);
        sum / (n as f32)
    }
}

/// ELU activation: elu(x) = x if x >= 0, alpha * (exp(x) - 1) if x < 0
///
/// Requires three buffers: dst, src, tmp (workspace).
/// **src is destroyed** (used as workspace for intermediate Add).
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn elu_f32(dst: &mut UbBuf, src: &mut UbBuf, tmp: &mut UbBuf, alpha: f32, n: u32) {
    unsafe {
        // tmp = exp(x)  — reads src
        crate::__tile_v_exp_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // dst = max(x, 0)  — last read of src for max
        crate::__tile_maxs_f32(*dst, *src, 0.0f32, n);
        crate::__tile_pipe_barrier();

        // tmp = exp(x) - 1
        crate::__tile_adds_f32(*tmp, *tmp, -1.0f32, n);
        crate::__tile_pipe_barrier();
        // tmp = alpha * (exp(x) - 1)
        crate::__tile_muls_f32(*tmp, *tmp, alpha, n);
        crate::__tile_pipe_barrier();
        // tmp = min(0, alpha*(exp(x)-1))
        crate::__tile_mins_f32(*tmp, *tmp, 0.0f32, n);
        crate::__tile_pipe_barrier();

        // src = max(x,0) + min(0, alpha*(exp(x)-1)) — ALL SEPARATE (src != dst != tmp)
        crate::__tile_v_add_f32(*src, *dst, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = result
        crate::__tile_muls_f32(*dst, *src, 1.0f32, n);
    }
}

/// HardSigmoid: hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
///
/// Result is written to dst. src is preserved.
#[inline(always)]
pub unsafe fn hardsigmoid_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        // dst = x / 6
        crate::__tile_muls_f32(dst, src, 1.0f32 / 6.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = x/6 + 0.5
        crate::__tile_adds_f32(dst, dst, 0.5f32, n);
        crate::__tile_pipe_barrier();
        // dst = max(0, dst)
        crate::__tile_maxs_f32(dst, dst, 0.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = min(1, dst) = clamp(x/6 + 0.5, 0, 1)
        crate::__tile_mins_f32(dst, dst, 1.0f32, n);
    }
}

/// Softsign: softsign(x) = x / (1 + |x|)
///
/// Requires three buffers: dst, src (preserved), and tmp (workspace).
/// Uses 3-buffer pattern to avoid aliased binary Mul on 310P.
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn softsign_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // tmp = |x|
        crate::__tile_v_abs_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // tmp = 1 + |x|
        crate::__tile_adds_f32(*tmp, *tmp, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // tmp = 1 / (1 + |x|)
        crate::__tile_reciprocal_f32(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = x * (1 / (1 + |x|))
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
    }
}

/// LogSoftmax: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
///
/// Numerically stable implementation.
/// Requires four buffers: dst, src (destroyed), work, and work2 (workspace).
///
/// ReduceSum requires all three buffer args to be distinct (aliased dst=src
/// crashes on 910B). This function needs 4 buffers to hold x-max(x) in dst
/// across the ReduceSum call.
///
/// **Buffer constraint:** dst, src, work, work2 must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn log_softmax_f32(
    dst: &mut UbBuf,
    src: &mut UbBuf,
    work: &mut UbBuf,
    work2: &mut UbBuf,
    n: u32,
) {
    unsafe {
        // Step 1: max_val = max(x)
        let max_val = crate::__tile_v_reduce_max_f32(*work, *src, *dst, n);
        crate::__tile_pipe_barrier();

        // Step 2: dst = x - max(x)
        crate::__tile_adds_f32(*dst, *src, -max_val, n);
        crate::__tile_pipe_barrier();

        // Step 3: work = exp(x - max(x))
        crate::__tile_v_exp_f32(*work, *dst, n);
        crate::__tile_pipe_barrier();

        // Step 4: sum = reduce_sum(exp(x - max(x)))
        // All three args must be distinct to avoid 910B crash.
        let sum = crate::__tile_v_reduce_sum_f32(*src, *work, *work2, n);
        crate::__tile_pipe_barrier();

        // Step 5: dst still has (x - max(x)) from step 2
        let log_sum = crate::core::builtins::logf(sum);
        crate::__tile_adds_f32(*dst, *dst, -log_sum, n);
    }
}

/// MinGPT new GELU (tanh approximation):
///   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Requires three buffers: dst, src (preserved), and tmp (workspace).
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn gelu_tanh_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // sqrt(2/pi) ≈ 0.7978845608
        let sqrt_2_over_pi: f32 = 0.7978845608f32;

        // tmp = x^2
        crate::__tile_v_mul_f32(*tmp, *src, *src, n);
        crate::__tile_pipe_barrier();
        // dst = x^3 = x^2 * x — all separate (dst != tmp != src)
        crate::__tile_v_mul_f32(*dst, *tmp, *src, n);
        crate::__tile_pipe_barrier();
        // dst = 0.044715 * x^3
        crate::__tile_muls_f32(*dst, *dst, 0.044715f32, n);
        crate::__tile_pipe_barrier();
        // tmp = x + 0.044715 * x^3 — all separate (tmp != src != dst)
        crate::__tile_v_add_f32(*tmp, *src, *dst, n);
        crate::__tile_pipe_barrier();
        // tmp = sqrt(2/pi) * (x + 0.044715 * x^3)
        crate::__tile_muls_f32(*tmp, *tmp, sqrt_2_over_pi, n);
        crate::__tile_pipe_barrier();
        // tmp = tanh(...)
        tanh_f32(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // tmp = 1 + tanh(...)
        crate::__tile_adds_f32(*tmp, *tmp, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = x * (1 + tanh(...))
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = 0.5 * x * (1 + tanh(...))
        crate::__tile_muls_f32(*dst, *dst, 0.5f32, n);
    }
}

/// Mish activation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// Requires three buffers: dst, src (preserved), and tmp (workspace).
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn mish_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // tmp = softplus(x) = ln(1 + exp(x))
        softplus_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // tmp = tanh(softplus(x))
        tanh_f32(*tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // dst = x * tanh(softplus(x))
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
    }
}

/// HardSwish: hardswish(x) = x * clamp(x/6 + 0.5, 0, 1) = x * hardsigmoid(x)
///
/// Requires three buffers: dst, src (preserved), and tmp (workspace).
/// Uses 3-buffer pattern to avoid aliased binary Mul on 310P.
///
/// **Buffer constraint:** dst, src, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn hardswish_f32(dst: &mut UbBuf, src: &UbBuf, tmp: &mut UbBuf, n: u32) {
    unsafe {
        // tmp = hardsigmoid(x)
        hardsigmoid_f32(*tmp, *src, n);
        crate::__tile_pipe_barrier();
        // dst = x * hardsigmoid(x)
        crate::__tile_v_mul_f32(*dst, *src, *tmp, n);
    }
}

/// RMS Normalization: rms_norm(x) = x / sqrt(mean(x^2) + eps)
///
/// Requires three buffers: dst, src (preserved), and work (workspace).
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn rms_norm_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32, eps: f32) {
    unsafe {
        // work = x^2
        crate::__tile_v_mul_f32(*work, *src, *src, n);
        crate::__tile_pipe_barrier();

        // mean_sq = sum(x^2) / n
        let sum_sq = crate::__tile_v_reduce_sum_f32(*work, *work, *dst, n);
        let mean_sq = sum_sq / (n as f32);
        crate::__tile_pipe_barrier();

        // inv_rms = 1 / sqrt(mean(x^2) + eps)
        let inv_rms = 1.0f32 / crate::core::builtins::sqrtf(mean_sq + eps);
        crate::__tile_muls_f32(*dst, *src, inv_rms, n);
    }
}

/// L1 Norm: l1_norm(x) = sum(|x|)
///
/// Returns scalar result. Requires three buffers.
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn l1_norm_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32) -> f32 {
    unsafe {
        // dst = |x|
        crate::__tile_v_abs_f32(*dst, *src, n);
        crate::__tile_pipe_barrier();
        // sum(|x|)
        crate::__tile_v_reduce_sum_f32(*dst, *dst, *work, n)
    }
}

/// L2 Norm (Frobenius norm for vectors): l2_norm(x) = sqrt(sum(x^2))
///
/// Returns scalar result. Requires three buffers.
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn l2_norm_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32) -> f32 {
    unsafe {
        // dst = x^2
        crate::__tile_v_mul_f32(*dst, *src, *src, n);
        crate::__tile_pipe_barrier();
        // sum(x^2)
        let sum_sq = crate::__tile_v_reduce_sum_f32(*dst, *dst, *work, n);
        crate::core::builtins::sqrtf(sum_sq)
    }
}

/// L2 Normalize: l2_normalize(x) = x / l2_norm(x)
///
/// Requires three buffers: dst, src (preserved), and work (workspace).
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn l2_normalize_f32(dst: &mut UbBuf, src: &UbBuf, work: &mut UbBuf, n: u32, eps: f32) {
    unsafe {
        let norm = l2_norm_f32(work, src, dst, n);
        crate::__tile_pipe_barrier();
        let inv_norm = 1.0f32 / (norm + eps);
        crate::__tile_muls_f32(*dst, *src, inv_norm, n);
    }
}

/// MSE Loss: mse(pred, target) = mean((pred - target)^2)
///
/// Returns scalar result. Requires pred, target, and work buffers.
///
/// **Buffer constraint:** work, pred, target, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn mse_loss_f32(
    work: &mut UbBuf,
    pred: &UbBuf,
    target: &UbBuf,
    tmp: &mut UbBuf,
    n: u32,
) -> f32 {
    unsafe {
        // work = pred - target
        crate::__tile_v_sub_f32(*work, *pred, *target, n);
        crate::__tile_pipe_barrier();
        // work = (pred - target)^2
        crate::__tile_v_mul_f32(*work, *work, *work, n);
        crate::__tile_pipe_barrier();
        // mean
        let sum = crate::__tile_v_reduce_sum_f32(*work, *work, *tmp, n);
        sum / (n as f32)
    }
}

/// Huber Loss: huber(pred, target, delta) =
///   0.5 * (pred - target)^2            if |pred - target| <= delta
///   delta * (|pred - target| - 0.5 * delta)  otherwise
///
/// Equivalent to: 0.5 * min(|diff|, delta)^2 + delta * max(|diff| - delta, 0)
/// Requires four buffers: dst, pred (destroyed, used as workspace), target (preserved), tmp.
/// Returns scalar mean loss.
///
/// **Buffer constraint:** dst, pred, target, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn huber_loss_f32(
    dst: &mut UbBuf,
    pred: &mut UbBuf,
    target: &UbBuf,
    tmp: &mut UbBuf,
    delta: f32,
    n: u32,
) -> f32 {
    unsafe {
        // dst = pred - target
        crate::__tile_v_sub_f32(*dst, *pred, *target, n);
        crate::__tile_pipe_barrier();
        // dst = |pred - target|
        crate::__tile_v_abs_f32(*dst, *dst, n);
        crate::__tile_pipe_barrier();

        // tmp = min(|diff|, delta)  (clamped)
        crate::__tile_mins_f32(*tmp, *dst, delta, n);
        crate::__tile_pipe_barrier();
        // tmp = clamped^2
        crate::__tile_v_mul_f32(*tmp, *tmp, *tmp, n);
        crate::__tile_pipe_barrier();
        // tmp = 0.5 * clamped^2  (quadratic part)
        crate::__tile_muls_f32(*tmp, *tmp, 0.5f32, n);
        crate::__tile_pipe_barrier();

        // dst = |diff| - delta
        crate::__tile_adds_f32(*dst, *dst, -delta, n);
        crate::__tile_pipe_barrier();
        // dst = max(|diff| - delta, 0)  (excess)
        crate::__tile_maxs_f32(*dst, *dst, 0.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = delta * excess  (linear part)
        crate::__tile_muls_f32(*dst, *dst, delta, n);
        crate::__tile_pipe_barrier();

        // pred = quadratic + linear = huber_element — all separate (pred != dst != tmp)
        // (pred no longer needed after initial Sub)
        crate::__tile_v_add_f32(*pred, *dst, *tmp, n);
        crate::__tile_pipe_barrier();

        // mean
        let sum = crate::__tile_v_reduce_sum_f32(*pred, *pred, *tmp, n);
        sum / (n as f32)
    }
}

/// Hinge Loss: hinge(pred, target) = mean(max(0, 1 - pred * target))
///
/// Returns scalar result. target should be +1 or -1.
///
/// **Buffer constraint:** dst, pred, target, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn hinge_loss_f32(
    dst: &mut UbBuf,
    pred: &UbBuf,
    target: &UbBuf,
    tmp: &mut UbBuf,
    n: u32,
) -> f32 {
    unsafe {
        // dst = pred * target
        crate::__tile_v_mul_f32(*dst, *pred, *target, n);
        crate::__tile_pipe_barrier();
        // dst = -pred * target
        crate::__tile_muls_f32(*dst, *dst, -1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = 1 - pred * target
        crate::__tile_adds_f32(*dst, *dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        // dst = max(0, 1 - pred * target)
        crate::__tile_maxs_f32(*dst, *dst, 0.0f32, n);
        crate::__tile_pipe_barrier();
        // mean
        let sum = crate::__tile_v_reduce_sum_f32(*dst, *dst, *tmp, n);
        sum / (n as f32)
    }
}

/// Cosine similarity: cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b))
///
/// Returns scalar result.
///
/// **Buffer constraint:** dst, a, b, tmp must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn cosine_similarity_f32(
    dst: &mut UbBuf,
    a: &UbBuf,
    b: &UbBuf,
    tmp: &mut UbBuf,
    n: u32,
) -> f32 {
    unsafe {
        // dot product: dst = a * b element-wise, then sum
        crate::__tile_v_mul_f32(*dst, *a, *b, n);
        crate::__tile_pipe_barrier();
        let dot = crate::__tile_v_reduce_sum_f32(*dst, *dst, *tmp, n);
        crate::__tile_pipe_barrier();

        // norm(a)
        let norm_a = l2_norm_f32(dst, a, tmp, n);
        crate::__tile_pipe_barrier();

        // norm(b)
        let norm_b = l2_norm_f32(dst, b, tmp, n);

        dot / (norm_a * norm_b + 1e-8f32)
    }
}

/// SGD update: param = param - lr * grad
///
/// Updates param buffer in-place. **grad is destroyed**.
#[inline(always)]
pub unsafe fn sgd_update_f32(param: &mut UbBuf, grad: &mut UbBuf, lr: f32, n: u32) {
    unsafe {
        // grad = lr * grad
        crate::__tile_muls_f32(*grad, *grad, lr, n);
        crate::__tile_pipe_barrier();
        // grad = param - lr*grad (all distinct: output=grad, src1=param, src2=grad)
        crate::__tile_v_sub_f32(*grad, *param, *grad, n);
        crate::__tile_pipe_barrier();
        // param = result (copy via scalar multiply by 1.0)
        crate::__tile_muls_f32(*param, *grad, 1.0f32, n);
    }
}

/// Reduce product: prod(x) = x[0] * x[1] * ... * x[n-1]
///
/// Computed as exp(sum(log(|x|))). Only correct for positive inputs.
/// Returns scalar result.
///
/// **Buffer constraint:** dst, src, work must all be distinct (aliased buffers cause crashes/wrong values on NPU).
#[inline(always)]
pub unsafe fn reduce_prod_f32(dst: &mut UbBuf, src: &mut UbBuf, work: &mut UbBuf, n: u32) -> f32 {
    unsafe {
        // dst = ln(x) — assumes positive inputs
        crate::__tile_ln_f32(*dst, *src, n);
        crate::__tile_pipe_barrier();
        // ReduceSum requires dst != src (aliased ReduceSum crashes on 910B).
        // Use work as dst, src as workspace (src no longer needed).
        let log_sum = crate::__tile_v_reduce_sum_f32(*work, *dst, *src, n);
        crate::core::builtins::expf(log_sum)
    }
}

// =========================================================================
// Cube engine (matrix multiply) composite operations
// =========================================================================

/// Matrix multiply: C[m,n] = A[m,k] * B[k,n]
///
/// A, B are f16 (GM pointers to u16); C is f32 (GM pointer to f32).
/// Handles buffer allocation and data movement through the cube pipeline:
///   GM → L1 → L0A/L0B → Mmad → L0C → UB → GM
///
/// For small matrices that fit in L0 memory (m*k, k*n < ~32K elements each).
/// For larger matrices, use the low-level __tile_mmad_f16 with manual tiling.
#[inline(always)]
pub unsafe fn matmul_f16(c: *mut f32, a: *const u16, b: *const u16, m: u32, k: u32, n: u32) {
    unsafe {
        // Single macro intrinsic: generates the complete reference-style cube matmul
        // pipeline including CopyND2NZ, SplitA, B-split loop, Mmad, Aggregate, CopyOut.
        crate::__tile_matmul_cube(c, a, b, m, k, n);
    }
}

/// Matrix multiply with transpose B: C[m,n] = A[m,k] * B^T where B is stored as [n,k]
///
/// A is f16 [m,k], B is f16 [n,k] (transposed layout in GM), C is f32 [m,n].
/// Uses hardware transpose during L1→L0B copy (LoadData2dParams.ifTranspose=true)
/// and MmadParams.enBTranspose=true.
#[inline(always)]
pub unsafe fn matmul_f16_transpose_b(
    c: *mut f32,
    a: *const u16,
    b: *const u16,
    m: u32,
    k: u32,
    n: u32,
) {
    unsafe {
        // Allocate cube pipeline buffers
        let l1_a = crate::__tile_buf_alloc_l1(m * k);
        let l1_b = crate::__tile_buf_alloc_l1(n * k); // B stored as [n,k]
        let l0a = crate::__tile_buf_alloc_l0a(m * k);
        let l0b = crate::__tile_buf_alloc_l0b(k * n);
        let l0c = crate::__tile_buf_alloc_l0c(m * n);
        let ub_c = crate::__tile_buf_alloc(m * n);

        // Stage 1: GM → L1
        crate::__tile_load_gm_to_l1_f16(l1_a, a, m * k);
        crate::__tile_load_gm_to_l1_f16(l1_b, b, n * k);
        crate::__tile_pipe_barrier();

        // Stage 2: L1 → L0A (normal), L1 → L0B (with hardware transpose)
        crate::__tile_copy_l1_to_l0a_f16(l0a, l1_a, m * k);
        crate::__tile_copy_l1_to_l0b_f16_2d(l0b, l1_b, 1, 1, 1); // transpose=1
        crate::__tile_pipe_barrier();

        // Stage 3: Mmad with transpose_b=1
        crate::__tile_mmad_f16_ex(l0c, l0a, l0b, m, k, n, 1, 0, 0, 1);
        crate::__tile_pipe_barrier();

        // Stage 4: L0C → UB → GM
        crate::__tile_copy_l0c_to_ub_f32(ub_c, l0c, m * n);
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f32(c, ub_c, m * n);
    }
}

/// Matrix multiply with transpose A: C[m,n] = A^T * B where A is stored as [k,m]
///
/// A is f16 [k,m] (transposed layout in GM), B is f16 [k,n], C is f32 [m,n].
/// Uses hardware transpose during L1→L0A copy (LoadData2dParams.ifTranspose=true)
/// and MmadParams.enATranspose=true.
#[inline(always)]
pub unsafe fn matmul_f16_transpose_a(
    c: *mut f32,
    a: *const u16,
    b: *const u16,
    m: u32,
    k: u32,
    n: u32,
) {
    unsafe {
        // Allocate cube pipeline buffers
        let l1_a = crate::__tile_buf_alloc_l1(k * m); // A stored as [k,m]
        let l1_b = crate::__tile_buf_alloc_l1(k * n);
        let l0a = crate::__tile_buf_alloc_l0a(m * k);
        let l0b = crate::__tile_buf_alloc_l0b(k * n);
        let l0c = crate::__tile_buf_alloc_l0c(m * n);
        let ub_c = crate::__tile_buf_alloc(m * n);

        // Stage 1: GM → L1
        crate::__tile_load_gm_to_l1_f16(l1_a, a, k * m);
        crate::__tile_load_gm_to_l1_f16(l1_b, b, k * n);
        crate::__tile_pipe_barrier();

        // Stage 2: L1 → L0A (with hardware transpose), L1 → L0B (normal)
        crate::__tile_copy_l1_to_l0a_f16_2d(l0a, l1_a, 1, 1, 1); // transpose=1
        crate::__tile_copy_l1_to_l0b_f16(l0b, l1_b, k * n);
        crate::__tile_pipe_barrier();

        // Stage 3: Mmad with transpose_a=1
        crate::__tile_mmad_f16_ex(l0c, l0a, l0b, m, k, n, 1, 0, 1, 0);
        crate::__tile_pipe_barrier();

        // Stage 4: L0C → UB → GM
        crate::__tile_copy_l0c_to_ub_f32(ub_c, l0c, m * n);
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f32(c, ub_c, m * n);
    }
}

// ---------------------------------------------------------------------------
// Simple unary ops (composites of low-level intrinsics)
// ---------------------------------------------------------------------------

/// Negate: neg(x) = -x = Muls(x, -1)
#[inline(always)]
pub unsafe fn neg_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_muls_f32(dst, src, -1.0f32, n);
    }
}

/// Negate (f16): neg(x) = -x
#[inline(always)]
pub unsafe fn neg_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_muls_f16(dst, src, -1.0f32, n);
    }
}

/// ReLU (f16): relu(x) = max(x, 0)
#[inline(always)]
pub unsafe fn relu_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_maxs_f16(dst, src, 0.0f32, n);
    }
}

/// Sigmoid (f16): sigmoid(x) = 1 / (1 + exp(-x))
#[inline(always)]
pub unsafe fn sigmoid_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_muls_f16(dst, src, -1.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_v_exp_f16(dst, dst, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f16(dst, dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_reciprocal_f16(dst, dst, n);
    }
}

/// Tanh (f16): tanh(x) = 2*sigmoid(2x) - 1
#[inline(always)]
pub unsafe fn tanh_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_muls_f16(dst, src, -2.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_v_exp_f16(dst, dst, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f16(dst, dst, 1.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_reciprocal_f16(dst, dst, n);
        crate::__tile_pipe_barrier();
        crate::__tile_muls_f16(dst, dst, 2.0f32, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f16(dst, dst, -1.0f32, n);
    }
}

/// expm1 (f32): expm1(x) = exp(x) - 1
#[inline(always)]
pub unsafe fn expm1_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_v_exp_f32(dst, src, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f32(dst, dst, -1.0f32, n);
    }
}

/// expm1 (f16): expm1(x) = exp(x) - 1
#[inline(always)]
pub unsafe fn expm1_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        crate::__tile_v_exp_f16(dst, src, n);
        crate::__tile_pipe_barrier();
        crate::__tile_adds_f16(dst, dst, -1.0f32, n);
    }
}
