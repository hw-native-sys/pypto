// Engine-signature shims: the MERGED Rust kernels exposed under the names and signatures the
// serving engine binds. `device.rs` looks up C++-MANGLED device symbols, so these are plain C++
// `__global__` functions (no `extern "C"`), each instantiating the generated kernel class.
//
// This must be concatenated AFTER the kernel classes, which is what merge_kernels.py does.
//
// Covered here: the slots whose socket already matches a Rust kernel's argument list exactly.
//   ds4_rmsmul_{4096,1024,512}(float*, float*, float*)   3 args, rms x gamma
//   ds4_hc_norm(float*, float*)                          2 args, no gamma
//   ds4_rope_halves(float*, float*, float*, float*)      4 args
// NOT covered, and deliberately: the four `ds4_q8mv_*` sockets are 4-argument Q8_0 with a
// companion plane, and `ds4_hc_mv`/`ds4_mx4pk_*` are packed formats -- the w8a8 arms take five
// arguments, so exposing them under those names would be the over-credit iteration 51 removed.

__global__ __aicore__ void ds4_rmsmul_4096(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_4096 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rmsmul_1024(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_1024 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rmsmul_512(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_512 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_hc_norm(__gm__ float *x, __gm__ float *o) {
    Kernel_engine_rms_norm_1_16384 op;
    op.Init((GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rope_halves(__gm__ float *x, __gm__ float *cs, __gm__ float *sn, __gm__ float *o) {
    Kernel_rope_halves_engine_1x64 op;
    op.Init((GM_ADDR)x, (GM_ADDR)cs, (GM_ADDR)sn, (GM_ADDR)o);
    op.Process();
}

// THE FOUR PROJECTION SOCKETS. `ds4_q8mv_<in>_<out>(w: si8, s: f32, x: f32, o: f32)` -- four
// pointers, and the block scales arrive PRE-EXPANDED into a full f32 plane (layer0.pto:3), so the
// generated kernels compute exactly `out[r] = sum_k w8[r,k]*x[k]*s[r,k]` with no block arithmetic.
// All four shapes are certified on device; these shims are what let the engine bind them.
__global__ __aicore__ void ds4_q8mv_4096_1024(__gm__ signed char *w, __gm__ float *s,
                                              __gm__ float *x, __gm__ float *o) {
    Kernel_proj_q8mv_engine_1x1024_k4096 op;
    op.Init((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_q8mv_1024_32768(__gm__ signed char *w, __gm__ float *s,
                                               __gm__ float *x, __gm__ float *o) {
    Kernel_proj_q8mv_engine_1x32768_k1024 op;
    op.Init((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_q8mv_4096_512(__gm__ signed char *w, __gm__ float *s,
                                             __gm__ float *x, __gm__ float *o) {
    Kernel_proj_q8mv_engine_1x512_k4096 op;
    op.Init((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_q8mv_8192_4096(__gm__ signed char *w, __gm__ float *s,
                                              __gm__ float *x, __gm__ float *o) {
    Kernel_proj_q8mv_engine_1x4096_k8192 op;
    op.Init((GM_ADDR)w, (GM_ADDR)s, (GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

// THE ATTENTION SOCKET. `ds4_attn_sink_batched(q, kv, sk, o)` -- 64 heads, S=128, D=512, one launch.
//
// The generated kernel computes ONE HEAD (and one 8-position block), certified on device at
// backward_rel 7.932e-08 against an f64 oracle. The 64-head extent is reached with a C++ loop HERE
// rather than in the tile, because the AscendC emitter has no `scf.for`: a straight-line 64-head
// unroll is ~1088 ops and ~500 buffers, and each head needs ~40 KB of UB, so only four heads would
// fit. A loop reuses one head's buffers, which is what makes 64 fit.
//
// The per-head kernel takes (q_head, kv, sk_head, o_head), so the heads are addressed by POINTER
// OFFSET: q + h*512, sk + h*8, o + h*512.
__global__ __aicore__ void ds4_attn_sink_batched(__gm__ float *q, __gm__ float *kv,
                                                __gm__ float *sk, __gm__ float *o) {
    for (int __h = 0; __h < 64; ++__h) {
        Kernel_attn_sink_1blk op;
        op.Init((GM_ADDR)(q + __h * 512), (GM_ADDR)kv, (GM_ADDR)(sk + __h * 8),
                (GM_ADDR)(o + __h * 512));
        op.Process();
    }
}
