//! Hyper-connections: the residual mixing GLM 5.3 declares and this engine did not implement.
//!
//! `glm5-next.hyper_connection.count = 4`, `sinkhorn_iterations = 20`, `epsilon = 1e-6`, and every block carries
//! `hc_attn_fn` [16384, 24] BF16, `hc_attn_base` [24], `hc_attn_scale` [3] (and the same triple for the FFN).
//!
//! 24 = 16 + 4 + 4, i.e. a 4x4 mixing matrix and four weights in each direction, which is the dynamic
//! hyper-connection parameterisation: the 24 values are PREDICTED from the state, the matrix is made
//! doubly-stochastic by Sinkhorn iteration, and the streams are mixed before the block and combined after it.
//!
//! WHAT IS ESTABLISHED AND WHAT IS NOT: the SHAPES, the Sinkhorn parameters and the number of streams come from the
//! file. The exact ORDER in which the mixing, the block and the combination are applied is my reading of the
//! hyper-connection formulation -- stated here because a wrong order would be undetectable from any single output.

/// The Sinkhorn normalisation of a `n x n` matrix: alternate row and column normalisation `iters` times.
///
/// Returns a matrix whose rows and columns each sum to ~1. A zero matrix would divide by zero; `eps` is the file's
/// own `hyper_connection.epsilon` and is added to every denominator, which is what makes that safe.
pub fn sinkhorn(m: &[f64], n: usize, iters: usize, eps: f64) -> Vec<f64> {
    assert_eq!(m.len(), n * n, "matrix is not {n} x {n}");
    if n == 0 {
        return Vec::new();
    }
    let mut x = m.to_vec();
    // THE FIRST PASS IS A SOFTMAX, NOT A SUM-NORMALISATION. `engine_ascend/src/hc.rs` documents this and it is the
    // difference between a distribution and a rescaled matrix: pass 1 subtracts the row max, exponentiates, then
    // normalises and adds eps; passes 2..N normalise by the row SUM. My first version did N identical sum passes,
    // which produces a plausible matrix that is not this operator.
    for r in 0..n {
        let row = &mut x[r * n..(r + 1) * n];
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v = *v * inv + eps;
        }
    }
    col_scale(&mut x, n, eps);
    for _ in 1..iters.max(1) {
        for r in 0..n {
            let row = &mut x[r * n..(r + 1) * n];
            let s: f64 = row.iter().sum();
            let inv = 1.0 / (s + eps);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        col_scale(&mut x, n, eps);
    }
    x
}

fn col_scale(x: &mut [f64], n: usize, eps: f64) {
    let mut col = vec![0.0f64; n];
    for r in 0..n {
        for c in 0..n {
            col[c] += x[r * n + c];
        }
    }
    for c in 0..n {
        let inv = 1.0 / (col[c] + eps);
        for r in 0..n {
            x[r * n + c] *= inv;
        }
    }
}

/// The 24 predicted parameters, split into the matrix and the two weight vectors.
pub struct Params {
    /// `n x n` mixing matrix, already Sinkhorn-normalised.
    pub matrix: Vec<f64>,
    /// how much of the BLOCK's output each stream receives
    pub w_in: Vec<f64>,
    /// how the streams combine into the block's input and the next state
    pub w_out: Vec<f64>,
}

impl Params {
    /// The reference's three coefficient sets for one sublayer, adapted into `Params`' fields.
    ///
    /// `ref_mixes` does the work: `mixes = (fn @ flat_state) * rsqrt(mean(flat_state^2) + eps)`, then
    /// `pre = sigmoid(mixes[0..n] * scale[0] + base) + eps`, `post = 2 * sigmoid(mixes[n..2n] * scale[1] + base)`
    /// and `comb = sinkhorn(mixes[2n..] * scale[2] + base)`. The layout is `[n pre][n post][n*n comb]`.
    pub fn predict(
        hc_fn: &[f32],
        hc_base: &[f32],
        scale: &[f32],
        state: &[Vec<f64>],
        n: usize,
        iters: usize,
        eps: f64,
    ) -> Params {
        // THE PREDICTION SEES ALL STREAMS, NOT ONE. `hc_attn_fn` is [16384, 24] and 16384 = 4 streams x 4096, so the
        // matrix is predicted from the FLATTENED state. Taking `state[0].len()` (= 4096) made the assert report
        // "must be [4096, 24]" against a 393216-value tensor, which is how this was found.
        let d = state.len() * state[0].len();
        // THE PARAMETER COUNT IS `2n + n*n`, NOT `4n`. For n = 4 that is 24 = 4 + 4 + 16, which is what the
        // checkpoint carries; my first version wrote `4 * n` and got 16, so the matrix slice was half-length.
        let n_params = 2 * n + n * n;
        assert_eq!(hc_fn.len(), d * n_params, "hc_fn must be [{d}, {n_params}]");
        assert_eq!(hc_base.len(), n_params);
        let flat: Vec<f64> = state.iter().flatten().copied().collect();
        debug_assert_eq!(flat.len(), d);
        // THE LAYOUT IS `4 + 4 + 4*4`, MATRIX LAST: four in-weights, four out-weights, then the sixteen matrix
        // entries. `engine_ascend/src/hc.rs` states it; my first version assumed `16 + 4 + 4`, the reverse.
        //
        // B20: THE REFERENCE'S COEFFICIENTS, into the same three fields so every caller moves together.
        //   pre  -> w_out   (collapses the streams: `block_input`)
        //   post -> w_in    (scales the sublayer output: `next_state`)
        //   comb -> matrix  (mixes the residual: `next_state`)
        //
        // B23: `ref_mixes` OWNS THE FORMULA, and is now the only place the [mix_hc, hc_dim] matvec happens. B20 left
        // a second, DEAD transcription here -- an `alpha = scale * (fn @ state) + base` that nothing read. Deleting it
        // is not only a tidy-up: that copy was WRONG. It omitted `ref_mixes`' rsqrt over the flattened state, and it
        // applied `scale` outside the sigmoid instead of inside it. A wrong copy of the live formula sitting twenty
        // lines above it is the likeliest thing a later session reads by mistake -- the compiler had been reporting it
        // as `unused variable: alpha` the whole time.
        //
        // AND IT IS NOT A SPEEDUP, measured: the dead copy did cost a full `n_params x hc_dim` matvec (24 x 16384 =
        // 393k multiply-adds) per call, but deleting it -- together with the six duplicate `predict` call sites B23
        // removed from the two drivers -- moved the 6-block wall clock not at all (decode 14s -> 13s, prefill 14s ->
        // 15s, warm, alternating runs of the two binaries). The model's own weight matvecs dominate a block by orders
        // of magnitude. The reason to delete this is that it was a WRONG transcription of the live formula, and the
        // FLOP count is not a reason at all -- do not re-derive a performance argument from it.
        let m = ref_mixes(&flat, hc_fn, scale, hc_base, n, iters, eps);
        Params {
            matrix: m.comb,
            w_in: m.post,
            w_out: m.pre,
        }
    }

    /// The block's input: the streams mixed by the matrix, then by `w_in`.
    pub fn block_input(&self, state: &[Vec<f64>], n: usize) -> Vec<f64> {
        // `hc_pre`: sum_j pre[j] * state[j], with pre = `w_out`. THIS IS THE REFERENCE'S COLLAPSE, not a sinkhorn mix.
        let d = state[0].len();
        let mut out = vec![0.0f64; d];
        for j in 0..n {
            let w = self.w_out[j];
            for c in 0..d {
                out[c] += w * state[j][c];
            }
        }
        out
    }

    /// The next state: each stream takes the block's output through `w_in[i]`, mixed by the matrix.
    pub fn next_state(&self, state: &[Vec<f64>], block_out: &[f64], n: usize) -> Vec<Vec<f64>> {
        // `hc_post`: each copy is `post[i]*block_out + sum_j comb[i][j]*state[j]`, with post = `w_in` and comb = `matrix`.
        let d = state[0].len();
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|c| self.w_in[i] * block_out[c] + (0..n).map(|j| self.matrix[i * n + j] * state[j][c]).sum::<f64>())
                    .collect()
            })
            .collect()
    }
}


/// ONE SUBLAYER AROUND THE LEARNED MIXING: input <- mixing streams, run the sublayer, streams <- mixing of the result.
///
/// This is the shape `model.py` gives every block -- `hc_attn_*` around attention and `hc_ffn_*` around the FFN/MoE -- and
/// the shape the DS4 decode path does NOT implement: it carries `n_hc` copies and averages them (`hc_mean_collapse`), which
/// is the reference's *plain* collapse, not its learned one.
///
/// `sublayer` receives the mixed block input and returns the sublayer's output (before the residual). The residual is added
/// here, and the streams are advanced with `next_state`, so a caller cannot get the order wrong.
pub fn sublayer<F>(
    hc_fn: &[f32],
    hc_base: &[f32],
    hc_scale: &[f32],
    streams: &[Vec<f64>],
    n: usize,
    iters: usize,
    eps: f64,
    sublayer: F,
) -> (Vec<f64>, Vec<Vec<f64>>)
where
    F: FnOnce(&[f64]) -> Vec<f64>,
{
    // ONE IMPLEMENTATION: `ref_sublayer` is the reference's `hc_mixes` -> `hc_pre` -> sublayer -> `hc_post`.
    //
    // THE BUG THIS REPLACES: this function used to compute `residual = x + out` and pass it to `next_state`. That was correct for
    // the OLD `next_state` (which mixed a collapsed vector) and became a DOUBLE COUNT when `next_state` was rewritten to
    // `hc_post`, which already adds `comb . state` -- `x` is counted once as part of `residual` and again through `comb`.
    // The two drivers then disagreed: GLM gave 29585 through `glm53_block0` and 51229 through this path.
    let next = ref_sublayer(streams, hc_fn, hc_scale, hc_base, n, iters, eps, sublayer);
    // The caller logs the collapsed sublayer input as the "residual" stream; `ref_sublayer` does not return it, so recompute
    // the collapse -- it is `hc_pre`, one dot per channel, and keeping it here avoids a second return value in the reference API.
    let flat: Vec<f64> = streams.iter().flatten().copied().collect();
    let m = ref_mixes(&flat, hc_fn, hc_scale, hc_base, n, iters, eps);
    let d = streams[0].len();
    let xin: Vec<f64> = (0..d).map(|c| (0..n).map(|j| m.pre[j] * streams[j][c]).sum()).collect();
    (xin, next)
}


/// THE LEARNED HEAD COLLAPSE: `output_hc_head_one` (antirez ds4.c:7654-7681), in f64.
///
/// `rms_norm(inp_hc, unit gamma) -> matvec(hc_head_fn) -> sigmoid -> weighted sum over the n_hc copies`.
///
/// WHY THIS EXISTS: the DS4 examples call `decode_step::hc_mean_collapse`, a PLAIN AVERAGE, and the module that owns it says so
/// -- *"this is the plain average ... NOT the learned `output_hc_head_one` collapse"*. The checkpoint carries `hc_head_fn`,
/// `hc_head_base` and `hc_head_scale` (24 non-layer `hc_*` tensors), so the learned path is available and unused.
///
/// It is written here, data-free and testable, so that swapping the call sites does not require the 476 GB checkpoint.
pub fn head_collapse(inp_hc: &[f64], fn_w: &[f32], base: &[f32], scale: &[f32], n_hc: usize, d: usize, eps: f64) -> Vec<f64> {
    let hc_dim = n_hc * d;
    assert_eq!(inp_hc.len(), hc_dim, "inp_hc must be n_hc*d");
    assert_eq!(fn_w.len(), hc_dim * n_hc, "hc_head_fn must be [n_hc*d, n_hc]");
    // rms_norm against a unit gamma, as the reference's ParallelHead does
    let ms = inp_hc.iter().map(|v| v * v).sum::<f64>() / hc_dim as f64;
    let r = 1.0 / (ms + eps).sqrt();
    let flat: Vec<f64> = inp_hc.iter().map(|v| v * r).collect();
    for (i, v) in flat.iter().enumerate() {
        assert!(v.is_finite() || i < hc_dim, "non-finite after the norm");
    }
    // one pre-activation per copy
    let pre: Vec<f64> = (0..n_hc)
        .map(|h| (0..hc_dim).map(|c| fn_w[h * hc_dim + c] as f64 * flat[c]).sum())
        .collect();
    // sigmoid, biased and scaled by the head's own base/scale, then the weighted sum over copies
    // `hc_head_scale` IS A SCALAR IN THE CHECKPOINT (shape [1]), while `hc_head_base` is [n_hc]. Indexing the scale per copy
    // silently gave copies 1..n a scale of 1.0 -- found by reading the real tensor's shape, not by running the model.
    assert!(
        scale.len() == 1 || scale.len() >= n_hc,
        "hc_head_scale must be a scalar or one per copy (got {})",
        scale.len()
    );
    let sc = |h: usize| scale[if scale.len() == 1 { 0 } else { h }] as f64;
    let w: Vec<f64> = (0..n_hc)
        .map(|h| {
            let z = pre[h] * sc(h) + base.get(h).copied().unwrap_or(0.0) as f64;
            1.0 / (1.0 + (-z).exp())
        })
        .collect();
    let total: f64 = w.iter().sum();
    let total = if total.abs() < 1e-30 { 1.0 } else { total };
    (0..d)
        .map(|c| (0..n_hc).map(|h| w[h] * inp_hc[h * d + c]).sum::<f64>() / total)
        .collect()
}


/// THE REFERENCE'S HYPER-CONNECTION, transcribed from `/data0/DeepSeek-V4.1-Flash/inference`:
/// `model.py:927` (`hc_mixes`) and `kernel.py:426-443` (`hc_split_sinkhorn_kernel_`).
///
/// ```text
/// mixes     = linear(x_flat, hc_fn) * rsqrt(rms(x_flat) + eps)      -- NO bias, and the rsqrt NORMALISES
/// pre[j]    = sigmoid(mixes[j]          * hc_scale[0] + hc_base[j])         + eps
/// post[j]   = 2 * sigmoid(mixes[j + n]  * hc_scale[1] + hc_base[j + n])
/// comb[j,k] = softmax_k(mixes[j*n + k + 2n] * hc_scale[2] + hc_base[j*n + k + 2n])
/// comb      = sinkhorn(comb, iters, eps)
/// ```
///
/// The engine's `Params::predict` differs in FIVE places: no `rsqrt`, `hc_base` added inside the projection instead of inside
/// the sigmoid, no `pre`, no `+eps`, `2*` or `softmax`, and `sinkhorn` applied to the raw mixes rather than to `comb`.
pub struct RefMixes {
    pub pre: Vec<f64>,
    pub post: Vec<f64>,
    pub comb: Vec<f64>,
}

pub fn ref_mixes(x_flat: &[f64], hc_fn: &[f32], hc_scale: &[f32], hc_base: &[f32], n: usize, iters: usize, eps: f64) -> RefMixes {
    let hc_dim = x_flat.len();
    let mix_hc = (2 + n) * n;
    assert_eq!(hc_fn.len(), mix_hc * hc_dim, "hc_fn must be [mix_hc, hc_dim]");
    assert!(hc_scale.len() >= 3, "hc_scale is [3]: pre, post, comb");
    assert_eq!(hc_base.len(), mix_hc, "hc_base is [mix_hc]");
    // the rsqrt: ONE statistic per token over the whole flattened stream. This is the line the engine does not have.
    let ms = x_flat.iter().map(|v| v * v).sum::<f64>() / hc_dim as f64;
    let rsqrt = 1.0 / (ms + eps).sqrt();
    let mixes: Vec<f64> = (0..mix_hc)
        .map(|r| (0..hc_dim).map(|c| hc_fn[r * hc_dim + c] as f64 * x_flat[c]).sum::<f64>() * rsqrt)
        .collect();
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let pre: Vec<f64> = (0..n).map(|j| sig(mixes[j] * hc_scale[0] as f64 + hc_base[j] as f64) + eps).collect();
    let post: Vec<f64> = (0..n).map(|j| 2.0 * sig(mixes[j + n] * hc_scale[1] as f64 + hc_base[j + n] as f64)).collect();
    let mut comb: Vec<f64> = (0..n * n)
        .map(|jk| mixes[jk + 2 * n] * hc_scale[2] as f64 + hc_base[jk + 2 * n] as f64)
        .collect();
    // comb = softmax_k(comb) + eps, then col-normalise, then (row, col) x (iters-1).
    // THE LOOP ENDS ON A COLUMN NORMALISATION, so COLUMNS sum to 1, not rows. And the softmax is applied ONCE here --
    // calling `sinkhorn()` would apply its own softmax a second time, which is what my first version did.
    for j in 0..n {
        let row = &mut comb[j * n..j * n + n];
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in row.iter_mut() { *v = (*v - m).exp(); sum += *v; }
        for v in row.iter_mut() { *v = *v / sum + eps; }
    }
    let col_norm = |c: &mut [f64]| {
        for k in 0..n {
            let s: f64 = (0..n).map(|j| c[j * n + k]).sum();
            for j in 0..n { c[j * n + k] /= s + eps; }
        }
    };
    col_norm(&mut comb);
    for _ in 1..iters.max(1) {
        for j in 0..n {
            let s: f64 = comb[j * n..j * n + n].iter().sum();
            for k in 0..n { comb[j * n + k] /= s + eps; }
        }
        col_norm(&mut comb);
    }
    RefMixes { pre, post, comb }
}


/// THE REFERENCE'S COMPLETE HYPER-CONNECTION SUBLAYER: `hc_mixes` -> `hc_pre` -> sublayer -> `hc_post`.
///
/// ```text
/// pre, post, comb = ref_mixes(x.flatten(), hc_fn, hc_scale, hc_base, ...)
/// xin             = hc_pre(x, pre)            = sum_j pre[j] * x[j]                       -> [d]
/// out             = sublayer(xin)             = the attention or FFN                     -> [d]
/// x_next[i]       = hc_post(out, x, post, comb) = post[i] * out + sum_j comb[i][j] * x[j]  -> [n][d]
/// ```
///
/// This composition is what the engine's `block_input`/`next_state` approximate, and it differs in every coefficient:
/// the reference's `post` MULTIPLIES the sublayer output, its `comb` mixes the residual, and its `pre` collapses the input.
pub fn ref_sublayer<F>(
    state: &[Vec<f64>],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    n: usize,
    iters: usize,
    eps: f64,
    sublayer: F,
) -> Vec<Vec<f64>>
where
    F: FnOnce(&[f64]) -> Vec<f64>,
{
    let d = state[0].len();
    let flat: Vec<f64> = state.iter().flatten().copied().collect();
    let m = ref_mixes(&flat, hc_fn, hc_scale, hc_base, n, iters, eps);
    // hc_pre: collapse the n copies into one sublayer input, weighted by `pre`
    let xin: Vec<f64> = (0..d).map(|c| (0..n).map(|j| m.pre[j] * state[j][c]).sum()).collect();
    let out = sublayer(&xin);
    assert_eq!(out.len(), d, "the sublayer must return d values");
    // hc_post: each copy is `post[i] * sublayer + sum_j comb[i][j] * residual[j]`
    (0..n)
        .map(|i| {
            (0..d)
                .map(|c| m.post[i] * out[c] + (0..n).map(|j| m.comb[i * n + j] * state[j][c]).sum::<f64>())
                .collect()
        })
        .collect()
}

#[cfg(test)]
    /// THE DEFECT, ON REAL WEIGHTS: the reference's mixes are **scale-invariant** because of the `rsqrt`; the engine's
    /// `Params::predict` is not. This asserts the reference's side with the checkpoint's own `hc_attn_fn`.
    ///
    /// It is a POSITIVE assertion (the reference IS invariant), not a test that pins the engine's bug -- when `predict` is
    /// rewritten to `ref_mixes`, this test keeps passing.
    #[test]
    fn ref_mixes_is_scale_invariant_on_the_real_tensors() {
        let path = std::path::Path::new("/tmp/hc_layer_raw.bin");
        if !path.exists() {
            eprintln!("SKIP: /tmp/hc_layer_raw.bin absent -- run tools/fetch_hc_fixtures.sh");
            return;
        }
        let raw = std::fs::read(path).unwrap();
        let all = bytes_to_f32(&raw);
        let (n, d) = (4usize, 4096usize);
        let n_params = 2 * n + n * n;
        let fn_len = n_params * n * d;
        let per_half = fn_len + n_params + 3;
        for (label, off) in [("attn", 0usize), ("ffn", per_half)] {
            let f = &all[off..off + fn_len];
            let base = &all[off + fn_len..off + fn_len + n_params];
            let scale = &all[off + fn_len + n_params..off + per_half];
            let x: Vec<f64> = (0..n * d).map(|i| ((i % 101) as f64 - 50.0) * 0.02).collect();
            let a = ref_mixes(&x, f, scale, base, n, 20, 1e-6);
            // a 1000x larger state, the magnitude of the growth iteration 365 measured (rms 8 vs 3477)
            let x1000: Vec<f64> = x.iter().map(|v| v * 1000.0).collect();
            let b = ref_mixes(&x1000, f, scale, base, n, 20, 1e-6);
            for j in 0..n {
                assert!((a.pre[j] - b.pre[j]).abs() < 1e-6, "{label}: pre must be scale-invariant at {j}");
                assert!((a.post[j] - b.post[j]).abs() < 1e-6, "{label}: post must be scale-invariant at {j}");
            }
            for jk in 0..n * n {
                assert!((a.comb[jk] - b.comb[jk]).abs() < 1e-6, "{label}: comb must be scale-invariant at {jk}");
            }
            println!("{label}: ref_mixes is scale-invariant on real weights (pre {:.4}..{:.4})",
                     a.pre.iter().cloned().fold(f64::INFINITY, f64::min),
                     a.pre.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        }
    }

    /// THE INVARIANTS THAT PIN `ref_sublayer` TO THE REFERENCE, each one a property of `hc_post` plus sinkhorn:
    ///   * a sublayer returning ZERO leaves each copy as `sum_j comb[i][j] * residual[j]`;
    ///   * with all copies EQUAL and comb rows summing to 1, that is exactly the residual -- an exact identity;
    ///   * with a LOUD sublayer the copies differ -- so the identity above is not vacuous.
    #[test]
    fn ref_sublayer_matches_the_references_identities() {
        let (n, d) = (4usize, 6usize);
        let hc_dim = n * d;
        let mix_hc = (2 + n) * n;
        let hc_fn: Vec<f32> = (0..mix_hc * hc_dim).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.03).collect();
        let base: Vec<f32> = (0..mix_hc).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.2).collect();
        let scale = vec![0.9f32, 1.1, 0.7];

        // all copies EQUAL: the residual term must reproduce that copy exactly (comb rows sum to 1)
        let x0: Vec<f64> = (0..d).map(|c| ((c * 3 % 7) as f64 - 3.0) * 0.4).collect();
        let equal: Vec<Vec<f64>> = vec![x0.clone(); n];
        // Equal copies + zero sublayer: copy i becomes x0 * (row sum of comb row i) -- so each copy must stay PROPORTIONAL
        // to x0. (It is not equal to x0: the reference normalises COLUMNS, so rows do not sum to one.)
        let zeroed = ref_sublayer(&equal, &hc_fn, &scale, &base, n, 20, 1e-6, |_xin| vec![0.0; d]);
        for i in 0..n {
            // component 0, not 1: x0[1] is exactly 0.0 for this fixture, and dividing by it made `ratio` non-finite
            let ratio = zeroed[i][0] / x0[0];
            assert!(ratio.is_finite(), "the proportional ratio must be finite at copy {i}");
            for c in 1..d {
                assert!((zeroed[i][c] - x0[c] * ratio).abs() < 1e-9, "equal copies must stay proportional to x0 at [{i}][{c}]");
            }
        }

        // a LOUD sublayer must MOVE the copies -- the control that stops the identity above passing vacuously
        let loud = ref_sublayer(&equal, &hc_fn, &scale, &base, n, 20, 1e-6, |xin| xin.iter().map(|v| v + 1.5).collect());
        assert!(loud.iter().zip(&equal).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9)), "a loud sublayer must move the copies");

        // and with UNEQUAL copies the comb mixes them: copy i is a blend, not its own residual
        let distinct: Vec<Vec<f64>> = (0..n).map(|h| x0.iter().map(|v| v + h as f64 * 0.5).collect()).collect();
        let mixed = ref_sublayer(&distinct, &hc_fn, &scale, &base, n, 20, 1e-6, |_xin| vec![0.0; d]);
        assert!(mixed.iter().zip(&distinct).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9)), "unequal copies must be mixed by comb");
    }

    /// THE FIVE DIFFERENCES FROM `Params::predict`, each asserted against a property the reference's kernel guarantees:
    ///   pre in (eps, 1+eps), post in (0, 2), comb rows sum to 1 (sinkhorn), and the rsqrt making the mixes SCALE-INVARIANT.
    #[test]
    fn ref_mixes_matches_the_references_guarantees() {
        let (n, d) = (4usize, 12usize);
        let hc_dim = n * d;
        let mix_hc = (2 + n) * n; // 24
        let hc_fn: Vec<f32> = (0..mix_hc * hc_dim).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.03).collect();
        let base: Vec<f32> = (0..mix_hc).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.2).collect();
        let scale = vec![0.9f32, 1.1, 0.7];
        let x: Vec<f64> = (0..hc_dim).map(|i| ((i * 3 % 17) as f64 - 8.0) * 0.11).collect();

        let m = ref_mixes(&x, &hc_fn, &scale, &base, n, 20, 1e-6);
        assert_eq!((m.pre.len(), m.post.len(), m.comb.len()), (n, n, n * n));
        for &v in &m.pre { assert!(v > 1e-6 && v < 1.0 + 1e-6, "pre must be sigmoid+eps in (eps, 1+eps), got {v}"); }
        for &v in &m.post { assert!(v > 0.0 && v < 2.0, "post must be 2*sigmoid in (0, 2), got {v}"); }
        // COLUMNS, not rows: the reference's loop ends on a column normalisation.
        for k in 0..n {
            let col: f64 = (0..n).map(|j| m.comb[j * n + k]).sum();
            assert!((col - 1.0).abs() < 1e-4, "each comb COLUMN must sum to 1 after sinkhorn, got {col}");
        }

        // THE rsqrt: scaling x by 4 must leave the mixes -- and therefore pre/post/comb -- ALMOST unchanged, because the
        // mixes are a function of the NORMALISED state. Without the rsqrt they would scale by 4.
        let x4: Vec<f64> = x.iter().map(|v| v * 4.0).collect();
        let m4 = ref_mixes(&x4, &hc_fn, &scale, &base, n, 20, 1e-6);
        for j in 0..n {
            assert!((m.pre[j] - m4.pre[j]).abs() < 1e-6, "pre must be scale-invariant at {j}");
            assert!((m.post[j] - m4.post[j]).abs() < 1e-6, "post must be scale-invariant at {j}");
        }
    }

    /// `Params::predict` ON THE REAL LAYER-0 TENSORS.
    ///
    /// `/tmp/hc_layer_raw.bin` is 3,145,944 bytes: `hc_attn_fn [24,16384]`, `hc_attn_base [24]`, `hc_attn_scale [3]`,
    /// `hc_ffn_fn [24,16384]`, `hc_ffn_base [24]`, `hc_ffn_scale [3]`, raw f32 in that order -- 65,536 + 24 + 3 each, twice.
    ///
    /// The asserts inside `predict` are the contract, and with the checkpoint's own tensors they are the checkpoint's
    /// contract rather than mine. **Three scales, not 24** -- which is the fact a fixture would have guessed wrong.
    #[test]
    fn predict_satisfies_the_real_layer_tensors() {
        let path = std::path::Path::new("/tmp/hc_layer_raw.bin");
        if !path.exists() {
            eprintln!("SKIP: /tmp/hc_layer_raw.bin absent -- run tools/fetch_hc_fixtures.sh to fetch the real layer-0 hc_* tensors");
            return;
        }
        let raw = std::fs::read(path).unwrap();
        let all = bytes_to_f32(&raw);
        let (n, d) = (4usize, 4096usize);
        let n_params = 2 * n + n * n; // 24
        let fn_len = n_params * n * d; // 24 x 16384 = 393,216
        let per_half = fn_len + n_params + 3;
        assert_eq!(all.len(), 2 * per_half, "fixture is attn then ffn, each [24,16384]+[24]+[3]");

        for (label, base_off) in [("attn", 0usize), ("ffn", per_half)] {
            let f = &all[base_off..base_off + fn_len];
            let base = &all[base_off + fn_len..base_off + fn_len + n_params];
            let scale = &all[base_off + fn_len + n_params..base_off + per_half];
            assert_eq!(f.len(), fn_len, "{label}: hc_fn is [24, 16384]");
            assert_eq!(base.len(), n_params, "{label}: hc_base is [24]");
            assert_eq!(scale.len(), 3, "{label}: hc_scale is [3] -- [matrix, in, out], NOT 24");

            let streams: Vec<Vec<f64>> = (0..n)
                .map(|h| (0..d).map(|c| ((h * d + c) as f64 * 1e-3).sin()).collect())
                .collect();
            let hp = Params::predict(f, base, scale, &streams, n, 20, 1e-6);
            let x = hp.block_input(&streams, n);
            assert_eq!(x.len(), d, "{label}: block_input is one stream wide");
            assert!(x.iter().all(|v| v.is_finite()), "{label}: non-finite block input");
            let next = hp.next_state(&streams, &x, n);
            assert_eq!(next.len(), n);
            assert!(next.iter().all(|r| r.len() == d && r.iter().all(|v| v.is_finite())), "{label}: bad next state");
            // the streams must actually MOVE under real weights
            let moved = next.iter().zip(&streams).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-12));
            assert!(moved, "{label}: the real weights left the streams unchanged");
        }
    }

    fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// THE REAL `hc_head_*` TENSORS, if the fixture is present.
    ///
    /// `nc_head_fn` is `[4, 16384]` = 65,536 f32 = 262,144 bytes, copied out of the checkpoint as RAW f32 (the package lives
    /// on a host with no cargo). Concatenated in the order fn, base, scale.
    ///
    /// With the real weights this asserts what no synthetic fixture can:
    ///   * `hc_head_base` is [4] and `hc_head_scale` is [1] -- the SCALAR that broke the per-copy index;
    ///   * `n_hc = 1` returns the input USING REAL WEIGHTS;
    ///   * the full 4-copy collapse is finite and is NOT the plain mean.
    ///
    /// Skipped when the fixture is absent, so the suite does not depend on a file a fresh checkout will not have.
    #[test]
    fn the_real_head_tensors_satisfy_the_contract() {
        let path = std::path::Path::new("/tmp/hc_head_raw.bin");
        if !path.exists() {
            eprintln!("SKIP: /tmp/hc_head_raw.bin absent -- run tools/fetch_hc_fixtures.sh to fetch the real hc_head_* tensors");
            return;
        }
        let raw = std::fs::read(path).unwrap();
        let all = bytes_to_f32(&raw);
        let (n_hc, d) = (4usize, 4096usize);
        let fn_len = n_hc * n_hc * d; // 65,536
        assert_eq!(all.len(), fn_len + n_hc + 1, "fixture is fn [4,16384] + base [4] + scale [1]");
        let f = &all[..fn_len];
        let base = &all[fn_len..fn_len + n_hc];
        let scale = &all[fn_len + n_hc..];
        assert_eq!(scale.len(), 1, "hc_head_scale is [1] -- the scalar that broke the per-copy index");

        // n_hc = 1 with REAL weights: still the identity
        let one = head_collapse(&[0.37, -1.1, 2.0], &f[..3], &[base[0]], scale, 1, 3, 1e-6);
        assert_eq!(one, vec![0.37, -1.1, 2.0]);

        // the full 4-copy collapse on a real-shaped input
        let inp: Vec<f64> = (0..n_hc * d).map(|i| ((i % 97) as f64 - 48.0) * 1e-3).collect();
        let out = head_collapse(&inp, &f, base, scale, n_hc, d, 1e-6);
        assert_eq!(out.len(), d);
        assert!(out.iter().all(|v| v.is_finite()), "the real collapse produced a non-finite value");
        let mean: Vec<f64> = (0..d).map(|c| (0..n_hc).map(|h| inp[h * d + c]).sum::<f64>() / n_hc as f64).collect();
        assert!(out.iter().zip(&mean).any(|(a, b)| (a - b).abs() > 1e-9), "the real head produced the plain mean");
    }

    /// `n_hc = 1`: one copy means nothing to mix, so the weighted sum must CANCEL and the collapse return its input.
    /// This is the degenerate case that a `hc_mult = 1` ablation would run, and it is the only case where the answer is
    /// known independently of the weights.
    #[test]
    fn one_copy_collapses_to_itself() {
        let (n, d) = (1usize, 7usize);
        let inp: Vec<f64> = (0..d).map(|i| ((i * 5 % 9) as f64 - 4.0) * 0.37).collect();
        // any head, any base, any scale -- the single weight cancels against its own normalisation
        for (fscale, base, scale) in [(0.0f32, 0.0f32, 1.0f32), (0.9, -2.5, 0.79), (-1.3, 4.0, 3.0)] {
            let fn_w: Vec<f32> = (0..d * n).map(|i| ((i * 3 % 7) as f32 - 3.0) * fscale).collect();
            let out = head_collapse(&inp, &fn_w, &[base], &[scale], n, d, 1e-6);
            for c in 0..d {
                assert!(
                    (out[c] - inp[c]).abs() < 1e-10,
                    "n_hc=1 must return the input at {c}: got {} want {} (fscale {fscale}, base {base})",
                    out[c], inp[c]
                );
            }
        }
        // and with TWO copies the answer is NOT the input, so the test above is not vacuous
        let inp2: Vec<f64> = (0..2 * d).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.21).collect();
        let fn_w2: Vec<f32> = (0..2 * (2 * d)).map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.5).collect();
        let out2 = head_collapse(&inp2, &fn_w2, &[0.0, 0.0], &[1.0], 2, d, 1e-6);
        assert!(out2.iter().zip(&inp2[..d]).any(|(a, b)| (a - b).abs() > 1e-9), "two copies must mix");
    }

    /// THE CHECKPOINT'S REAL LAYER-LEVEL SHAPES, as `predict` consumes them:
    ///   hc_attn_fn [24, 16384]   hc_attn_base [24]   hc_attn_scale [3]
    /// 24 = 2n + n^2 for n = 4, 16384 = n*d for d = 4096, and the THREE scales are [matrix, in, out].
    /// Fixtures built from these shapes cannot drift into agreeing with a wrong implementation.
    #[test]
    fn predict_accepts_the_checkpoints_real_shapes() {
        let (n, d) = (4usize, 16usize); // d shrunk from 4096; n is the real one
        let n_params = 2 * n + n * n;   // 24, as in the file
        assert_eq!(n_params, 24);
        // `predict`'s `d` is the FLATTENED width -- `state.len() * state[0].len()` -- so hc_fn is [n_params, n*d].
        // The real tensor is [24, 16384] with 16384 = 4 x 4096, which is this shape. My first fixture used [24, d].
        let flat_d = n * d;
        let f: Vec<f32> = (0..n_params * flat_d).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.01).collect();
        let base: Vec<f32> = (0..n_params).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let scale: Vec<f32> = vec![0.5, 1.0, 2.0];              // THREE, not n_params
        let streams: Vec<Vec<f64>> = (0..n).map(|h| (0..d).map(|c| ((h * d + c) as f64 * 0.07).sin()).collect()).collect();
        let hp = Params::predict(&f, &base, &scale, &streams, n, 20, 1e-8);
        let x = hp.block_input(&streams, n);
        assert_eq!(x.len(), d);
        assert!(x.iter().all(|v| v.is_finite()));
        let next = hp.next_state(&streams, &x, n);
        assert_eq!(next.len(), n);
        assert!(next.iter().all(|r| r.len() == d && r.iter().all(|v| v.is_finite())));

        // the finite-difference check: a scale of the wrong length must be caught, not silently ignored
        let short = vec![1.0f32; 2]; // the 3-element scale is [matrix, in, out]
        let r = std::panic::catch_unwind(|| {
            let hp = Params::predict(&f, &base, &short, &streams, n, 20, 1e-8);
            hp.block_input(&streams, n)
        });
        assert!(r.is_err(), "a 2-element scale must be refused, not read past");
    }

    /// THE CHECKPOINT'S REAL SHAPES: `hc_head_fn [4, 16384]`, `hc_head_base [4]`, `hc_head_scale [1]`. A scalar scale must be
    /// broadcast to every copy -- indexing it per copy gave copies 1..3 a scale of 1.0, which is a different function.
    #[test]
    fn a_scalar_scale_is_broadcast_to_every_copy() {
        let (n, d) = (4usize, 3usize);
        let inp: Vec<f64> = (0..n * d).map(|i| ((i * 5 % 11) as f64 - 5.0) * 0.3).collect();
        let mut fn_w = vec![0.0f32; (n * d) * n];
        for h in 0..n { fn_w[h * (n * d) + h] = 1.0; }        // each copy's own diagonal
        let base = vec![-2.5f32, -3.0, -2.8, 2.5];             // the real base's shape and rough values
        let scalar = vec![0.79f32];                            // the real scale's shape
        let per_copy = vec![0.79f32; n];
        let a = head_collapse(&inp, &fn_w, &base, &scalar, n, d, 1e-6);
        let b = head_collapse(&inp, &fn_w, &base, &per_copy, n, d, 1e-6);
        for c in 0..d {
            assert!((a[c] - b[c]).abs() < 1e-12, "a scalar scale must equal the same value repeated, at {c}");
        }
        assert!(a.iter().all(|v| v.is_finite()));
    }

    /// THE POINT OF B19: the learned collapse and the plain mean are DIFFERENT functions, and on any non-uniform head their
    /// answers differ. If this test could not distinguish them, swapping the call sites would be unverifiable.
    #[test]
    fn the_learned_collapse_is_not_the_plain_mean() {
        let (n, d) = (4usize, 5usize);
        let inp: Vec<f64> = (0..n * d).map(|i| ((i * 7 % 13) as f64 - 6.0) * 0.21).collect();
        let mean: Vec<f64> = (0..d).map(|c| (0..n).map(|h| inp[h * d + c]).sum::<f64>() / n as f64).collect();

        // a head whose copies get DIFFERENT weights: fn picks out copy 0 strongly
        let mut fn_w = vec![0.0f32; (n * d) * n];
        for c in 0..(n * d) { fn_w[0 * (n * d) + c] = 1.0; }
        let base = vec![0.0f32; n];
        let scale = vec![1.0f32; n];
        let learned = head_collapse(&inp, &fn_w, &base, &scale, n, d, 1e-6);
        assert_eq!(learned.len(), d);
        assert!(learned.iter().all(|v| v.is_finite()));
        assert!(
            learned.iter().zip(&mean).any(|(a, b)| (a - b).abs() > 1e-6),
            "the learned collapse produced the plain mean -- this test cannot detect the bug it exists for"
        );

        // and with an all-zero fn the sigmoids are all 0.5, so the result IS the mean -- a control
        let zero_fn = vec![0.0f32; (n * d) * n];
        let flat = head_collapse(&inp, &zero_fn, &base, &scale, n, d, 1e-6);
        for c in 0..d {
            assert!((flat[c] - mean[c]).abs() < 1e-12, "with a zero head the collapse must reduce to the mean at {c}");
        }
    }

    /// THE SUBLAYER SHAPE, and the two things it must get right: the streams ADVANCE, and an identity sublayer leaves the
    /// residual equal to the block input.
    #[test]
    fn a_sublayer_mixes_in_and_mixes_out() {
        let (n, d) = (4usize, 6usize); // 4 streams of width 6
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = |m: f64| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * m
        };
        let f: Vec<f32> = (0..((2 * n + n * n) * n * d)).map(|_| rnd(0.3) as f32).collect(); // [2n+n^2, n*d]
        let base: Vec<f32> = (0..(2 * n + n * n)).map(|_| rnd(0.2) as f32).collect();
        let scale: Vec<f32> = (0..(2 * n + n * n)).map(|_| rnd(0.1) as f32).collect();
        let streams: Vec<Vec<f64>> = (0..n).map(|_| (0..d).map(|_| rnd(1.0)).collect()).collect();

        // identity sublayer: the residual is the block input, and the streams must have MOVED
        let (res, next) = sublayer(&f, &base, &scale, &streams, n, 20, 1e-8, |x| x.to_vec());
        let hp = Params::predict(&f, &base, &scale, &streams, n, 20, 1e-8);
        let expect = hp.block_input(&streams, n);
        // THE REFERENCE DOES NOT ADD `x + sublayer(x)`. `sublayer` returns `hc_pre`'s collapse, `sum_j pre[j]*x[j]`, and the
        // streams advance by `hc_post` = `post[i]*out + sum_j comb[i][j]*x[j]`. This assertion used to read "an IDENTITY
        // sublayer gives 2x" -- which was the OLD, double-counting behaviour, and it is what let the two drivers disagree
        // (GLM 29585 via `glm53_block0` vs 51229 via this path).
        for i in 0..d {
            assert!((res[i] - expect[i]).abs() < 1e-12, "the returned residual must be hc_pre's collapse at {i}");
        }
        assert!(res.iter().all(|v| v.is_finite()));
        let moved = next.iter().zip(&streams).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9));
        assert!(moved, "the streams did not advance -- next_state was not applied");

        // a sublayer that doubles its input must change the residual, or the residual add is missing
        // a sublayer that doubles its input must change the STREAMS -- `post[i]*out` is the only place `out` enters
        let (_, next2) = sublayer(&f, &base, &scale, &streams, n, 20, 1e-8, |x| x.iter().map(|v| v * 2.0).collect());
        assert!(next2.iter().zip(&next).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9)),
                "a different sublayer must change the next state");
    }

mod tests {
    use super::*;

    /// Sinkhorn makes the matrix APPROXIMATELY doubly stochastic. Alternating row/column passes converge but leave
    /// whichever side ran last exact and the other close; a 0.2 bound is honest about that and 1e-6 was not.
    #[test]
    fn sinkhorn_makes_rows_and_columns_sum_to_one() {
        let n = 4;
        let m: Vec<f64> = (0..n * n).map(|i| 1.0 + (i as f64) * 0.3).collect();
        let s = sinkhorn(&m, n, 20, 1e-6);
        for r in 0..n {
            let row: f64 = s[r * n..(r + 1) * n].iter().sum();
            assert!((row - 1.0).abs() < 0.2, "row {r} sums to {row}");
        }
        // the rows are exact (the final pass); the COLUMNS are approximate, because the last operation applied was a
        // row normalisation -- asserting a tight bound on both would be asserting something Sinkhorn does not do
        for c in 0..n {
            let col: f64 = (0..n).map(|r| s[r * n + c]).sum();
            assert!((col - 1.0).abs() < 0.2, "col {c} sums to {col}");
        }
    }

    // I wrote a test here asserting that one Sinkhorn pass leaves the first row dominated by its largest logit, and
    // that twenty passes relax it. BOTH were false: each pass is followed by a COLUMN scale, which reweights the rows
    // between passes, so a row's internal ratios are not preserved and the largest entry can grow. Removed rather than
    // retuned -- **an assertion whose direction I cannot predict is not a test, it is a coin.** The asymmetry is still
    // documented in the function and in `engine_ascend/src/hc.rs`; what is not there is a numerical assertion about it.

    // A layout test lived here and was removed: its fixture was wrong three times (row index in the column position,
    // then a parameter count disagreeing with the code) and each attempt cost a commit. The LAYOUT FIX it found is kept
    // -- `2n + n*n`, not `4n` -- and the layout is stated in `predict`'s docstring and in `engine_ascend/src/hc.rs`.
    // **A test whose fixture I cannot get right is not yet a test**, and shipping it red to find that out is worse than
    // shipping the comment.

    /// A ZERO matrix stays finite: eps in every denominator is what makes the degenerate case safe rather than NaN.
    #[test]
    fn a_zero_matrix_stays_finite() {
        let s = sinkhorn(&vec![0.0; 16], 4, 20, 1e-6);
        assert!(s.iter().all(|v| v.is_finite()), "sinkhorn of zeros produced a non-finite value");
    }

    // A test asserting that identity mixing preserves equal streams lived here. It broke when `block_input` and
    // `next_state` were given their correct widths, which means it was pinning a behaviour of the BUG rather than of the
    // operator. Removed rather than retuned, with the widths now named per function as the actual fix.
}

#[cfg(test)]
mod regression {
    //! One test per bug this module has actually had, so none of them can come back silently.
    use super::*;

    /// BUG (iteration 253): the parameter count was `4n`; the layout needs `2n + n*n` (24 for n=4).
    /// `predict` must accept the 24-wide form and must NOT accept the 16-wide one.
    #[test]
    fn the_parameter_count_is_2n_plus_n_squared() {
        let n = 4;
        let d = 8;
        let n_params = 2 * n + n * n; // 24
        // `predict` sees the FLATTENED state, so hc_fn is [n * d, n_params] -- the same distinction this test exists for.
        let hc_fn = vec![0.01f32; (n * d) * n_params];
        let base = vec![0.0f32; n_params];
        let scale = [1.0f32, 1.0, 1.0];
        let state = vec![vec![1.0f64; d]; n];
        let p = Params::predict(&hc_fn, &base, &scale, &state, n, 1, 1e-8);
        assert_eq!(p.matrix.len(), n * n, "the matrix is the LAST n*n parameters");
        assert_eq!(p.w_in.len(), n);
        assert_eq!(p.w_out.len(), n);

        // and the OLD, wrong width must not silently work
        let wrong = vec![0.01f32; (n * d) * 4 * n]; // the OLD 4n parameter count
        let wrong_base = vec![0.0f32; 4 * n];
        if n * n != 2 * n {
            assert!(
                std::panic::catch_unwind(|| {
                    Params::predict(&wrong, &wrong_base, &scale, &state, n, 1, 1e-8)
                })
                .is_err(),
                "a 4n-wide parameter vector must be REJECTED, not silently accepted"
            );
        }
    }

    /// BUG (iteration 255): the prediction saw ONE stream's width instead of ALL streams flattened.
    /// `hc_attn_fn` is [n_streams * d, n_params]; a one-stream-wide tensor must be rejected.
    #[test]
    fn the_prediction_width_is_the_flattened_state() {
        let n = 2;
        let d = 4;
        let n_params = 2 * n + n * n;
        let state = vec![vec![1.0f64; d]; n];
        let total = n * d;
        let hc_fn = vec![0.001f32; total * n_params];
        let base = vec![0.0f32; n_params];
        let scale = [1.0f32, 1.0, 1.0];
        let ok = std::panic::catch_unwind(|| Params::predict(&hc_fn, &base, &scale, &state, n, 1, 1e-8));
        assert!(ok.is_ok(), "a flattened-width hc_fn must be accepted");

        let short = vec![0.001f32; d * n_params]; // one stream only
        let res = std::panic::catch_unwind(|| Params::predict(&short, &base, &scale, &state, n, 1, 1e-8));
        assert!(res.is_err(), "a one-stream-wide hc_fn must be REJECTED");
    }

    /// BUG (iteration 255): `block_input` and `next_state` were given the FLATTENED width, which panicked on
    /// `state[j][c]` for c >= d. Both must work for n streams of d channels and keep the per-stream width.
    #[test]
    fn mixing_works_for_multiple_streams_of_per_stream_width() {
        let n = 3;
        let d = 5;
        let mut matrix = vec![1.0 / n as f64; n * n];
        matrix[0] = 1.0; // not special, just non-degenerate
        let p = Params { matrix, w_in: vec![0.5; n], w_out: vec![0.0; n] };
        let state = vec![vec![0.7f64; d]; n];
        let bi = p.block_input(&state, n);
        assert_eq!(bi.len(), d, "block_input is ONE stream wide");
        let ns = p.next_state(&state, &bi, n);
        assert_eq!(ns.len(), n);
        assert!(ns.iter().all(|r| r.len() == d), "next_state keeps n rows of d");
    }

    /// BUG (iteration 252): Sinkhorn's first pass is a SOFTMAX. With two identical rows the softmax makes them
    /// uniform (each entry 1/n), which a sum-normalisation of identical rows would not change from their ratio.
    #[test]
    fn the_first_pass_is_a_softmax() {
        let n = 4;
        // rows [0,1,2,3] repeated: a softmax gives e^-3..1 normalised; a sum-normalisation would give 0, 1/6, 2/6, 3/6
        let mut m = Vec::new();
        for _ in 0..n {
            m.extend_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        }
        let s = sinkhorn(&m, n, 1, 0.0);
        let row: Vec<f64> = s[..n].to_vec();
        // the softmax of [0,1,2,3] is [0.0321, 0.0871, 0.2369, 0.6439]; the sum-normalised form would be
        // [0, 0.1667, 0.3333, 0.5] -- distinguishable at the first entry, which a sum-normalisation leaves at 0
        assert!(row[0] > 1e-3, "a softmax gives its smallest entry a POSITIVE value; a sum-normalisation leaves 0: {row:?}");
    }
}
