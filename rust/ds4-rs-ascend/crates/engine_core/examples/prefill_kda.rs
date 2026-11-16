//! THE CAUSALITY INVARIANT, on GLM 5.3's real KDA blocks.
//!
//! With two tokens, POSITION 0's residual must be IDENTICAL to the single-token run of token 0, because attention
//! and the delta recurrence are causal. A prefill that leaks future tokens -- the classic off-by-one in a conv's
//! history window or in the recurrence's ordering -- fails this, and nothing else would notice.
//!
//! usage: prefill_kda <gguf>

use engine_core::gguf::GgufFile;
use engine_core::glm5_ref::DefaultsGlm5;
use engine_core::glm5_stream::ShardedGguf;
use engine_core::kda_block::{block, gated_rms_norm, BlockParams};
use engine_core::tokenizer::Vocab;
use std::path::PathBuf;

struct Ctx {
    d: usize,
    heads: usize,
    head_dim: usize,
    eps: f64,
}

fn embed(g: &ShardedGguf, tok: u32, d: usize) -> anyhow::Result<Vec<f64>> {
    let emb = g.dequant_name("token_embd.weight")?;
    let cols = emb.len() / d;
    let id = tok as usize;
    anyhow::ensure!(id < cols, "token {id} outside the {cols}-wide embedding");
    let v: Vec<f64> = (0..d).map(|c| emb[c * cols + id] as f64).collect();
    if std::env::var_os("KDA_INTER").is_some() {
        println!("EMB prefill: token {id}, d {d}, cols {cols}, len {}, first {:?}", emb.len(), &v[..3]);
    }
    Ok(v)
}


/// ONE MLA BLOCK over N positions, causally: positions `s <= t` only.
fn mla_block_n(g: &ShardedGguf, b: usize, c: &Ctx, xs: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = xs.len();
    // EVERY MLA WIDTH IS DERIVED -- the decode driver had these removed in rounds 291-292 and the prefill never did, which
    // is why the prefill could not serve Kimi (B15).
    //   hd  from `attn_k_b`: [lat, hd, heads] contiguous, so hd = len / (lat * heads)
    //   lat from `attn_kv_a_norm`, which is the width the k/v projections consume (kv_a_mqa emits MORE: Kimi 576 = 512+64)
    //   q_out from `attn_q_b`
    let heads = c.heads;
    let mv = |w: &[f32], xv: &[f64], k: usize, nn: usize| -> Vec<f64> {
        (0..nn).map(|r| (0..k).map(|cc| w[r * k + cc] as f64 * xv[cc]).sum()).collect()
    };
    let norm_w = engine_core::model_names::dequant_any(g, b, "norm")?;
    let nw: Vec<f64> = norm_w.iter().map(|v| *v as f64).collect();
    let wqa = engine_core::model_names::dequant_any(g, b, "q_a")?;
    let wqb = engine_core::model_names::dequant_any(g, b, "q_b")?;
    let wkva = engine_core::model_names::dequant_any(g, b, "kv_a_mqa")?;
    let wkb = engine_core::model_names::dequant_any(g, b, "k_b")?;
    let wvb = engine_core::model_names::dequant_any(g, b, "v_b")?;
    let wo = engine_core::model_names::dequant_any(g, b, "output")?;
    let lat = engine_core::model_names::dequant_any(g, b, "kv_a_norm")?.len();
    anyhow::ensure!(wkb.len() % (lat * heads) == 0, "k_b is {} not {lat} x hd x {heads}", wkb.len());
    let hd = wkb.len() / (lat * heads);
    let mut qs = Vec::with_capacity(n);
    let mut ks = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    for t in 0..n {
        let xin = gated_rms_norm(&xs[t], &nw, &vec![1.0; c.d], c.eps);
        // THE LATENT IS THE TENSOR'S WIDTH, NOT 1536. Derived so the prefill driver does not carry a GLM constant that
        // the decode driver had to have removed (round 291) -- the two drivers agree on GLM only because both were right,
        // and one hardcoded width is all it takes for them to stop agreeing on a second model.
        let q_lora = wqa.len() / c.d;
        let qa = mv(&wqa, &xin, c.d, q_lora);
        qs.push(mv(&wqb, &qa, q_lora, heads * hd));
        let kva = mv(&wkva, &xin, c.d, lat);
        let mut k = vec![0.0f64; heads * hd];
        let mut v = vec![0.0f64; heads * hd];
        for h in 0..heads {
            for r in 0..hd {
                let (mut ak, mut av) = (0.0, 0.0);
                for cc in 0..lat {
                    ak += wkb[(h * hd + r) * lat + cc] as f64 * kva[cc];
                    av += wvb[(h * hd + r) * lat + cc] as f64 * kva[cc];
                }
                k[h * hd + r] = ak;
                v[h * hd + r] = av;
            }
        }
        ks.push(k);
        vs.push(v);
    }
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let mut o = vec![0.0f64; heads * hd];
        for h in 0..heads {
            let mut sc = Vec::with_capacity(t + 1);
            for s in 0..=t {
                let mut acc = 0.0;
                for r in 0..hd { acc += qs[t][h * hd + r] * ks[s][h * hd + r]; }
                sc.push(acc / (hd as f64).sqrt());
            }
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for s in 0..=t {
                let w = ex[s] / z;
                for r in 0..hd { o[h * hd + r] += w * vs[s][h * hd + r]; }
            }
        }
        out.push(mv(&wo, &o, heads * hd, c.d));
    }
    Ok(out)
}

/// Run `blocks` blocks over `xs` (one row per token), handling each block by the schedule.
fn run(g: &ShardedGguf, c: &Ctx, plan: &[engine_core::glm53_plan::BlockPlan], blocks: usize, xs: Vec<Vec<f64>>, scale: f64) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = xs.len();
    let mut x = xs;
    // HYPER-CONNECTIONS, the same state decode carries (B14). One token at a time, because `hc_attn_fn` is [4*d, 24]: it is
    // predicted from a FLATTENED 4-stream state of width d, so flattening across TOKENS as well would give [4*n*d, 24] and
    // `Params::predict` refuses it. That is why this is per token and not one call for the batch.
    let hc_n: usize = if std::env::var_os("GLM53_HC").is_some() { 4 } else { 1 };
    let mut hc: Vec<Vec<Vec<f64>>> = (0..n).map(|t| vec![x[t].clone(); hc_n]).collect();
    if hc_n > 1 {
        println!("hyper-connections: {hc_n} streams x {n} token(s), 20 Sinkhorn iterations");
    }
    for b in 0..blocks {
        // THE STREAMS BECOME THE BLOCK'S INPUT.
        //
        // A BLOCK-INPUT MIX, not a sublayer: this only produces the block's input, so it calls predict+block_input
        // directly. The streams are advanced after the attention residual, per arm, below.
        //
        // B23: ONE `predict` PER TOKEN PER BLOCK for the attention sublayer, held in `attn_hps`. The MLA arm used to
        // recompute exactly this -- the same three tensors, the same unmodified `hc[t]` -- and the KDA arm needed a
        // third `predict` for its advance. Nothing writes `hc[t]` between the collapse and the advance, so one
        // `Params` per token serves both; that is also why the duplicates could never have disagreed, and so why
        // nothing caught them.
        let mut attn_hps: Vec<engine_core::hyper_connection::Params> = Vec::new();
        if hc_n > 1 {
            let f = g.dequant_name(&format!("blk.{b}.hc_attn_fn.weight"))?;
            let base = g.dequant_name(&format!("blk.{b}.hc_attn_base.weight"))?;
            let sc = g.dequant_name(&format!("blk.{b}.hc_attn_scale.weight"))?;
            for t in 0..n {
                let hp = engine_core::hyper_connection::Params::predict(&f, &base, &sc, &hc[t], hc_n, 20, 1e-8);
                x[t].copy_from_slice(&hp.block_input(&hc[t], hc_n));
                attn_hps.push(hp);
            }
        }
        // B22: the KDA arm collapses the streams (block input, above) but never advanced them -- only the MLA arm did.
        let kda_x0: Vec<Vec<f64>> = x.clone();
        let mut is_kda = false;
        match plan[b].attn {
            engine_core::glm53_plan::Attn::Mla => {
                // B22: THE ATTENTION MUST RUN ON `hc_pre(streams)`, NOT ON `x`. The old version called `mla_block_n` on `x`
                // first and then advanced the streams with a ZERO sublayer output, so `hc_post`'s `post` term was multiplied
                // by zero and `comb` mixed the residual forward alone. Reference order: hc_pre -> attention -> hc_post.
                if hc_n > 1 {
                    // B23: `x` IS ALREADY `hc_pre(streams)` -- the hoisted block above wrote it. This arm used to build
                    // an identical `xin` from the same state with its own predict+block_input.
                    let outs = mla_block_n(g, b, c, &x)?;
                    println!("MLA-TENSOR prefill blk{b}: {:?}", &outs[0][..3]);
                    for t in 0..n {
                        // hc_post takes the attention OUTPUT, not the residual.
                        //
                        // B23: AND NO RECOLLAPSE HERE. This used to predict a third time on the NEW streams and write
                        // the collapse into `x[t]` "for the FFN" -- but the FFN's own `hc_ffn_*` mix below overwrites
                        // `x[t]` before anything reads it, with the FFN's coefficients, which are the correct ones.
                        // The attention-weight collapse was computed and discarded unread, on every MLA block.
                        hc[t] = attn_hps[t].next_state(&hc[t], &outs[t], hc_n);
                    }
                } else {
                    let outs = mla_block_n(g, b, c, &x)?;
                    println!("MLA-TENSOR prefill blk{b}: {:?}", &outs[0][..3]);
                    for t in 0..n {
                        for i in 0..c.d {
                            x[t][i] += outs[t][i];
                        }
                    }
                }
            }
            engine_core::glm53_plan::Attn::Kda => {
                is_kda = true;
                let norm_w = engine_core::model_names::dequant_any(g, b, "norm")?;
                let nw: Vec<f64> = norm_w.iter().map(|v| *v as f64).collect();
                let mut q = Vec::with_capacity(n);
                let mut k = Vec::with_capacity(n);
                let mut v = Vec::with_capacity(n);
                let mut gate = Vec::with_capacity(n);
                for t in 0..n {
                    let xin = gated_rms_norm(&x[t], &nw, &vec![1.0; c.d], c.eps);
                    // THE PROJECTIONS GO THROUGH THE LIBRARY RESOLVER TOO. These three were formatted differently from the
                    // 17 that were routed last round, which is why the prefill still asked for `blk.0.kda_q.weight` after
                    // Kimi's config and schedule both worked.
                    for (dst, base) in [(&mut q, "q"), (&mut k, "k"), (&mut v, "v")] {
                        let w = engine_core::model_names::dequant_any(g, b, base)?;
                        dst.push((0..c.heads * c.head_dim)
                            .map(|r| (0..c.d).map(|cc| w[r * c.d + cc] as f64 * xin[cc]).sum())
                            .collect::<Vec<f64>>());
                    }
                    // THE REAL f-GATE per position: dt = f_b(f_a(normed x)) + dt_bias
                    let fa = engine_core::model_names::dequant_any(g, b, "f_a")?;
                    let fb = engine_core::model_names::dequant_any(g, b, "f_b")?;
                    let hid: Vec<f64> = (0..128).map(|r| (0..c.d).map(|cc| fa[r * c.d + cc] as f64 * xin[cc]).sum()).collect();
                    gate.push((0..c.heads * c.head_dim)
                        .map(|r| (0..128).map(|cc| fb[r * 128 + cc] as f64 * hid[cc]).sum())
                        .collect::<Vec<f64>>());
                }
                let flat = |rows: &Vec<Vec<f64>>| -> Vec<f64> { rows.iter().flatten().copied().collect() };
                let params = BlockParams {
                    // THE REAL CONSTANTS, the same ones the decode example loads: a_log per head.
                    heads: c.heads, head_dim: c.head_dim,
                    a_log: {
                        let raw = engine_core::model_names::dequant_any(g, b, "a_log")?;
                        anyhow::ensure!(raw.len() == c.heads, "kda_a_log is {} not {}", raw.len(), c.heads);
                        raw.iter().map(|v| *v as f64).collect()
                    },
                    // THE REAL PER-CHANNEL CONVS. `kda_{q,k,v}_conv` are [4, 1, 8192]: four taps for EACH of the
            // 8192 channels. Loaded from the file rather than shared.
            conv_q: load_conv(g, b, "kda_q_conv", c.heads * c.head_dim)?,
            conv_k: load_conv(g, b, "kda_k_conv", c.heads * c.head_dim)?,
            conv_v: load_conv(g, b, "kda_v_conv", c.heads * c.head_dim)?,
                    dt_bias: {
                        let raw = engine_core::model_names::dequant_any(g, b, "dt_bias")?;
                        raw.iter().map(|v| *v as f64).collect()
                    },
                    // THE REAL beta, as a per-head projection of the FIRST position's input (the decode path's
                    // reading), and the real output gate for the block.
                    beta: {
                        let raw = engine_core::model_names::dequant_any(g, b, "beta")?;
                        let xin0 = gated_rms_norm(&x[0], &nw, &vec![1.0; c.d], c.eps);
                        (0..c.heads)
                            .map(|h| {
                                // THE NORMALISED input, matching the decode driver. Using the raw residual here made the two drivers
                                // differ by 0.1% at the MLA output, which compounds to a different token over six blocks.
                                let z: f64 = (0..c.d).map(|cc| raw[cc * c.heads + h] as f64 * xin0[cc]).sum();
                                1.0 / (1.0 + (-z).exp())
                            })
                            .collect()
                    },
                    eps: c.eps,
                };
                let onw = vec![1.0f64; c.head_dim];
                // THE OUTPUT GATE COMES IN TWO SHAPES, and the shapes say which: GLM factorises it (`kda_g_a` + `kda_g_b`,
                // low-rank) and Kimi applies ONE matrix (`ssm_g`, direct). Same difference as the decode driver's B14, in a
                // second place -- which is exactly what "two copies of the naming" costs.
                let ogate = if let Ok(wg) = engine_core::model_names::dequant_any(g, b, "g") {
                    anyhow::ensure!(wg.len() == c.heads * c.head_dim * c.d, "direct gate is {} not {}x{}", wg.len(), c.heads * c.head_dim, c.d);
                    let mut v = Vec::with_capacity(n * c.heads * c.head_dim);
                    for t in 0..n {
                        let xin = gated_rms_norm(&x[t], &nw, &vec![1.0; c.d], c.eps);
                        for r in 0..c.heads * c.head_dim {
                            let z: f64 = (0..c.d).map(|cc| wg[r * c.d + cc] as f64 * xin[cc]).sum();
                            v.push(1.0 / (1.0 + (-z).exp()));
                        }
                    }
                    v
                } else {
                    // THE REAL output gate per position (the low-rank pair)
                    let ga = engine_core::model_names::dequant_any(g, b, "g_a")?;
                    let gb = engine_core::model_names::dequant_any(g, b, "g_b")?;
                    let mut v = Vec::with_capacity(n * c.heads * c.head_dim);
                    for t in 0..n {
                        let xin = gated_rms_norm(&x[t], &nw, &vec![1.0; c.d], c.eps);
                        let hid: Vec<f64> = (0..128).map(|r| (0..c.d).map(|cc| ga[r * c.d + cc] as f64 * xin[cc]).sum()).collect();
                        for r in 0..c.heads * c.head_dim {
                            let z: f64 = (0..128).map(|cc| gb[r * 128 + cc] as f64 * hid[cc]).sum();
                            v.push(1.0 / (1.0 + (-z).exp()));
                        }
                    }
                    v
                };
                if std::env::var_os("KDA_INTER").is_some() {
                    let xinp = gated_rms_norm(&x[0], &nw, &vec![1.0; c.d], c.eps);
                    println!("KDA-INTER blk{b} prefill: x0={:?} xin0={:?} q0={:?} beta0={:?}",
                        x[0][0], xinp[0], q[0][0], params.beta[0]);
                }
                if std::env::var_os("GATE_DBG").is_some() {
                    println!("GATE prefill blk{b}: gate {:?} ogate {:?}", &gate[0][..3], &ogate[..3]);
                }
                if std::env::var_os("CONST_DBG").is_some() {
            println!("CONST prefill blk{b}: conv_q[0] {:?} a_log[0] {:?} dt_bias[0] {:?}",
                params.conv_q[0], params.a_log[0], params.dt_bias[0]);
        }
        let out = block(&params, &flat(&q), &flat(&k), &flat(&v), &flat(&gate), n, &onw, &ogate);
                if std::env::var_os("CONST_DBG").is_some() {
                    println!("CONST prefill blk{b}: block_out {:?}", &out[..3]);
                }
                let wo = engine_core::model_names::dequant_any(g, b, "output")?;
                for t in 0..n {
                    let o = &out[t * c.heads * c.head_dim..(t + 1) * c.heads * c.head_dim];
                    for r in 0..c.d {
                        let mut acc = 0.0;
                        for cc in 0..o.len() { acc += wo[r * o.len() + cc] as f64 * o[cc]; }
                        x[t][r] += acc;
                    }
                }
            }
        }
        // B22: the KDA attention's OUTPUT is `x - kda_x0`; advance the streams with it.
        //
        // B23: with `attn_hps` hoisted, and without the recollapse the FFN's own mix overwrote -- the comment here used
        // to say "then re-collapse for the FFN", which was the stale half of that discarded write.
        if is_kda && hc_n > 1 {
            for t in 0..n {
                let out: Vec<f64> = x[t].iter().zip(&kda_x0[t]).map(|(p, q)| p - q).collect();
                hc[t] = attn_hps[t].next_state(&hc[t], &out, hc_n);
            }
        }
        // B22: THE FFN HAD NO HYPER-CONNECTION HERE EITHER -- same wrap as the decode driver, same differencing trick.
        let mut ffn_hp: Vec<Option<engine_core::hyper_connection::Params>> = Vec::new();
        let mut ffn_x0: Vec<Vec<f64>> = Vec::new();
        if hc_n > 1 {
            let f = g.dequant_name(&format!("blk.{b}.hc_ffn_fn.weight"))?;
            let base = g.dequant_name(&format!("blk.{b}.hc_ffn_base.weight"))?;
            let sc = g.dequant_name(&format!("blk.{b}.hc_ffn_scale.weight"))?;
            for t in 0..n {
                let hp = engine_core::hyper_connection::Params::predict(&f, &base, &sc, &hc[t], hc_n, 20, 1e-8);
                x[t] = hp.block_input(&hc[t], hc_n);
                ffn_x0.push(x[t].clone());
                ffn_hp.push(Some(hp));
            }
        }
        // the FFN, per position
        match plan[b].ffn {
            engine_core::glm53_plan::Ffn::Dense => {
                let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
                let fnw: Vec<f64> = fnorm_w.iter().map(|v| *v as f64).collect();
                let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
                let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
                let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
                let inter = wg.len() / c.d;
                for t in 0..n {
                    let fin = gated_rms_norm(&x[t], &fnw, &vec![1.0; c.d], c.eps);
                    let gv: Vec<f64> = (0..inter).map(|r| (0..c.d).map(|cc| wg[r * c.d + cc] as f64 * fin[cc]).sum()).collect();
                    let uv: Vec<f64> = (0..inter).map(|r| (0..c.d).map(|cc| wu[r * c.d + cc] as f64 * fin[cc]).sum()).collect();
                    let act: Vec<f64> = gv.iter().zip(&uv).map(|(a, bb)| (a / (1.0 + (-a).exp())) * bb).collect();
                    for r in 0..c.d {
                        let mut acc = 0.0;
                        for cc in 0..act.len() { acc += wd[r * act.len() + cc] as f64 * act[cc]; }
                        x[t][r] += acc;
                    }
                }
            }
            engine_core::glm53_plan::Ffn::Moe => {
                // THE SAME LIBRARY FUNCTION the decode driver calls. It was missing here entirely until iteration 213,
                // which is what made the two drivers disagree at block 3.
                for t in 0..n {
                    let (dn, _used) = engine_core::glm53_ffn::moe_ffn(g, scale, b, &x[t], c.d, c.eps)?;
                    for i in 0..c.d { x[t][i] += dn[i]; }
                }
            }
            _ => {}
        }
        for t in 0..n {
            if let Some(hp) = ffn_hp.get(t).and_then(|o| o.as_ref()) {
                let ffn_out: Vec<f64> = x[t].iter().zip(&ffn_x0[t]).map(|(a, b)| a - b).collect();
                hc[t] = hp.next_state(&hc[t], &ffn_out, hc_n);
            }
        }
        if b <= 4 { println!("BLK{b}-TENSOR prefill(after attn+FFN): {:?}", &x[0][..3]); }
    }
    Ok(x)
}


/// `kda_*_conv` is [4, 1, C]: the FOUR taps of each of C channels, stored tap-major.
fn load_conv(g: &ShardedGguf, b: usize, which: &str, channels: usize) -> anyhow::Result<Vec<[f64; 4]>> {
    let raw = engine_core::model_names::dequant_any(g, b, &which.replace("kda_", ""))?;
    anyhow::ensure!(raw.len() == 4 * channels, "{which} is {} not 4 x {channels}", raw.len());
    Ok((0..channels)
        .map(|c| [raw[c] as f64, raw[channels + c] as f64, raw[2 * channels + c] as f64, raw[3 * channels + c] as f64])
        .collect())
}

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: prefill_kda <gguf>"))?);
    let g0 = GgufFile::open(&path)?;
    // THE KIMI CONFIG PATH. Without it the prefill cannot open Kimi at all ("missing indexer.head_count"), which is why the
    // campaign's cross-check was GLM-only and Kimi's tokens came from a single implementation (B15).
    let cfg = if std::env::var_os("DS4_KIMI").is_some() {
        let u = |suffix: &str| -> u32 {
            g0.meta.iter().find(|(k, _)| k.ends_with(suffix))
                .and_then(|(_, v)| match v { engine_core::gguf::MetaValue::U32(x) => Some(*x), _ => None })
                .unwrap_or(0)
        };
        let f = |suffix: &str| -> f32 {
            g0.meta.iter().find(|(k, _)| k.ends_with(suffix))
                .and_then(|(_, v)| match v { engine_core::gguf::MetaValue::F32(x) => Some(*x), _ => None })
                .unwrap_or(1e-5)
        };
        let mut hidden = 0usize;
        for pp in std::iter::once(&path).chain(std::env::args().skip(2).map(std::path::PathBuf::from).collect::<Vec<_>>().iter()) {
            if let Ok(gf) = GgufFile::open(pp) {
                if let Some(t) = gf.tensors.iter().find(|t| t.name == "token_embd.weight") {
                    hidden = t.dims.first().copied().unwrap_or(0) as usize;
                    break;
                }
            }
        }
        println!("kimi config: {} blocks, {} heads, hidden {}", u("block_count"), u("attention.head_count"), hidden);
        DefaultsGlm5 {
            hidden_size: hidden,
            rms_norm_eps: f("attention.layer_norm_rms_epsilon"),
            vocab_size: u("vocab_size") as usize,
            num_hidden_layers: u("block_count") as usize,
            ..DefaultsGlm5::from_gguf(&GgufFile::open(std::path::Path::new("/Volumes/Data/models/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf"))?)?
        }
    } else {
        DefaultsGlm5::from_gguf(&g0)?
    };
    // THE SCHEDULE COMES FROM THE MERGED TENSOR TABLE, not from the first path: Kimi's shard 1 carries the metadata and NO
    // tensors, so reading it alone said "no blk.<n>.* tensors" -- the same bug the decode driver had in iteration 280.
    let merged: Vec<engine_core::gguf::TensorInfo> = std::iter::once(path.clone())
        .chain(std::env::args().skip(2).map(std::path::PathBuf::from))
        .filter_map(|pp| GgufFile::open(&pp).ok())
        .flat_map(|gf| gf.tensors)
        .collect();
    println!("merged {} tensors for the schedule", merged.len());
    let plan = engine_core::glm53_plan::schedule_of(&merged)?;
    let expert_scale = engine_core::glm53_ffn::expert_weights_scale(&g0)?;
    println!("expert_weights_scale = {expert_scale} (read from the file)");
    let v = Vocab::from_gguf(&g0)?;
    // THE SHARD LIST IS WHATEVER WAS GIVEN, not a directory scan: `ShardedGguf::open` insists on all 14 Kimi shards
    // ("missing shard 00010, split.count=14") while 10 are present and blocks 0-61 are complete. `open_paths` is what the
    // decode driver has used since iteration 279.
    let g = {
        let extra: Vec<std::path::PathBuf> = std::env::args().skip(2).map(std::path::PathBuf::from).collect();
        if extra.is_empty() {
            ShardedGguf::open(&path)?
        } else {
            let mut all = vec![path.clone()];
            all.extend(extra);
            ShardedGguf::open_paths(&all)?
        }
    };
    // THE HEAD COUNT IS THE MODEL'S, NOT GLM's. Hardcoded 64 made Kimi fail at "kda_a_log is 96 not 64" -- its a_log is
    // per head (96), so the count has to come from the config.
    let heads = if std::env::var_os("DS4_KIMI").is_some() {
        g0.meta.iter().find(|(k, _)| k.ends_with("attention.head_count"))
            .and_then(|(_, v)| match v { engine_core::gguf::MetaValue::U32(x) => Some(*x as usize), _ => None })
            .unwrap_or(64)
    } else {
        64
    };
    let c = Ctx { d: cfg.hidden_size, heads, head_dim: 128, eps: cfg.rms_norm_eps as f64 };
    anyhow::ensure!(plan[0].attn == engine_core::glm53_plan::Attn::Kda, "block 0 must be KDA");

    // THE SAME PROMPT AS THE DECODE DRIVER. This hardcoded "Hello world" while decode reads GLM53_PROMPT (default "Hello"),
    // so the two drivers embedded DIFFERENT TOKENS -- the prefill id 19180 against decode's 9703 -- and every comparison
    // between them was comparing different inputs. That is what the "disagreement on Kimi" was.
    let prompt = std::env::var("GLM53_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let ids = v.encode(&prompt);
    println!("prompt -> {ids:?}");
    // ONE TOKEN IS THE CROSS-CHECK AGAINST DECODE, TWO IS THE CAUSALITY CHECK. The example used to demand exactly two, which
    // made the one-token comparison against `glm53_block0` impossible -- and that comparison is the only way to test the two
    // drivers against each other.
    anyhow::ensure!(ids.len() >= 1, "the prompt tokenised to nothing");

    // single-token run of token 0
    let one = run(&g, &c, &plan, 1, vec![embed(&g, ids[0], c.d)?], expert_scale)?;
    // two-token run of the same token followed by another
    // THE CAUSALITY CHECK NEEDS TWO TOKENS; THE CROSS-CHECK AGAINST DECODE NEEDS ONE. A one-token prompt used to reach
    // `ids[1]` and panic ("len is 1 but the index is 1"), so the causality probe is now conditional on there being a second
    // token. Kimi's "Hello" is one token, which is what made the cross-check impossible before this round.
    let two = if ids.len() >= 2 {
        run(&g, &c, &plan, 1, vec![embed(&g, ids[0], c.d)?, embed(&g, ids[1], c.d)?], expert_scale)?
    } else {
        one.clone()
    };

    let diff = one[0].iter().zip(&two[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    println!("position 0 after block 0: |1-token - 2-token| = {diff:.3e}");
    anyhow::ensure!(diff < 1e-12, "the two-token prefill CHANGED position 0 by {diff:.3e} -- attention is not causal");
    println!("PASS: position 0 is identical whether or not a later token is present -- the recurrence is causal");

    // THE SAME INVARIANT ON THE MLA PATH, at block 3 (the first MLA block per the file's own schedule).
    // THE SECOND `ids[1]` SITE -- the MLA causality probe. Guarded like the first: a one-token prompt (Kimi's "Hello") has
    // no second token, and this panicked at it AFTER the KDA probe had already passed.
    match plan.iter().position(|b| b.attn == engine_core::glm53_plan::Attn::Mla) {
        Some(mla) if ids.len() >= 2 => {
            let e0 = vec![embed(&g, ids[0], c.d)?];
            let e01 = vec![embed(&g, ids[0], c.d)?, embed(&g, ids[1], c.d)?];
            let m1 = mla_block_n(&g, mla, &c, &e0)?;
            let m2 = mla_block_n(&g, mla, &c, &e01)?;
            let dm = m1[0].iter().zip(&m2[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            println!("MLA block {mla}: position 0 |1-token - 2-token| = {dm:.3e}");
            anyhow::ensure!(dm < 1e-12, "MLA attention leaked the future: {dm:.3e}");
            println!("PASS: MLA position 0 is also independent of a later token -- attention is causal");
        }
        Some(mla) => println!("MLA block {mla}: causality probe skipped (the prompt is one token)"),
        None => println!("no MLA block in this checkpoint: MLA causality probe skipped"),
    }

    // THE INVARIANT AT THE OUTPUT: six blocks, both runs, then the head on position 0. The ARGMAX must match --
    // a prefill that leaked the future would have moved it, and this is the check a person would actually make.
    let blocks = std::env::var("PREFILL_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);
    // PREFILL_ONE=1 runs the SINGLE-token chain, which is the only fair comparison with the decode driver.
    if std::env::var_os("PREFILL_ONE").is_some() {
        let one = run(&g, &c, &plan, blocks, vec![embed(&g, ids[0], c.d)?], expert_scale)?;
        // THE FINAL NORM, which the decode driver applies and this one did not. It preserves direction and changes
        // magnitude, which is exactly the signature the comparison found: identical residuals, 4x logits.
        let xf = match g.dequant_name("output_norm.weight") {
            Ok(w) => gated_rms_norm(&one[0], &w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0; c.d], c.eps),
            Err(_) => one[0].clone(),
        };
        let head = g.dequant_name("token_embd.weight")?;
        let hw = head.len() / c.d;
        let mut best = (f64::NEG_INFINITY, 0usize);
        for r in 0..hw {
            let mut acc = 0.0;
            for cc in 0..c.d { acc += head[r * c.d + cc] as f64 * xf[cc]; }
            if acc > best.0 { best = (acc, r); }
        }
        let rms = (xf.iter().map(|v| v * v).sum::<f64>() / c.d as f64).sqrt();
        println!("ONE-TOKEN prefill after {blocks} blocks: {} ({:.6})  rms {rms:.6}", best.1, best.0);
        println!("FINAL-VEC prefill: {:?}", &one[0][..4]);
        return Ok(());
    }
    let h1 = run(&g, &c, &plan, blocks, vec![embed(&g, ids[0], c.d)?], expert_scale)?;
    let h2 = run(&g, &c, &plan, blocks, vec![embed(&g, ids[0], c.d)?, embed(&g, ids[1], c.d)?], expert_scale)?;
    let head = g.dequant_name("token_embd.weight")?;
    let hw = head.len() / c.d;
    let logits = |h: &[f64]| -> (f64, usize) {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for r in 0..hw {
            let mut acc = 0.0;
            for cc in 0..c.d { acc += head[r * c.d + cc] as f64 * h[cc]; }
            if acc > best.0 { best = (acc, r); }
        }
        best
    };
    let (l1, i1) = logits(&h1[0]);
    let (l2, i2) = logits(&h2[0]);
    println!("after {blocks} blocks: 1-token -> {i1} ({l1:.6}), 2-token -> {i2} ({l2:.6})");
    anyhow::ensure!(i1 == i2, "the two-token prefill changed position 0's token: {i1} vs {i2}");
    anyhow::ensure!((l1 - l2).abs() < 1e-9, "the top logit moved by {:.3e}", (l1 - l2).abs());
    println!("PASS: after {blocks} blocks position 0's TOKEN is unchanged by a later token -- prefill is causal end to end");
    Ok(())
}
