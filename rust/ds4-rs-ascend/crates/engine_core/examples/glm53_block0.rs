//! ONE REAL KDA BLOCK of GLM 5.3, on real weights, on the CPU.
//!
//! This is the last step before a forward pass: the schedule says block 0 is KDA/Dense, the block's pieces are
//! assembled in `kda_block`, and the projections are certified kernels on the device -- so the one thing never
//! tried is the assembly on the CHECKPOINT'S OWN TENSORS.
//!
//! What it can prove without a reference implementation: the projections produce the widths the file states, the
//! block produces a finite output of the right shape, and the whole chain is deterministic. What it CANNOT prove
//! is that the numbers match GLM 5.3 -- there is no reference here, and claiming otherwise would be the
//! over-credit this campaign has removed four times.
//!
//! usage: glm53_block0 <gguf>

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::Attn;
use engine_core::glm5_ref::DefaultsGlm5;
use engine_core::glm5_stream::ShardedGguf;
use engine_core::kda_block::{block, BlockParams};
use std::path::PathBuf;


/// ONE TOKEN through an MLA block. With a single position the attention over that position is the identity on
/// `v`, so the block reduces to its four certified projections plus the norms -- which is why a decode-shaped
/// MLA block is tractable while a prefill one is not.
fn mla_block_1token(g: &ShardedGguf, b: usize, x: &[f64], d: usize, eps: f64) -> anyhow::Result<Vec<f64>> {
    // THE TESTED CODE IS THE CODE THAT RUNS. The arithmetic moved to `engine_core::mla_block`, whose three regression tests
    // pin the flat matmul to the per-head loop it replaced, the two-latent rule, and whole-block sanity. This function now
    // only fetches tensors and hands them over -- if the numbers move, the library changed, not the glue.
    let norm_w = engine_core::model_names::dequant_any(&g, b, "norm")?;
    let xin = engine_core::kda_block::gated_rms_norm(
        x,
        &norm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &vec![1.0f64; d],
        eps,
    );
    let q_a = engine_core::model_names::dequant_any(&g, b, "q_a")?;
    let q_b = engine_core::model_names::dequant_any(&g, b, "q_b")?;
    let kv_a = engine_core::model_names::dequant_any(&g, b, "kv_a_mqa")?;
    let kv_a_norm = engine_core::model_names::dequant_any(&g, b, "kv_a_norm")?.len();
    let v_b = engine_core::model_names::dequant_any(&g, b, "v_b")?;
    let out = engine_core::model_names::dequant_any(&g, b, "output")?;
    Ok(engine_core::mla_block::mla_1token(
        &q_a, &q_b, &kv_a, kv_a_norm, &v_b, &out, &xin, d,
    ))
}

/// How many blocks to run: all of them unless GLM53_BLOCKS says otherwise.
fn env_blocks() -> usize {
    std::env::var("GLM53_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX)
}


/// THE MoE FEED-FORWARD: router -> top-k -> the selected experts -> weighted sum, then the SHARED expert.
///
/// Every shape is the file's: `ffn_gate_inp [d, n_exp]` and `exp_probs_b.bias [n_exp]` route, and the stacked
/// `ffn_{gate,up,down}_exps` hold `n_exp` experts. The shared expert is three `shexp` planes applied
/// unconditionally. GLM's note says 288 experts and 8 used; both are read from the file rather than assumed.


/// `kda_*_conv` is [4, 1, C]: the FOUR taps of each of C channels, stored tap-major.
fn load_conv(g: &ShardedGguf, b: usize, which: &str, channels: usize) -> anyhow::Result<Vec<[f64; 4]>> {
    // `kda_q_conv` (GLM) and `ssm_conv1d_q` (Kimi) are the same tensor. Hardcoded aliases: a chained `replace` built
    // "ssm_conv1d_q_conv" and stripped BOTH "_conv"s. (Also re-applied after iteration 254's checkout took it.)
    let alias = match which {
        "kda_q_conv" => "ssm_conv1d_q",
        "kda_k_conv" => "ssm_conv1d_k",
        "kda_v_conv" => "ssm_conv1d_v",
        other => other,
    };
    let mut raw = None;
    for name in [format!("blk.{b}.{which}.weight"), format!("blk.{b}.{alias}.weight")] {
        if let Ok(v) = g.dequant_name(&name) {
            raw = Some(v);
            break;
        }
    }
    let raw = raw.ok_or_else(|| anyhow::anyhow!("no conv spelling for {which} in block {b}"))?;
    anyhow::ensure!(raw.len() == 4 * channels, "{which} is {} not 4 x {channels}", raw.len());
    Ok((0..channels)
        .map(|c| [raw[c] as f64, raw[channels + c] as f64, raw[2 * channels + c] as f64, raw[3 * channels + c] as f64])
        .collect())
}


/// DEQUANTISE A TENSOR UNDER EITHER SPELLING.
///
/// GLM 5.3 names the linear projections `kda_q`/`kda_k`/`kda_v`/`kda_output`; Kimi K3 names the same tensors
/// `attn_q`/`attn_k`/`attn_v`/`attn_output`. The forward was GLM-spelled throughout and failed on Kimi with
/// "tensor not found: blk.0.kda_q.weight" -- the same class of mismatch the SCHEDULE hit one round earlier.

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: glm53_block0 <gguf>"))?;
    // TWO CONFIG READERS. `DefaultsGlm5` requires GLM's indexer keys, which Kimi K3's checkpoint does not carry at all --
    // and it took a failed run to learn that rather than assuming the two models shared a config path. DS4_KIMI=1 reads
    // the `kimi-k3.*` keys instead: 93 blocks, 96 heads, kda.head_dim 128, vocab 163840.
    let cfg = if std::env::var_os("DS4_KIMI").is_some() {
        let g = GgufFile::open(&path)?;
        let u = |suffix: &str| -> u32 {
            g.meta
                .iter()
                .find(|(k, _)| k.ends_with(suffix))
                .and_then(|(_, v)| match v {
                    engine_core::gguf::MetaValue::U32(x) => Some(*x),
                    _ => None,
                })
                .unwrap_or(0)
        };
        let f = |suffix: &str| -> f32 {
            g.meta
                .iter()
                .find(|(k, _)| k.ends_with(suffix))
                .and_then(|(_, v)| match v {
                    engine_core::gguf::MetaValue::F32(x) => Some(*x),
                    _ => None,
                })
                .unwrap_or(1e-5)
        };
        // hidden_size is taken from the embedding's own dims rather than from a key Kimi does not carry.
        // THE EMBEDDING IS IN A SHARD, NOT IN SHARD 1. Kimi's shard 1 carries metadata and zero tensors, so reading
        // `token_embd`'s dims from it gave hidden = 0. Scan the other paths too.
        let mut hidden = 0usize;
        let mut extra_paths: Vec<std::path::PathBuf> = std::env::args().skip(2).map(std::path::PathBuf::from).collect();
        extra_paths.insert(0, path.clone());
        for pp in &extra_paths {
            if let Ok(gf) = GgufFile::open(pp) {
                if let Some(t) = gf.tensors.iter().find(|t| t.name == "token_embd.weight") {
                    hidden = t.dims.first().copied().unwrap_or(0) as usize;
                    break;
                }
            }
        }
        println!(
            "kimi config: {} blocks, {} heads, kda head_dim {}, vocab {}, hidden {}, eps {:.1e}",
            u("block_count"),
            u("attention.head_count"),
            u("kda.head_dim"),
            u("vocab_size"),
            hidden,
            f("attention.layer_norm_rms_epsilon")
        );
        DefaultsGlm5 {
            hidden_size: hidden,
            rms_norm_eps: f("attention.layer_norm_rms_epsilon"),
            vocab_size: u("vocab_size") as usize,
            num_hidden_layers: u("block_count") as usize,
            ..DefaultsGlm5::from_gguf(&GgufFile::open(std::path::Path::new("/Volumes/Data/models/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf"))?)?
        }
    } else {
        DefaultsGlm5::from_gguf(&GgufFile::open(&path)?)?
    };
    // THE CONFIG AND THE SCHEDULE COME FROM DIFFERENT SHARDS. Kimi's shard 1 carries the metadata and NO tensors; the block
    // tensors are in 2..13. So the keys are read from the first path and the TENSOR TABLE is merged across all of them --
    // which is what `schedule_of` exists for. Reading both from the first path reported "hidden 0" and "no blk tensors",
    // which were one fact seen twice.
    let all_paths: Vec<PathBuf> = {
        let mut v = vec![path.clone()];
        v.extend(std::env::args().skip(2).map(PathBuf::from));
        v
    };
    let merged_tensors: Vec<engine_core::gguf::TensorInfo> = {
        let mut t = Vec::new();
        for pp in &all_paths {
            match GgufFile::open(pp) {
                Ok(gf) => {
                    println!("  shard {}: {} tensors", pp.file_name().and_then(|n| n.to_str()).unwrap_or("?"), gf.tensors.len());
                    t.extend(gf.tensors);
                }
                Err(e) => println!("  shard {}: FAILED TO OPEN: {e}", pp.display()),
            }
        }
        t
    };
    println!("merged {} tensors from {} shard(s)", merged_tensors.len(), all_paths.len());
    let plan = engine_core::glm53_plan::schedule_of(&merged_tensors)?;
    let expert_scale = engine_core::glm53_ffn::expert_weights_scale(&GgufFile::open(&path)?)?;
    println!("expert_weights_scale = {expert_scale} (read from the file)");
    anyhow::ensure!(plan[0].attn == Attn::Kda, "block 0 is not KDA in this checkpoint");
    println!("config: hidden {} layers {}; block 0 = KDA", cfg.hidden_size, cfg.num_hidden_layers);

    let g = {
        // MULTI-SHARD. A sharded checkpoint (Kimi K3 ships as 14 files, 10 of which are present) cannot be opened by path
        // alone, so any further arguments are treated as additional shards.
        let extra: Vec<PathBuf> = std::env::args().skip(2).map(PathBuf::from).collect();
        if extra.is_empty() {
            ShardedGguf::open(&path)?
        } else {
            let mut all = vec![path.clone()];
            all.extend(extra);
            println!("opening {} shard(s)", all.len());
            ShardedGguf::open_paths(&all)?
        }
    };
    let d = cfg.hidden_size;
    // The tensors the file itself names for block 0 (read, not assumed).
    let wq = engine_core::model_names::dequant_any(&g, 0, "q")?;
    println!("kda_q dequantised: {} values", wq.len());
    // THE HEAD COUNT COMES FROM THE TENSOR. `kda_q` is [d, heads*head_dim]: 8192 for GLM (64 x 128), 12288 for Kimi
    // (96 x 128). A hardcoded 64 refused Kimi's weights with "88080384 not 8192 x 7168" -- and my previous attempt at this
    // line replaced the MLA helper's `heads` instead, which did not compile.
    let heads = wq.len() / (d * 128);
    let head_dim = 128usize;
    anyhow::ensure!(wq.len() == heads * 128 * d, "kda_q is {} not {heads} x 128 x {d}", wq.len());
    anyhow::ensure!(heads * head_dim * d == wq.len(), "kda_q is {wq_len} not {heads} x {head_dim} x {d}", wq_len = wq.len());

    // A deterministic input; the VALUES do not matter, the SHAPES and the finiteness do.
    // THE INPUT: a REAL PROMPT, tokenised by the repository's own tokeniser, embedded through `token_embd`.
    // Before this, every run used a synthetic vector and the token it produced answered nothing. GLM53_PROMPT
    // selects the text; unset falls back to the synthetic vector so the earlier measurements stay reproducible.
    let mut x: Vec<f64> = match std::env::var("GLM53_PROMPT") {
        Ok(prompt) => {
            use engine_core::tokenizer::Vocab;
            let v = Vocab::from_gguf(&GgufFile::open(&path)?)?;
            let ids = v.encode(&prompt);
            println!("prompt {prompt:?} -> {ids:?}");
            anyhow::ensure!(!ids.is_empty(), "the prompt tokenised to nothing");
            // A single-token prompt keeps the block loop's arithmetic identical to every earlier run; a longer one
            // would need a prefill loop the block assembly does not have yet, so it is refused by name.
            anyhow::ensure!(
                ids.len() == 1,
                "the prompt is {} tokens; this decode path takes ONE (a prefill loop is not implemented)",
                ids.len()
            );
            let emb = g.dequant_name("token_embd.weight")?;
            let vocab_cols = emb.len() / d;
            let id = ids[0] as usize;
            anyhow::ensure!(id < vocab_cols, "token {id} is outside the {vocab_cols}-wide embedding");
            let v: Vec<f64> = {
                let v: Vec<f64> = (0..d).map(|c| emb[c * vocab_cols + id] as f64).collect();
                if std::env::var_os("KDA_INTER").is_some() {
                    println!("EMB decode: token {id}, d {d}, cols {vocab_cols}, first {:?}", &v[..3]);
                }
                v
            };
            if std::env::var_os("KDA_INTER").is_some() {
                println!("EMB decode: token {id}, d {d}, cols {vocab_cols}, first {:?}", &v[..3]);
            }
            v
        }
        Err(_) => (0..d).map(|i| ((i % 11) as f64 - 5.0) * 0.01).collect(),
    };
    // HYPER-CONNECTIONS, when GLM53_HC=1: the residual becomes N parallel streams that are mixed per sublayer by a
    // Sinkhorn-normalised 4x4 matrix predicted from the state (`hc_attn_fn/base/scale`). Without the flag the driver adds
    // the residual directly, which is what every earlier token was produced with.
    let hc_n: usize = if std::env::var_os("GLM53_HC").is_some() { 4 } else { 1 };
    let mut streams: Vec<Vec<f64>> = vec![x.clone(); hc_n];
    let hc_eps = 1e-8f64;
    let hc_iters: usize = 20;
    if hc_n > 1 { println!("hyper-connections: {hc_n} streams, {hc_iters} Sinkhorn iterations"); }

    // 8192-wide projections from the real weight matrix (f64, one row at a time).
    let proj = |w: &[f32]| -> Vec<f64> {
        (0..heads * head_dim)
            .map(|r| (0..d).map(|c| w[r * d + c] as f64 * x[c]).sum())
            .collect()
    };
    let q = proj(&wq);
    let k = proj(&engine_core::model_names::dequant_any(&g, 0, "k")?);
    let v = proj(&engine_core::model_names::dequant_any(&g, 0, "v")?);
    println!("projections: q {} k {} v {}", q.len(), k.len(), v.len());

    let onw = vec![1.0f64; head_dim];
    // THE BLOCK'S REAL STRUCTURE: attn_norm -> KDA -> output projection -> residual add. GLM 5.3's leading
    // blocks are KDA/Dense (read from the file), so the chain runs until the first MLA block and no further --
    // a partial chain that says where it stopped is worth more than one that guesses.
    println!("running blocks 0..{} of {}", plan.len().min(4), plan.len());
    let mut moe_used = 0usize;
    let t0 = std::time::Instant::now();
    // THE TRUNK, NOT THE FILE. `glm5-next.block_count = 46` but `trunk_block_count = 45`: the last block is not
    // part of the decode trunk, and running 46 puts one extra block in the residual stream. Found by auditing
    // every metadata key against the implementation (skill rule 13c), not by any arithmetic check.
    let trunk = std::env::var("GLM53_TRUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(45usize);
    let upto = plan.len().min(trunk).min(env_blocks());
    println!("trunk: {upto} of {} blocks (block_count 46, trunk_block_count {trunk})", plan.len());
    for b in 0..upto {
        // THE STREAMS BECOME THE BLOCK'S INPUT when hyper-connections are on.
        //
        // B23: ONE `predict` PER BLOCK FOR THE ATTENTION SUBLAYER, hoisted to here. Both arms used to re-read the three
        // `hc_attn_*` tensors and call `predict` a second time on the SAME, unmodified `streams` -- an identical result
        // for a second dequant of a 393216-value BF16 tensor and a second [24, 16384] matvec. Holding one `Params` is
        // safe precisely because nothing writes `streams` between the collapse and the advance, which is also why the
        // duplication was invisible: it could never disagree.
        let mut attn_hp = None;
        if hc_n > 1 {
            let f = g.dequant_name(&format!("blk.{b}.hc_attn_fn.weight"))?;
            let base = g.dequant_name(&format!("blk.{b}.hc_attn_base.weight"))?;
            let sc = g.dequant_name(&format!("blk.{b}.hc_attn_scale.weight"))?;
            let hp = engine_core::hyper_connection::Params::predict(&f, &base, &sc, &streams, hc_n, hc_iters, hc_eps);
            x = hp.block_input(&streams, hc_n);
            attn_hp = Some(hp);
        }
        if plan[b].attn == Attn::Mla {
            let op = mla_block_1token(&g, b, &x, d, cfg.rms_norm_eps as f64)?;
            println!("MLA-TENSOR decode blk{b}: {:?}", &op[..3]);
            for i in 0..d { x[i] += op[i]; }
            if let Some(hp) = attn_hp.as_ref() {
                // `next_state` IS `hc_post`: `post[i]*block_out + sum_j comb[i][j]*state[j]`. Passing `x` here -- which is
                // already `xin + op` -- counts `xin` a SECOND time through `comb`. The block output is `op`.
                streams = hp.next_state(&streams, &op, hc_n);
            }
            let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
            println!("block {b}: MLA, rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
            if b <= 4 { println!("BLK{b}-TENSOR decode (MLA): {:?}", &x[..3]); }
            anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after MLA block {b} is not finite");
            // THE FFN IS NOT OPTIONAL ON AN MLA BLOCK. `continue` used to skip straight to the next block, so every
            // MLA block ran its attention and NOT its feed-forward -- which is why this driver and the prefill driver
            // disagreed by 40x on the logit. The `continue` was written to skip the KDA code below, not the FFN.
            // B22: THE MLA FFN HAD NO HYPER-CONNECTION EITHER -- the same wrap as the KDA branch, the same differencing trick.
            let mut ffn_hp = None;
            let mut ffn_x0: Vec<f64> = Vec::new();
            if hc_n > 1 {
                let f = g.dequant_name(&format!("blk.{b}.hc_ffn_fn.weight"))?;
                let base = g.dequant_name(&format!("blk.{b}.hc_ffn_base.weight"))?;
                let sc = g.dequant_name(&format!("blk.{b}.hc_ffn_scale.weight"))?;
                let hp = engine_core::hyper_connection::Params::predict(&f, &base, &sc, &streams, hc_n, hc_iters, hc_eps);
                x = hp.block_input(&streams, hc_n);
                ffn_x0 = x.clone();
                ffn_hp = Some(hp);
            }
            match plan[b].ffn {
                engine_core::glm53_plan::Ffn::Dense => {
                    let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
                    let fnw: Vec<f64> = fnorm_w.iter().map(|v| *v as f64).collect();
                    let fin = engine_core::kda_block::gated_rms_norm(&x, &fnw, &vec![1.0f64; d], cfg.rms_norm_eps as f64);
                    let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
                    let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
                    let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
                    let inter = wg.len() / d;
                    let gv: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wg[r * d + c] as f64 * fin[c]).sum()).collect();
                    let uv: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wu[r * d + c] as f64 * fin[c]).sum()).collect();
                    let act: Vec<f64> = gv.iter().zip(&uv).map(|(a, bb)| (a / (1.0 + (-a).exp())) * bb).collect();
                    for r in 0..d {
                        let mut acc = 0.0;
                        for c in 0..act.len() { acc += wd[r * act.len() + c] as f64 * act[c]; }
                        x[r] += acc;
                    }
                }
                engine_core::glm53_plan::Ffn::Moe => {
                    let (dn, used) = engine_core::glm53_ffn::moe_ffn(&g, expert_scale, b, &x, d, cfg.rms_norm_eps as f64)?;
                    for i in 0..d { x[i] += dn[i]; }
                    moe_used = used;
                }
                engine_core::glm53_plan::Ffn::None => {}
            }
            if let Some(hp) = ffn_hp {
                let ffn_out: Vec<f64> = x.iter().zip(&ffn_x0).map(|(a, b)| a - b).collect();
                streams = hp.next_state(&streams, &ffn_out, hc_n);
            }
            continue;
        }
        // B22: THE KDA BRANCH HAD NO HYPER-CONNECTION AT ALL. The reference applies `hc_mixes`/`hc_pre` before EVERY sublayer,
        // on every block; only the MLA branch did, so KDA blocks carried `streams` forward untouched.
        //
        // B23: `x` IS ALREADY `attn_hp`'s collapse -- the hoisted block above did it. This arm used to repeat the whole
        // predict + block_input, which is why the KDA path paid for the attention coefficients twice.
        let kda_hp = attn_hp;
        let norm_w = g.dequant_name(&format!("blk.{b}.attn_norm.weight"))?;
        let xin = engine_core::kda_block::gated_rms_norm(&x, &norm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64);
        let wq = engine_core::model_names::dequant_any(&g, b, "q")?;
        let wk = engine_core::model_names::dequant_any(&g, b, "k")?;
        let wv = engine_core::model_names::dequant_any(&g, b, "v")?;
        let proj = |w: &[f32], xv: &[f64]| -> Vec<f64> {
            (0..heads * head_dim).map(|r| (0..d).map(|c| w[r * d + c] as f64 * xv[c]).sum()).collect()
        };
        let q = proj(&wq, &xin);
        let k = proj(&wk, &xin);
        let v = proj(&wv, &xin);
        let params = BlockParams {
            // `kda_a_log` IS in the file and its shape matches a_log exactly: 64 values, one per head. The
            // other KDA constants are NOT yet loaded -- see the note in the commit; the conv is per-channel
            // ([4,1,8192]) and beta is [4096,64], neither of which `BlockParams` can hold.
            heads, head_dim,
            a_log: {
                let raw = engine_core::model_names::dequant_any(&g, b, "a_log")?;
                anyhow::ensure!(raw.len() == heads, "kda_a_log is {} not {heads}", raw.len());
                raw.iter().map(|v| *v as f64).collect()
            },
            // THE REAL PER-CHANNEL CONVS. `kda_{q,k,v}_conv` are [4, 1, 8192]: four taps for EACH of the
            // 8192 channels. Loaded from the file rather than shared.
            conv_q: load_conv(&g, b, "kda_q_conv", heads * head_dim)?,
            conv_k: load_conv(&g, b, "kda_k_conv", heads * head_dim)?,
            conv_v: load_conv(&g, b, "kda_v_conv", heads * head_dim)?,
            // THE REAL dt BIAS: kda_dt_bias is [8192], one value per channel.
            dt_bias: {
                let raw = engine_core::model_names::dequant_any(&g, b, "dt_bias")?;
                anyhow::ensure!(raw.len() == heads * head_dim, "kda_dt_bias is {} not {}", raw.len(), heads * head_dim);
                raw.iter().map(|v| *v as f64).collect()
            },
            // `kda_beta` is [4096, 64] = d x heads, so it reads as a PER-HEAD PROJECTION of x, sigmoid-ed into
            // (0,1) the way a write strength must be. This is a MODELLING decision, stated rather than assumed:
            // the alternative reading is per (channel, head) values used directly, which the recurrence's
            // per-head `beta` cannot consume. If the reading is wrong every block is wrong in the same way.
            beta: {
                let raw = engine_core::model_names::dequant_any(&g, b, "beta")?;
                anyhow::ensure!(raw.len() == d * heads, "kda_beta is {} not {d} x {heads}", raw.len());
                (0..heads)
                    .map(|h| {
                        let z: f64 = (0..d).map(|c| raw[c * heads + h] as f64 * xin[c]).sum();
                        1.0 / (1.0 + (-z).exp())
                    })
                    .collect()
            },
            eps: cfg.rms_norm_eps as f64,
        };
        // THE TIME STEP, from the block's own f-gate: dt = f_b(f_a(normed x)) + dt_bias, 8192 channels.
        let fa = engine_core::model_names::dequant_any(&g, b, "f_a")?;
        let fb = engine_core::model_names::dequant_any(&g, b, "f_b")?;
        let hid: Vec<f64> = (0..128)
            .map(|r| (0..d).map(|c| fa[r * d + c] as f64 * xin[c]).sum())
            .collect();
        let gate: Vec<f64> = (0..heads * head_dim)
            .map(|r| (0..128).map(|c| fb[r * 128 + c] as f64 * hid[c]).sum())
            .collect();
        // THE OUTPUT GATE COMES IN TWO SHAPES, and the shapes say which.
        //   GLM 5.3: a LOW-RANK pair -- kda_g_a [d,128] then kda_g_b [128, heads*head_dim]
        //   Kimi K3: ONE DIRECT matrix -- ssm_g [d, heads*head_dim]   (its f-gate is still low-rank)
        // Reading only the pair failed on Kimi with "tensor not found: blk.0.ssm_dt.g_a"; a direct matrix is tried FIRST
        // and the pair is the fallback, because `ssm_g` and `kda_g_a` cannot both exist for one block.
        let ogate: Vec<f64> = if let Ok(wg) = engine_core::model_names::dequant_any(&g, b, "g") {
            if wg.len() == (heads * head_dim) * d {
                (0..heads * head_dim)
                    .map(|r| {
                        let z: f64 = (0..d).map(|c| wg[r * d + c] as f64 * xin[c]).sum();
                        1.0 / (1.0 + (-z).exp())
                    })
                    .collect()
            } else {
                anyhow::bail!("gate matrix is {} not {}", wg.len(), (heads * head_dim) * d)
            }
        } else {
            // the low-rank pair
            let ga = engine_core::model_names::dequant_any(&g, b, "g_a")?;
            let gb = engine_core::model_names::dequant_any(&g, b, "g_b")?;
            let ghid: Vec<f64> = (0..128).map(|r| (0..d).map(|c| ga[r * d + c] as f64 * xin[c]).sum()).collect();
            (0..heads * head_dim)
                .map(|r| {
                    let z: f64 = (0..128).map(|c| gb[r * 128 + c] as f64 * ghid[c]).sum();
                    1.0 / (1.0 + (-z).exp())
                })
                .collect()
        };
        // B23: the gate was `b >= 0 && ...`, always true on a `usize` -- the leftover of a block filter the dump no
        // longer applies. `KDA_INTER` alone is the gate. (`warning: comparison is useless due to type limits`.)
        if std::env::var_os("KDA_INTER").is_some() {
            println!("KDA-INTER blk{b} decode: x0={:?} xin0={:?} q0={:?} beta0={:?}",
                x[0], xin[0], q[0], params.beta[0]);
        }
        if std::env::var_os("GATE_DBG").is_some() {
            println!("GATE decode blk{b}: gate {:?} ogate {:?}", &gate[..3], &ogate[..3]);
        }
        if std::env::var_os("CONST_DBG").is_some() {
            println!("CONST decode blk{b}: conv_q[0] {:?} a_log[0] {:?} dt_bias[0] {:?}",
                params.conv_q[0], params.a_log[0], params.dt_bias[0]);
        }
        let o = block(&params, &q, &k, &v, &gate, 1, &onw, &ogate);
        if std::env::var_os("CONST_DBG").is_some() {
            println!("CONST decode blk{b}: block_out {:?}", &o[..3]);
        }
        anyhow::ensure!(o.iter().all(|v| v.is_finite()), "block {b} produced a non-finite value");
        // SPLIT READ FROM DEQUANT. The per-block time grows with depth (100s -> 172s -> 309s) and MADV_DONTNEED did not fix
        // it (iteration 273), so the next question is WHICH half grows: mmap page-in (I/O) or IQ1_S -> f32 (CPU).
        if std::env::var_os("DS4_GLM_PROF").is_some() {
            let (rd, dq, bytes, calls) = engine_core::glm5_stream::glm_prof_report();
            println!(
                "  blk{b} prof: read {:.2}s  dequant {:.2}s  {:.1} MB in {calls} calls",
                // glm_prof_report returns MILLISECONDS (ns / 1e6), not nanoseconds. I divided by 1e9 and printed 0.00,
                // then wrote "the profiler's read timer is broken" -- the profiler was right and my arithmetic was not.
                rd / 1e3, dq / 1e3, bytes as f64 / 1e6
            );
            engine_core::glm5_stream::glm_prof_reset();
        }
        // RELEASE THIS BLOCK'S PAGES. Without it the page cache evicts what the NEXT block needs and every block re-reads
        // from the SSD once the model exceeds RAM (Kimi: 100s -> 172s -> 327s per block).
        if std::env::var_os("DS4_RELEASE_PAGES").is_some() {
            let freed = g.advice_release_prefix(&format!("blk.{b}."));
            println!("  released {:.1} MB of block {b} pages", freed as f64 / 1e6);
        }
        // output projection (8192 -> d) then the residual
        let wo = engine_core::model_names::dequant_any(&g, b, "output")?;
        // attn_output is [12288, 7168] for Kimi and [8192, 4096]-shaped for GLM; take the leading d columns
        // THE OUTPUT PROJECTION'S WIDTH IS THE TENSOR'S, NOT GLM's. This hardcoded 8192 -- GLM's `heads*head_dim` --
        // and indexed `wo` with stride 8192, so on Kimi (12288 wide) it summed only the first 8192 of 12288 values AND
        // read the wrong rows. GLM agreed with the prefill because 8192 IS its width; Kimi did not (10374 vs 36382).
        // The prefill always used `o.len()`; this is the same expression.
        let op = engine_core::mla_block::output_projection(&wo, &o, d);
        for i in 0..d { x[i] += op[i]; }
        if let Some(hp) = kda_hp {
            // `next_state` is `hc_post`; the block output is `op`, not `x` (which is already `xin + op`)
            streams = hp.next_state(&streams, &op, hc_n);
        }
        // THE DENSE FFN, when the block has one: ffn_norm -> gate/up -> SiLU -> down -> residual. GLM's leading
        // blocks carry `ffn_gate`/`ffn_up`/`ffn_down` at 12288 wide; the MoE blocks carry stacked experts instead,
        // which are NOT implemented, so their FFN is skipped and SAID so.
        // B22: THE FFN HAD NO HYPER-CONNECTION EITHER. The reference applies `hc_mixes`/`hc_pre` before the FFN and
        // `hc_post` after it, exactly as it does for attention. `ffn_x0` captures the input so the FFN's OUTPUT can be
        // recovered by differencing without rewriting each branch's `x[i] += ...`.
        let mut ffn_hp = None;
        let mut ffn_x0: Vec<f64> = Vec::new();
        if hc_n > 1 {
            let f = g.dequant_name(&format!("blk.{b}.hc_ffn_fn.weight"))?;
            let base = g.dequant_name(&format!("blk.{b}.hc_ffn_base.weight"))?;
            let sc = g.dequant_name(&format!("blk.{b}.hc_ffn_scale.weight"))?;
            let hp = engine_core::hyper_connection::Params::predict(&f, &base, &sc, &streams, hc_n, hc_iters, hc_eps);
            x = hp.block_input(&streams, hc_n);
            ffn_x0 = x.clone();
            ffn_hp = Some(hp);
        }
        if plan[b].ffn == engine_core::glm53_plan::Ffn::Dense {
            let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
            let fin = engine_core::kda_block::gated_rms_norm(&x, &fnorm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64);
            let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
            let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
            let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
            let inter = wg.len() / d;
            let g_: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wg[r * d + c] as f64 * fin[c]).sum()).collect();
            let u_: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wu[r * d + c] as f64 * fin[c]).sum()).collect();
            let act: Vec<f64> = g_.iter().zip(&u_).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
            let dn: Vec<f64> = (0..d).map(|r| (0..act.len()).map(|c| wd[r * act.len() + c] as f64 * act[c]).sum()).collect();
            for i in 0..d { x[i] += dn[i]; }
        } else if plan[b].ffn == engine_core::glm53_plan::Ffn::Moe {
            let (dn, used) = engine_core::glm53_ffn::moe_ffn(&g, expert_scale, b, &x, d, cfg.rms_norm_eps as f64)?;
            for i in 0..d { x[i] += dn[i]; }
            moe_used = used;
        } else {
            println!("  block {b}: FFN is {:?} -- skipped", plan[b].ffn);
        }
        if let Some(hp) = ffn_hp {
            // the FFN's output is `x - x0`; `next_state` is `hc_post` and takes the OUTPUT, not the incremented vector
            let ffn_out: Vec<f64> = x.iter().zip(&ffn_x0).map(|(a, b)| a - b).collect();
            streams = hp.next_state(&streams, &ffn_out, hc_n);
        }
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
        println!("block {b}: rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
        if b <= 5 { println!("BLK{b}-TENSOR decode: {:?}", &x[..3]); }
        if b == 5 { println!("FINAL-VEC decode: {:?}", &x[..4]); }
        anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after block {b} is not finite");
        anyhow::ensure!(rms > 0.0, "the residual after block {b} is all zero");
    }
    // THE HEAD. GLM 5.3 TIES it to the embedding: `token_embd.weight [4096, 154880]`. The final norm is applied
    // first if the checkpoint carries one by that name -- checked, not assumed.
    let out_norm = g.dequant_name("output_norm.weight").ok();
    let xf = match &out_norm {
        Some(w) => engine_core::kda_block::gated_rms_norm(
            &x, &w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64),
        None => {
            println!("no output_norm.weight in this checkpoint -- the trunk output is used as-is");
            x.clone()
        }
    };
    let head = g.dequant_name("token_embd.weight")?;
    let vocab = head.len() / d;
    println!("head: token_embd, {vocab} outputs (tied embedding)");
    anyhow::ensure!(vocab == cfg.vocab_size, "head is {vocab} wide, config says {}", cfg.vocab_size);
    let t1 = std::time::Instant::now();
    let mut best = (f64::NEG_INFINITY, 0usize);
    let mut checked = 0usize;
    for r in 0..vocab {
        let mut acc = 0.0f64;
        for c in 0..d {
            acc += head[r * d + c] as f64 * xf[c];
        }
        if !acc.is_finite() {
            // one non-finite logit would make the argmax meaningless, so it is a refusal, not a warning
            anyhow::bail!("logit {r} is not finite");
        }
        checked += 1;
        if acc > best.0 {
            best = (acc, r);
        }
    }
    anyhow::ensure!(checked == vocab, "only {checked} of {vocab} logits were computed");
    println!("logits: {checked} finite, best {:.4} at id {} (MoE used {moe_used} experts per block)", best.0, best.1);
    // THE LABEL NAMES THE MODEL THAT WAS LOADED, not the one the config reader is named after. A Kimi run printed
    // "TOKEN from GLM 5.3" for a whole round because the string was hardcoded beside a shared code path.
    let model = if std::env::var_os("DS4_KIMI").is_some() { "Kimi K3" } else { "GLM 5.3" };
    println!("TOKEN from {model}: {} (vocab {vocab})  ({:.1}s)", best.1, t1.elapsed().as_secs_f64());
    anyhow::ensure!(best.1 < vocab, "token id outside the vocabulary");
    println!("PASS: {model} produced a token from its own weights, through the assembled blocks and its own head");
    Ok(())
}
