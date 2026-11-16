//! A ONE-LAYER model through the engine's token path.
//!
//! The layer's weights are ZERO. With zero weights every sub-block contributes nothing and the
//! residual carries `x` through unchanged, so a one-layer model must produce the SAME token as a
//! zero-layer one -- an exact expectation, not an approximation. What it exercises is everything a
//! real layer would: attention norm + matvec, the router, softplus_sqrt, router_finalize and
//! moe_routed_step with real expert buffers.
//!
//! usage: w8a8_layer_token

use engine_core::decode_step::{decode_one_token, DecodeConfig, LayerWeights, ModelWeights};
use std::borrow::Cow;
use std::num::NonZeroUsize;

fn main() -> anyhow::Result<()> {
    let d_model = 16usize;
    let d_ffn = 8usize;
    let n_experts = 4usize;
    let n_used = 2usize;
    let vocab = 128usize;

    let hot = 7usize % d_model;
    let mut x = vec![0.01f32; d_model];
    x[hot] = 3.0;

    let layer = LayerWeights {
        d_model,
        d_ffn: NonZeroUsize::new(d_ffn),
        n_experts,
        n_experts_used: n_used,
        attn_norm_gamma: vec![1.0; d_model],
        w_attn: vec![0.0; d_model * d_model],
        ffn_norm_gamma: vec![1.0; d_model],
        w_router: vec![0.0; n_experts * d_model],
        w_router_f16: Cow::Borrowed(&[]),
        router_bias: vec![0.0; n_experts],
        routing_table: None,
        w_gate_exps: vec![0.0; n_experts * d_ffn * d_model],
        w_up_exps: vec![0.0; n_experts * d_ffn * d_model],
        w_down_exps: vec![0.0; n_experts * d_model * d_ffn],
    };

    let mut lm_head = vec![0.0f32; vocab * d_model];
    for v in 0..vocab {
        lm_head[v * d_model + (v % d_model)] = 1.0;
    }

    let zero_layer = ModelWeights {
        layers: Vec::new(),
        final_norm_gamma: vec![1.0; d_model],
        lm_head: lm_head.clone(),
        vocab_size: vocab,
        d_model,
    };
    let one_layer = ModelWeights {
        layers: vec![layer],
        final_norm_gamma: vec![1.0; d_model],
        lm_head,
        vocab_size: vocab,
        d_model,
    };
    let cfg = DecodeConfig { eps_rms: 1e-6 };

    // A layer whose attention projection is large: if the weights FLOW, this cannot produce the same
    // token as a zero-weight layer. Without this, a sub-block that silently does nothing would pass
    // the residual check above.
    let mut loud = one_layer.layers[0].clone();
    // ONE entry, not a scale: the head is a permutation, so scaling the whole hidden state leaves the
    // argmax unchanged and would have "passed" a broken kernel either way. A single large cross-term
    // moves ONE dimension, which is what an argmax can see.
    loud.w_attn[0 * d_model + hot] = 8.0;
    let loud_model = ModelWeights {
        layers: vec![loud],
        final_norm_gamma: one_layer.final_norm_gamma.clone(),
        lm_head: one_layer.lm_head.clone(),
        vocab_size: vocab,
        d_model,
    };

    let t0 = decode_one_token(x.clone(), &zero_layer, &cfg)?;
    let t1 = decode_one_token(x.clone(), &one_layer, &cfg)?;
    let t2 = decode_one_token(x.clone(), &loud_model, &cfg)?;

    println!("d_model {d_model}, d_ffn {d_ffn}, experts {n_experts} (top {n_used}), vocab {vocab}");
    println!("token with 0 layers: {t0}");
    println!("token with 1 layer of ZERO weights: {t1}");
    anyhow::ensure!(t0 < vocab && t1 < vocab, "a token id is out of range");
    anyhow::ensure!(
        t0 == t1,
        "a zero-weight layer changed the token ({t0} -> {t1}), so a sub-block is not a pure residual"
    );
    println!("token with 1 layer of LARGE attention weights: {t2}");
    anyhow::ensure!(
        t2 != t0,
        "a layer with 8x attention weights produced the same token as no layer, so the weights are \
         not reaching the kernels"
    );
    println!("PASS: a zero-weight layer is a pure residual, and a loud layer changes the token");
    Ok(())
}
