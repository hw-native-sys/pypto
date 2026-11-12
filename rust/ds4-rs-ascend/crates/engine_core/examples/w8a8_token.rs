//! The engine generates a TOKEN: final norm, head, argmax, through the engine's own code path.
//!
//! A zero-layer model isolates the TAIL of `decode_one_token` from the 288 GB of package weights, so
//! the token mechanism itself is demonstrated and checkable by construction: with an identity head,
//! the argmax of the normalised hidden state is the index of its largest component, which this
//! example computes independently and compares.
//!
//! usage: w8a8_token [d_model] [vocab]

use engine_core::decode_step::{decode_one_token, DecodeConfig, ModelWeights};

fn main() -> anyhow::Result<()> {
    let d_model: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let vocab: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    // Hidden state with a deliberately unique maximum at index `hot`.
    let hot = 7usize % d_model;
    let mut x = vec![0.01f32; d_model];
    x[hot] = 3.0;

    // gamma = 1, identity head rows: lm_head[v][d] = 1 when v == d % d_model else 0.
    let mut lm_head = vec![0.0f32; vocab * d_model];
    for v in 0..vocab {
        lm_head[v * d_model + (v % d_model)] = 1.0;
    }

    let model = ModelWeights {
        layers: Vec::new(),
        final_norm_gamma: vec![1.0; d_model],
        lm_head,
        vocab_size: vocab,
        d_model,
    };
    let cfg = DecodeConfig { eps_rms: 1e-6 };

    let token = decode_one_token(x.clone(), &model, &cfg)?;

    // Independent expectation: the head is a permutation, so the winning row is the argmax column
    // after the final RMS-norm, i.e. `hot` contributes to every v with v % d_model == hot.
    let expected_class = hot;
    let candidates = (0..vocab).filter(|v| v % d_model == expected_class).max();
    println!("d_model {d_model}, vocab {vocab}, hot {hot}");
    println!("token from the engine: {token}");
    println!("independently expected class: {:?}", candidates);
    anyhow::ensure!(
        candidates.map(|c| c == token).unwrap_or(false),
        "the engine's token {token} is not the class the head should select"
    );
    println!("PASS: the engine produced the token its own head selects");
    Ok(())
}
