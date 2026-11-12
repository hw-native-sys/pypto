//! Generate tokens from the REAL GLM 5.3 checkpoint through the engine's own streaming decode.
//!
//! Item (4): "the same for GLM 5.3 ... once their weights are on the host". The weights arrived, the
//! engine reads the container and derives its own config (iteration 124), and `decode_step` dequantises
//! per tensor from a `ShardedGguf` -- so 96.5 GB never has to be resident.
//!
//! usage: glm53_token <gguf> [n_tokens] [prompt_token]

use engine_core::gguf::GgufFile;
use engine_core::glm5_ref::DefaultsGlm5;
use engine_core::glm5_stream::{decode_step, Glm5KvCache, ShardedGguf};
use std::path::PathBuf;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; best = i; }
    }
    best
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: glm53_token <gguf> [n] [prompt]"))?;
    let n_tok: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let prompt: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let cfg = DefaultsGlm5::from_gguf(&GgufFile::open(&path)?)?;
    println!("config: {} layers, hidden {}, vocab {}", cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size);

    let g = ShardedGguf::open(&path)?;
    // The KV cache is sized for the TRANSFORMER layers the decode loop walks.
    let n_layers = cfg.num_hidden_layers;
    let mut cache = Glm5KvCache::new(n_layers);

    let mut tok = prompt;
    let t0 = std::time::Instant::now();
    let mut generated = Vec::new();
    for step in 0..n_tok {
        let t = std::time::Instant::now();
        let logits = decode_step(tok, &cfg, &g, &mut cache)?;
        anyhow::ensure!(logits.len() == cfg.vocab_size, "logits {} != vocab {}", logits.len(), cfg.vocab_size);
        let finite = logits.iter().filter(|x| x.is_finite()).count();
        anyhow::ensure!(finite == logits.len(), "{} of {} logits are not finite", logits.len() - finite, logits.len());
        let next = argmax(&logits);
        generated.push(next);
        println!("step {step}: in {tok} -> out {next}   ({:.1}s)", t.elapsed().as_secs_f64());
        tok = next;
    }
    println!("generated {:?} in {:.1}s total", generated, t0.elapsed().as_secs_f64());
    anyhow::ensure!(!generated.is_empty(), "no tokens generated");
    anyhow::ensure!(generated.iter().all(|&t| t < cfg.vocab_size), "a token id exceeded the vocabulary");
    println!("PASS: GLM 5.3 generated tokens through the engine's streaming decode");
    Ok(())
}
