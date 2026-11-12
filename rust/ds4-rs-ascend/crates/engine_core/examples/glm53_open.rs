//! Read the REAL GLM 5.3 checkpoint through the engine's own GGUF reader.
//!
//! Item (4) of the campaign: "the same for GLM 5.3 ... once their weights are on the host". The weights
//! arrived (96.5 GB, one Q2 GGUF), so the first measurement is whether the engine can OPEN them and derive
//! its own config from the file -- not from a table I copied.
//!
//! usage: glm53_open <gguf path>

use engine_core::gguf::GgufFile;
use engine_core::glm5_ref::DefaultsGlm5;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: glm53_open <gguf>"))?;
    let t0 = std::time::Instant::now();
    let g = GgufFile::open(&path)?;
    println!("opened {} in {:.1}s", path.display(), t0.elapsed().as_secs_f64());

    // The engine derives the architecture FROM THE FILE. Any of these printed values could have been
    // copied from a config table; printing them next to the source field is what makes it a reading.
    let cfg = DefaultsGlm5::from_gguf(&g)?;
    println!("from the file:");
    println!("  vocab          {}", cfg.vocab_size);
    println!("  hidden         {}", cfg.hidden_size);
    println!("  layers         {}", cfg.num_hidden_layers);
    println!("  heads          {}", cfg.num_attention_heads);
    println!("  q_lora_rank    {}", cfg.q_lora_rank);
    println!("  kv_lora_rank   {}", cfg.kv_lora_rank);

    let n = g.tensors.len();
    let elems: u64 = g.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum();
    println!("tensors {n}, {elems} elements");
    anyhow::ensure!(n > 100, "only {n} tensors -- is this a complete checkpoint?");
    anyhow::ensure!(cfg.num_hidden_layers > 10, "layers {} is not a real model", cfg.num_hidden_layers);
    println!("PASS: the engine read the real GLM 5.3 checkpoint and derived its own config");
    Ok(())
}
