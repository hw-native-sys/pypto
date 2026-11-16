//! Read GLM 5.3's / Kimi K3's per-block schedule FROM THE CHECKPOINT and print the census.
//!
//! Accepts MULTIPLE gguf paths, because a sharded checkpoint distributes the block tensors across its shards
//! (Kimi K3 has 14, and shard 1 carries metadata only). Merging the tensor tables is what makes the schedule a
//! property of the model rather than of whichever shard happened to be opened.
//!
//! usage: glm53_schedule <gguf> [gguf ...]

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::{schedule_from, Census};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    anyhow::ensure!(!paths.is_empty(), "usage: glm53_schedule <gguf> [gguf ...]");
    let mut tensors = Vec::new();
    let mut meta = Vec::new();
    for p in &paths {
        let g = GgufFile::open(p)?;
        if meta.is_empty() {
            meta = g.meta.clone();   // the metadata lives in the first shard
        }
        tensors.extend(g.tensors);
    }
    // COMPLETENESS, from the file names. Kimi K3 ships as `...-NNNNN-of-00014.gguf`; if ten of the fourteen are
    // present the census below describes an INCOMPLETE model, and the difference matters -- a schedule read from
    // ten shards is not a schedule, it is a schedule-shaped number. (Ten of fourteen is what is on disk today,
    // and the feed-forward census was wrong until this check existed.)
    let expected: usize = paths
        .iter()
        .filter_map(|p| p.file_name()?.to_str()?.split("-of-").nth(1))
        .filter_map(|s| s.trim_end_matches(".gguf").parse().ok())
        .max()
        .unwrap_or(paths.len());
    println!("{} of {} shard(s) present, {} tensors", paths.len(), expected, tensors.len());
    anyhow::ensure!(
        paths.len() == expected,
        "INCOMPLETE checkpoint: {expected} shards expected, {} present. The census below describes what is here, \
         not the model.",
        paths.len()
    );
    let (s, source) = schedule_from(&tensors, &meta)?;
    let c = Census::of(&s);
    println!("  schedule source: {source}");
    println!(
        "  {} blocks: {} KDA + {} MLA, {} MoE + {} dense",
        c.blocks, c.kda, c.mla, c.moe, c.dense
    );
    for (i, b) in s.iter().take(4).enumerate() {
        println!("  blk {i}: {:?} / {:?}", b.attn, b.ffn);
    }
    anyhow::ensure!(c.blocks > 10, "only {} blocks -- is this a complete checkpoint?", c.blocks);
    anyhow::ensure!(c.kda + c.mla == c.blocks, "every block must have exactly one attention kind");
    println!("PASS: the schedule was read from the checkpoint's own tensor names");
    Ok(())
}
