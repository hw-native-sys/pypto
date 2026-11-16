//! Encode text with the REPOSITORY's tokeniser and round-trip it.
//!
//! Written after I destroyed this module trying to build my own: the repo already had a full BPE tokeniser with
//! GLM-specific special tokens, so the only thing needed here is to USE it on the real checkpoint.
//!
//! usage: vocab_check <gguf> <text> [text ...]

use engine_core::gguf::GgufFile;
use engine_core::tokenizer::Vocab;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: vocab_check <gguf> <text>"))?);
    let g = GgufFile::open(&path)?;
    let v = Vocab::from_gguf(&g)?;
    println!("vocabulary: {} tokens, {} merges", v.id_to_token.len(), v.merge_rank.len());
    let mut checked = 0usize;
    for text in std::env::args().skip(2) {
        let ids = v.encode(&text);
        let back = v.decode(&ids)?;
        println!("encode({text:?}) -> {ids:?}\ndecode -> {back:?}");
        anyhow::ensure!(back == text, "roundtrip changed the text: {back:?} vs {text:?}");
        anyhow::ensure!(v.encode(&back) == ids, "encode(decode(ids)) differs from ids");
        checked += 1;
    }
    anyhow::ensure!(checked > 0, "no text supplied");
    println!("PASS: {checked} string(s) round-trip through the repository tokeniser");
    Ok(())
}
