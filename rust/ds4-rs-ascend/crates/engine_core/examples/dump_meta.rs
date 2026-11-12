//! Dump a GGUF's metadata keys, so the engine's config reader can learn the file's own vocabulary.
use engine_core::gguf::GgufFile;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| anyhow::anyhow!("usage: dump_meta <gguf>"))?;
    let g = GgufFile::open(&p)?;
    println!("version {}  tensors {}  meta {}", g.version, g.n_tensors, g.n_meta_kv);
    for (k, v) in &g.meta {
        let s = format!("{v:?}");
        if k.contains("block_count") || k.contains("head") || k.contains("hidden")
            || k.contains("expert") || k.contains("vocab") || k.contains("arch")
            || k.contains("lora") || k.contains("rope") || k.contains("norm") {
            println!("{k} = {}", &s[..s.len().min(90)]);
        }
    }
    // The hybrid schedule: an MQA/MLA flag per layer. Counting it is the difference between
    // "Kimi K3 is a hybrid" and knowing HOW MANY of its 93 blocks use which attention.
    for (k, v) in &g.meta {
        if k.ends_with("head_count_kv") {
            if let engine_core::gguf::MetaValue::Array { values, .. } = v {
                let ones = values.iter().filter(|x| format!("{x:?}").contains("I32(1)")).count();
                println!("{k}: {} layers, {ones} with kv=1 (MLA), {} with kv=0 (linear/KDA)", values.len(), values.len() - ones);
            }
        }
    }
    println!("--- names containing expert/shexp in block 3 ---");
    for t in g.tensors.iter().filter(|t| t.name.starts_with("blk.4.") && (t.name.contains("exp") || t.name.contains("shexp"))) {
        println!("{}", t.name);
    }
    println!("--- every tensor of block 0 ---");
    for t in g.tensors.iter().filter(|t| t.name.starts_with("blk.4.")) {
        println!("{} {:?} {:?}", t.name, t.dims, t.ttype);
    }
    println!("--- first 24 tensor names ---");
    for t in g.tensors.iter().take(24) {
        println!("{} {:?} {:?}", t.name, t.dims, t.ttype);
    }
    Ok(())
}
