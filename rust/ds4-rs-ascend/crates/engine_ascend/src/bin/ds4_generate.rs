// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Minimal end-to-end generation for DeepSeek-V4-Flash on the engine_ascend decode
//! surface: tokenize a prompt, prefill, greedy-decode, and stream text out.
//!
//!   ds4_generate <model.gguf> [--prompt "..."] [--max N] [--raw]
//!
//! CPU by default (weights streamed from the GGUF mmap — slow but any size).
//! With the `ascend` feature and `DS4_DEVICE_ELF` set, the proven kernels run on
//! the NPU (device 0). This is the "serving" core the controller plans for.

use anyhow::{bail, Result};
use engine_ascend::decode_runner::{DecodeRunner, DecodeSession};
use engine_ascend::sampling::argmax;
use engine_core::gguf::GgufFile;
use engine_core::tokenizer::{StreamingDecoder, Vocab};
use std::io::Write as _;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = String::new();
    let mut prompt = "Give me a short introduction to large language models.".to_string();
    let mut max_new = 64usize;
    let mut raw = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned().unwrap_or_default();
            }
            "--max" => {
                i += 1;
                max_new = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(64);
            }
            "--raw" => raw = true,
            s if !s.starts_with("--") => model = s.to_string(),
            other => bail!("unknown flag {other}"),
        }
        i += 1;
    }
    if model.is_empty() {
        bail!("usage: ds4_generate <model.gguf> [--prompt \"...\"] [--max N] [--raw]");
    }
    let path = Path::new(&model);

    // Tokenizer from the GGUF's embedded vocab+merges (antirez-faithful joyai-llm BPE).
    let g = GgufFile::open(path)?;
    let vocab = Vocab::from_gguf(&g)?;
    let sp = vocab.special_tokens()?;

    // Build the prompt token stream. Chat mode wraps the user text in the DS4
    // turn tokens and closes </think> so the model answers directly; --raw just
    // encodes the text after BOS.
    let mut ids: Vec<i32> = Vec::new();
    ids.push(sp.bos);
    if raw {
        ids.extend(vocab.encode(&prompt).into_iter().map(|t| t as i32));
    } else {
        ids.push(sp.user);
        ids.extend(vocab.encode(&prompt).into_iter().map(|t| t as i32));
        ids.push(sp.assistant);
        ids.push(sp.think_end);
    }

    // Decode surface: open the model, optionally install the NPU.
    let runner = DecodeRunner::open(path, 256)?;
    let mut sess = DecodeSession::new(&runner);
    #[cfg(feature = "ascend")]
    if let Ok(elf) = std::env::var("DS4_DEVICE_ELF") {
        let ids = engine_ascend::decode_runner::env_device_ids();
        sess.install_devices(Path::new(&elf), &ids)?;
        eprintln!(
            "[ds4_generate] NPU offload: {:?} on chips {:?}",
            sess.offloaded_by(),
            ids
        );
    }

    eprintln!(
        "[ds4_generate] prefill {} prompt tokens (max_new={max_new})…",
        ids.len()
    );
    let t0 = std::time::Instant::now();
    sess.prefill(&ids)?;
    eprintln!("[ds4_generate] prefill done in {:.1}s", t0.elapsed().as_secs_f64());

    let mut dec = StreamingDecoder::new(&vocab);
    let mut out = std::io::stdout();
    let mut next = argmax(sess.logits());
    let mut n = 0usize;
    let g0 = std::time::Instant::now();
    while n < max_new {
        if next == sp.eos {
            break;
        }
        let piece = dec.push(next as u32);
        print!("{piece}");
        out.flush().ok();
        sess.step(next)?;
        next = argmax(sess.logits());
        n += 1;
    }
    print!("{}", dec.flush());
    println!();
    let dt = g0.elapsed().as_secs_f64();
    eprintln!(
        "[ds4_generate] {n} tokens in {:.1}s ({:.3} tok/s)",
        dt,
        n as f64 / dt.max(1e-9)
    );
    Ok(())
}
