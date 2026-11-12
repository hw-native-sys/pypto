// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Single-stream serving: open the model once and generate. This is the actual
//! decode surface (`engine_ascend::decode_runner`) the controller plans for — the
//! planned multi-chip split is the throughput path and is not yet wired, so this
//! runs on device 0 (weights resident if they fit host RAM, else SSD-streamed).

use anyhow::Result;
use engine_ascend::decode_runner::{DecodeRunner, DecodeSession};
use engine_ascend::sampling::argmax;
use engine_core::gguf::GgufFile;
use engine_core::tokenizer::{StreamingDecoder, Vocab};
use std::io::Write as _;
use std::path::Path;

/// Serve `model`. With `prompt`, answer it once; otherwise read prompts from
/// stdin until EOF. `max_new` bounds tokens generated per prompt.
pub fn run(model: &Path, prompt: Option<&str>, max_new: usize) -> Result<()> {
    let g = GgufFile::open(model)?;
    let vocab = Vocab::from_gguf(&g)?;
    let sp = vocab.special_tokens()?;

    eprintln!("\nloading model for single-stream serve…");
    // A full DS4-Flash exceeds single-device host RAM, so stream weights from SSD
    // unless the operator already chose otherwise. The load budget is checked deep
    // in prefill, so the flag has to be set before any weights are touched.
    // (Resident multi-chip placement is the throughput path, not yet wired.)
    if std::env::var("DS4_SSD_STREAM").is_err() {
        std::env::set_var("DS4_SSD_STREAM", "1");
    }
    let runner = DecodeRunner::open(model, 256)?;

    let generate = |text: &str| -> Result<()> {
        // DS4 chat wrap: BOS, User turn, the prompt, Assistant, close </think>.
        let mut ids = vec![sp.bos, sp.user];
        ids.extend(vocab.encode(text).into_iter().map(|t| t as i32));
        ids.push(sp.assistant);
        ids.push(sp.think_end);

        let mut sess = DecodeSession::new(&runner);
        #[cfg(feature = "ascend")]
        if let Ok(elf) = std::env::var("DS4_DEVICE_ELF") {
            let ids = engine_ascend::decode_runner::env_device_ids();
            sess.install_devices(Path::new(&elf), &ids)?;
        }
        sess.prefill(&ids)?;

        let mut dec = StreamingDecoder::new(&vocab);
        let mut out = std::io::stdout();
        let mut next = argmax(sess.logits());
        // Per-token timing. Without it a run reports a word and no rate, which is
        // what made the first dry run unreadable: 480 s of wall clock and a single
        // token, with no way to tell generation from waiting on stdin.
        let timing = std::env::var("DS4_TIMING").is_ok();
        let t_start = std::time::Instant::now();
        let mut n_tok: u32 = 0;
        for _ in 0..max_new {
            if next == sp.eos {
                break;
            }
            print!("{}", dec.push(next as u32));
            out.flush().ok();
            sess.step(next)?;
            next = argmax(sess.logits());
            n_tok += 1;
            if timing {
                let el = t_start.elapsed().as_secs_f64();
                eprintln!(
                    "token={} elapsed={:.3}s ms_per_step={:.1} tok_per_s={:.3}",
                    n_tok,
                    el,
                    1000.0 * el / n_tok as f64,
                    n_tok as f64 / el
                );
            }
        }
        let total = t_start.elapsed().as_secs_f64();
        if n_tok > 0 {
            eprintln!(
                "decode: tokens={} elapsed={:.3}s ms_per_step={:.1} tok_per_s={:.3}",
                n_tok,
                total,
                1000.0 * total / n_tok as f64,
                n_tok as f64 / total
            );
        }
        print!("{}", dec.flush());
        println!();
        Ok(())
    };

    match prompt {
        Some(p) => generate(p)?,
        None => {
            use std::io::BufRead as _;
            eprintln!("interactive — type a prompt and press Enter (Ctrl-D to exit):");
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                generate(&line)?;
            }
        }
    }
    Ok(())
}
