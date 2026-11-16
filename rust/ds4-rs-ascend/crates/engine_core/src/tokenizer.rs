// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// check-english-only: allow -- this file spells the MODEL's own special tokens,
// which use U+FF5C (`<｜begin▁of▁sentence｜>`) and U+3008/U+3009 (`〈|EOS|〉`), and its
// byte-level tests use real multibyte input. Those are identifiers and test data,
// not prose: rewriting them would change which token ids the tokenizer resolves.
//! DS4-compatible byte-level BPE tokenizer.
//!
//! Faithful Rust port of `bpe_tokenize_text` + `bpe_emit_piece` from antirez's
//! `ds4.c:13413-13876`. DS4 declares `tokenizer.ggml.pre = "joyai-llm"`; the
//! pre-tokenizer is a hand-rolled state machine, not a regex.
//!
//! Pipeline per text input:
//!   1. JoyAI pre-tokenizer splits the input into pieces (digits-up-to-3,
//!      CJK runs, punctuation+ASCII-letter, letter runs, space-leading words,
//!      etc.). The split shape matters — see ds4.c:13788-13805.
//!   2. Each piece is byte-encoded via the GPT-2 byte→codepoint map
//!      (printable ASCII / Latin-1 punct stay put; the other ~68 bytes go to
//!      codepoints 256+n).
//!   3. The byte-encoded piece is split into single UTF-8 chars, then
//!      iteratively merged using the lowest-rank pair from `tokenizer.ggml.merges`.
//!   4. Final symbols are looked up in `tokenizer.ggml.tokens`; if a symbol
//!      isn't in the vocab, fall back to per-byte token lookup.
//!
//! The CPU path is end-to-end vendor-only (`anyhow` is the only dep).

#![allow(dead_code)]

use std::collections::HashMap;

use anyhow::{anyhow, Result};

/// A loaded BPE vocabulary.
pub struct Vocab {
    /// Token string → token id. Strings are stored in their GGUF representation
    /// (post byte-encoding: e.g. " a" appears as "Ġa").
    pub token_to_id: HashMap<String, u32>,
    /// Reverse map (for detokenization).
    pub id_to_token: Vec<String>,
    /// `"a b"` → merge rank (smaller = applied first). The space separator
    /// matches the GGUF `tokenizer.ggml.merges` format.
    pub merge_rank: HashMap<String, u32>,
}

impl Vocab {
    pub fn new(tokens: Vec<String>, merges: Vec<String>) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            merge_rank.insert(m.clone(), i as u32);
        }
        Self {
            token_to_id,
            id_to_token: tokens,
            merge_rank,
        }
    }

    pub fn n_vocab(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut pos = 0;

        while pos < len {
            let start = pos;
            let c = bytes[pos];

            if ascii_digit(c) {
                let mut n = 0;
                while pos < len && ascii_digit(bytes[pos]) && n < 3 {
                    pos += 1;
                    n += 1;
                }
            } else if joyai_cjk_at(bytes, pos) {
                loop {
                    pos = next_utf8_char(bytes, pos);
                    if pos >= len || !joyai_cjk_at(bytes, pos) {
                        break;
                    }
                }
            } else if joyai_ascii_punct_symbol(c) && pos + 1 < len && ascii_alpha(bytes[pos + 1]) {
                pos += 1;
                while pos < len && ascii_alpha(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_letter_like_at(bytes, pos) {
                pos = joyai_consume_letters(bytes, pos);
            } else if !ascii_newline(c)
                && !joyai_ascii_punct_symbol(c)
                && pos + 1 < len
                && joyai_letter_like_at(bytes, pos + 1)
            {
                pos += 1;
                pos = joyai_consume_letters(bytes, pos);
            } else if c == b' ' && pos + 1 < len && joyai_ascii_punct_symbol(bytes[pos + 1]) {
                pos += 1;
                while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_ascii_punct_symbol(c) {
                while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if ascii_space(c) {
                let mut p = pos;
                let mut last_newline_end = 0;
                while p < len && ascii_space(bytes[p]) {
                    let sc = bytes[p];
                    p += 1;
                    if ascii_newline(sc) {
                        last_newline_end = p;
                    }
                }
                if last_newline_end != 0 {
                    pos = last_newline_end;
                } else if p < len
                    && p > pos + 1
                    && (joyai_letter_like_at(bytes, p) || joyai_ascii_punct_symbol(bytes[p]))
                {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(bytes, pos);
            }

            if pos == start {
                pos = next_utf8_char(bytes, pos);
            }
            self.bpe_emit_piece(&bytes[start..pos], &mut out);
        }
        out
    }

    /// BPE-merge one pre-tokenized piece, then look up each merged symbol.
    fn bpe_emit_piece(&self, raw_piece: &[u8], out: &mut Vec<u32>) {
        let encoded = byte_encode(raw_piece);
        let mut sym: Vec<String> = utf8_split_chars(&encoded);

        loop {
            let mut best_i: Option<usize> = None;
            let mut best_rank = u32::MAX;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(r) = self.merge_rank_of(&sym[i], &sym[i + 1]) {
                    if r < best_rank {
                        best_rank = r;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else {
                break;
            };
            let merged = format!("{}{}", sym[i], sym[i + 1]);
            sym[i] = merged;
            sym.remove(i + 1);
        }

        for s in &sym {
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else {
                for b in s.bytes() {
                    let mut buf = [0u8; 1];
                    buf[0] = b;
                    if let Ok(s1) = std::str::from_utf8(&buf) {
                        if let Some(&id) = self.token_to_id.get(s1) {
                            out.push(id);
                        }
                    }
                    // multi-byte UTF-8 fallback: try each codepoint slice
                }
                // Fallback for multi-byte chars: walk char boundaries
                if !s.is_empty() {
                    for ch in s.chars() {
                        let cs = ch.to_string();
                        if let Some(&id) = self.token_to_id.get(&cs) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn merge_rank_of(&self, a: &str, b: &str) -> Option<u32> {
        let key = format!("{a} {b}");
        self.merge_rank.get(&key).copied()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut buf = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            let s = self
                .id_to_token
                .get(id as usize)
                .ok_or_else(|| anyhow!("token id {id} out of range"))?;
            // Reverse byte-encoding: walk codepoints and map back to bytes.
            for ch in s.chars() {
                let cp = ch as u32;
                let b = gpt2_codepoint_to_byte(cp);
                buf.push(b);
            }
        }
        String::from_utf8(buf).map_err(|e| anyhow!("decode utf8: {e}"))
    }

    /// Raw bytes of a single token id (GPT-2 codepoint→byte map). Used by
    /// [`StreamingDecoder`] for incremental detokenize. Unknown id → empty.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match self.id_to_token.get(id as usize) {
            Some(s) => s
                .chars()
                .map(|ch| gpt2_codepoint_to_byte(ch as u32))
                .collect(),
            None => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF loading + DS4 chat template + streaming detokenize
// (ports of ds4.c encode_chat_prompt / ds4_chat_append_message, 14822-14970,
//  and ds4_server.c append_tools_prompt_text, 1995-2020)
// ---------------------------------------------------------------------------

use crate::gguf::{GgufFile, MetaValue};

/// DS4 reasoning mode (mirrors `ds4_think_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    /// True iff a `<think>` block is opened (High or Max).
    pub fn enabled(self) -> bool {
        matches!(self, ThinkMode::High | ThinkMode::Max)
    }
}

/// One chat message for prompt construction.
#[derive(Clone, Copy, Debug)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Special-token ids resolved from the vocab (ds4.c:14822).
#[derive(Clone, Copy, Debug)]
pub struct SpecialTokens {
    pub bos: i32,
    pub eos: i32,
    pub user: i32,
    pub assistant: i32,
    pub think_start: i32,
    pub think_end: i32,
    pub dsml: i32,
}

/// Mirrors `DS4_REASONING_EFFORT_MAX_PREFIX` (ds4.c:63-66).
const REASONING_EFFORT_MAX_PREFIX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

impl Vocab {
    /// Build a `Vocab` from a GGUF's `tokenizer.ggml.tokens` + `.merges` arrays.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let str_array = |key: &str| -> Result<Vec<String>> {
            match g.get_meta(key) {
                Some(MetaValue::Array { values, .. }) => values
                    .iter()
                    .map(|v| match v {
                        MetaValue::String(s) => Ok(s.clone()),
                        other => Err(anyhow!("{key}: non-string array element {other:?}")),
                    })
                    .collect(),
                Some(other) => Err(anyhow!("{key}: expected String array, got {other:?}")),
                None => Err(anyhow!("missing GGUF metadata: {key}")),
            }
        };
        let tokens = str_array("tokenizer.ggml.tokens")?;
        let merges = str_array("tokenizer.ggml.merges")?;
        Ok(Self::new(tokens, merges))
    }

    /// Look up a token id by its exact (byte-encoded) string, or -1 if absent.
    pub fn token_id(&self, s: &str) -> i32 {
        self.token_to_id.get(s).map(|&id| id as i32).unwrap_or(-1)
    }

    /// Resolve DS4 special-token ids (ds4.c:14822). Errors if the required
    /// role markers are missing.
    pub fn special_tokens(&self) -> Result<SpecialTokens> {
        let st = SpecialTokens {
            bos: self.token_id("<｜begin▁of▁sentence｜>"),
            eos: self.token_id("<｜end▁of▁sentence｜>"),
            user: self.token_id("<｜User｜>"),
            assistant: self.token_id("<｜Assistant｜>"),
            think_start: self.token_id("<think>"),
            think_end: self.token_id("</think>"),
            dsml: self.token_id("｜DSML｜"),
        };
        for (name, id) in [
            ("bos", st.bos),
            ("eos", st.eos),
            ("user", st.user),
            ("assistant", st.assistant),
            ("think_end", st.think_end),
        ] {
            anyhow::ensure!(id >= 0, "vocab missing required special token: {name}");
        }
        Ok(st)
    }

    /// Resolve GLM-5.2 (`glm-dsa`) special-token ids. GLM uses a different marker
    /// set than DeepSeek: `[gMASK]`/`<sop>` prefix, `<|user|>`/`<|assistant|>`
    /// role markers, `<|endoftext|>` eos, `<think>`/`</think>`. `bos` (the gMASK
    /// prefix) is optional. Errors only if the role markers or eos are missing.
    pub fn special_tokens_glm(&self) -> Result<SpecialTokens> {
        let g_mask = self.token_id("[gMASK]");
        let eos = {
            let e = self.token_id("<|endoftext|>");
            if e >= 0 {
                e
            } else {
                self.token_id("<|user|>")
            }
        };
        let st = SpecialTokens {
            bos: g_mask, // [gMASK] prefix (optional; -1 tolerated by the GLM prompt builder)
            eos,
            user: self.token_id("<|user|>"),
            assistant: self.token_id("<|assistant|>"),
            think_start: self.token_id("<think>"),
            think_end: self.token_id("</think>"),
            dsml: -1,
        };
        for (name, id) in [
            ("eos", st.eos),
            ("user", st.user),
            ("assistant", st.assistant),
        ] {
            anyhow::ensure!(id >= 0, "GLM vocab missing required special token: {name}");
        }
        Ok(st)
    }

    /// Build the "## Tools" instruction block injected into the prompt when the
    /// request carries tools (ds4_server.c:1995-2020). `tool_schemas` is the
    /// caller-formatted schema text (one per tool).
    pub fn tools_prompt_block(tool_schemas: &str) -> String {
        format!(
            "## Tools\n\nYou have access to a set of tools to help answer the user question. \
You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
...\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
String parameters: raw text, set `string=\"true\"`. Other types (numbers, booleans, arrays, objects): \
JSON format, set `string=\"false\"`.\n\n\
### Available Tool Schemas\n\n{tool_schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas.\n\n"
        )
    }

    /// Build the full prompt token stream from chat `messages`, mirroring
    /// `encode_chat_prompt` + `ds4_chat_append_message` (ds4.c:14841-14970):
    /// `bos` + [max-reasoning prefix] + system text + per-message role markers +
    /// The tools block for CHATML models (Qwen), which are trained on the
    /// Hermes convention and know nothing about DSML.
    ///
    /// Handing a Qwen model the DSML instructions — which is what happened
    /// before this existed — asks it to emit a format it has never seen. It
    /// improvises: measured with Codex on Qwen3.6, one turn produced a
    /// `<tool_call>` with no `name` field and the next produced no call at all,
    /// just the program in a markdown fence. Neither can be executed, so the
    /// agent silently does nothing while looking like it works.
    ///
    /// The wording mirrors the Qwen chat template, because that is the text the
    /// model was tuned against; paraphrasing it is a good way to lose the
    /// behaviour again.
    pub fn tools_prompt_block_chatml(tool_schemas: &str) -> String {
        format!(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool_schemas}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>\n"
        )
    }

    /// `<｜Assistant｜>` + (`<think>` if thinking else `</think>`). `tools`, when
    /// present, is appended as the "## Tools" block after the system text.
    pub fn build_chat_prompt(
        &self,
        st: &SpecialTokens,
        messages: &[ChatMessage<'_>],
        think: ThinkMode,
        tools: Option<&str>,
    ) -> Vec<i32> {
        self.build_chat_prompt_boundary(st, messages, think, tools).0
    }

    /// Like [`build_chat_prompt`], but also returns the token index where the
    /// system section ends: after `bos` + the reasoning-effort prefix + every
    /// `system`/`developer` message + the tools block, before the first
    /// conversation turn. This is the boundary a system-prompt KV snapshot is
    /// taken at (see docs/design/system-prompt-kv-snapshot.md). With no system
    /// message the boundary is `bos (+ prefix)` — the tools block, emitted
    /// after history in that case, is not part of the reusable prefix.
    pub fn build_chat_prompt_boundary(
        &self,
        st: &SpecialTokens,
        messages: &[ChatMessage<'_>],
        think: ThinkMode,
        tools: Option<&str>,
    ) -> (Vec<i32>, usize) {
        let mut out: Vec<i32> = Vec::new();
        out.push(st.bos);
        if think == ThinkMode::Max {
            self.encode_into(REASONING_EFFORT_MAX_PREFIX, &mut out);
        }
        let mut sys_end = out.len();
        // System/developer text goes in raw (no marker) first; tools block is
        // appended to the system section.
        let mut tools_emitted = false;
        let emit_tools = |this: &Self, out: &mut Vec<i32>| {
            if let Some(schemas) = tools {
                this.encode_into(&Self::tools_prompt_block(schemas), out);
            }
        };
        for m in messages {
            match m.role {
                "system" | "developer" => {
                    self.encode_into(m.content, &mut out);
                    if !tools_emitted {
                        emit_tools(self, &mut out);
                        tools_emitted = true;
                    }
                    sys_end = out.len();
                }
                "assistant" => {
                    out.push(st.assistant);
                    if !m.content.starts_with("<think>") && !m.content.starts_with("</think>") {
                        out.push(st.think_end);
                    }
                    self.encode_into(m.content, &mut out);
                }
                "tool" | "function" => {
                    out.push(st.user);
                    self.encode_into("Tool: ", &mut out);
                    self.encode_into(m.content, &mut out);
                }
                _ => {
                    // user (and any unknown role)
                    out.push(st.user);
                    self.encode_into(m.content, &mut out);
                }
            }
        }
        // If there was no system message, still emit the tools block (after
        // history, before the assistant turn) so the model sees it.
        if !tools_emitted {
            emit_tools(self, &mut out);
        }
        // Start the assistant turn.
        out.push(st.assistant);
        if think.enabled() {
            out.push(st.think_start);
        } else {
            out.push(st.think_end);
        }
        (out, sys_end)
    }

    /// Build a GLM-5.2 chat prompt: `[gMASK]<sop>` prefix, then per message a
    /// `<|role|>\n{content}` block, ending with `<|assistant|>\n` to start the
    /// model's turn. Mirrors the GLM-4.5/5 chat template (thinking left to the
    /// model). `system`/`developer` → `<|system|>`, `tool`/`function` →
    /// `<|observation|>`, everything else → `<|user|>`.
    pub fn build_chat_prompt_glm(
        &self,
        st: &SpecialTokens,
        messages: &[ChatMessage<'_>],
    ) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        if st.bos >= 0 {
            out.push(st.bos); // [gMASK]
        }
        let sop = self.token_id("<sop>");
        if sop >= 0 {
            out.push(sop);
        }
        let sys = self.token_id("<|system|>");
        let obs = self.token_id("<|observation|>");
        for m in messages {
            let marker = match m.role {
                "system" | "developer" => sys,
                "assistant" => st.assistant,
                "tool" | "function" => obs,
                _ => st.user,
            };
            if marker >= 0 {
                out.push(marker);
            }
            self.encode_into(&format!("\n{}", m.content), &mut out);
        }
        // Start the assistant turn.
        out.push(st.assistant);
        self.encode_into("\n", &mut out);
        out
    }

    /// Resolve Qwen2 (`DeepSeek-R1-Distill-Qwen`) special-token ids. R1-Distill
    /// uses the **DeepSeek** chat template (NOT Qwen `<|im_start|>`): the same
    /// `<｜begin▁of▁sentence｜>` bos, `<｜User｜>`/`<｜Assistant｜>` role markers,
    /// `<think>`/`</think>` reasoning, `<｜end▁of▁sentence｜>` eos. So this is a
    /// thin wrapper over [`Self::special_tokens`]; kept as a named entry point so
    /// the server's arch dispatch reads symmetrically (`_qwen` alongside `_glm`).
    pub fn special_tokens_qwen(&self) -> Result<SpecialTokens> {
        // R1-Distill-Qwen shares the DeepSeek-R1 markers (bos + `<｜User｜>` /
        // `<｜Assistant｜>`). Standard Qwen2.5 (coder / instruct) uses ChatML
        // (`<|im_start|>` / `<|im_end|>`, NO bos), so resolve those instead.
        if self.token_id("<｜Assistant｜>") >= 0 {
            return self.special_tokens();
        }
        let im_start = self.token_id("<|im_start|>");
        let im_end = self.token_id("<|im_end|>");
        let eot = self.token_id("<|endoftext|>");
        let eos = if im_end >= 0 { im_end } else { eot };
        anyhow::ensure!(
            im_start >= 0 && eos >= 0,
            "vocab has neither DeepSeek-R1 nor ChatML (<|im_start|>/<|im_end|>) chat markers"
        );
        Ok(SpecialTokens {
            bos: -1, // Qwen2.5/Qwen3 have no BOS
            eos,
            user: -1,
            assistant: -1,
            // Qwen3 has native thinking (`<think>`/`</think>`); Qwen2.5 lacks them (→ -1).
            think_start: self.token_id("<think>"),
            think_end: self.token_id("</think>"),
            dsml: -1,
        })
    }

    /// Build a Qwen2 chat prompt. R1-Distill-Qwen shares the DeepSeek-R1 template
    /// (bos + `<｜User｜>`/`<｜Assistant｜>` + `<think>`); standard Qwen2.5 (coder /
    /// instruct) uses **ChatML**: `<|im_start|>{role}\n{content}<|im_end|>\n` turns
    /// then a trailing `<|im_start|>assistant\n`. Detected by the vocab's markers.
    pub fn build_chat_prompt_qwen(
        &self,
        st: &SpecialTokens,
        messages: &[ChatMessage<'_>],
        think: ThinkMode,
        tools: Option<&str>,
    ) -> Vec<i32> {
        if self.token_id("<｜Assistant｜>") >= 0 {
            return self.build_chat_prompt(st, messages, think, tools);
        }
        // ChatML (Qwen2.5). `<|im_start|>`/`<|im_end|>` pushed as ids; role names +
        // content + newlines encoded as text.
        let im_start = self.token_id("<|im_start|>");
        let im_end = self.token_id("<|im_end|>");
        let mut out: Vec<i32> = Vec::new();
        let turn = |this: &Self, out: &mut Vec<i32>, role: &str, content: &str| {
            if im_start >= 0 {
                out.push(im_start);
            }
            this.encode_into(role, out);
            this.encode_into("\n", out);
            this.encode_into(content, out);
            if im_end >= 0 {
                out.push(im_end);
            }
            this.encode_into("\n", out);
        };
        // TOOLS GO IN THE SYSTEM TURN, and they were previously not emitted at
        // all on this path: the ChatML builder took no `tools` argument, so an
        // agent's tool definitions were dropped and the model was asked to use
        // tools it had never been told existed. Appending to the caller's own
        // system turn (rather than adding a second one) keeps the ChatML shape
        // the template expects.
        let tools_block = tools.map(Self::tools_prompt_block_chatml);
        let has_system = messages
            .iter()
            .any(|m| m.role == "system" || m.role == "developer");
        if !has_system {
            let mut sys =
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string();
            if let Some(t) = &tools_block {
                sys.push_str("\n\n");
                sys.push_str(t);
            }
            turn(self, &mut out, "system", &sys);
        }
        let mut tools_emitted = has_system && tools_block.is_none();
        for m in messages {
            let role = match m.role {
                "system" | "developer" => "system",
                "assistant" => "assistant",
                _ => "user", // user / tool / function / unknown
            };
            if role == "system" && !tools_emitted {
                tools_emitted = true;
                let mut c = m.content.to_string();
                if let Some(t) = &tools_block {
                    c.push_str("\n\n");
                    c.push_str(t);
                }
                turn(self, &mut out, role, &c);
            } else {
                turn(self, &mut out, role, m.content);
            }
        }
        if im_start >= 0 {
            out.push(im_start);
        }
        self.encode_into("assistant\n", &mut out);
        out
    }

    /// Resolve Laguna (Poolside `laguna` arch) special-token ids. Mirrors the
    /// `DS4_MODEL_FAMILY_LAGUNA` branch of `vocab_load` (ds4.c:36636-36660): BOS/EOS
    /// are both the `〈|EOS|〉` marker (bos_token_id == eos_token_id == 2 in the gguf),
    /// the assistant turn opens with `<assistant>` and closes with `</assistant>`
    /// (eot), reasoning uses `<think>`/`</think>`. Unlike GLM/DeepSeek there are NO
    /// `<system>`/`<user>` special ids — those tags are plain BPE text (`user`/`dsml`
    /// left -1), matching `special_token_at`'s `token < 0` skip.
    pub fn special_tokens_laguna(&self) -> Result<SpecialTokens> {
        let eos = self.token_id("〈|EOS|〉");
        let st = SpecialTokens {
            bos: eos, // laguna prefixes the EOS marker as its BOS (bos_id == eos_id)
            eos,
            user: -1, // <user>/<system> are plain BPE text in laguna (no special id)
            assistant: self.token_id("<assistant>"),
            think_start: self.token_id("<think>"),
            think_end: self.token_id("</think>"),
            dsml: -1,
        };
        for (name, id) in [
            ("eos", st.eos),
            ("assistant", st.assistant),
            ("think_end", st.think_end),
        ] {
            anyhow::ensure!(
                id >= 0,
                "Laguna vocab missing required special token: {name}"
            );
        }
        Ok(st)
    }

    /// Build a Laguna chat prompt. Faithful port of antirez's
    /// `render_laguna_chat_prompt_text` (ds4_server.c:2658) + the special-token-aware
    /// `tokenize_rendered_chat_vocab` (ds4.c:36871): the text is rendered once
    /// (`〈|EOS|〉` prefix, `<system>…</system>` block with the Poolside default when no
    /// system message is given, `<user>…</user>` turns, and a trailing `<assistant>` +
    /// `<think>`/`</think>` to start the model's turn), then tokenized in a SINGLE pass
    /// so BPE merges span the `<system>`/`<user>` tags exactly as the engine does. Only
    /// `〈|EOS|〉`, `<assistant>`, `</assistant>`, `<think>`, `</think>`, `<tool_call>`,
    /// `</tool_call>` are emitted as special ids; everything else is BPE.
    pub fn build_chat_prompt_laguna(
        &self,
        _st: &SpecialTokens,
        messages: &[ChatMessage<'_>],
        think: ThinkMode,
    ) -> Vec<i32> {
        let text = self.render_laguna_text(messages, think);
        self.tokenize_laguna_rendered(&text)
    }

    /// Render the laguna chat prompt to text (mirrors `render_laguna_chat_prompt_text`).
    fn render_laguna_text(&self, messages: &[ChatMessage<'_>], think: ThinkMode) -> String {
        let thinking = think.enabled();
        let default_system =
            "You are a helpful, conversationally-fluent assistant made by Poolside. \
You are here to be helpful to users through natural language conversations.";
        // A leading system/developer message overrides the Poolside default.
        let mut system: &str = default_system;
        let mut start = 0;
        if let Some(m) = messages.first() {
            if m.role == "system" || m.role == "developer" {
                system = m.content;
                start = 1;
            }
        }
        let sys_nonspace = system.chars().any(|c| !c.is_whitespace());
        let mut out = String::new();
        out.push_str("〈|EOS|〉");
        // The <system> block is emitted when there is system text or thinking is on
        // (tools are not part of the count path).
        if sys_nonspace || thinking {
            out.push_str("<system>");
            if sys_nonspace {
                out.push_str(system.trim_end());
            }
            out.push_str("</system>\n");
        }
        let mut pending_assistant = false;
        for m in &messages[start..] {
            match m.role {
                "system" | "developer" => {
                    out.push_str("<system>");
                    out.push_str(m.content);
                    out.push_str("</system>\n");
                }
                "assistant" => {
                    out.push_str("<assistant>");
                    out.push_str(if thinking {
                        "<think></think>"
                    } else {
                        "</think>"
                    });
                    out.push_str(m.content);
                    out.push_str("</assistant>\n");
                    pending_assistant = false;
                }
                "tool" | "function" => {
                    out.push_str("<tool_response>");
                    out.push_str(m.content);
                    out.push_str("</tool_response>\n");
                    pending_assistant = true;
                }
                _ => {
                    // user (and any unknown role)
                    out.push_str("<user>");
                    out.push_str(m.content);
                    out.push_str("</user>\n");
                    pending_assistant = true;
                }
            }
        }
        if pending_assistant {
            out.push_str("<assistant>");
            out.push_str(if thinking { "<think>" } else { "</think>" });
        }
        out
    }

    /// Special-token-aware tokenize of a rendered laguna prompt (mirrors
    /// `tokenize_rendered_chat_vocab`): non-special spans are BPE-encoded contiguously
    /// so merges cross the `<system>`/`<user>` tags, and each recognised marker becomes
    /// its single id. First-match-in-order wins, matching `special_token_at`.
    fn tokenize_laguna_rendered(&self, text: &str) -> Vec<i32> {
        let specials: Vec<(&str, i32)> = [
            "〈|EOS|〉",
            "<assistant>",
            "</assistant>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
        ]
        .iter()
        .map(|s| (*s, self.token_id(s)))
        .filter(|(_, id)| *id >= 0)
        .collect();

        let bytes = text.as_bytes();
        let mut out: Vec<i32> = Vec::new();
        let mut span_start = 0usize;
        let mut p = 0usize;
        'outer: while p < bytes.len() {
            for (tok, id) in &specials {
                if bytes[p..].starts_with(tok.as_bytes()) {
                    if p > span_start {
                        self.encode_into(&text[span_start..p], &mut out);
                    }
                    out.push(*id);
                    p += tok.len();
                    span_start = p;
                    continue 'outer;
                }
            }
            p += 1;
        }
        if span_start < bytes.len() {
            self.encode_into(&text[span_start..], &mut out);
        }
        out
    }

    /// `encode` into an existing i32 buffer (used by the chat template).
    fn encode_into(&self, text: &str, out: &mut Vec<i32>) {
        for id in self.encode(text) {
            out.push(id as i32);
        }
    }
}

/// Incremental, UTF-8-safe detokenizer for streaming output. Byte-level BPE can
/// split a multi-byte char across tokens, so we buffer trailing incomplete
/// bytes and only emit complete UTF-8 prefixes.
pub struct StreamingDecoder<'v> {
    vocab: &'v Vocab,
    buf: Vec<u8>,
}

impl<'v> StreamingDecoder<'v> {
    pub fn new(vocab: &'v Vocab) -> Self {
        Self {
            vocab,
            buf: Vec::new(),
        }
    }

    /// Push one token id; return any newly-complete UTF-8 text (may be empty if
    /// the token only added a partial multi-byte char).
    pub fn push(&mut self, id: u32) -> String {
        self.buf.extend_from_slice(&self.vocab.token_bytes(id));
        self.take_complete()
    }

    /// Drain whatever complete UTF-8 is buffered; lossily decode any trailing
    /// incomplete bytes (call at end of stream).
    pub fn flush(&mut self) -> String {
        let mut s = self.take_complete();
        if !self.buf.is_empty() {
            s.push_str(&String::from_utf8_lossy(&self.buf));
            self.buf.clear();
        }
        s
    }

    /// Split off the longest valid-UTF-8 prefix of `buf`, leaving any trailing
    /// incomplete bytes buffered.
    fn take_complete(&mut self) -> String {
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_string();
                self.buf.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    return String::new();
                }
                let out = String::from_utf8_lossy(&self.buf[..valid]).into_owned();
                self.buf.drain(..valid);
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-tokenizer primitives (mirror ds4.c)
// ---------------------------------------------------------------------------

fn ascii_alpha(c: u8) -> bool {
    (b'A'..=b'Z').contains(&c) || (b'a'..=b'z').contains(&c)
}
fn ascii_digit(c: u8) -> bool {
    (b'0'..=b'9').contains(&c)
}
fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}
fn ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        1
    }
}

fn next_utf8_char(bytes: &[u8], pos: usize) -> usize {
    let mut n = utf8_len_from_first_byte(bytes[pos]);
    if pos + n > bytes.len() {
        n = 1;
    }
    pos + n
}

fn utf8_peek_one(bytes: &[u8], pos: usize) -> (u32, usize) {
    let c0 = bytes[pos];
    let mut n = utf8_len_from_first_byte(c0);
    if pos + n > bytes.len() {
        n = 1;
    }
    let cp = match n {
        1 => c0 as u32,
        2 => ((c0 & 0x1f) as u32) << 6 | (bytes[pos + 1] & 0x3f) as u32,
        3 => {
            ((c0 & 0x0f) as u32) << 12
                | ((bytes[pos + 1] & 0x3f) as u32) << 6
                | (bytes[pos + 2] & 0x3f) as u32
        }
        _ => {
            ((c0 & 0x07) as u32) << 18
                | ((bytes[pos + 1] & 0x3f) as u32) << 12
                | ((bytes[pos + 2] & 0x3f) as u32) << 6
                | (bytes[pos + 3] & 0x3f) as u32
        }
    };
    (cp, pos + n)
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn joyai_cjk_at(bytes: &[u8], pos: usize) -> bool {
    if bytes[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(bytes, pos);
    utf8_is_cjk_hira_kata(cp)
}

fn joyai_letter_like_at(bytes: &[u8], pos: usize) -> bool {
    let c = bytes[pos];
    if c < 128 {
        return ascii_alpha(c);
    }
    true
}

fn joyai_consume_letters(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && joyai_letter_like_at(bytes, pos) {
        pos = next_utf8_char(bytes, pos);
    }
    pos
}

// ---------------------------------------------------------------------------
// GPT-2 byte ↔ codepoint mapping
// ---------------------------------------------------------------------------

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return b as u32;
    }
    let mut n: u32 = 0;
    for x in 0u32..256 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 {
            continue;
        }
        if x == b as u32 {
            return 256 + n;
        }
        n += 1;
    }
    b as u32
}

fn gpt2_codepoint_to_byte(cp: u32) -> u8 {
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return cp as u8;
    }
    if cp < 256 {
        return cp as u8;
    }
    let n = cp - 256;
    let mut i = 0u32;
    for x in 0u32..256 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 {
            continue;
        }
        if i == n {
            return xb;
        }
        i += 1;
    }
    0
}

fn byte_encode(raw: &[u8]) -> String {
    let mut s = String::with_capacity(raw.len() * 2);
    for &b in raw {
        let cp = gpt2_byte_to_codepoint(b);
        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
    }
    s
}

fn utf8_split_chars(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_classifiers_basics() {
        assert!(ascii_digit(b'5'));
        assert!(!ascii_digit(b'a'));
        assert!(ascii_alpha(b'A'));
        assert!(ascii_alpha(b'z'));
        assert!(!ascii_alpha(b'5'));
        assert!(ascii_space(b' '));
        assert!(ascii_space(b'\t'));
        assert!(ascii_newline(b'\n'));
        assert!(!ascii_newline(b' '));
    }

    #[test]
    fn joyai_punct_symbol_table() {
        // ! @ [ { are all in.
        assert!(joyai_ascii_punct_symbol(b'!'));
        assert!(joyai_ascii_punct_symbol(b'@'));
        assert!(joyai_ascii_punct_symbol(b'['));
        assert!(joyai_ascii_punct_symbol(b'{'));
        // Letters & digits are out.
        assert!(!joyai_ascii_punct_symbol(b'a'));
        assert!(!joyai_ascii_punct_symbol(b'0'));
        // Space is out (handled separately).
        assert!(!joyai_ascii_punct_symbol(b' '));
    }

    #[test]
    fn gpt2_byte_codepoint_roundtrip_printable() {
        // Printable ASCII bytes map to themselves.
        for b in 33u8..=126 {
            let cp = gpt2_byte_to_codepoint(b);
            assert_eq!(cp, b as u32, "byte {b} → cp {cp}");
            assert_eq!(gpt2_codepoint_to_byte(cp), b);
        }
    }

    #[test]
    fn gpt2_space_maps_to_unicode_codepoint() {
        // Space (0x20) is *not* in the printable set per gpt-2's convention.
        let cp = gpt2_byte_to_codepoint(b' ');
        assert!(
            cp >= 256,
            "space should map to remapped codepoint, got {cp}"
        );
        // Round-trip
        assert_eq!(gpt2_codepoint_to_byte(cp), b' ');
    }

    #[test]
    fn gpt2_byte_to_codepoint_remaps_control_bytes() {
        // 0x00 (NUL) is not printable → must map into ≥256 range.
        assert!(gpt2_byte_to_codepoint(0) >= 256);
        assert!(gpt2_byte_to_codepoint(b'\n') >= 256);
    }

    #[test]
    fn utf8_len_from_first_byte_known_cases() {
        assert_eq!(utf8_len_from_first_byte(b'A'), 1);
        assert_eq!(utf8_len_from_first_byte(0xc3), 2); // start of Latin-1 ext
        assert_eq!(utf8_len_from_first_byte(0xe4), 3); // start of CJK
        assert_eq!(utf8_len_from_first_byte(0xf0), 4); // start of 4-byte plane
    }

    #[test]
    fn utf8_peek_one_decodes_cjk_codepoint() {
        // U+4E2D (中) = 0xE4 0xB8 0xAD
        let bytes = "中".as_bytes();
        let (cp, _) = utf8_peek_one(bytes, 0);
        assert_eq!(cp, 0x4e2d);
        assert!(utf8_is_cjk_hira_kata(cp));
    }

    #[test]
    fn joyai_cjk_at_detects_chinese_char() {
        let s = "中文".as_bytes();
        assert!(joyai_cjk_at(s, 0));
        assert!(joyai_cjk_at(s, 3)); // second CJK char starts at byte 3
    }

    #[test]
    fn joyai_cjk_at_rejects_ascii() {
        let s = "hello".as_bytes();
        assert!(!joyai_cjk_at(s, 0));
    }

    #[test]
    fn joyai_letter_like_at_handles_ascii_and_non_ascii() {
        let s = "Hé".as_bytes();
        assert!(joyai_letter_like_at(s, 0)); // 'H'
        assert!(joyai_letter_like_at(s, 1)); // 'é' first byte (non-ASCII)
    }

    #[test]
    fn byte_encode_roundtrip_via_codepoint_to_byte() {
        // "Hi!" → byte-encoded then decoded should recover the original bytes.
        let raw = b"Hi!".to_vec();
        let enc = byte_encode(&raw);
        let mut back = Vec::new();
        for ch in enc.chars() {
            back.push(gpt2_codepoint_to_byte(ch as u32));
        }
        assert_eq!(back, raw);
    }

    #[test]
    fn pre_tokenizer_splits_digits_in_runs_of_three() {
        // Build a trivial vocab so we can call encode(); just verify the piece
        // boundaries via a smoke test on the pre-tokenizer.
        // We'll instrument by counting splits via a custom helper.
        let bytes = b"1234567";
        let mut pieces = Vec::new();
        let mut pos = 0;
        let len = bytes.len();
        while pos < len {
            let start = pos;
            let c = bytes[pos];
            if ascii_digit(c) {
                let mut n = 0;
                while pos < len && ascii_digit(bytes[pos]) && n < 3 {
                    pos += 1;
                    n += 1;
                }
            } else {
                pos = next_utf8_char(bytes, pos);
            }
            if pos == start {
                pos = next_utf8_char(bytes, pos);
            }
            pieces.push(&bytes[start..pos]);
        }
        // Expect ["123","456","7"]
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0], b"123");
        assert_eq!(pieces[1], b"456");
        assert_eq!(pieces[2], b"7");
    }

    #[test]
    fn vocab_encode_falls_back_to_per_char_when_no_merges() {
        // Build a 1-char vocab where merges are empty. Each character in the
        // byte-encoded piece is its own token.
        // ASCII letters 'a'..'z' map straight through byte_encode (printable
        // range), so single-char tokens are 'a','b',...
        let tokens: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
        let merges: Vec<String> = vec![];
        let vocab = Vocab::new(tokens, merges);

        let ids = vocab.encode("cat");
        assert_eq!(ids.len(), 3);
        assert_eq!(vocab.id_to_token[ids[0] as usize], "c");
        assert_eq!(vocab.id_to_token[ids[1] as usize], "a");
        assert_eq!(vocab.id_to_token[ids[2] as usize], "t");
    }

    #[test]
    fn vocab_encode_applies_single_merge_in_priority_order() {
        // Tokens: 'a','b','c','ab','bc'. Merges: "a b" rank=0, "b c" rank=1.
        // Encoding "abc": both pairs available, "a b" wins → ["ab","c"].
        let tokens: Vec<String> = ["a", "b", "c", "ab", "bc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let merges: Vec<String> = ["a b", "b c"].iter().map(|s| s.to_string()).collect();
        let vocab = Vocab::new(tokens, merges);

        let ids = vocab.encode("abc");
        assert_eq!(ids.len(), 2);
        assert_eq!(vocab.id_to_token[ids[0] as usize], "ab");
        assert_eq!(vocab.id_to_token[ids[1] as usize], "c");
    }

    #[test]
    fn vocab_encode_restarts_after_merge_to_form_longer_token() {
        let tokens: Vec<String> = ["a", "b", "c", "ab", "abc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let merges: Vec<String> = ["a b", "ab c"].iter().map(|s| s.to_string()).collect();
        let vocab = Vocab::new(tokens, merges);

        let ids = vocab.encode("abc");
        assert_eq!(
            ids.len(),
            1,
            "expected restarted merge to produce abc, got {ids:?}"
        );
        assert_eq!(vocab.id_to_token[ids[0] as usize], "abc");
    }

    #[test]
    fn vocab_encode_prefers_lower_rank_pair_before_longer_result() {
        let tokens: Vec<String> = ["a", "b", "c", "ab", "bc", "abc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let merges: Vec<String> = ["b c", "a b", "a bc", "ab c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let vocab = Vocab::new(tokens, merges);

        let ids = vocab.encode("abc");
        assert_eq!(ids.len(), 1, "expected rank-0 b c then a bc, got {ids:?}");
        assert_eq!(vocab.id_to_token[ids[0] as usize], "abc");
    }

    #[test]
    fn vocab_decode_roundtrips_ascii_text() {
        // Single-char ASCII vocab; encode + decode should be identity for letters.
        let tokens: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
        let merges: Vec<String> = vec![];
        let vocab = Vocab::new(tokens, merges);
        let ids = vocab.encode("hello");
        let text = vocab.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn pre_tokenizer_treats_space_then_word_as_joined_piece() {
        // " a" should be a single piece per JoyAI's "space joins word" rule
        // (ds4.c:13839-13843 — when ascii_space sees " " followed by letter,
        // pos = p - 1, so the space is included in the *next* piece).
        // This test reproduces just the splitting behavior at the piece level.
        // We use the encode() path with a vocab that has " a" mapped explicitly.
        let space_a = format!("{}a", char::from_u32(gpt2_byte_to_codepoint(b' ')).unwrap());
        let tokens: Vec<String> = vec!["a".into(), space_a.clone()];
        let merges: Vec<String> = vec![format!(
            "{} a",
            char::from_u32(gpt2_byte_to_codepoint(b' ')).unwrap()
        )];
        let vocab = Vocab::new(tokens, merges);

        // Pre-token for "   a" (3 spaces + a) per JoyAI: ["  ", " a"]
        // → byte-encoded then BPE'd → at minimum should resolve to "  " + " a"
        // when both are present as merges. Our 1-merge vocab doesn't have "  "
        // available, so the leading 2 spaces will become per-byte tokens.
        // The test asserts: the final-piece tokenization contains the " a" token id.
        let ids = vocab.encode("   a");
        // Decode and verify suffix is " a" (byte-decode of space_a token).
        let text = vocab.decode(&ids).unwrap();
        assert!(text.ends_with(" a"), "decoded text was: {:?}", text);
    }

    #[test]
    fn pre_tokenizer_joyai_punct_then_alpha_groups() {
        // ds4.c:13826-13830 — when a punct symbol is immediately followed by an
        // ASCII letter, the piece is `<punct><letters>` (e.g., "[abc"). ds4.c
        // bpe_emit_piece always splits to chars first, then BPE-merges; the
        // grouping only manifests when merges actually produce the joined form.
        // We supply merges that BPE-collapse "[abc" → "[abc" (id=2) to verify
        // both pre-tok piece bounds AND merge precedence.
        let tokens: Vec<String> = vec![
            "[".into(),
            "abc".into(),
            "[abc".into(),
            "a".into(),
            "b".into(),
            "c".into(),
            "[a".into(),
            "[ab".into(),
            "ab".into(),
        ];
        // Rank order: "a b" (rank 0) → "ab", then "ab c" (rank 1) → "abc",
        // then "[ abc" (rank 2) → "[abc". Final single token id=2.
        let merges: Vec<String> = vec!["a b".into(), "ab c".into(), "[ abc".into()];
        let vocab = Vocab::new(tokens, merges);
        let ids = vocab.encode("[abc");
        assert_eq!(ids.len(), 1, "expected single merged token, got {ids:?}");
        assert_eq!(ids[0], 2, "expected '[abc' token id = 2, got {ids:?}");
    }

    #[test]
    fn empty_input_returns_empty_id_list() {
        let vocab = Vocab::new(vec!["a".into()], vec![]);
        assert!(vocab.encode("").is_empty());
    }
}
