// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! S3 round-trip verifier.
//!
//! Catches silent information loss in `parser → emitter_pto → parser` by
//! comparing the ASTs before and after a serialize/reparse cycle. Useful as
//! a regression harness for every fixture we add: if the emitter forgets a
//! field or the parser starts dropping one, the diff surfaces it.
//!
//! The round-trip is AST-level, not byte-level. Canonical PTO emitted by
//! `emitter_pto` drops non-load-bearing attributes (v_row, blayout, ...)
//! and re-uses operands where the AST doesn't track a scratch SSA, so
//! text-to-text equality would be over-restrictive.

use crate::{ast::*, emitter_pto, parser};

#[derive(Debug)]
pub struct RoundTripMismatch {
    pub before: PtoModule,
    pub after: PtoModule,
    pub reemitted: String,
    /// Set when the cycle never completed -- the source, the re-emitted text,
    /// or the reverse emitter was rejected. `before`/`after` are then empty
    /// and carry no information; this string is the whole result.
    pub failure: Option<String>,
}

impl RoundTripMismatch {
    fn parse_failed(msg: String) -> Self {
        Self {
            before: PtoModule::default(),
            after: PtoModule::default(),
            reemitted: String::new(),
            failure: Some(msg),
        }
    }

    /// One line naming what went wrong, whichever kind of failure this is.
    pub fn summary(&self) -> String {
        self.failure
            .clone()
            .unwrap_or_else(|| "AST diverged across round-trip".to_string())
    }
}

/// Parse `src`, emit canonical PTO, reparse, and assert AST equality.
/// Returns `Ok(())` on success, a `RoundTripMismatch` describing the
/// divergence on failure.
/// How many of `src`'s ops this crate actually models: `(modelled, unmodelled)`.
///
/// Ported from `pto-rs-viewpoint-rebase`, which needed it for exactly the reason
/// this branch needed the `MODELLED_OPS` gate it replaces: carrying unmodelled
/// ops verbatim makes a round trip faithful, but a file that is *entirely*
/// unmodelled round-trips for the trivial reason that nothing was interpreted.
/// Byte passthrough proves no fidelity. A ratio keeps that distinction visible
/// instead of letting green stand in for coverage.
///
/// Counts ops inside `scf.for` and `scf.if` regions too -- a flat walk would
/// read a loop body as one op and report a kernel whose whole computation sits
/// in a loop as almost fully modelled. That exact flat-walk mistake broke two
/// other checks when regions landed.
pub fn op_model_coverage(src: &str) -> Result<(usize, usize), String> {
    let m = parser::parse_module(src)?;
    let mut modelled = 0;
    let mut unmodelled = 0;
    for f in &m.funcs {
        count_ops(&f.ops, &mut modelled, &mut unmodelled);
    }
    Ok((modelled, unmodelled))
}

fn count_ops(ops: &[PtoOp], modelled: &mut usize, unmodelled: &mut usize) {
    for op in ops {
        match op {
            PtoOp::Unmodelled { .. } => *unmodelled += 1,
            PtoOp::For { body, .. } => {
                *modelled += 1;
                count_ops(body, modelled, unmodelled);
            }
            PtoOp::If {
                then_body,
                else_body,
                ..
            } => {
                *modelled += 1;
                count_ops(then_body, modelled, unmodelled);
                if let Some(els) = else_body {
                    count_ops(els, modelled, unmodelled);
                }
            }
            _ => *modelled += 1,
        }
    }
}

pub fn verify_roundtrip(src: &str) -> Result<(), RoundTripMismatch> {
    // We deliberately do NOT run the fusion pass here -- the reverse PTO
    // emitter only handles the literal ops that exist in the source, and
    // the AST's synthetic `Softmax` variant has no PTO spelling.
    // These three were `panic!`. A sweep over every artifact in the repo
    // cannot report how many fail if the first unparseable one aborts the
    // process -- the `bad constant \u{60}128.0\u{60}` failure hid the rest of the
    // corpus for as long as it stood. They are returned now.
    let before = parser::parse_module(src)
        .map_err(|e| RoundTripMismatch::parse_failed(format!("initial parse failed: {e}")))?;

    let reemitted = emitter_pto::emit_module_pto(&before)
        .map_err(|e| RoundTripMismatch::parse_failed(format!("reverse emit failed: {e}")))?;

    let after = parser::parse_module(&reemitted).map_err(|e| {
        RoundTripMismatch::parse_failed(format!(
            "re-parse failed: {e}\n--- re-emitted ---\n{reemitted}"
        ))
    })?;

    if before == after {
        Ok(())
    } else {
        Err(RoundTripMismatch {
            before,
            after,
            reemitted,
            failure: None,
        })
    }
}
