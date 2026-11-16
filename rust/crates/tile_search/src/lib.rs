// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Enumerate the tilings a kernel admits, keep the ones the target accepts, and
//! rank what is left.
//!
//! # Why a declared alignment rule, and not a threshold
//!
//! Measured on the bench's `gate_up_silu` kernel by sweeping its column extent
//! through the emitter by repeated halving:
//!
//! | columns | verdict |
//! |---|---|
//! | 8960 (as captured) | refused -- unified-buffer budget |
//! | 4480 | **accepted** |
//! | 2240 | refused -- tile bound |
//! | 1120 | refused -- tile bound |
//! | 896 | **accepted** |
//! | 560, 448, 224 | refused -- tile bound |
//!
//! Read naively that is a ragged feasible set, and the first version of this
//! crate said so. It is not. Every refusal in that table is a MISALIGNED
//! extent: a row-major data tile has to span whole 512-byte fractals, so
//! `columns * sizeof(dtype)` must be a multiple of 512. 4480*4 is 35 fractals
//! and 896*4 is 7; 2240*4 is 17.5, which is why it is refused while a smaller
//! extent is not. Once [`knob::ladder`] applies that rule the remaining
//! candidates are all accepted -- the set is contiguous in the filtered space
//! and ragged only in the unfiltered one.
//!
//! So the alignment rule is what the search is really built on, and it carries
//! its provenance ([`Provenance::Correctness`]) for that reason. Two mistakes
//! it rules out, both of which a plain sweep makes:
//!
//! * **Halving is not enumeration.** 896 is 8960/10, so no sequence of halvings
//!   reaches it. A halving ladder reports one feasible tiling where there are
//!   several. [`knob::ladder`] enumerates divisors.
//! * **A threshold is not a filter.** Binary-searching for a largest-feasible
//!   extent lands on a misaligned value and concludes the kernel cannot be
//!   tiled below it.
//!
//! # The feasibility oracle is the emitter, not a model
//!
//! Every candidate is decided by running the real PTO emitter and reading its
//! answer. That is deliberate. A model of the bounds would be a second place for
//! them to be written down, and the two would drift -- and the drift would be
//! invisible, because a search whose model is wrong still returns a confident
//! answer. The emitter's refusal messages are kept verbatim for the same reason.
//!
//! # What the ranking is, and is not
//!
//! Ranking is a **declared cost model**, not a measurement. No candidate is run
//! on hardware here, so this module cannot tell which accepted tiling is faster;
//! it can only order them by a rule stated in the open. Calling that an
//! optimisation would be claiming a measurement nobody took.
//!
//! [`Cost`] carries its own reasoning, and the ordering it induces is reported
//! alongside the winner so a caller can disagree with the rule rather than only
//! with the result.

pub mod knob;
pub mod search;

pub use knob::{Knob, Provenance};
pub use search::{Candidate, Cost, Oracle, Outcome, SearchReport, search, search_with};
