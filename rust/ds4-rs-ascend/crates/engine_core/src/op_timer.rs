// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Per-op cost attribution timer for decode_step.
//!
//! Gated on `DS4_TIME_OPS=1`. When off, all `time!` invocations are
//! near-zero cost (one env-var check on first use cached in a thread-local).
//!
//! Usage:
//!
//! ```text
//! let r = crate::op_timer::time!("moe_routed_step", { k.moe_routed_step(...) });
//! ```
//!
//! At end of decode loop, call `crate::op_timer::report_and_reset()` to
//! print a CSV-ish summary to stderr:
//!
//! ```text
//! OP_TIME_TOTAL,op_name,calls,total_ms,avg_us,share_pct
//! ```
//!
//! The aggregator is global (`Mutex<HashMap<...>>`) because decode_step is
//! single-threaded and we want layer×op aggregation across the whole token.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

thread_local! {
    static ENABLED: Cell<Option<bool>> = const { Cell::new(None) };
}

fn enabled() -> bool {
    ENABLED.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        let v = std::env::var("DS4_TIME_OPS").ok().as_deref() == Some("1");
        c.set(Some(v));
        v
    })
}

static AGG: Mutex<Option<HashMap<&'static str, (u64, u128)>>> = Mutex::new(None);

pub fn record(name: &'static str, ns: u128) {
    let mut guard = AGG.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let e = map.entry(name).or_insert((0u64, 0u128));
    e.0 += 1;
    e.1 += ns;
}

/// Print aggregated per-op timings to stderr as CSV rows, then clear.
/// Header row precedes the data. Sorted by total_ns descending.
pub fn report_and_reset() {
    if !enabled() {
        return;
    }
    let mut guard = AGG.lock().unwrap();
    let map = guard.take().unwrap_or_default();
    drop(guard);
    let mut rows: Vec<(&'static str, u64, u128)> = map
        .into_iter()
        .map(|(name, (calls, ns))| (name, calls, ns))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.2));
    let total_ns: u128 = rows.iter().map(|r| r.2).sum();
    eprintln!("OP_TIME_HEADER,op,calls,total_ms,avg_us,share_pct");
    for (name, calls, ns) in rows {
        let total_ms = (ns as f64) / 1.0e6;
        let avg_us = if calls > 0 {
            (ns as f64) / (calls as f64) / 1.0e3
        } else {
            0.0
        };
        let share_pct = if total_ns > 0 {
            (ns as f64) / (total_ns as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "OP_TIME_ROW,{},{},{:.3},{:.1},{:.2}",
            name, calls, total_ms, avg_us, share_pct
        );
    }
    let total_ms = (total_ns as f64) / 1.0e6;
    eprintln!(
        "OP_TIME_TOTAL,_total,1,{:.3},{:.1},100.00",
        total_ms,
        total_ms * 1000.0
    );
}

/// Time `$body`; record under `$name` when enabled. Returns the body's value.
#[macro_export]
macro_rules! op_time {
    ($name:expr, $body:expr) => {{
        if $crate::op_timer::__is_on() {
            let __t = std::time::Instant::now();
            let __r = { $body };
            $crate::op_timer::record($name, __t.elapsed().as_nanos());
            __r
        } else {
            $body
        }
    }};
}

#[doc(hidden)]
pub fn __is_on() -> bool {
    enabled()
}

// Keep an unused Instant import so the macro expansion compiles even when
// the body uses none of std::time directly.
#[doc(hidden)]
pub fn __touch_instant() -> Instant {
    Instant::now()
}
