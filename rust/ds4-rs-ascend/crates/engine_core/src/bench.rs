// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Bench-harness helpers — produce CSV rows matching antirez baseline schema
//! so A/B comparison against `benchmarks/ds4_aws/results/antirez_baseline.csv`
//! is a one-liner (`paste` or `csvkit join`).
//!
//! Antirez schema (from `scripts/aws_ds4/03_run_benchmark.sh:22`):
//!   `scenario,run,prefill_tokens,prefill_tok_s,decode_tokens,decode_tok_s,total_s,wall_clock_iso`

#![allow(dead_code)]

use std::time::Instant;

/// One row of the tile-rs bench CSV.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub scenario: String,
    pub run: u32,
    pub prefill_tokens: Option<u32>,
    pub prefill_tok_s: Option<f64>,
    pub decode_tokens: u32,
    /// `None` when decode never ran, or ran but produced no tokens.
    ///
    /// Deliberately `Option`, like its `prefill_tok_s` sibling: a bare f64 falls back to
    /// 0.0, and 0.000 in a results CSV is indistinguishable from a genuine collapse. The
    /// A1 harness (2026-07-14) reported "0.013 tok/s" for two different configs that had
    /// measured nothing at all, and it nearly drove a wrong engineering conclusion. No
    /// data must render as NA.
    pub decode_tok_s: Option<f64>,
    pub total_s: f64,
    pub wall_clock_iso: String,
}

impl BenchRow {
    /// Schema-compatible header (drops into `antirez_baseline.csv` as-is).
    pub const HEADER: &'static str =
        "scenario,run,prefill_tokens,prefill_tok_s,decode_tokens,decode_tok_s,total_s,wall_clock_iso";

    pub fn to_csv(&self) -> String {
        fn opt_u32(v: Option<u32>) -> String {
            v.map(|n| n.to_string()).unwrap_or_else(|| "NA".into())
        }
        fn opt_f64(v: Option<f64>) -> String {
            v.map(|n| format!("{n:.3}")).unwrap_or_else(|| "NA".into())
        }
        format!(
            "{},{},{},{},{},{},{:.3},{}",
            self.scenario,
            self.run,
            opt_u32(self.prefill_tokens),
            opt_f64(self.prefill_tok_s),
            self.decode_tokens,
            opt_f64(self.decode_tok_s),
            self.total_s,
            self.wall_clock_iso
        )
    }
}

/// Tiny wall-clock timer for the decode loop. The expected use pattern is:
///
/// ```ignore
/// let mut t = DecodeTimer::new("short", 1);
/// t.mark_prefill_start();
/// // ... prefill (loaded from snapshot in our decode-only design — usually 0s)
/// t.mark_decode_start(prefill_tokens);
/// for _ in 0..max_tokens { engine.step(); t.tick_decode(); }
/// let row = t.finish();
/// println!("{}", row.to_csv());
/// ```
pub struct DecodeTimer {
    scenario: String,
    run: u32,
    start: Instant,
    prefill_start: Option<Instant>,
    decode_start: Option<Instant>,
    prefill_tokens: Option<u32>,
    decode_tokens: u32,
}

impl DecodeTimer {
    pub fn new(scenario: &str, run: u32) -> Self {
        Self {
            scenario: scenario.into(),
            run,
            start: Instant::now(),
            prefill_start: None,
            decode_start: None,
            prefill_tokens: None,
            decode_tokens: 0,
        }
    }

    pub fn mark_prefill_start(&mut self) {
        self.prefill_start = Some(Instant::now());
    }

    pub fn mark_decode_start(&mut self, prefill_tokens: u32) {
        self.prefill_tokens = Some(prefill_tokens);
        self.decode_start = Some(Instant::now());
    }

    pub fn tick_decode(&mut self) {
        self.decode_tokens += 1;
    }

    pub fn finish(self) -> BenchRow {
        /// A rate exists only if time passed AND tokens came out; otherwise there is no
        /// measurement to report.
        fn rate(tokens: u32, secs: f64) -> Option<f64> {
            (secs > 0.0 && tokens > 0).then(|| tokens as f64 / secs)
        }

        let now = Instant::now();
        let total_s = now.duration_since(self.start).as_secs_f64();

        let (prefill_tok_s, decode_tok_s) = match (self.prefill_start, self.decode_start) {
            (Some(ps), Some(ds)) => {
                let prefill_s = ds.duration_since(ps).as_secs_f64();
                let decode_s = now.duration_since(ds).as_secs_f64();
                let prefill_rate = self.prefill_tokens.and_then(|t| rate(t, prefill_s));
                let decode_rate = rate(self.decode_tokens, decode_s);
                (prefill_rate, decode_rate)
            }
            (None, Some(ds)) => {
                let decode_s = now.duration_since(ds).as_secs_f64();
                (None, rate(self.decode_tokens, decode_s))
            }
            // Decode never started: there is nothing to report, not a rate of zero.
            _ => (None, None),
        };

        BenchRow {
            scenario: self.scenario,
            run: self.run,
            prefill_tokens: self.prefill_tokens,
            prefill_tok_s,
            decode_tokens: self.decode_tokens,
            decode_tok_s,
            total_s,
            wall_clock_iso: chrono_like_utc_iso(),
        }
    }
}

/// Tiny UTC ISO-8601 stamp without pulling in `chrono` (vendor-only constraint).
pub fn utc_iso_now() -> String {
    chrono_like_utc_iso()
}

fn chrono_like_utc_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert via standard date math — avoids pulling chrono. Year-month-day from
    // days since 1970-01-01 using civil_from_days (Howard Hinnant). Sufficient
    // precision (seconds) matches antirez `date -u +%FT%TZ` output.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Hinnant's algorithm: convert days since 1970-01-01 to (year, month, day).
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn bench_row_csv_matches_antirez_header() {
        // Field count must match antirez schema exactly.
        let row = BenchRow {
            scenario: "short".into(),
            run: 1,
            prefill_tokens: Some(64),
            prefill_tok_s: Some(123.456),
            decode_tokens: 64,
            decode_tok_s: Some(33.5),
            total_s: 2.1,
            wall_clock_iso: "2026-05-12T10:00:00Z".into(),
        };
        let csv = row.to_csv();
        let header_cols = BenchRow::HEADER.split(',').count();
        let row_cols = csv.split(',').count();
        assert_eq!(
            header_cols, row_cols,
            "header has {header_cols} cols, row has {row_cols}: '{csv}'"
        );
    }

    #[test]
    fn bench_row_renders_na_for_missing_prefill() {
        let row = BenchRow {
            scenario: "decode_only".into(),
            run: 1,
            prefill_tokens: None,
            prefill_tok_s: None,
            decode_tokens: 32,
            decode_tok_s: Some(30.0),
            total_s: 1.0,
            wall_clock_iso: "2026-05-12T10:00:00Z".into(),
        };
        let csv = row.to_csv();
        assert!(
            csv.contains(",NA,NA,32,"),
            "expected NA prefill fields: {csv}"
        );
    }

    #[test]
    fn decode_timer_counts_ticks() {
        let mut t = DecodeTimer::new("test", 1);
        t.mark_decode_start(0);
        for _ in 0..5 {
            t.tick_decode();
        }
        sleep(Duration::from_millis(10));
        let row = t.finish();
        assert_eq!(row.decode_tokens, 5);
        let rate = row.decode_tok_s.expect("a real decode must report a rate");
        assert!(rate > 0.0, "rate = {rate}");
    }

    #[test]
    fn decode_timer_records_prefill_rate_when_marked() {
        let mut t = DecodeTimer::new("with_prefill", 1);
        t.mark_prefill_start();
        sleep(Duration::from_millis(10));
        t.mark_decode_start(100);
        sleep(Duration::from_millis(10));
        t.tick_decode();
        let row = t.finish();
        assert!(row.prefill_tok_s.is_some());
        assert!(row.prefill_tok_s.unwrap() > 0.0);
        assert_eq!(row.prefill_tokens, Some(100));
    }

    #[test]
    fn civil_from_days_epoch_is_1970_01_01() {
        let (y, m, d) = civil_from_days(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn civil_from_days_handles_today() {
        // 2026-05-12 = 20585 days since 1970-01-01 (verified manually).
        let (y, m, d) = civil_from_days(20585);
        assert_eq!((y, m, d), (2026, 5, 12));
    }

    #[test]
    fn iso_stamp_has_z_suffix_and_zulu_format() {
        let s = chrono_like_utc_iso();
        assert!(s.ends_with('Z'));
        assert_eq!(s.len(), 20, "expected YYYY-MM-DDTHH:MM:SSZ, got {s}");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[13..14], ":");
        assert_eq!(&s[16..17], ":");
    }
}
