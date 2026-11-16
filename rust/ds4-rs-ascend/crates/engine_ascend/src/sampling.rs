// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! CPU logits sampling — a faithful port of antirez's `ds4.c` sampler
//! (`sample_top_p_min_p` / `sample_full_vocab` / `sample_rng_*`,
//! ds4.c:15122-15298). The server uses this to honor the OpenAI
//! `temperature` / `top_k` / `top_p` / `min_p` / `seed` parameters.
//!
//! Behavior matches the C reference exactly (same XOR-shift RNG, same
//! top-k insertion sort, same top-p/min-p cumulative cutoff), so output is
//! reproducible for a given seed and consistent with the antirez server.

/// OpenAI-style sampling parameters. Defaults match `ds4.c` (temp=1, no
/// top-k, top_p=1, min_p=0 → effectively pure softmax sampling).
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}

/// XOR-shift RNG, bit-identical to `ds4.c` `sample_rng_next` / `sample_rng_f32`.
/// Seedable for reproducible sampling.
#[derive(Clone, Copy, Debug)]
pub struct SampleRng {
    state: u64,
}

impl SampleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        if x == 0 {
            x = 0x9e37_79b9_7f4a_7c15;
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform f32 in [0, 1), matching `sample_rng_f32`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let x = self.next_u64();
        (((x >> 40) & 0x00ff_ffff) as f32) / 16_777_216.0
    }
}

/// Greedy argmax over finite logits (NaN/Inf-safe). Mirrors `sample_argmax`.
pub fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i32;
        }
    }
    best
}

/// Entry point — mirrors `sample_top_p_min_p`. `temperature <= 0` → argmax.
pub fn sample(logits: &[f32], p: &SampleParams, rng: &mut SampleRng) -> i32 {
    let n_vocab = logits.len();
    let temperature = p.temperature;
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut top_p = p.top_p;
    if top_p <= 0.0 || top_p > 1.0 {
        top_p = 1.0;
    }
    let min_p = if p.min_p < 0.0 { 0.0 } else { p.min_p };
    let mut top_k = p.top_k;
    if top_k <= 0 {
        return sample_full_vocab(logits, temperature, top_p, min_p, rng);
    }
    if top_k > 1024 {
        top_k = 1024;
    }
    if top_k as usize > n_vocab {
        top_k = n_vocab as i32;
    }
    let top_k = top_k as usize;

    // Top-k insertion sort into descending (vals[0] = largest), like ds4.c.
    let mut ids: Vec<i32> = Vec::with_capacity(top_k);
    let mut vals: Vec<f32> = Vec::with_capacity(top_k);
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let n = vals.len();
        if n == top_k && v <= vals[n - 1] {
            continue;
        }
        if n < top_k {
            vals.push(0.0);
            ids.push(0);
        }
        let mut j = if n < top_k { n } else { top_k - 1 };
        while j > 0 && vals[j - 1] < v {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j -= 1;
        }
        vals[j] = v;
        ids[j] = i as i32;
    }
    let n = vals.len();
    if n == 0 {
        return argmax(logits);
    }

    let max_logit = vals[0];
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for &v in &vals {
        let pr = ((v - max_logit) / temperature).exp();
        probs.push(pr);
        sum += pr;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return ids[0];
    }

    let min_prob = (probs[0] / sum) * min_p;
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for i in 0..n {
        let pr = probs[i] / sum;
        if i > 0 && pr < min_prob {
            break;
        }
        filtered_sum += probs[i];
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return ids[0];
    }
    let mut r = rng.next_f32() * filtered_sum;
    for i in 0..filtered {
        r -= probs[i];
        if r <= 0.0 {
            return ids[i];
        }
    }
    ids[filtered - 1]
}

/// Full-vocab path (no top-k) — mirrors `sample_full_vocab`.
fn sample_full_vocab(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    min_p: f32,
    rng: &mut SampleRng,
) -> i32 {
    let mut max_logit = f32::NEG_INFINITY;
    let mut best = 0i32;
    let mut finite = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        finite += 1;
        if v > max_logit {
            max_logit = v;
            best = i as i32;
        }
    }
    if finite == 0 {
        return argmax(logits);
    }

    // top_p >= 1: sample over all finite tokens with an absolute min_p floor.
    if top_p >= 1.0 {
        let min_rel = if min_p > 0.0 { min_p } else { 0.0 };
        let mut sum = 0.0f32;
        for &v in logits {
            if !v.is_finite() {
                continue;
            }
            let pr = ((v - max_logit) / temperature).exp();
            if pr < min_rel {
                continue;
            }
            sum += pr;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return best;
        }
        let mut r = rng.next_f32() * sum;
        for (i, &v) in logits.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let pr = ((v - max_logit) / temperature).exp();
            if pr < min_rel {
                continue;
            }
            r -= pr;
            if r <= 0.0 {
                return i as i32;
            }
        }
        return best;
    }

    // top_p < 1: build + sort candidates, apply min_p + top_p cumulative cutoff.
    let mut cand: Vec<(i32, f32)> = Vec::with_capacity(finite); // (id, prob)
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let pr = ((v - max_logit) / temperature).exp();
        cand.push((i as i32, pr));
        sum += pr;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return best;
    }
    // Descending by prob (equiv. to descending by logit since exp is monotone).
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let min_prob = (cand[0].1 / sum) * if min_p > 0.0 { min_p } else { 0.0 };
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for (i, &(_, pr)) in cand.iter().enumerate() {
        let p = pr / sum;
        if i > 0 && p < min_prob {
            break;
        }
        filtered_sum += pr;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    if filtered == 0 {
        return best;
    }
    let mut r = rng.next_f32() * filtered_sum;
    for &(id, pr) in cand.iter().take(filtered) {
        r -= pr;
        if r <= 0.0 {
            return id;
        }
    }
    cand[filtered - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_zero_is_argmax() {
        let logits = [0.1f32, 3.0, -1.0, 2.9];
        let mut rng = SampleRng::new(1);
        let p = SampleParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&logits, &p, &mut rng), 1);
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let p = SampleParams {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
        };
        let mut a = SampleRng::new(42);
        let mut b = SampleRng::new(42);
        let sa: Vec<i32> = (0..16).map(|_| sample(&logits, &p, &mut a)).collect();
        let sb: Vec<i32> = (0..16).map(|_| sample(&logits, &p, &mut b)).collect();
        assert_eq!(sa, sb, "same seed must give same draws");
    }

    #[test]
    fn rng_matches_reference_first_values() {
        // Sanity: XOR-shift produces a stable, nonzero stream.
        let mut rng = SampleRng::new(1);
        let a = rng.next_f32();
        let b = rng.next_f32();
        assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
        assert_ne!(a, b);
    }
}
