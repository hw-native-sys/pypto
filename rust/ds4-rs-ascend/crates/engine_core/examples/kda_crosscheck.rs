//! Cross-check `kda_block::step` against an INDEPENDENT numpy implementation (see ~/kda/ref.py).
//!
//! The numpy version is written from the formula with an outer-product state update and `S.T @ k` reads; this one
//! uses explicit loops. Agreement to 1e-12 across three shapes is what makes the recurrence's arithmetic a
//! CORRECTNESS check rather than a consistency check -- the only discharge available on a host where llama.cpp
//! refuses GLM 5.3's architecture.
//!
//! usage: kda_crosscheck <case-file>

use engine_core::kda_block::{new_state, step};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/kda_case.txt".into());
    let text = std::fs::read_to_string(&path)?;
    let mut lines = text.lines();
    let mut cases = 0usize;
    let mut worst = 0.0f64;
    while let Some(head) = lines.next() {
        if !head.starts_with("CASE ") {
            continue;
        }
        let h: Vec<&str> = head.split_whitespace().collect();
        let d: usize = h[1].parse()?;
        let a_log: f64 = h[2].parse()?;
        let beta: f64 = h[3].parse()?;
        let mut get = |want: &str| -> anyhow::Result<Vec<f64>> {
            let line = lines.next().ok_or_else(|| anyhow::anyhow!("truncated at {want}"))?;
            let mut it = line.split_whitespace();
            let name = it.next().unwrap_or("");
            anyhow::ensure!(name == want, "expected {want}, got {name}");
            Ok(it.map(|v| v.parse::<f64>().unwrap_or(f64::NAN)).collect())
        };
        let q = get("q")?;
        let k = get("k")?;
        let v = get("v")?;
        let dt = get("dt")?;
        let g = get("g")?;
        let want = get("o")?;
        anyhow::ensure!(q.len() == d && want.len() == d, "case {cases}: expected width {d}");
        let mut s = new_state(d);
        let got = step(&mut s, &q, &k, &v, a_log, &dt, beta, &g);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        worst = worst.max(err);
        cases += 1;
        println!("case {cases} (d={d}): worst |rust - numpy| = {err:.3e}");
        anyhow::ensure!(err < 1e-12, "case {cases} disagrees by {err:.3e} -- one implementation is wrong");
    }
    anyhow::ensure!(cases > 0, "no cases parsed from {path}");
    println!("PASS: {cases} cases agree with the independent numpy recurrence, worst {worst:.3e}");

    // THE OTHER PIECES, same treatment: a conv, a low-rank gate and a gated RMS norm, each against numpy.
    if let Some(path2) = std::env::args().nth(2) {
        let t2 = std::fs::read_to_string(&path2)?;
        let mut it = t2.lines();
        let mut worst2 = 0.0f64;
        while let Some(line) = it.next() {
            let mut nums = |line: &str| -> Vec<f64> {
                line.split_whitespace().skip(1).filter_map(|v| v.parse::<f64>().ok()).collect()
            };
            if let Some(rest) = line.strip_prefix("CONV ") {
                let wv: Vec<f64> = rest.split_whitespace().filter_map(|v| v.parse().ok()).collect();
                let w: [f64; 4] = [wv[0], wv[1], wv[2], wv[3]];
                let x = nums(it.next().unwrap());
                let want = nums(it.next().unwrap());
                let got = engine_core::kda_block::conv4_causal(&x, &w);
                let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                worst2 = worst2.max(e);
                println!("conv:   worst = {e:.3e}");
                anyhow::ensure!(e < 1e-12, "conv disagrees by {e:.3e}");
            } else if line == "GATE" {
                let x = nums(it.next().unwrap());
                let aflat = nums(it.next().unwrap());
                let bflat = nums(it.next().unwrap());
                let bias = nums(it.next().unwrap());
                let want = nums(it.next().unwrap());
                let rows = 3;
                let a: Vec<Vec<f64>> = (0..rows).map(|r| aflat[r * x.len()..(r + 1) * x.len()].to_vec()).collect();
                let b: Vec<Vec<f64>> = (0..bias.len()).map(|r| bflat[r * rows..(r + 1) * rows].to_vec()).collect();
                let got = engine_core::kda_block::low_rank_gate(&x, &a, &b, &bias);
                let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                worst2 = worst2.max(e);
                println!("gate:   worst = {e:.3e}");
                anyhow::ensure!(e < 1e-12, "gate disagrees by {e:.3e}");
            } else if let Some(rest) = line.strip_prefix("RMS ") {
                let eps: f64 = rest.trim().parse()?;
                let x = nums(it.next().unwrap());
                let w = nums(it.next().unwrap());
                let g = nums(it.next().unwrap());
                let want = nums(it.next().unwrap());
                let got = engine_core::kda_block::gated_rms_norm(&x, &w, &g, eps);
                let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                worst2 = worst2.max(e);
                println!("rmsnorm:worst = {e:.3e}");
                anyhow::ensure!(e < 1e-12, "rmsnorm disagrees by {e:.3e}");
            }
        }
        println!("PASS: conv, gate and rmsnorm agree with numpy, worst {worst2:.3e}");
    }

    // THE ROUTER, against numpy, with a deliberate TIE in every case so both implementations must break it the
    // same way -- the rule that decides which expert a tied score selects.
    if let Some(path3) = std::env::args().nth(3) {
        let t3 = std::fs::read_to_string(&path3)?;
        let mut it = t3.lines();
        let mut worst3 = 0.0f64;
        let mut n = 0usize;
        while let Some(head) = it.next() {
            let Some(rest) = head.strip_prefix("ROUTE ") else { continue };
            let v: Vec<usize> = rest.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            let (d, n_exp, k) = (v[0], v[1], v[2]);
            let mut get = |want: &str| -> Vec<f64> {
                let line = it.next().unwrap_or("");
                let mut sp = line.split_whitespace();
                debug_assert_eq!(sp.next().unwrap_or(""), want);
                sp.filter_map(|x| x.parse().ok()).collect()
            };
            let x = get("x");
            let router_f = get("router");
            let bias_f = get("bias");
            let want_w = get("w");
            let want_i: Vec<usize> = get("idx").iter().map(|v| *v as usize).collect();
            let router: Vec<f32> = router_f.iter().map(|v| *v as f32).collect();
            let bias: Vec<f32> = bias_f.iter().map(|v| *v as f32).collect();
            let scored = engine_core::moe_route::scores(&router, &bias, &x, d, n_exp);
            let got = engine_core::moe_route::top_k_weights(&scored, k)?;
            anyhow::ensure!(got.len() == want_i.len(), "route selected {} of {}", got.len(), want_i.len());
            for (i, (gw, gi)) in got.iter().enumerate() {
                anyhow::ensure!(
                    *gi == want_i[i],
                    "case {n}: position {i} chose expert {gi}, numpy chose {} -- the TIE rule differs",
                    want_i[i]
                );
                worst3 = worst3.max((gw - want_w[i]).abs());
            }
            n += 1;
        }
        anyhow::ensure!(n > 0, "no ROUTE cases parsed");
        anyhow::ensure!(worst3 < 1e-12, "router weights disagree by {worst3:.3e}");
        println!("PASS: {n} router cases agree with numpy (indices identical, weights within {worst3:.3e})");
    }
    Ok(())
}
