//! The MoE router as a pure function: scores -> top-k -> softmax weights.
//!
//! Extracted from the example that first implemented it so it can be cross-checked against an independent
//! implementation, the way the KDA block's arithmetic was. The routing is where a silent error is cheapest to
//! make and hardest to see: a wrong top-k still produces a finite, plausible-looking output.

/// One expert's score and index.
pub type Scored = (f64, usize);

/// Score every expert: `score[e] = <row_e, x> + bias[e]`.
///
/// `router` is `[d, n_exp]` in the file's layout -- the tensor is `ffn_gate_inp.weight [4096, 288]`, i.e.
/// `d` rows of `n_exp` values, so expert `e`'s row is the `e`-th column of that matrix.
pub fn scores(router: &[f32], bias: &[f32], x: &[f64], d: usize, n_exp: usize) -> Vec<Scored> {
    assert_eq!(router.len(), d * n_exp);
    (0..n_exp)
        .map(|e| {
            let s: f64 = (0..d).map(|c| router[c * n_exp + e] as f64 * x[c]).sum::<f64>()
                + bias[e] as f64;
            (s, e)
        })
        .collect()
}

/// The top `k` by score, as softmax weights summing to 1, in descending score order.
///
/// Ties break by LOWER INDEX so the result is deterministic; a NaN score would make every comparison false and
/// leave `best` at its initial value, so NaN is refused rather than sorted through.
pub fn top_k_weights(scored: &[Scored], k: usize) -> anyhow::Result<Vec<(f64, usize)>> {
    anyhow::ensure!(!scored.is_empty(), "no experts to route");
    anyhow::ensure!(
        scored.iter().all(|(s, _)| s.is_finite()),
        "a router score is not finite: {:?}",
        scored.iter().find(|(s, _)| !s.is_finite())
    );
    let mut v = scored.to_vec();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let k = k.min(v.len());
    let mx = v[0].0;
    let exps: Vec<(f64, usize)> = v[..k].iter().map(|(s, e)| ((s - mx).exp(), *e)).collect();
    let z: f64 = exps.iter().map(|(w, _)| w).sum();
    anyhow::ensure!(z > 0.0, "the softmax normaliser is zero");
    Ok(exps.into_iter().map(|(w, e)| (w / z, e)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weights sum to ONE, are descending, and there are exactly k of them.
    #[test]
    fn top_k_weights_are_normalised_and_ordered() {
        let scored: Vec<Scored> = vec![(0.1, 0), (2.0, 1), (1.0, 2), (-1.0, 3)];
        let w = top_k_weights(&scored, 2).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].1, 1);
        assert_eq!(w[1].1, 2);
        assert!((w.iter().map(|(x, _)| x).sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(w[0].0 > w[1].0);
    }

    /// A tie breaks by LOWER INDEX, so the route does not depend on sort stability.
    #[test]
    fn ties_break_by_index() {
        let scored: Vec<Scored> = vec![(1.0, 7), (1.0, 2), (1.0, 5)];
        let w = top_k_weights(&scored, 3).unwrap();
        assert_eq!(w.iter().map(|(_, e)| *e).collect::<Vec<_>>(), vec![2, 5, 7]);
    }

    /// `scores` reads the FILE's layout: `ffn_gate_inp.weight` is `[d, n_exp]`, so expert e is COLUMN e. A row-major
    /// misreading would give a different answer for every e, and this case is built so the two readings differ.
    #[test]
    fn scores_uses_the_column_of_each_expert() {
        // d = 2, n_exp = 3; router[c * 3 + e]
        //   row c=0: [1, 10, 100]      row c=1: [2, 20, 200]
        let router = vec![1.0f32, 10.0, 100.0, 2.0, 20.0, 200.0];
        let bias = vec![0.0f32, 0.5, 1.0];
        let x = vec![1.0f64, 1.0];
        let got = scores(&router, &bias, &x, 2, 3);
        // column e dotted with [1,1] + bias[e]:  e=0 -> 1+2+0 = 3 ; e=1 -> 10+20+0.5 = 30.5 ; e=2 -> 100+200+1 = 301
        assert!((got[0].0 - 3.0).abs() < 1e-6, "expert 0 = {:?}", got[0]);
        assert!((got[1].0 - 30.5).abs() < 1e-6, "expert 1 = {:?}", got[1]);
        assert!((got[2].0 - 301.0).abs() < 1e-6, "expert 2 = {:?}", got[2]);
        assert_eq!(got.iter().map(|(_, e)| *e).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    /// A non-finite score is REFUSED: `partial_cmp` returns None for NaN, which would silently leave the sort in
    /// an arbitrary order.
    #[test]
    fn a_non_finite_score_is_refused() {
        let scored: Vec<Scored> = vec![(1.0, 0), (f64::NAN, 1)];
        let err = top_k_weights(&scored, 1).expect_err("NaN must be refused");
        assert!(format!("{err}").contains("not finite"), "got: {err}");
    }
}
