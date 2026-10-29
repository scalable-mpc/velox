//! Online signing: Quorus Protocol 5 (sign) with the rejection test of
//! Protocol 4 (ΠRejSamp). It starts when `w1` of all attempts is open.
//!
//! For each attempt, each party computes the challenge `c` from the opened
//! `w1`, then `[z] = c·[s] + [y]` and `[w̄0] = [w0] − c·[e]` on its shares.
//! A coefficient `x` passes its bound test when `‖x‖ < B`, which is
//! `(x + B − 1) mod q < 2B − 1`: the Planner's `LessThanPub` gives `[ok]`.
//! `B = γ1 − β` for `z` and `B = γ2 − β` for `w̄0`.
//!
//! The failure count `f = Σ (1 − ok)` of an attempt is at most
//! [`TESTS_PER_ATTEMPT`] `< q`, so `f = 0` exactly when every coefficient
//! passes. Protocol 4 needs a batched OR for this; the count makes it
//! unnecessary. The parties open `f·u` for [`VERDICT_MASKS`] independent
//! random sharings `u`: each product is 0 when `f = 0`, and uniform in
//! `F_q \ {0}` when `f ≠ 0`, except when its `u` is 0. A failing attempt
//! passes only when all `u` are 0, with a probability of `q^−3 < 2^−69`
//! (decision D5).

use velox::fields::ml_dsa::RING_DEGREE;

use crate::{
    mldsa::{challenge, ring_product, Fq, Poly, ML_DSA_44},
    preprocessing::{Attempt, W_COEFFICIENTS_PER_ATTEMPT},
};

/// The bound tests of one attempt: the coefficients of `z` and `w̄0`.
pub const TESTS_PER_ATTEMPT: usize = (ML_DSA_44.l + ML_DSA_44.k) * RING_DEGREE;

/// The random sharings `u` for the verdict of one attempt.
pub const VERDICT_MASKS: usize = 3;

/// The shared responses of all attempts, after `w1` is open.
pub struct Responses {
    /// `c̃` of each attempt.
    pub c_tilde: Vec<Vec<u8>>,
    /// `[z] = c·[s] + [y]` of each attempt.
    pub z: Vec<Vec<Poly>>,
    /// `[w̄0] = [w0] − c·[e]` of each attempt.
    w0_bar: Vec<Vec<Poly>>,
}

impl Responses {
    /// `w1`: the opened `w1` of all attempts, in the order of `attempts`.
    pub fn new(mu: &[u8], w1: &[Fq], attempts: &[Attempt], s: &[Poly], e: &[Poly]) -> Self {
        let mut responses = Self { c_tilde: Vec::new(), z: Vec::new(), w0_bar: Vec::new() };
        for (attempt, w1) in attempts.iter().zip(w1.chunks_exact(W_COEFFICIENTS_PER_ATTEMPT)) {
            let w1: Vec<[u32; RING_DEGREE]> =
                w1.chunks_exact(RING_DEGREE).map(|p| std::array::from_fn(|i| p[i].representative())).collect();
            let (c_tilde, c) = challenge(mu, &w1, &ML_DSA_44);
            let z = attempt.y.iter().zip(s).map(|(y, s)| add(y, &ring_product(&c, s), Fq::one())).collect();
            let w0_bar = attempt.w0.iter().zip(e).map(|(w0, e)| add(w0, &ring_product(&c, e), -Fq::one())).collect();
            responses.c_tilde.push(c_tilde);
            responses.z.push(z);
            responses.w0_bar.push(w0_bar);
        }
        responses
    }

    /// The operands of `LessThanPub` for all attempts: `x + B − 1` and
    /// `2B − 1`, attempt by attempt, `z` first.
    pub fn bound_tests(&self) -> (Vec<Fq>, Vec<Fq>) {
        let beta = ML_DSA_44.beta;
        let (bound_z, bound_w) = (ML_DSA_44.gamma1 - beta, ML_DSA_44.gamma2 - beta);
        let tests = |polys: &[Poly], bound: u32| -> Vec<(Fq, Fq)> {
            let (shift, limit) = (Fq::from(u64::from(bound - 1)), Fq::from(u64::from(2 * bound - 1)));
            polys.iter().flatten().map(|x| (x + &shift, limit.clone())).collect()
        };
        self.z.iter().zip(&self.w0_bar).flat_map(|(z, w0_bar)| [tests(z, bound_z), tests(w0_bar, bound_w)].concat()).unzip()
    }
}

/// `a + sign·b`, coefficient by coefficient.
fn add(a: &Poly, b: &Poly, sign: Fq) -> Poly {
    std::array::from_fn(|i| &a[i] + &b[i] * &sign)
}

/// `[f] = Σ (1 − [ok])` of each attempt, from the results of the bound
/// tests.
pub fn failure_counts(in_bounds: &[Fq]) -> Vec<Fq> {
    let tests = Fq::from(TESTS_PER_ATTEMPT as u64);
    in_bounds.chunks_exact(TESTS_PER_ATTEMPT).map(|ok| ok.iter().fold(tests.clone(), |f, ok| f - ok)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In the clear: a coefficient at `±(B − 1)` passes, and one at `±B`
    /// fails, for `z` and for `w̄0`.
    #[test]
    fn bound_tests_at_the_edges() {
        let poly = |values: &[i64]| -> Poly {
            std::array::from_fn(|i| {
                let v = values.get(i).copied().unwrap_or(0);
                Fq::from(v.rem_euclid(i64::from(crate::mldsa::Q)) as u64)
            })
        };
        let (bz, bw) = (i64::from(ML_DSA_44.gamma1 - ML_DSA_44.beta), i64::from(ML_DSA_44.gamma2 - ML_DSA_44.beta));
        let z = vec![poly(&[bz - 1, 1 - bz, bz, -bz]); ML_DSA_44.l];
        let w0_bar = vec![poly(&[bw - 1, 1 - bw, bw, -bw]); ML_DSA_44.k];
        let responses = Responses { c_tilde: vec![Vec::new()], z: vec![z], w0_bar: vec![w0_bar] };
        let (x, c) = responses.bound_tests();
        let ok: Vec<Fq> = x
            .iter()
            .zip(&c)
            .map(|(x, c)| Fq::from(u64::from(x.representative() < c.representative())))
            .collect();
        for block in ok.chunks_exact(RING_DEGREE) {
            assert_eq!(block[..4], [Fq::one(), Fq::one(), Fq::zero(), Fq::zero()]);
        }
        // Two failures in each of the 8 ring elements.
        assert_eq!(failure_counts(&ok), vec![Fq::from(16u64)]);
    }
}
