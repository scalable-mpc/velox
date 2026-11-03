//! Sharings of coefficients that are uniform in `[−η, η]`, `η = 2`: Quorus
//! Protocol 10 (ΠRand[S_η]). Key generation takes the secrets `s` and `e`
//! from them, and preprocessing takes the error `e_w`.
//!
//! A candidate is `r = b0 + 2·b1 + 4·b2` for three random bits. The parties
//! keep it when `r < 5`, and the coefficient is `r − 2`. They open only the
//! reject bit `b2·(b0 + b1 − b0·b1)`, which is 1 when `r ≥ 5`: one
//! multiplication for `b0·b1`, one for the product with `b2`, then one
//! reveal. A kept `r` is uniform in `[0, 5)`, so the opened bits tell
//! nothing about the kept coefficients.

use anyhow::{bail, Result};

use crate::mldsa::{Fq, ML_DSA_44};

const _: () = assert!(ML_DSA_44.eta == 2, "the reject bit is for 2η + 1 = 5 values");

/// The random bits of one candidate.
pub const BITS_PER_CANDIDATE: usize = 3;

/// The candidates give too few coefficients with a probability of at most
/// `2^−SHORTFALL_BITS`.
const SHORTFALL_BITS: f64 = 40.0;

/// The number of candidates for `count` coefficients: the least `C` with
/// `5C/8 > count` and `2·(5C/8 − count)² ≥ 40·ln 2·C`. By Hoeffding's
/// inequality, fewer than `count` of the `C` candidates pass with a
/// probability of at most `2^−40`.
pub fn candidates_for(count: usize) -> usize {
    (count..)
        .find(|&c| {
            let surplus = 5.0 * c as f64 / 8.0 - count as f64;
            surplus > 0.0 && 2.0 * surplus * surplus >= SHORTFALL_BITS * std::f64::consts::LN_2 * c as f64
        })
        .expect("a large enough count exists")
}

/// The candidates, as sharings of their three bits (0 or 1).
pub struct Candidates {
    bits: Vec<[Fq; BITS_PER_CANDIDATE]>,
}

impl Candidates {
    /// `bits`: sharings of 0 or 1, three for each candidate, `b0` first.
    pub fn new(bits: &[Fq]) -> Self {
        let bits = bits.chunks_exact(BITS_PER_CANDIDATE).map(|b| [b[0].clone(), b[1].clone(), b[2].clone()]).collect();
        Self { bits }
    }

    /// The operands of the first multiplication: `b0·b1`.
    pub fn low_product_operands(&self) -> (Vec<Fq>, Vec<Fq>) {
        self.bits.iter().map(|[b0, b1, _]| (b0.clone(), b1.clone())).unzip()
    }

    /// The operands of the second multiplication, whose product is the
    /// reject bit: `b2·(b0 + b1 − b0·b1)`.
    pub fn reject_operands(&self, low_products: &[Fq]) -> (Vec<Fq>, Vec<Fq>) {
        self.bits.iter().zip(low_products).map(|([b0, b1, b2], b0b1)| (b2.clone(), b0 + b1 - b0b1)).unzip()
    }

    /// The first `count` kept candidates, as sharings of `r − 2`, from the
    /// opened reject bits.
    pub fn kept(&self, reject: &[Fq], count: usize) -> Result<Vec<Fq>> {
        let two = Fq::from(2u64);
        let kept: Vec<Fq> = self
            .bits
            .iter()
            .zip(reject)
            .filter(|(_, reject)| **reject == Fq::zero())
            .map(|([b0, b1, b2], _)| b0 + b1.double() + b2.double().double() - &two)
            .take(count)
            .collect();
        if kept.len() < count {
            bail!("{} of {} candidates passed, {} are needed", kept.len(), self.bits.len(), count);
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::centered;

    /// Each of the 8 candidates `r` in the clear: the reject bit is
    /// `[r ≥ 5]`, and a kept `r` gives `r − 2`.
    #[test]
    fn rejects_exactly_r_at_least_5() {
        let bits: Vec<Fq> = (0..8u64).flat_map(|r| (0..3).map(move |i| Fq::from((r >> i) & 1))).collect();
        let candidates = Candidates::new(&bits);
        let (x, y) = candidates.low_product_operands();
        let low: Vec<Fq> = x.iter().zip(&y).map(|(x, y)| x * y).collect();
        let (x, y) = candidates.reject_operands(&low);
        let reject: Vec<Fq> = x.iter().zip(&y).map(|(x, y)| x * y).collect();
        let want: Vec<Fq> = (0..8u64).map(|r| Fq::from(u64::from(r >= 5))).collect();
        assert_eq!(reject, want);
        let kept: Vec<i64> = candidates.kept(&reject, 5).unwrap().iter().map(centered).collect();
        assert_eq!(kept, vec![-2, -1, 0, 1, 2]);
        assert!(candidates.kept(&reject, 6).is_err(), "only 5 of 8 pass");
    }

    /// Key generation needs 2048 coefficients (`s` and `e` of ML-DSA-44):
    /// 3637 candidates, about 1.78 for each coefficient.
    #[test]
    fn candidates_for_key_generation() {
        assert_eq!(candidates_for(2048), 3637);
        assert!(candidates_for(1) > 1);
    }
}
