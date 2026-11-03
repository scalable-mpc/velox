//! The preprocessing of the signing attempts: Quorus Protocol 8 (ΠPrep-44).
//! It does not depend on the message.
//!
//! For each attempt the parties hold:
//!
//! - `[y]`, with coefficients uniform in `(−γ1, γ1]`: `γ1 − r` for an
//!   18-bit `r` made of random bits;
//! - `[w] = A·[y] + [e_w]`, with `e_w` from the
//!   [`small_coefficients`](crate::small_coefficients);
//! - `[w1] = ((−44·[w] + (q − 1)/2) mod q) mod 44`, the Planner's
//!   `CanonicalMod` with `d = 44`. This is `HighBits(w)` for every `w` (the
//!   test `rounding_over_all_of_f_q` of [`mldsa`](crate::mldsa));
//! - `[w0] = [w] − 2γ2·[w1]`, which is `LowBits(w)` modulo `q`.
//!
//! Nothing is opened here: signing opens `w1` of all attempts.

use velox::fields::ml_dsa::RING_DEGREE;

use crate::mldsa::{into_polys, matrix_vector_product, Fq, Poly, ML_DSA_44, Q};

/// The random bits of one coefficient of `y`: `bitlen(2γ1 − 1)`.
pub const Y_BITS: usize = ML_DSA_44.gamma1.trailing_zeros() as usize + 1;

/// The random bits of `y` for one attempt.
pub const Y_BITS_PER_ATTEMPT: usize = Y_BITS * ML_DSA_44.l * RING_DEGREE;

/// The coefficients of `e_w`, and of `w`, for one attempt.
pub const W_COEFFICIENTS_PER_ATTEMPT: usize = ML_DSA_44.k * RING_DEGREE;

/// `(q − 1)/(2γ2)`: the modulus of `w1`, the `d` of `CanonicalMod`.
pub const W1_MODULUS: usize = ((Q - 1) / (2 * ML_DSA_44.gamma2)) as usize;

/// The sharings of one signing attempt.
pub struct Attempt {
    pub y: Vec<Poly>,
    pub w0: Vec<Poly>,
    pub w1: Vec<Poly>,
}

/// The attempts before `[w1]` is known.
pub struct PendingAttempts {
    y: Vec<Vec<Poly>>,
    w: Vec<Vec<Poly>>,
}

impl PendingAttempts {
    /// `[y]` from `y_bits` (sharings of 0 or 1, `Y_BITS_PER_ATTEMPT` for each
    /// attempt, least significant bit first), and `[w] = A·[y] + [e_w]` with
    /// `errors` (`W_COEFFICIENTS_PER_ATTEMPT` for each attempt).
    pub fn new(a_hat: &[Vec<Poly>], y_bits: &[Fq], errors: &[Fq]) -> Self {
        let gamma1 = Fq::from(u64::from(ML_DSA_44.gamma1));
        let y: Vec<Vec<Poly>> = y_bits
            .chunks_exact(Y_BITS_PER_ATTEMPT)
            .map(|bits| {
                let coefficients: Vec<Fq> = bits
                    .chunks_exact(Y_BITS)
                    .map(|r| &gamma1 - r.iter().rev().fold(Fq::zero(), |acc, bit| acc.double() + bit))
                    .collect();
                into_polys(&coefficients)
            })
            .collect();
        let w = y
            .iter()
            .zip(errors.chunks_exact(W_COEFFICIENTS_PER_ATTEMPT))
            .map(|(y, e_w)| {
                matrix_vector_product(a_hat, y)
                    .into_iter()
                    .zip(into_polys(e_w))
                    .map(|(a_y, e_w)| std::array::from_fn(|i| &a_y[i] + &e_w[i]))
                    .collect()
            })
            .collect();
        Self { y, w }
    }

    /// The input of `CanonicalMod`, for all attempts: `−44·[w] + (q − 1)/2`.
    pub fn high_bits_input(&self) -> Vec<Fq> {
        let (scale, offset) = (-Fq::from(W1_MODULUS as u64), Fq::from(u64::from((Q - 1) / 2)));
        self.w.iter().flatten().flatten().map(|w| w * &scale + &offset).collect()
    }

    /// The attempts, with `[w1]` from `CanonicalMod`.
    pub fn complete(self, w1: &[Fq]) -> Vec<Attempt> {
        let alpha = Fq::from(2 * u64::from(ML_DSA_44.gamma2));
        self.y
            .into_iter()
            .zip(self.w)
            .zip(w1.chunks_exact(W_COEFFICIENTS_PER_ATTEMPT))
            .map(|((y, w), w1)| {
                let w1 = into_polys(w1);
                let w0 = w.iter().zip(&w1).map(|(w, w1)| std::array::from_fn(|i| &w[i] - &w1[i] * &alpha)).collect();
                Attempt { y, w0, w1 }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::{centered, expand_a, high_bits, low_bits};

    #[test]
    fn constants() {
        assert_eq!((Y_BITS, W1_MODULUS), (18, 44));
    }

    /// In the clear: `y = γ1 − r` covers both ends of `(−γ1, γ1]`, and with
    /// the true `w1 = HighBits(w)`, `w0` is `LowBits(w)` modulo `q`.
    #[test]
    fn y_ends_and_the_decomposition() {
        let a_hat = expand_a(&[1u8; 32], &ML_DSA_44);
        // The first coefficient has r = 0, the second r = 2^18 − 1.
        let mut y_bits = vec![Fq::zero(); Y_BITS_PER_ATTEMPT];
        for bit in &mut y_bits[Y_BITS..2 * Y_BITS] {
            *bit = Fq::one();
        }
        for (i, bit) in y_bits.iter_mut().enumerate().skip(2 * Y_BITS) {
            *bit = Fq::from((i % 3 == 0) as u64);
        }
        let errors: Vec<Fq> = (0..W_COEFFICIENTS_PER_ATTEMPT).map(|i| Fq::from((i % 5) as u64) - Fq::from(2u64)).collect();
        let pending = PendingAttempts::new(&a_hat, &y_bits, &errors);
        let gamma1 = i64::from(ML_DSA_44.gamma1);
        assert_eq!((centered(&pending.y[0][0][0]), centered(&pending.y[0][0][1])), (gamma1, 1 - gamma1));
        let input = pending.high_bits_input();
        let w: Vec<u32> = pending.w.iter().flatten().flatten().map(|w| w.representative()).collect();
        let w1: Vec<Fq> = w.iter().map(|w| Fq::from(u64::from(high_bits(*w, ML_DSA_44.gamma2)))).collect();
        for (input, w1) in input.iter().zip(&w1) {
            assert_eq!(Fq::from(u64::from(input.representative() % W1_MODULUS as u32)), *w1);
        }
        let attempt = pending.complete(&w1).remove(0);
        let w0: Vec<i64> = attempt.w0.iter().flatten().map(centered).collect();
        for (w, w0) in w.iter().zip(&w0) {
            assert_eq!(*w0, low_bits(*w, ML_DSA_44.gamma2), "w = {w}");
        }
    }
}
