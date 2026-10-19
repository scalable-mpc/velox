// The number-theoretic transform of FIPS 204 (Algorithms 41 and 42) on the
// ML-DSA ring R_q = F_q[X] / (X^256 + 1).
//
// ζ = 1753 is a primitive 512th root of unity mod q, so X^256 + 1 splits into
// the 256 linear factors X − ζ^(2·brv8(k)+1). The transform maps a polynomial
// to its values at these roots, and a product in R_q becomes a product of the
// values, one per root: 256 field products in place of 65,536.
//
// Both transforms are linear, so a party applies them to its shares of a
// polynomial and gets shares of the transformed polynomial, with no
// communication and no preprocessing.

use lambdaworks_math::field::element::FieldElement;

use super::field::{pow_mod, MlDsaField};

type FqE = FieldElement<MlDsaField>;

/// Coefficients of a ring element.
pub const RING_DEGREE: usize = 256;

/// A primitive 512th root of unity mod q (FIPS 204, §7.5).
const ZETA: u32 = 1753;

/// `256^(−1) mod q`, the scale of the inverse transform.
const RING_DEGREE_INV: u32 = 8_347_681;

/// `ZETAS[k] = ζ^brv8(k)`, with `brv8` the reversal of the 8 bits of `k`:
/// the table of FIPS 204, Appendix B, calculated by the compiler.
const ZETAS: [u32; RING_DEGREE] = {
    let mut zetas = [0u32; RING_DEGREE];
    let mut k = 0;
    while k < RING_DEGREE {
        zetas[k] = pow_mod(ZETA, (k as u8).reverse_bits() as u64);
        k += 1;
    }
    zetas
};

/// The forward transform, in place: FIPS 204 Algorithm 41. The output is in
/// bit-reversed order, as in FIPS 204. 1,024 field products.
pub fn ntt(w: &mut [FqE; RING_DEGREE]) {
    let mut m = 0;
    let mut len = RING_DEGREE / 2;
    while len >= 1 {
        for start in (0..RING_DEGREE).step_by(2 * len) {
            m += 1;
            let z = FqE::from_raw(ZETAS[m]);
            for j in start..start + len {
                let t = &z * &w[j + len];
                w[j + len] = &w[j] - &t;
                w[j] = &w[j] + &t;
            }
        }
        len /= 2;
    }
}

/// The inverse transform, in place: FIPS 204 Algorithm 42.
/// `intt(ntt(w)) = w`. 1,280 field products.
pub fn intt(w: &mut [FqE; RING_DEGREE]) {
    let mut m = RING_DEGREE;
    let mut len = 1;
    while len < RING_DEGREE {
        for start in (0..RING_DEGREE).step_by(2 * len) {
            m -= 1;
            let z = -FqE::from_raw(ZETAS[m]);
            for j in start..start + len {
                let t = w[j].clone();
                w[j] = &t + &w[j + len];
                w[j + len] = &z * &(&t - &w[j + len]);
            }
        }
        len *= 2;
    }
    let scale = FqE::from_raw(RING_DEGREE_INV);
    for c in w.iter_mut() {
        *c = &*c * &scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_dsa::field::ML_DSA_PRIME_FIELD_ORDER as Q;

    /// xorshift: deterministic, reproducible failures.
    fn random_poly(seed: u64) -> [FqE; RING_DEGREE] {
        let mut state = seed | 1;
        std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            FqE::from(state % Q as u64)
        })
    }

    /// The product in `F_q[X] / (X^256 + 1)`: `X^256 = −1`.
    fn negacyclic_product(a: &[FqE; RING_DEGREE], b: &[FqE; RING_DEGREE]) -> [FqE; RING_DEGREE] {
        let mut c: [FqE; RING_DEGREE] = std::array::from_fn(|_| FqE::zero());
        for i in 0..RING_DEGREE {
            for j in 0..RING_DEGREE {
                let p = &a[i] * &b[j];
                if i + j < RING_DEGREE {
                    c[i + j] = &c[i + j] + &p;
                } else {
                    c[i + j - RING_DEGREE] = &c[i + j - RING_DEGREE] - &p;
                }
            }
        }
        c
    }

    #[test]
    fn constants_match_fips_204() {
        assert_eq!(&ZETAS[..10], &[1, 4808194, 3765607, 3761513, 5178923, 5496691, 5234739, 5178987, 7778734, 3542485]);
        assert_eq!(ZETAS[255], 7648983);
        assert_eq!(pow_mod(ZETA, 256), Q - 1, "ζ has order 512");
        assert_eq!(FqE::from(256) * FqE::from_raw(RING_DEGREE_INV), FqE::one());
    }

    #[test]
    fn inverse_undoes_forward() {
        for seed in 1..20 {
            let a = random_poly(seed);
            let mut w = a.clone();
            ntt(&mut w);
            assert_ne!(w, a);
            intt(&mut w);
            assert_eq!(w, a, "seed {seed}");
        }
    }

    #[test]
    fn pointwise_product_is_the_ring_product() {
        for seed in 1..6 {
            let (a, b) = (random_poly(seed), random_poly(seed + 100));
            let (mut a_hat, mut b_hat) = (a.clone(), b.clone());
            ntt(&mut a_hat);
            ntt(&mut b_hat);
            let mut c: [FqE; RING_DEGREE] = std::array::from_fn(|i| &a_hat[i] * &b_hat[i]);
            intt(&mut c);
            assert_eq!(c, negacyclic_product(&a, &b), "seed {seed}");
        }
        // X · X^255 = X^256 = −1.
        let mut x = std::array::from_fn(|_| FqE::zero());
        let mut x255 = x.clone();
        x[1] = FqE::one();
        x255[255] = FqE::one();
        let (mut x_hat, mut x255_hat) = (x.clone(), x255.clone());
        ntt(&mut x_hat);
        ntt(&mut x255_hat);
        let mut c: [FqE; RING_DEGREE] = std::array::from_fn(|i| &x_hat[i] * &x255_hat[i]);
        intt(&mut c);
        assert_eq!(c[0], -FqE::one());
        assert!(c[1..].iter().all(|v| *v == FqE::zero()));
    }

    #[test]
    fn linear_on_shares() {
        // A sum of shares transforms to the sum of the transformed shares.
        let (a, b) = (random_poly(7), random_poly(8));
        let mut sum: [FqE; RING_DEGREE] = std::array::from_fn(|i| &a[i] + &b[i]);
        let (mut a_hat, mut b_hat) = (a, b);
        ntt(&mut sum);
        ntt(&mut a_hat);
        ntt(&mut b_hat);
        assert!((0..RING_DEGREE).all(|i| sum[i] == &a_hat[i] + &b_hat[i]));
    }
}
