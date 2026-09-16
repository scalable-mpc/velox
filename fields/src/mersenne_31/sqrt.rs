//! Square root over the Mersenne-31 Fp8 tower.
//!
//! `rand_bit` needs `sqrt` over whichever field the protocol shares in, and
//! lambdaworks's inherent `FieldElement::sqrt` is gated on `IsPrimeField`, so
//! the base field is covered but the tower is not. The Mersenne-61 tower uses
//! Scott's complex method, which needs the algebra of each layer written out
//! (`mersenne_61::sqrt`). Fp8 here is three layers deep with a full-Fp4
//! non-residue at the top, so it uses plain Tonelli–Shanks instead: generic
//! over the field, its only inputs are the group order `q = p^8` and one
//! quadratic non-residue, which is found once by Euler's criterion.
//!
//! Cost: `q − 1 = 2^34 · m`, so one call is an exponentiation by the ~248-bit
//! `m` plus a loop bounded by `34²/2` multiplications. Milliseconds per call,
//! called once per random bit in a batch that is already a rayon job.

use std::sync::OnceLock;

use lambdaworks_math::unsigned_integer::element::U256;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::{
    extension_fp8::{Fp, Fp8},
    ser::fp8_from_limbs,
};

/// The order of the base field, `2^31 − 1`.
const P: u64 = (1u64 << 31) - 1;

/// `q − 1 = 2^s · m` with `m` odd, plus the non-residue Tonelli–Shanks needs.
struct Params {
    s: u32,
    m: U256,
    /// `(m + 1) / 2`, the exponent of the first root guess.
    m_plus_one_half: U256,
    /// `(q − 1) / 2`, Euler's criterion exponent.
    euler: U256,
    /// A quadratic non-residue, `z^((q−1)/2) = −1`.
    non_residue: Fp8,
}

fn params() -> &'static Params {
    static PARAMS: OnceLock<Params> = OnceLock::new();
    PARAMS.get_or_init(|| {
        // q − 1 = p^8 − 1, well inside 256 bits.
        let p = U256::from_u64(P);
        let mut q = U256::from_u64(1);
        for _ in 0..8 {
            q = &q * &p;
        }
        let q_minus_one = q - U256::from_u64(1);

        let mut m = q_minus_one.clone();
        let mut s = 0u32;
        while m.limbs[3] & 1 == 0 {
            m = m >> 1;
            s += 1;
        }
        let m_plus_one_half = (&m + &U256::from_u64(1)) >> 1;
        let euler = &q_minus_one >> 1;

        // Every base-field element is a square in Fp2 already, so the
        // candidates have to be genuine Fp8 elements. Half of Fp8 qualifies;
        // a fixed seed keeps the choice deterministic across parties and runs.
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        let non_residue = loop {
            let candidate = fp8_from_limbs(std::array::from_fn(|_| Fp::from(rng.next_u64() & (P - 1))));
            if candidate.pow(euler.clone()) == -Fp8::one() {
                break candidate;
            }
        };

        Params { s, m, m_plus_one_half, euler, non_residue }
    })
}

/// Tonelli–Shanks over Fp8. Returns `Some((r, −r))` with `r² = a`, `None` if
/// `a` is a non-residue; `sqrt(0) = (0, 0)`.
pub fn sqrt_fp8(a: &Fp8) -> Option<(Fp8, Fp8)> {
    if *a == Fp8::zero() {
        return Some((Fp8::zero(), Fp8::zero()));
    }
    let prm = params();
    if a.pow(prm.euler.clone()) != Fp8::one() {
        return None;
    }

    let mut order = prm.s;
    let mut c = prm.non_residue.pow(prm.m.clone());
    let mut t = a.pow(prm.m.clone());
    let mut r = a.pow(prm.m_plus_one_half.clone());

    while t != Fp8::one() {
        // Least i with t^(2^i) = 1; exists and is < order because t is in the
        // 2-power subgroup of size 2^order.
        let mut i = 0u32;
        let mut probe = t.clone();
        while probe != Fp8::one() {
            probe = probe.square();
            i += 1;
        }
        debug_assert!(i < order, "t escaped the 2-Sylow subgroup");
        let mut b = c;
        for _ in 0..(order - i - 1) {
            b = b.square();
        }
        order = i;
        c = b.square();
        t = t * &c;
        r = r * &b;
    }
    Some((r.clone(), -r))
}

/// Exposes the decomposition for tests.
#[cfg(test)]
pub(crate) fn two_adicity() -> u32 {
    params().s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_fp8(rng: &mut ChaCha20Rng) -> Fp8 {
        fp8_from_limbs(std::array::from_fn(|_| Fp::from(rng.next_u64() & (P - 1))))
    }

    /// `p+1 = 2^31` contributes 31, and `p−1`, `p²+1`, `p⁴+1` one each.
    #[test]
    fn q_minus_one_has_two_adicity_34() {
        assert_eq!(two_adicity(), 34);
    }

    #[test]
    fn roots_square_back_and_non_residues_are_rejected() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let mut residues = 0;
        for _ in 0..64 {
            let a = rand_fp8(&mut rng);
            let sq = a.square();
            let (r, neg_r) = sqrt_fp8(&sq).expect("a square has a root");
            assert_eq!(r.square(), sq);
            assert_eq!(neg_r, -r.clone());
            // Of a random element, half are residues: sqrt agrees with Euler.
            match sqrt_fp8(&a) {
                Some((r, _)) => {
                    assert_eq!(r.square(), a);
                    residues += 1;
                }
                None => assert_ne!(a.pow(params().euler.clone()), Fp8::one()),
            }
        }
        assert!((16..48).contains(&residues), "residue count {residues} implausible for 64 draws");
        assert_eq!(sqrt_fp8(&Fp8::zero()), Some((Fp8::zero(), Fp8::zero())));
        assert_eq!(sqrt_fp8(&Fp8::one()).map(|(r, _)| r.square()), Some(Fp8::one()));
    }
}
