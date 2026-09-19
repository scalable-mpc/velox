//! The local half of Liu et al.'s truncation (Protocol 3.1, ΠTrunc), shared
//! by `Truncate` and `FixedMul`.
//!
//! Both open a public `c = v + 2^{ℓ−2} + r` — `v` the value to truncate,
//! `r` an edaBit — and then compute, locally,
//!
//! ```text
//! [Trunc_d(v)] = Trunc_d(c) − [Trunc_d(r)] + (1 − [r_msb]) · c_msb · (2^{ℓ−d} − 1) − 2^{ℓ−d−2}
//! ```
//!
//! up to an error in `{0, ±1, ±2}`. `2^{ℓ−2}` shifts `v ∈ [−2^{ℓ−2}, 2^{ℓ−2})`
//! into `[0, (p−1)/2]`, where Theorem 3.2 holds for any `r`; the last term
//! shifts back (Corollary 3.3). The `(1 − r_msb) · c_msb` term is the one
//! wrap-around case, detected from the public MSB of `c`.

use fields::{MersennePrimeField, ProtocolField};

use crate::primitives::edabit::EdaBit;

use super::E;

/// `Trunc_d(c)` of a public `c` (Theorem 3.1): shift down by `d`, fill the
/// top `d` bits with the MSB.
pub fn trunc_public<F: ProtocolField + MersennePrimeField>(c: &E<F>, d: usize) -> E<F> {
    let ell = F::BITS;
    let value = F::to_canonical_u64(c);
    let msb = value >> (ell - 1);
    let mut out = value >> d;
    for i in (ell - d)..ell {
        out |= msb << i;
    }
    E::<F>::from(out)
}

pub fn pow2<F: ProtocolField>(exp: usize) -> E<F> {
    E::<F>::from(1u64 << exp)
}

/// The offset that makes the value non-negative before it is opened.
pub fn offset<F: ProtocolField + MersennePrimeField>() -> E<F> {
    pow2::<F>(F::BITS - 2)
}

/// Steps 6–8 of ΠTrunc on the opened `c`.
pub fn unmask<F: ProtocolField + MersennePrimeField>(c: &E<F>, eda: &EdaBit<F>, d: usize) -> E<F> {
    let ell = F::BITS;
    let one = E::<F>::one();
    let c_msb = F::bit(c, ell - 1);
    // e = (1 − r_msb) · c_msb, with c_msb public.
    let e = if c_msb == 1 { &one - eda.msb() } else { E::<F>::zero() };
    let correction = pow2::<F>(ell - d) - &one;
    trunc_public(c, d) - eda.trunc_shift(d) + e * correction - pow2::<F>(ell - d - 2)
}
