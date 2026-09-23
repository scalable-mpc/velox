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

use fields::ProtocolField;

use crate::primitives::mersenne;

use crate::primitives::edabit::EdaBit;

use super::E;

pub fn pow2<F: ProtocolField>(exp: usize) -> E<F> {
    E::<F>::from(1u64 << exp)
}

/// The offset that makes the value non-negative before it is opened.
pub fn offset<F: ProtocolField>() -> E<F> {
    pow2::<F>(mersenne::ell::<F>() - 2)
}

/// Steps 6–8 of ΠTrunc on the opened `c`.
pub fn unmask<F: ProtocolField>(c: &E<F>, eda: &EdaBit<F>, d: usize) -> E<F> {
    let ell = mersenne::ell::<F>();
    let one = E::<F>::one();
    let c_msb = mersenne::bit::<F>(c, ell - 1);
    // e = (1 − r_msb) · c_msb, with c_msb public.
    let e = if c_msb == 1 { &one - eda.msb() } else { E::<F>::zero() };
    let correction = pow2::<F>(ell - d) - &one;
    // `Trunc_d(c)` of the public `c` (Theorem 3.1): shift down by `d`, fill
    // the top `d` bits with the MSB.
    let c_value = mersenne::canonical::<F>(c);
    let mut c_trunc = c_value >> d;
    for i in (ell - d)..ell {
        c_trunc |= (c_value >> (ell - 1)) << i;
    }
    E::<F>::from(c_trunc) - eda.trunc_shift(d) + e * correction - pow2::<F>(ell - d - 2)
}
