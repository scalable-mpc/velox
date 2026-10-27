//! The local half of Liu et al.'s truncation (Protocol 3.1, ΠTrunc), shared
//! by `Truncate` and `FixedMul`.
//!
//! Both open a public `c = v + 2^{p_bits−2} + r` — `v` the value to truncate,
//! `r` an edaBit — and then compute, locally,
//!
//! ```text
//! [Trunc_d(v)] = Trunc_d(c) − [Trunc_d(r)] + (1 − [r_msb]) · c_msb · (2^{p_bits−d} − 1) − 2^{p_bits−d−2}
//! ```
//!
//! up to an error in `{0, ±1, ±2}`. `2^{p_bits−2}` shifts
//! `v ∈ [−2^{p_bits−2}, 2^{p_bits−2})` into `[0, (p−1)/2]`, where Theorem 3.2
//! holds for any `r`; the last term shifts back (Corollary 3.3). The
//! `(1 − r_msb) · c_msb` term is the one wrap-around case, detected from the
//! public MSB of `c`.

use fields::ProtocolField;

use crate::primitives::mersenne;

use crate::primitives::edabit::EdaBit;

use super::E;

/// The offset that makes the value non-negative before it is opened.
pub fn offset<F: ProtocolField>() -> E<F> {
    mersenne::pow2::<F>(mersenne::p_bits::<F>() - 2)
}

/// Steps 6–8 of ΠTrunc on the opened `c`.
pub fn unmask<F: ProtocolField>(c: &E<F>, eda: &EdaBit<F>, d: usize) -> E<F> {
    let p_bits = mersenne::p_bits::<F>();
    let one = E::<F>::one();
    let c_msb = mersenne::bit::<F>(c, p_bits - 1);
    // e = (1 − r_msb) · c_msb, with c_msb public.
    let e = if c_msb == 1 { &one - eda.msb() } else { E::<F>::zero() };
    let correction = mersenne::pow2::<F>(p_bits - d) - &one;
    // `Trunc_d(c)` of the public `c` (Theorem 3.1): shift down by `d`, fill
    // the top `d` bits with the MSB.
    let c_value = mersenne::canonical::<F>(c);
    let mut c_trunc = c_value >> d;
    for i in (p_bits - d)..p_bits {
        c_trunc |= (c_value >> (p_bits - 1)) << i;
    }
    mersenne::from_u128::<F>(c_trunc) - eda.trunc_shift(d) + e * correction - mersenne::pow2::<F>(p_bits - d - 2)
}
