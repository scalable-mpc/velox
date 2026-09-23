//! The Mersenne-prime arithmetic the comparison and truncation ops need,
//! read through `ProtocolField`'s optional capability so the Planner itself
//! stays generic over every protocol field.
//!
//! Every function here assumes the field is a Mersenne prime field. That is
//! checked once, where it can be refused cleanly: `Plan::compile` rejects a
//! declaration with a Mersenne-only op over any other field, and
//! `Op::validate` rejects the op itself. Past those two gates nothing in the
//! Planner reaches these functions over a non-Mersenne field, which is why
//! a `None` here is a panic rather than an error.

use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// `ℓ` in `p = 2^ℓ − 1`.
pub fn ell<F: ProtocolField>() -> usize {
    F::MERSENNE_BITS.expect("a Mersenne-only op reached a non-Mersenne field past Plan::compile")
}

/// The integer in `[0, p)` that `elem` represents.
pub fn canonical<F: ProtocolField>(elem: &FieldElement<F>) -> u64 {
    F::mersenne_canonical(elem).expect("a Mersenne-only op reached a non-Mersenne field past Plan::compile")
}

/// Bit `i` of the canonical representative, `0` for `i ≥ ℓ`.
pub fn bit<F: ProtocolField>(elem: &FieldElement<F>, i: usize) -> u64 {
    if i >= ell::<F>() {
        0
    } else {
        (canonical(elem) >> i) & 1
    }
}
