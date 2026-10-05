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
pub fn canonical<F: ProtocolField>(elem: &FieldElement<F>) -> u128 {
    F::mersenne_canonical(elem).expect("a Mersenne-only op reached a non-Mersenne field past Plan::compile")
}

/// Bit `i` of the canonical representative, `0` for `i ≥ ℓ`.
pub fn bit<F: ProtocolField>(elem: &FieldElement<F>, i: usize) -> u64 {
    if i >= ell::<F>() {
        0
    } else {
        ((canonical(elem) >> i) & 1) as u64
    }
}

/// The element `x` represents. A public value or a power of two at
/// `ℓ = 127` does not fit the `u64` that `FieldElement::from` takes, so the
/// high word goes in as a multiple of `2^64 = (2^32)^2`.
pub fn from_u128<F: ProtocolField>(x: u128) -> FieldElement<F> {
    let low = FieldElement::<F>::from(x as u64);
    match (x >> 64) as u64 {
        0 => low,
        high => FieldElement::<F>::from(high) * FieldElement::<F>::from(1u64 << 32).square() + low,
    }
}

/// `2^exp`, for `exp < 128`.
pub fn pow2<F: ProtocolField>(exp: usize) -> FieldElement<F> {
    from_u128(1u128 << exp)
}
