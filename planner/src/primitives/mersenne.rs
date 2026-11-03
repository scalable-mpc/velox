//! The Mersenne-prime arithmetic the comparison and truncation ops need,
//! read through `ProtocolField`'s optional capability so the Planner itself
//! stays generic over every protocol field.
//!
//! `p_bits` and `bit` assume that the field is a Mersenne prime field.
//! `canonical`, `from_u128` and `pow2` do not read `p_bits`, so the
//! uniform-mask prefix can use `pow2` over the ML-DSA field. The Mersenne
//! assumption is checked once, where it can be refused cleanly:
//! `Plan::compile` rejects a declaration with a Mersenne-only op over any
//! other field, and `Op::validate` rejects the op itself. Past those two gates
//! nothing in the Planner reaches `p_bits` or `bit` over a non-Mersenne field,
//! which is why a `None` there is a panic rather than an error.

use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// `p_bits` in `p = 2^p_bits − 1`.
pub fn p_bits<F: ProtocolField>() -> usize {
    F::MERSENNE_BITS.expect("a Mersenne-only op reached a non-Mersenne field past Plan::compile")
}

/// The integer in `[0, p)` that `elem` represents. Any prime field has it,
/// not only a Mersenne field.
pub fn canonical<F: ProtocolField>(elem: &FieldElement<F>) -> u128 {
    F::canonical_u128(elem).expect("an op that reads integers reached a field that is not a prime field")
}

/// Bit `i` of the canonical representative, `0` for `i ≥ p_bits`.
pub fn bit<F: ProtocolField>(elem: &FieldElement<F>, i: usize) -> u64 {
    if i >= p_bits::<F>() {
        0
    } else {
        ((canonical(elem) >> i) & 1) as u64
    }
}

/// The element `x` represents. A public value or a power of two at
/// `p_bits = 127` does not fit the `u64` that `FieldElement::from` takes, so
/// the high word goes in as a multiple of `2^64 = (2^32)^2`.
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
