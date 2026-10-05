//! Fields whose modulus is a Mersenne prime `p = 2^ℓ − 1`.
//!
//! The comparison and truncation protocols (Liu–Xie–Yu, USENIX Sec'24; issue
//! #5) work only over such a field, and they rest on two facts about it:
//!
//! - **`p` in binary is all ones**, so `bits(p − b) = ¬bits(b)`: the bit
//!   sharings of `p − b` are `1 − [b_i]`, for free.
//! - **`p` is odd**, so `msb(a) = lsb(2a)`: sign extraction is one doubling
//!   plus a least-significant-bit test.
//!
//! Both are used on *public* values — the revealed `y = 2a + r` and
//! `c = a + r` — which is why the trait exposes the canonical integer an
//! element encodes and its bits, and nothing about sharings. The arithmetic
//! that consumes them (the carry tree, `Trunc_d` on public values) lives in
//! the ops layer, where it is tested against integer references.
//!
//! # A trait and a capability
//!
//! This trait is the static statement — `F: MersennePrimeField` — for code
//! that only ever runs over a Mersenne prime field, such as a circuit's
//! cleartext reference evaluator. Code that is generic over every protocol
//! field and needs Mersenne arithmetic for *some* of what it does — the
//! Planner, whose `Mul`/`Reveal` run over any field but whose comparisons do
//! not — reads the same facts at runtime through
//! [`ProtocolField::MERSENNE_BITS`](crate::ProtocolField::MERSENNE_BITS) and
//! [`ProtocolField::mersenne_canonical`](crate::ProtocolField::mersenne_canonical),
//! which the Mersenne base fields answer by delegating here and every other
//! field answers with `None`. The extension fields are deliberately excluded
//! from both: their elements are not integers mod a Mersenne prime, so
//! neither fact above holds for them.
//!
//! The supertrait is lambdaworks's [`IsPrimeField`], not `ProtocolField`, so
//! the trait is a statement about the field alone; a consumer that needs both
//! writes `F: ProtocolField + MersennePrimeField`. It is implemented for the
//! Mersenne-31, Mersenne-61 and Mersenne-127 base fields.

use lambdaworks_math::field::{element::FieldElement, traits::IsPrimeField};

/// A prime field with modulus `p = 2^BITS − 1`.
pub trait MersennePrimeField: IsPrimeField {
    /// `ℓ` in `p = 2^ℓ − 1`: the number of solved bits behind one random
    /// mask, and the width of the bitwise comparison's carry tree.
    const BITS: usize;

    /// The modulus, derived from [`BITS`](Self::BITS) so an implementor cannot
    /// claim a bit length its prime does not have. `u128` because the
    /// trait has one integer type for every implementor, and Mersenne-127
    /// does not fit a `u64`; the 31- and 61-bit fields widen at no cost.
    const MODULUS: u128 = (1u128 << Self::BITS) - 1;

    /// The unique integer in `[0, p)` that `elem` represents.
    ///
    /// Must go through the field's canonical form, not its raw storage: an
    /// implementation that lets the internal word hold `p` as a second
    /// encoding of zero would otherwise hand the bit layer `0b11…1` for `0`.
    fn to_canonical_u128(elem: &FieldElement<Self>) -> u128;

    /// Bit `i` of the canonical representative, `0` for `i ≥ BITS`.
    fn bit(elem: &FieldElement<Self>, i: usize) -> u64 {
        if i >= Self::BITS {
            0
        } else {
            ((Self::to_canonical_u128(elem) >> i) & 1) as u64
        }
    }
}
