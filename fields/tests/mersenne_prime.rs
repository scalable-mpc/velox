//! `MersennePrimeField` pins the two facts the comparison and truncation
//! protocols rest on — `p` is all ones in binary, and `p` is odd — plus the
//! canonical-integer view of an element they compute on. Checked here against
//! plain `u64` arithmetic for both base fields, so a field cannot quietly stop
//! satisfying them.

use fields::{mersenne_31::Mersenne31Field, MersennePrimeField, Mersenne61Field};
use lambdaworks_math::field::element::FieldElement;

/// xorshift64: deterministic, no dependency, reproducible failures.
fn samples(count: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed | 1;
    (0..count).map(move |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    })
}

/// Every check, for one field. Generic so the two fields run the identical
/// suite; the `From<u64>` bound is what lets a `u64` reference value be lifted.
fn check_field<F: MersennePrimeField>()
where
    FieldElement<F>: From<u64>,
    F::RepresentativeType: Into<u64>,
{
    let p = F::MODULUS;
    let fe = |x: u64| FieldElement::<F>::from(x);
    let canon = |e: &FieldElement<F>| F::to_canonical_u64(e);

    // -- The constants agree with the field's own description of itself.
    assert_eq!(p, (1u64 << F::BITS) - 1);
    assert_eq!(F::field_bit_size(), F::BITS, "field_bit_size disagrees with BITS");
    assert_eq!(p, Into::<u64>::into(F::modulus_minus_one()) + 1, "MODULUS disagrees with the field's p");

    // -- Canonical edges, including the non-canonical zero: `from(p)` reduces
    //    eagerly, so `(p − 1) + 1` is how the internal word actually reaches p.
    assert_eq!(canon(&fe(0)), 0);
    assert_eq!(canon(&fe(1)), 1);
    assert_eq!(canon(&fe(p - 1)), p - 1);
    assert_eq!(canon(&(fe(p - 1) + fe(1))), 0, "(p-1)+1 must read as 0, not p");
    assert_eq!(canon(&fe(p)), 0);
    assert_eq!(canon(&-fe(1)), p - 1);

    // -- Round trip through `from`, which reduces mod p. Inputs stay within
    //    `2·BITS` bits (capped at 63): each field's `from_u64` folds a fixed
    //    number of times — lambdaworks's M31 twice, so it is only correct below
    //    ~2^62 — and both carry a `+ 1` that overflows near `u64::MAX`. That is
    //    their contract, not this trait's.
    let width = (2 * F::BITS).min(63);
    let mask = (1u64 << width) - 1;
    let in_domain = samples(10_000, 0x5eed).map(|x| x & mask);
    for x in [p, p + 1, 2 * p, 2 * p + 1, mask].into_iter().chain(in_domain) {
        assert_eq!(canon(&fe(x)), x % p, "x={x}");
    }

    // -- Bits reassemble the canonical value; bits past BITS are zero.
    for x in samples(1_000, 0xb175).map(|x| x % p) {
        let e = fe(x);
        let reassembled: u64 = (0..F::BITS).map(|i| F::bit(&e, i) << i).sum();
        assert_eq!(reassembled, x);
        assert_eq!(F::bit(&e, F::BITS), 0);
        assert_eq!(F::bit(&e, 63), 0);
    }
    assert_eq!(F::bit(&fe(p - 1), F::BITS - 1), 1, "p-1 has its top bit set");

    // -- Fact 1: bits(p − b) = ¬bits(b) for b ≠ 0 (the BitLTL complement
    //    trick). For b = 0, p − b = p ≡ 0, which is exactly the 2^-ℓ case the
    //    protocols accept, so it is excluded here.
    for b in samples(1_000, 0xc0de).map(|b| b % p).filter(|&b| b != 0) {
        let neg = fe(p - b);
        for i in 0..F::BITS {
            assert_eq!(F::bit(&neg, i), 1 - F::bit(&fe(b), i), "b={b} bit {i}");
        }
        assert_eq!(canon(&neg), canon(&-fe(b)), "p - b is the field negation");
    }

    // -- Fact 2: msb(a) = lsb(2a) (the DReLU sign trick; needs p odd). The MSB
    //    is bit ℓ−1, set exactly when a > (p−1)/2, i.e. a is "negative".
    for a in samples(1_000, 0xd0ab).map(|a| a % p) {
        let doubled = fe(a) + fe(a);
        assert_eq!(F::bit(&fe(a), F::BITS - 1), F::bit(&doubled, 0), "a={a}");
        assert_eq!(F::bit(&fe(a), F::BITS - 1), (a > (p - 1) / 2) as u64, "a={a}");
    }
}

#[test]
fn mersenne_61() {
    assert_eq!(Mersenne61Field::BITS, 61);
    assert_eq!(Mersenne61Field::MODULUS, (1u64 << 61) - 1);
    check_field::<Mersenne61Field>();
}

#[test]
fn mersenne_31() {
    assert_eq!(Mersenne31Field::BITS, 31);
    assert_eq!(Mersenne31Field::MODULUS, (1u64 << 31) - 1);
    check_field::<Mersenne31Field>();
}
