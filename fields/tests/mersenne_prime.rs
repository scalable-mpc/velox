//! `MersennePrimeField` pins the two facts the comparison and truncation
//! protocols rest on — `p` is all ones in binary, and `p` is odd — plus the
//! canonical-integer view of an element they compute on. Checked here against
//! plain integer arithmetic for the three base fields, so a field cannot quietly stop
//! satisfying them.

use fields::{mersenne_31::Mersenne31Field, Mersenne127Field, MersennePrimeField, Mersenne61Field};
use lambdaworks_math::field::element::FieldElement;

/// xorshift64, two draws per value: deterministic, no dependency,
/// reproducible failures, and wide enough for the 127-bit field.
fn samples(count: usize, seed: u64) -> impl Iterator<Item = u128> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..count).map(move |_| ((next() as u128) << 64) | next() as u128)
}

/// Every check, for one field. Generic so the three fields run the identical
/// suite; the `From<u64>` bound is what lets a reference value be lifted.
fn check_field<F: MersennePrimeField>()
where
    FieldElement<F>: From<u64>,
    F::RepresentativeType: Into<u128>,
{
    let p = F::MODULUS;
    // A `u128` lifted as `hi·2^64 + lo`: field arithmetic, so it reduces mod p.
    let two_64 = FieldElement::<F>::from(1u64 << 32).square();
    let fe = |x: u128| FieldElement::<F>::from((x >> 64) as u64) * &two_64 + FieldElement::<F>::from(x as u64);
    let canon = |e: &FieldElement<F>| F::to_canonical_u128(e);

    // -- The constants agree with the field's own description of itself.
    assert_eq!(p, (1u128 << F::BITS) - 1);
    assert_eq!(F::field_bit_size(), F::BITS, "field_bit_size disagrees with BITS");
    assert_eq!(p, Into::<u128>::into(F::modulus_minus_one()) + 1, "MODULUS disagrees with the field's p");

    // -- Canonical edges, including the non-canonical zero: `from(p)` reduces
    //    eagerly, so `(p − 1) + 1` is how the internal word actually reaches p.
    assert_eq!(canon(&fe(0)), 0);
    assert_eq!(canon(&fe(1)), 1);
    assert_eq!(canon(&fe(p - 1)), p - 1);
    assert_eq!(canon(&(fe(p - 1) + fe(1))), 0, "(p-1)+1 must read as 0, not p");
    assert_eq!(canon(&fe(p)), 0);
    assert_eq!(canon(&-fe(1)), p - 1);

    // -- Round trip through `from(u64)`, which reduces mod p. Inputs stay within
    //    `2·BITS` bits (capped at 63): each field's `from_u64` folds a fixed
    //    number of times — lambdaworks's M31 twice, so it is only correct below
    //    ~2^62 — and both carry a `+ 1` that overflows near `u64::MAX`. That is
    //    their contract, not this trait's. Every `u64` is below the 127-bit p.
    let width = (2 * F::BITS).min(63);
    let mask = (1u64 << width) - 1;
    let in_domain = samples(10_000, 0x5eed).map(|x| x as u64 & mask);
    let p64 = u64::try_from(p).unwrap_or(u64::MAX);
    for x in [p64, p64.wrapping_add(1), p64.wrapping_mul(2), p64.wrapping_mul(2).wrapping_add(1), mask].into_iter().chain(in_domain).filter(|&x| x <= mask) {
        assert_eq!(canon(&FieldElement::<F>::from(x)), x as u128 % p, "x={x}");
    }

    // -- Round trip of full-width values through the field arithmetic.
    for x in samples(1_000, 0xf011) {
        assert_eq!(canon(&fe(x)), x % p, "x={x}");
    }

    // -- Bits reassemble the canonical value; bits past BITS are zero.
    for x in samples(1_000, 0xb175).map(|x| x % p) {
        let e = fe(x);
        let reassembled: u128 = (0..F::BITS).map(|i| (F::bit(&e, i) as u128) << i).sum();
        assert_eq!(reassembled, x);
        assert_eq!(F::bit(&e, F::BITS), 0);
        assert_eq!(F::bit(&e, 127), 0);
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
    assert_eq!(Mersenne61Field::MODULUS, (1u128 << 61) - 1);
    check_field::<Mersenne61Field>();
}

#[test]
fn mersenne_31() {
    assert_eq!(Mersenne31Field::BITS, 31);
    assert_eq!(Mersenne31Field::MODULUS, (1u128 << 31) - 1);
    check_field::<Mersenne31Field>();
}

#[test]
fn mersenne_127() {
    assert_eq!(Mersenne127Field::BITS, 127);
    assert_eq!(Mersenne127Field::MODULUS, (1u128 << 127) - 1);
    check_field::<Mersenne127Field>();
}
