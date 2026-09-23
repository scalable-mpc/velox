//! [`ProtocolField`] for 256-bit Montgomery prime fields — Stark252, BN254 and
//! the BLS12-381 scalar field.
//!
//! Written as one blanket impl over `MontgomeryBackendPrimeField<M, 4>` rather
//! than three copies, because nothing in it depends on the modulus: sampling,
//! serialization, square roots and the text packing are all the same
//! computation for any 256-bit prime. Naming another such field costs a type
//! alias and nothing else — which is all the BLS12-381 scalar field is.
//!
//! # How these differ from the Mersenne-61 Fp4 field
//!
//! Fp4_61 is a degree-4 *extension*, so it carries four independent 61-bit
//! limbs and its text packing has to dodge a per-limb modular fold. These are
//! *prime* fields with a single ~2^252 / ~2^254 element, so the packing is the
//! straightforward one — keep the value below the modulus and the round-trip is
//! exact. Square roots come from lambdaworks's Tonelli–Shanks on `IsPrimeField`
//! rather than the hand-rolled Scott's-method tower in `mersenne_61::sqrt`.

use core::fmt::Debug;

use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::{
        bls12_381::default_types::FrField as BLS12381FrField,
        bn_254::field_extension::BN254PrimeField,
    },
    errors::ByteConversionError,
    field::{
        element::FieldElement,
        fields::{
            fft_friendly::stark_252_prime_field::Stark252PrimeField,
            montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        },
    },
    traits::ByteConversion as LambdaByteConversion,
    unsigned_integer::element::U256,
};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::protocol_field::ProtocolField;

/// The STARK-friendly prime field, `p = 2^251 + 17·2^192 + 1`.
pub type Stark252Field = Stark252PrimeField;

/// The BN254 curve's base field, `p ≈ 2^253.6`.
pub type BN254Field = BN254PrimeField;

/// The BLS12-381 curve's *scalar* field, `r ≈ 2^254.9` (2-adicity 32).
///
/// This is the field the group exponents live in, so an application whose
/// outputs are later lifted into G1/G2 — the BTX/BTE setup exponentiates each
/// party's shares — has to compute over it, not over the curve's base field.
pub type BLS12381ScalarField = BLS12381FrField;

/// Serialized width of a 256-bit element.
const U256_BYTES: usize = 32;

/// Text capacity: the top byte is left clear so the packed value stays below
/// `2^248`, which is below both moduli, making the round-trip exact.
///
/// A `U256` field whose modulus is itself below `2^248` would not satisfy that,
/// so [`encode_ascii`](ProtocolField::encode_ascii) verifies the round-trip
/// rather than trusting this constant. Both fields named in this module clear
/// it with room to spare.
const U256_ASCII_PAYLOAD: usize = 31;

/// Number of 64-bit draws folded into one sampled element.
///
/// Five draws is 320 bits reduced into a ≤256-bit field, so the modular bias is
/// below `2^-64` — negligible, and cheaper than rejection sampling, which would
/// reject about 22% of draws for Stark252 (`p/2^252 ≈ 0.78`).
const SAMPLE_LIMBS: usize = 5;

/// Fold `SAMPLE_LIMBS` 64-bit draws into a field element by Horner's rule.
fn sample_from<M, R>(rng: &mut R) -> FieldElement<MontgomeryBackendPrimeField<M, 4>>
where
    M: IsModulus<U256> + Clone + Debug,
    R: RngCore,
{
    // 2^64 as a field element. Both moduli exceed 2^64, so this does not wrap.
    let shift = FieldElement::<MontgomeryBackendPrimeField<M, 4>>::from(u64::MAX)
        + FieldElement::<MontgomeryBackendPrimeField<M, 4>>::one();

    let mut acc = FieldElement::<MontgomeryBackendPrimeField<M, 4>>::zero();
    for _ in 0..SAMPLE_LIMBS {
        acc = acc * &shift + FieldElement::<MontgomeryBackendPrimeField<M, 4>>::from(rng.next_u64());
    }
    acc
}

/// A `RngCore` over the thread RNG, so `rand()` and `from_rng()` share one
/// sampling routine instead of drifting apart.
struct ThreadDraws;

impl RngCore for ThreadDraws {
    fn next_u32(&mut self) -> u32 {
        random::<u32>()
    }
    fn next_u64(&mut self) -> u64 {
        random::<u64>()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for b in dest.iter_mut() {
            *b = random::<u8>();
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl<M> ProtocolField for MontgomeryBackendPrimeField<M, 4>
where
    M: IsModulus<U256> + Clone + Debug + Send + Sync + 'static,
{
    const SER_BYTES: usize = U256_BYTES;
    const MAX_INPUT_PAYLOAD: usize = U256_ASCII_PAYLOAD;

    /// Stark252 is ~252-bit, BN254 ~254-bit and BLS12-381's scalar field
    /// ~255-bit, all already wide enough for a 2^-250-ish soundness bound, so
    /// these are their own extension.
    type Ext = Self;
    const CONV_RATIO: usize = 1;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        chunk.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        elem.clone()
    }

    /// Wide enough for statistical checks too: verification runs here as it is.
    type StatisticalExt = Self;
    const STATISTICAL_DEGREE: usize = 1;

    fn statistical_coeff(elem: &FieldElement<Self>, _k: usize) -> FieldElement<Self> {
        elem.clone()
    }

    fn from_statistical_coeffs(coeffs: &[FieldElement<Self>]) -> FieldElement<Self> {
        coeffs.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn rand() -> FieldElement<Self> {
        sample_from::<M, _>(&mut ThreadDraws)
    }

    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        sample_from::<M, _>(rng)
    }

    /// Tonelli–Shanks, via lambdaworks's inherent `sqrt` on `IsPrimeField`.
    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        elem.sqrt()
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        LambdaByteConversion::to_bytes_be(elem)
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        LambdaByteConversion::to_bytes_le(elem)
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as LambdaByteConversion>::from_bytes_be(bytes)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as LambdaByteConversion>::from_bytes_le(bytes)
    }

    /// Pack a text line into one element, right-aligned in the low 31 bytes.
    ///
    /// Unlike the Mersenne-61 Fp4 packing, there are no per-limb folds to dodge
    /// here — a prime field has one modular reduction, and clearing the top byte
    /// keeps the value below it. The round-trip is verified rather than assumed,
    /// so a `U256` field with an unexpectedly small modulus reports `None`
    /// instead of silently corrupting a party's input.
    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let bytes = input.as_bytes();
        if bytes.len() > Self::MAX_INPUT_PAYLOAD {
            return None;
        }
        let mut padded = [0u8; U256_BYTES];
        padded[U256_BYTES - bytes.len()..].copy_from_slice(bytes);

        let elem = <Self as ProtocolField>::from_bytes_be(&padded).ok()?;
        // `from_bytes_be` reduces mod p, so a value at or above the modulus
        // would come back as something else entirely.
        (<Self as ProtocolField>::to_bytes_be(&elem) == padded).then_some(elem)
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        let bytes = <Self as ProtocolField>::to_bytes_be(elem);
        let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
        bytes[first_nonzero..].iter().map(|&b| b as char).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    /// Every property is checked for every field through the same generic body:
    /// what the protocol relies on is the trait's contract, not any modulus.
    fn field_contract<F: ProtocolField>(name: &str) {
        let e = F::rand();
        assert_eq!(F::to_bytes_be(&e).len(), F::SER_BYTES, "{name}: width");
        assert_eq!(F::from_bytes_be(&F::to_bytes_be(&e)).unwrap(), e, "{name}: byte round-trip");

        for line in ["", "a", "hello world", "mountain grape desert sky", "~~~~~~~"] {
            let packed = F::encode_ascii(line).unwrap_or_else(|| panic!("{name}: {line:?} should fit"));
            assert_eq!(F::decode_ascii(&packed), line, "{name}: text round-trip");
        }

        assert!(F::encode_ascii(&"x".repeat(F::MAX_INPUT_PAYLOAD)).is_some(), "{name}: at capacity");
        assert!(F::encode_ascii(&"x".repeat(F::MAX_INPUT_PAYLOAD + 1)).is_none(), "{name}: over capacity");

        let draw = || {
            let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
            (0..4).map(|_| F::from_rng(&mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw(), "{name}: PRF sampling is seed-determined");

        let root = F::rand();
        let square = &root * &root;
        let (a, b) = F::sqrt(&square).unwrap_or_else(|| panic!("{name}: a square has a root"));
        assert!(a == root || b == root, "{name}: sqrt recovers the root");
    }

    #[test]
    fn stark252_satisfies_the_contract() {
        field_contract::<Stark252Field>("Stark252");
    }

    #[test]
    fn bn254_satisfies_the_contract() {
        field_contract::<BN254Field>("BN254");
    }

    #[test]
    fn bls12_381_scalar_satisfies_the_contract() {
        field_contract::<BLS12381ScalarField>("BLS12-381 Fr");
    }

    /// The fields must not collapse to the same arithmetic — a guard against
    /// a blanket impl accidentally keying off something modulus-independent.
    #[test]
    fn the_fields_are_distinct() {
        let text = "distinct";
        let s = Stark252Field::encode_ascii(text).unwrap();
        let b = BN254Field::encode_ascii(text).unwrap();
        let r = BLS12381ScalarField::encode_ascii(text).unwrap();
        // Same bytes in, same bytes out — but they are different types, and
        // their inverses differ because the moduli differ.
        assert_eq!(
            Stark252Field::to_bytes_be(&s),
            BN254Field::to_bytes_be(&b),
            "identical payloads encode identically"
        );
        assert_eq!(
            Stark252Field::to_bytes_be(&s),
            BLS12381ScalarField::to_bytes_be(&r),
            "identical payloads encode identically"
        );
        let s_inv = Stark252Field::to_bytes_be(&s.inv().unwrap());
        let b_inv = BN254Field::to_bytes_be(&b.inv().unwrap());
        let r_inv = BLS12381ScalarField::to_bytes_be(&r.inv().unwrap());
        assert_ne!(s_inv, b_inv, "different moduli must give different inverses");
        assert_ne!(s_inv, r_inv, "different moduli must give different inverses");
        assert_ne!(b_inv, r_inv, "different moduli must give different inverses");
    }
}
