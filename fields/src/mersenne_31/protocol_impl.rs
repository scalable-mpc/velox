//! [`ProtocolField`] for the Mersenne-31 base field and its Fp8 tower.
//!
//!
//! # A caveat that is the engine's, not this field's
//!
//! [`ProtocolField::Ext`] covers the ACSS DZK proof only. That is
//! acceptable for benchmarking the comparison layer at `ℓ = 31` (half the
//! carry tree of `ℓ = 61`) and not for a deployment; lifting verification is
//! tracked separately.

use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsSubFieldOf},
};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::{byte_conv::ByteConversion, protocol_field::ProtocolField};

use super::{
    extension_fp8::{Degree8ExtensionField, Fp, Mersenne31Field},
    ser::fp8_from_limbs,
    sqrt::sqrt_fp8,
};

/// `p = 2^31 − 1`. A draw of this many bits reduces with a `2^-31` bias
/// towards the two smallest residues — the same order as the field's own
/// statistical security, and the same trade the 61-bit base makes.
const LIMB_MASK: u64 = (1u64 << 31) - 1;

fn limb_from(x: u64) -> Fp {
    Fp::from(x & LIMB_MASK)
}

/// `limbs` limbs of 3 payload bytes each, high byte clear so the limb
/// stays below `2^24 < p` and the reduction is a no-op — the single-limb and
/// eight-limb versions of one packing.
fn encode_ascii_limbs(limbs: usize, input: &str) -> Option<Vec<u8>> {
    const PAYLOAD: usize = 3;
    let bytes = input.as_bytes();
    if bytes.len() > PAYLOAD * limbs {
        return None;
    }
    let mut padded = vec![0u8; 4 * limbs];
    let mut remaining = bytes;
    for limb in (0..limbs).rev() {
        let take = remaining.len().min(PAYLOAD);
        if take == 0 {
            break;
        }
        let base = limb * 4;
        // Byte 0 of the limb stays 0x00; payload sits in bytes 1..4,
        // right-aligned when partially filled.
        let dest_start = base + 1 + (PAYLOAD - take);
        padded[dest_start..base + 4].copy_from_slice(&remaining[remaining.len() - take..]);
        remaining = &remaining[..remaining.len() - take];
    }
    Some(padded)
}

fn decode_ascii_limbs(limbs: usize, bytes: &[u8]) -> String {
    let mut payload = Vec::with_capacity(3 * limbs);
    for limb in 0..limbs {
        payload.extend_from_slice(&bytes[limb * 4 + 1..limb * 4 + 4]);
    }
    let first_nonzero = payload.iter().position(|&b| b != 0).unwrap_or(payload.len());
    payload[first_nonzero..].iter().map(|&b| b as char).collect()
}

impl ProtocolField for Mersenne31Field {
    /// One 4-byte limb.
    const SER_BYTES: usize = 4;

    /// 3 payload bytes per limb; see [`encode_ascii_limbs`].
    const MAX_INPUT_PAYLOAD: usize = 3;

    /// 31 bits is nowhere near a soundness bound; challenges lift into Fp8.
    type Ext = Degree8ExtensionField;

    /// Eight base elements are the eight coefficients of one Fp8 element.
    const CONV_RATIO: usize = 8;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        debug_assert!(chunk.len() <= Self::CONV_RATIO, "chunk wider than one Ext element");
        let mut limbs: [Fp; 8] = std::array::from_fn(|_| Fp::zero());
        limbs[..chunk.len()].clone_from_slice(chunk);
        // Coefficient order matches `to_subfield_vec`, so `lift` is its inverse.
        fp8_from_limbs(limbs)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        FieldElement::new(<Self as IsSubFieldOf<Degree8ExtensionField>>::embed(*elem.value()))
    }

    fn rand() -> FieldElement<Self> {
        limb_from(random::<u64>())
    }

    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        limb_from(rng.next_u64())
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        // Tonelli-Shanks, from lambdaworks's inherent `sqrt` on `IsPrimeField`.
        elem.sqrt()
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        ByteConversion::to_bytes_be(elem)
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        ByteConversion::to_bytes_le(elem)
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as ByteConversion>::from_bytes_be(bytes)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as ByteConversion>::from_bytes_le(bytes)
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_limbs(1, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_limbs(1, &<Self as ProtocolField>::to_bytes_be(elem))
    }

    /// `simd::m31_avx2` single-limb kernel; scalar when the CPU has no AVX2
    /// or `VELOX_SIMD=off`.
    #[cfg(target_arch = "x86_64")]
    fn try_simd_gemm(
        matrix: &[Vec<FieldElement<Self>>],
        vectors: &[Vec<FieldElement<Self>>],
        row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        Some(crate::simd::gemm(matrix, vectors, row_major))
    }
}

impl ProtocolField for Degree8ExtensionField {
    /// 8 × 4-byte limbs.
    const SER_BYTES: usize = 32;

    /// 3 payload bytes in each of 8 limbs.
    const MAX_INPUT_PAYLOAD: usize = 24;

    type Ext = Self;
    const CONV_RATIO: usize = 1;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        chunk.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        elem.clone()
    }

    fn rand() -> FieldElement<Self> {
        fp8_from_limbs(std::array::from_fn(|_| limb_from(random::<u64>())))
    }

    /// Eight draws per element, in limb order.
    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        fp8_from_limbs(std::array::from_fn(|_| limb_from(rng.next_u64())))
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        sqrt_fp8(elem)
    }

    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8> {
        ByteConversion::to_bytes_be(elem)
    }

    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8> {
        ByteConversion::to_bytes_le(elem)
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as ByteConversion>::from_bytes_be(bytes)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError> {
        <FieldElement<Self> as ByteConversion>::from_bytes_le(bytes)
    }

    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let padded = encode_ascii_limbs(8, input)?;
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        decode_ascii_limbs(8, &<Self as ProtocolField>::to_bytes_be(elem))
    }

    /// Eight-lane Fp8 kernel (`simd::m31_avx2`).
    #[cfg(target_arch = "x86_64")]
    fn try_simd_gemm(
        matrix: &[Vec<FieldElement<Self>>],
        vectors: &[Vec<FieldElement<Self>>],
        row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        Some(crate::simd::gemm(matrix, vectors, row_major))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mersenne_31::ser::fp8_limbs;
    use rand_core::SeedableRng;

    type Base = Mersenne31Field;
    type Ext8 = Degree8ExtensionField;

    #[test]
    fn serialization_is_exactly_ser_bytes_wide() {
        let b = Base::rand();
        assert_eq!(Base::to_bytes_be(&b).len(), Base::SER_BYTES);
        assert_eq!(Base::to_bytes_le(&b).len(), Base::SER_BYTES);
        let e = Ext8::rand();
        assert_eq!(Ext8::to_bytes_be(&e).len(), Ext8::SER_BYTES);
        assert_eq!(Ext8::to_bytes_le(&e).len(), Ext8::SER_BYTES);
    }

    #[test]
    fn byte_round_trip() {
        for _ in 0..16 {
            let b = Base::rand();
            assert_eq!(Base::from_bytes_be(&Base::to_bytes_be(&b)).unwrap(), b);
            assert_eq!(Base::from_bytes_le(&Base::to_bytes_le(&b)).unwrap(), b);
            let e = Ext8::rand();
            assert_eq!(Ext8::from_bytes_be(&Ext8::to_bytes_be(&e)).unwrap(), e);
            assert_eq!(Ext8::from_bytes_le(&Ext8::to_bytes_le(&e)).unwrap(), e);
        }
        assert!(Base::from_bytes_be(&[0u8; 3]).is_err());
        assert!(Ext8::from_bytes_be(&[0u8; 31]).is_err());
    }

    /// Letters live in `0x60..0x7F`; with 4 payload bytes a limb would exceed
    /// `2^31 − 1` and fold. 3 bytes per limb keeps the reduction a no-op.
    #[test]
    fn ascii_round_trips_including_letters() {
        for line in ["", "a", "~~~", "hello world", "ZZZZZZZZZZZZZZZZZZZZZZZZ"] {
            let e = Ext8::encode_ascii(line).expect("within payload capacity");
            assert_eq!(Ext8::decode_ascii(&e), line, "Fp8 round-trip mangled {:?}", line);
        }
        for line in ["", "a", "~~~", "zzz"] {
            let e = Base::encode_ascii(line).expect("within payload capacity");
            assert_eq!(Base::decode_ascii(&e), line, "base round-trip mangled {:?}", line);
        }
    }

    #[test]
    fn ascii_rejects_oversized_input() {
        assert!(Base::encode_ascii(&"x".repeat(Base::MAX_INPUT_PAYLOAD + 1)).is_none());
        assert!(Base::encode_ascii(&"x".repeat(Base::MAX_INPUT_PAYLOAD)).is_some());
        assert!(Ext8::encode_ascii(&"x".repeat(Ext8::MAX_INPUT_PAYLOAD + 1)).is_none());
        assert!(Ext8::encode_ascii(&"x".repeat(Ext8::MAX_INPUT_PAYLOAD)).is_some());
    }

    #[test]
    fn from_rng_is_deterministic_in_the_seed() {
        let draw = || {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            let base: Vec<_> = (0..4).map(|_| Base::from_rng(&mut rng)).collect();
            let ext: Vec<_> = (0..2).map(|_| Ext8::from_rng(&mut rng)).collect();
            (base, ext)
        };
        assert_eq!(draw(), draw());
    }

    /// `lift` packs coefficient-wise and inverts `to_subfield_vec`; `embed_ext`
    /// is the field homomorphism, so it commutes with arithmetic.
    #[test]
    fn lift_and_embed_agree_with_the_tower() {
        let chunk: Vec<_> = (0..8).map(|_| Base::rand()).collect();
        let lifted = Base::lift(&chunk);
        let back: Vec<Fp> = <Base as IsSubFieldOf<Ext8>>::to_subfield_vec(lifted.value().clone())
            .into_iter()
            .map(Fp::from_raw)
            .collect();
        assert_eq!(back, chunk);
        assert_eq!(fp8_limbs(&lifted).to_vec(), chunk);
        // Short chunks zero-pad.
        assert_eq!(fp8_limbs(&Base::lift(&chunk[..3]))[3..], [Fp::zero(), Fp::zero(), Fp::zero(), Fp::zero(), Fp::zero()]);

        let (x, y) = (Base::rand(), Base::rand());
        assert_eq!(Base::embed_ext(&(&x + &y)), Base::embed_ext(&x) + Base::embed_ext(&y));
        assert_eq!(Base::embed_ext(&(&x * &y)), Base::embed_ext(&x) * Base::embed_ext(&y));
        assert_eq!(Base::embed_ext(&x), Base::lift(&[x.clone()]), "embedding is the first coefficient");
    }

    #[test]
    fn sqrt_round_trips_in_both_fields() {
        for _ in 0..16 {
            let b = Base::rand();
            let (r, _) = Base::sqrt(&b.square()).unwrap();
            assert_eq!(r.square(), b.square());
            let e = Ext8::rand();
            let (r, _) = Ext8::sqrt(&e.square()).unwrap();
            assert_eq!(r.square(), e.square());
        }
    }
}
