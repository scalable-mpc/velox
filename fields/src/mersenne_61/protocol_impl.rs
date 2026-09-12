//! [`ProtocolField`] for the degree-4 extension of Mersenne-61 — the field the
//! node currently runs the protocol over.
//!
//! Every body here was previously a free function pinned to `LargeField`
//! (`poly::rand_field_element`, the limb packing inside `poly::pseudorandom_lf`,
//! `mpc::input::convert_string_to_large_field`) or an inherent trait impl
//! (`ByteConversion`, `Sqrt`). Collecting them behind one trait is what lets the
//! rest of the workspace stop naming this field.

use lambdaworks_math::{errors::ByteConversionError, field::element::FieldElement};
use rand::random;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::{byte_conv::ByteConversion, protocol_field::ProtocolField};

use lambdaworks_math::field::traits::IsSubFieldOf;

use super::{
    extensions::{Fp2E, Mersenne61Degree4ExtensionField},
    field::Mersenne61Field,
    sqrt::Sqrt,
};

type FpE = FieldElement<Mersenne61Field>;

/// Build an Fp4 element from four independent u64 limbs. Each limb is folded
/// into the base field by `from_u64`; values ≥ p wrap, which is fine for
/// uniform sampling (the bias is 2^-61 per limb) but is exactly the fold that
/// [`ProtocolField::encode_ascii`] has to avoid.
fn fp4_from_limbs(limbs: [u64; 4]) -> FieldElement<Mersenne61Degree4ExtensionField> {
    let lo = Fp2E::new([FpE::from(limbs[0]), FpE::from(limbs[1])]);
    let hi = Fp2E::new([FpE::from(limbs[2]), FpE::from(limbs[3])]);
    FieldElement::new([lo, hi])
}

impl ProtocolField for Mersenne61Degree4ExtensionField {
    /// 4 × 8-byte limbs. Same width as the BN254 element this field replaced,
    /// so wire and serialization sizes were unchanged by that swap.
    const SER_BYTES: usize = 32;

    /// 4 limbs × 7 payload bytes — the high byte of each limb is reserved to
    /// keep the limb below 2^56 and the Mersenne reduction a no-op.
    const MAX_INPUT_PAYLOAD: usize = 28;

    /// ~244-bit, so challenges are already sound here and need no lift.
    type Ext = Self;
    const CONV_RATIO: usize = 1;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        chunk.first().cloned().unwrap_or_else(FieldElement::zero)
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        elem.clone()
    }

    fn rand() -> FieldElement<Self> {
        fp4_from_limbs(random::<[u64; 4]>())
    }

    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        fp4_from_limbs([
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ])
    }

    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)> {
        Sqrt::sqrt(elem)
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

    /// Pack a text line into one Fp4 element.
    ///
    /// The element is four 8-byte big-endian Mersenne-61 limbs. Reading 8 bytes
    /// of arbitrary text into a u64 and folding modulo `2^61 - 1` is lossy
    /// whenever the high byte has bit 5 or 6 set — which it always does for
    /// ASCII letters `0x60..0x7F` — so the round-trip text → field → text would
    /// mangle the input. We sidestep the fold by leaving the high byte of each
    /// limb at `0x00` and packing 7 payload bytes into the low 7 bytes. Every
    /// limb then stays below `2^56 < 2^61` and the reduction is a no-op.
    ///
    /// Inputs are right-aligned across the limbs: the *last* limb takes the
    /// last 7 input bytes (or fewer, with leading zeros), then the previous
    /// limb, and so on.
    fn encode_ascii(input: &str) -> Option<FieldElement<Self>> {
        let bytes = input.as_bytes();
        if bytes.len() > Self::MAX_INPUT_PAYLOAD {
            return None;
        }
        let mut padded = [0u8; 32];
        let mut remaining = bytes;
        for chunk in (0..4).rev() {
            let take = remaining.len().min(7);
            if take == 0 {
                break;
            }
            let chunk_base = chunk * 8;
            // Byte 0 of the chunk stays 0x00; payload occupies bytes 1..8,
            // right-aligned within those 7 bytes when partially filled.
            let dest_start = chunk_base + 1 + (7 - take);
            padded[dest_start..chunk_base + 8]
                .copy_from_slice(&remaining[remaining.len() - take..]);
            remaining = &remaining[..remaining.len() - take];
        }
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    /// Inverse of [`encode_ascii`](ProtocolField::encode_ascii): concatenate the
    /// four 7-byte limb payloads (skipping each limb's reserved high byte) and
    /// strip the left-side zero padding that encoding inserted.
    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        let bytes = <Self as ProtocolField>::to_bytes_be(elem);
        let mut payload = Vec::with_capacity(Self::MAX_INPUT_PAYLOAD);
        for chunk in 0..4 {
            payload.extend_from_slice(&bytes[chunk * 8 + 1..chunk * 8 + 8]);
        }
        let first_nonzero = payload.iter().position(|&b| b != 0).unwrap_or(payload.len());
        payload[first_nonzero..].iter().map(|&b| b as char).collect()
    }

    /// The CUDA kernel is compiled for this field's exact 32-byte layout
    /// (see `gpu_ffi`), so it is reachable only from here.
    #[cfg(feature = "gpu")]
    fn try_gpu_gemm(
        matrix: &[Vec<FieldElement<Self>>],
        vectors: &[Vec<FieldElement<Self>>],
        row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        Some(crate::gpu_ffi::gpu_matrix_matrix_multiply(
            matrix, vectors, row_major,
        ))
    }

    /// Lane-wise Fp4 kernel (`simd::m61_avx2::Fp4x4`); scalar when the CPU
    /// has no AVX2 or `VELOX_SIMD=off`.
    #[cfg(target_arch = "x86_64")]
    fn try_simd_gemm(
        matrix: &[Vec<FieldElement<Self>>],
        vectors: &[Vec<FieldElement<Self>>],
        row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        Some(crate::simd::gemm(matrix, vectors, row_major))
    }
}

/// [`ProtocolField`] for the *base* Mersenne-61 prime field, `p = 2^61 - 1`.
///
/// This is the configuration the soundness lift exists for. Shares are single
/// 61-bit elements — four times cheaper to multiply and a quarter the bytes on
/// the wire compared to Fp4 — but a Fiat-Shamir challenge drawn from a 61-bit
/// field gives a 2^-61 soundness bound, which is not enough. So `Ext` is the
/// degree-4 extension above, and every challenge and random linear combination
/// runs there at ~2^-244 while the shares stay small.
impl ProtocolField for Mersenne61Field {
    /// One 8-byte limb.
    const SER_BYTES: usize = 8;

    /// 7 payload bytes: the high byte stays clear so the limb stays below
    /// `2^56 < 2^61` and the Mersenne reduction is a no-op, exactly as in the
    /// Fp4 packing but for a single limb instead of four.
    const MAX_INPUT_PAYLOAD: usize = 7;

    /// 61 bits is far too narrow for a soundness bound, so challenges lift into
    /// the degree-4 extension.
    type Ext = Mersenne61Degree4ExtensionField;

    /// Four base elements are the four coefficients of one Fp4 element.
    const CONV_RATIO: usize = 4;

    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext> {
        debug_assert!(chunk.len() <= Self::CONV_RATIO, "chunk wider than one Ext element");
        let mut coeffs = [FpE::zero(), FpE::zero(), FpE::zero(), FpE::zero()];
        coeffs[..chunk.len()].clone_from_slice(chunk);
        // Coefficient order matches `to_subfield_vec`, so `lift` is its inverse.
        FieldElement::new([
            Fp2E::new([coeffs[0].clone(), coeffs[1].clone()]),
            Fp2E::new([coeffs[2].clone(), coeffs[3].clone()]),
        ])
    }

    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext> {
        // The subfield relation lambdaworks already knows about; unlike `lift`
        // this is a homomorphism, so evaluation points survive it.
        FieldElement::new(
            <Self as IsSubFieldOf<Mersenne61Degree4ExtensionField>>::embed(elem.value().clone()),
        )
    }

    fn rand() -> FieldElement<Self> {
        FieldElement::from(random::<u64>())
    }

    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self> {
        FieldElement::from(rng.next_u64())
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
        let bytes = input.as_bytes();
        if bytes.len() > Self::MAX_INPUT_PAYLOAD {
            return None;
        }
        let mut padded = [0u8; 8];
        padded[8 - bytes.len()..].copy_from_slice(bytes);
        <Self as ProtocolField>::from_bytes_be(&padded).ok()
    }

    fn decode_ascii(elem: &FieldElement<Self>) -> String {
        let bytes = <Self as ProtocolField>::to_bytes_be(elem);
        let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
        bytes[first_nonzero..].iter().map(|&b| b as char).collect()
    }

    /// Single-limb kernel (`simd::m61_avx2::M61x4`); scalar when the CPU
    /// has no AVX2 or `VELOX_SIMD=off`.
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
    use rand_core::SeedableRng;

    type F = Mersenne61Degree4ExtensionField;

    #[test]
    fn serialization_is_exactly_ser_bytes_wide() {
        let e = F::rand();
        assert_eq!(F::to_bytes_be(&e).len(), F::SER_BYTES);
        assert_eq!(F::to_bytes_le(&e).len(), F::SER_BYTES);
    }

    #[test]
    fn byte_round_trip() {
        let e = F::rand();
        assert_eq!(F::from_bytes_be(&F::to_bytes_be(&e)).unwrap(), e);
        assert_eq!(F::from_bytes_le(&F::to_bytes_le(&e)).unwrap(), e);
    }

    /// The reason `encode_ascii` packs 7 bytes per limb rather than 8: letters
    /// live in `0x60..0x7F`, which sets the bits that make an 8-byte limb
    /// exceed `2^61 - 1` and fold. A lossy encoding would silently mangle every
    /// party's input, so this checks the payload comes back byte-identical.
    #[test]
    fn ascii_round_trips_including_letters() {
        for line in ["", "a", "hello world", "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ", "~~~~~~~"] {
            let e = F::encode_ascii(line).expect("within payload capacity");
            let bytes = F::to_bytes_be(&e);
            let recovered: Vec<u8> = bytes.into_iter().filter(|b| *b != 0).collect();
            assert_eq!(
                recovered,
                line.as_bytes().iter().copied().filter(|b| *b != 0).collect::<Vec<u8>>(),
                "round-trip mangled {:?}",
                line
            );
        }
    }

    #[test]
    fn ascii_rejects_oversized_input() {
        let too_long = "x".repeat(F::MAX_INPUT_PAYLOAD + 1);
        assert!(F::encode_ascii(&too_long).is_none());
        assert!(F::encode_ascii(&"x".repeat(F::MAX_INPUT_PAYLOAD)).is_some());
    }

    /// PRF sampling must be a pure function of the seed — two parties derive
    /// their shared randomness from the same seed independently.
    #[test]
    fn from_rng_is_deterministic_in_the_seed() {
        let draw = || {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            (0..4).map(|_| F::from_rng(&mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    /// Four `u64` draws per element, in limb order — the layout the previous
    /// concrete `pseudorandom_lf` produced. Pinning it keeps this refactor
    /// wire-compatible with shares dealt by the pre-generic build.
    #[test]
    fn from_rng_consumes_four_limbs_in_order() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let got = F::from_rng(&mut rng);

        let mut expect_rng = ChaCha20Rng::from_seed([3u8; 32]);
        let expected = fp4_from_limbs([
            expect_rng.next_u64(),
            expect_rng.next_u64(),
            expect_rng.next_u64(),
            expect_rng.next_u64(),
        ]);
        assert_eq!(got, expected);
    }

    /// The lift's whole purpose: `Ext` must be wide enough that a challenge
    /// drawn from it gives a real soundness bound, while `Self` need not be.
    #[test]
    fn base_field_lifts_into_the_degree_four_extension() {
        type S = Mersenne61Field;
        assert_eq!(<S as ProtocolField>::CONV_RATIO, 4);
        assert_eq!(<S as ProtocolField>::SER_BYTES, 8);
        assert_eq!(<<S as ProtocolField>::Ext as ProtocolField>::SER_BYTES, 32);
    }

    /// `lift` must be injective — the soundness argument for the random linear
    /// combination rests on a modified share changing the packed value.
    #[test]
    fn lift_is_injective() {
        type S = Mersenne61Field;
        let base: Vec<FieldElement<S>> = (1..=4u64).map(FieldElement::from).collect();
        let lifted = S::lift(&base);
        for i in 0..4 {
            let mut perturbed = base.clone();
            perturbed[i] += FieldElement::<S>::one();
            assert_ne!(S::lift(&perturbed), lifted, "coefficient {i} did not move the packed value");
        }
    }

    /// `embed_ext` is a homomorphism and `lift` is not; both are needed, for
    /// different things (evaluation points vs. share packing).
    #[test]
    fn embed_ext_is_a_homomorphism_and_agrees_with_from_u64() {
        type S = Mersenne61Field;
        type L = Mersenne61Degree4ExtensionField;
        let x = FieldElement::<S>::from(7u64);
        let y = FieldElement::<S>::from(11u64);
        assert_eq!(<S as ProtocolField>::embed_ext(&(&x + &y)), <S as ProtocolField>::embed_ext(&x) + <S as ProtocolField>::embed_ext(&y));
        assert_eq!(<S as ProtocolField>::embed_ext(&(&x * &y)), <S as ProtocolField>::embed_ext(&x) * <S as ProtocolField>::embed_ext(&y));
        // Evaluation points are built with `from(u64)` on both sides of the
        // lift, so the two constructions have to land on the same element.
        assert_eq!(<S as ProtocolField>::embed_ext(&x), FieldElement::<L>::from(7u64));
    }

    /// The property the DZK depends on: packing evaluations coefficient-wise
    /// keeps them evaluations of a polynomial of the *same degree* over `Ext`.
    /// Without this, lifting the shares would destroy the degree the proof is
    /// about.
    #[test]
    fn lift_preserves_polynomial_degree() {
        use lambdaworks_math::polynomial::Polynomial;
        type S = Mersenne61Field;
        type L = Mersenne61Degree4ExtensionField;

        let degree = 3;
        // Four independent degree-3 polynomials over the base field.
        let polys: Vec<Polynomial<FieldElement<S>>> = (0..4)
            .map(|_| {
                Polynomial::new(&(0..=degree).map(|_| S::rand()).collect::<Vec<_>>())
            })
            .collect();

        // Evaluate all four at 8 points, then pack each point's four values.
        let points: Vec<u64> = (1..=8).collect();
        let packed: Vec<FieldElement<L>> = points
            .iter()
            .map(|&pt| {
                let at = FieldElement::<S>::from(pt);
                let vals: Vec<FieldElement<S>> =
                    polys.iter().map(|p| p.evaluate(&at)).collect();
                S::lift(&vals)
            })
            .collect();

        // The packed values must interpolate to a degree-3 polynomial over Ext:
        // build it from the first 4 points and check it predicts the other 4.
        let eval_pts: Vec<FieldElement<L>> =
            points.iter().map(|&pt| FieldElement::<L>::from(pt)).collect();
        let interpolated =
            Polynomial::interpolate(&eval_pts[..=degree], &packed[..=degree]).unwrap();
        assert_eq!(interpolated.degree(), degree);
        for i in (degree + 1)..points.len() {
            assert_eq!(
                interpolated.evaluate(&eval_pts[i]),
                packed[i],
                "packed evaluations left the degree-{degree} polynomial at point {i}"
            );
        }
    }

    /// The base field's own trait contract, at its narrower 8-byte width.
    #[test]
    fn base_field_satisfies_the_contract() {
        type S = Mersenne61Field;
        let e = S::rand();
        assert_eq!(S::to_bytes_be(&e).len(), 8);
        assert_eq!(S::from_bytes_be(&S::to_bytes_be(&e)).unwrap(), e);

        // 7 payload bytes, not 28 — one limb instead of four.
        for line in ["", "a", "hello", "ZZZZZZZ"] {
            let packed = S::encode_ascii(line).unwrap_or_else(|| panic!("{line:?} should fit"));
            assert_eq!(S::decode_ascii(&packed), line);
        }
        assert!(S::encode_ascii("12345678").is_none(), "8 bytes must not fit");

        let root = S::rand();
        let (a, b) = S::sqrt(&(&root * &root)).expect("a square has a root");
        assert!(a == root || b == root);
    }

    #[test]
    fn sqrt_round_trips() {
        // Squares are residues by construction, so this always finds a root.
        let root = F::rand();
        let square = &root * &root;
        let (a, b) = F::sqrt(&square).expect("a square has a root");
        assert!(a == root || b == root);
        assert_eq!(&a * &a, square);
    }
}
