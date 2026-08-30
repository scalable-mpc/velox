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
