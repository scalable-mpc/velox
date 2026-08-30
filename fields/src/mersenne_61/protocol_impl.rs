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
