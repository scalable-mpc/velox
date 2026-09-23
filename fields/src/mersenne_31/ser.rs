//! Byte layout for the Mersenne-31 base field and its Fp8 tower, mirroring
//! `mersenne_61::ser`: one 4-byte limb per base element, and eight of them,
//! in `to_subfield_vec` order, for an Fp8 element.

use lambdaworks_math::{errors::ByteConversionError, field::element::FieldElement};

use crate::byte_conv::ByteConversion;

use super::extension_fp8::{Degree8ExtensionField, Fp, Fp2, Fp4, Fp8, Mersenne31Field};

const LIMB_BYTES: usize = 4;
const FP8_LIMBS: usize = 8;

impl ByteConversion for FieldElement<Mersenne31Field> {
    fn to_bytes_be(&self) -> Vec<u8> {
        self.representative().to_be_bytes().to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.representative().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; LIMB_BYTES] = bytes
            .try_into()
            .map_err(|_| ByteConversionError::FromBEBytesError)?;
        Ok(Self::from(u32::from_be_bytes(bytes) as u64))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; LIMB_BYTES] = bytes
            .try_into()
            .map_err(|_| ByteConversionError::FromLEBytesError)?;
        Ok(Self::from(u32::from_le_bytes(bytes) as u64))
    }
}

/// The eight base coefficients of an Fp8 element, in `to_subfield_vec` order:
/// `c0.c0.[0,1], c0.c1.[0,1], c1.c0.[0,1], c1.c1.[0,1]`.
pub(super) fn fp8_limbs(elem: &Fp8) -> [Fp; FP8_LIMBS] {
    let [c0, c1] = elem.value();
    let (c00, c01) = (&c0.value()[0], &c0.value()[1]);
    let (c10, c11) = (&c1.value()[0], &c1.value()[1]);
    [
        c00.value()[0].clone(), c00.value()[1].clone(),
        c01.value()[0].clone(), c01.value()[1].clone(),
        c10.value()[0].clone(), c10.value()[1].clone(),
        c11.value()[0].clone(), c11.value()[1].clone(),
    ]
}

/// Inverse of [`fp8_limbs`].
pub(super) fn fp8_from_limbs(limbs: [Fp; FP8_LIMBS]) -> Fp8 {
    let [l0, l1, l2, l3, l4, l5, l6, l7] = limbs;
    Fp8::new([
        Fp4::new([Fp2::new([l0, l1]), Fp2::new([l2, l3])]),
        Fp4::new([Fp2::new([l4, l5]), Fp2::new([l6, l7])]),
    ])
}

impl ByteConversion for FieldElement<Degree8ExtensionField> {
    // Flat 32-byte layout: 8 × 4-byte limbs.
    fn to_bytes_be(&self) -> Vec<u8> {
        fp8_limbs(self)
            .iter()
            .flat_map(|limb| limb.representative().to_be_bytes())
            .collect()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        fp8_limbs(self)
            .iter()
            .flat_map(|limb| limb.representative().to_le_bytes())
            .collect()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        if bytes.len() < FP8_LIMBS * LIMB_BYTES {
            return Err(ByteConversionError::FromBEBytesError);
        }
        let mut limbs = std::array::from_fn(|_| Fp::zero());
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(LIMB_BYTES)) {
            *limb = <Fp as ByteConversion>::from_bytes_be(chunk)?;
        }
        Ok(fp8_from_limbs(limbs))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        if bytes.len() < FP8_LIMBS * LIMB_BYTES {
            return Err(ByteConversionError::FromLEBytesError);
        }
        let mut limbs = std::array::from_fn(|_| Fp::zero());
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(LIMB_BYTES)) {
            *limb = <Fp as ByteConversion>::from_bytes_le(chunk)?;
        }
        Ok(fp8_from_limbs(limbs))
    }
}
