// Extensions of the ML-DSA field: F_q[x] / (x^D − 2), for D = 3 and D = 11.
//
// A binomial x^D − w with D prime and D | q − 1 is irreducible exactly when w
// is not a D-th power. q − 1 = 2^13·3·11·31, and 2 is neither a cube nor an
// 11th power mod q, so both binomials are irreducible.
//
// - Degree 3 has q^3 ≈ 2^69 elements: the statistical extension, where the
//   verification checks run (at least 60 bits).
// - Degree 11 has q^11 ≈ 2^253 elements: the cryptographic extension, where
//   the ACSS DZK challenges live (at least 240 bits). It is the largest degree
//   whose elements still fit the 32 bytes of a hash, which the ACSS requires.

use lambdaworks_math::{
    errors::ByteConversionError,
    field::{errors::FieldError, traits::IsField},
    traits::ByteConversion as LambdaByteConversion,
};

use super::field::{pow_mod, MlDsaField, ML_DSA_PRIME_FIELD_ORDER};

const Q: u32 = ML_DSA_PRIME_FIELD_ORDER;

/// The constant term of the modulus: `x^D = W`.
const W: u64 = 2;

/// Bytes of a serialized extension element: `q^D < 2^256` for `D ≤ 11`.
const PACKED_BYTES: usize = 32;

/// An element `Σ c_i·x^i` as its coefficients, `c_i` at index `i`, each
/// fully reduced into `[0, q)`.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ExtCoeffs<const D: usize>(pub [u32; D]);

impl<const D: usize> Default for ExtCoeffs<D> {
    fn default() -> Self {
        Self([0; D])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MlDsaExtensionField<const D: usize>;

pub type MlDsaDegree3ExtensionField = MlDsaExtensionField<3>;
pub type MlDsaDegree11ExtensionField = MlDsaExtensionField<11>;

impl<const D: usize> MlDsaExtensionField<D> {
    /// `ω^k` for `k < D`, where `ω = W^((q−1)/D)` is a primitive `D`-th root
    /// of unity. The Frobenius `a ↦ a^q` fixes `F_q` and maps `x` to
    /// `x·(x^D)^((q−1)/D) = ω·x`, so its `k`-th power multiplies coefficient
    /// `i` by `ω^(i·k)`.
    const FROBENIUS_POWERS: [u32; D] = {
        assert!((Q - 1) % D as u32 == 0, "D must divide q − 1");
        let omega = pow_mod(W as u32, ((Q - 1) / D as u32) as u64);
        let mut powers = [1u32; D];
        let mut k = 1;
        while k < D {
            powers[k] = (powers[k - 1] as u64 * omega as u64 % Q as u64) as u32;
            k += 1;
        }
        powers
    };
}

impl<const D: usize> IsField for MlDsaExtensionField<D> {
    type BaseType = ExtCoeffs<D>;

    fn add(a: &ExtCoeffs<D>, b: &ExtCoeffs<D>) -> ExtCoeffs<D> {
        ExtCoeffs(std::array::from_fn(|i| MlDsaField::add(&a.0[i], &b.0[i])))
    }

    /// Schoolbook, with `x^(D+k) = W·x^k` folded in as the products are
    /// summed. Each product is below 2^46 and each coefficient sums `D` of
    /// them, some times `W = 2`, so it stays below `11·2^47 < 2^51` and is
    /// reduced once at the end.
    fn mul(a: &ExtCoeffs<D>, b: &ExtCoeffs<D>) -> ExtCoeffs<D> {
        let mut acc = [0u64; D];
        for i in 0..D {
            let ai = a.0[i] as u64;
            for j in 0..D - i {
                acc[i + j] += ai * b.0[j] as u64;
            }
            for j in D - i..D {
                acc[i + j - D] += W * ai * b.0[j] as u64;
            }
        }
        ExtCoeffs(acc.map(|c| (c % Q as u64) as u32))
    }

    fn sub(a: &ExtCoeffs<D>, b: &ExtCoeffs<D>) -> ExtCoeffs<D> {
        ExtCoeffs(std::array::from_fn(|i| MlDsaField::sub(&a.0[i], &b.0[i])))
    }

    fn neg(a: &ExtCoeffs<D>) -> ExtCoeffs<D> {
        ExtCoeffs(a.0.map(|c| MlDsaField::neg(&c)))
    }

    /// `a^(−1) = P / N(a)`, where `P = Π_{k=1}^{D−1} σ^k(a)` is the product of
    /// the other conjugates and `N(a) = a·P ∈ F_q` is the norm: `D − 1`
    /// extension products and one inversion in `F_q`, against about
    /// `1.5·log2(q^D)` products for Fermat.
    fn inv(a: &ExtCoeffs<D>) -> Result<ExtCoeffs<D>, FieldError> {
        if a.0.iter().all(|&c| c == 0) {
            return Err(FieldError::InvZeroError);
        }
        let conjugate =
            |k: usize| ExtCoeffs(std::array::from_fn(|i| MlDsaField::mul(&a.0[i], &Self::FROBENIUS_POWERS[i * k % D])));
        let others = (2..D).fold(conjugate(1), |acc, k| Self::mul(&acc, &conjugate(k)));
        let norm = Self::mul(a, &others).0[0];
        let norm_inv = MlDsaField::inv(&norm)?;
        Ok(ExtCoeffs(others.0.map(|c| MlDsaField::mul(&c, &norm_inv))))
    }

    fn div(a: &ExtCoeffs<D>, b: &ExtCoeffs<D>) -> Result<ExtCoeffs<D>, FieldError> {
        let b_inv = Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(Self::mul(a, &b_inv))
    }

    fn eq(a: &ExtCoeffs<D>, b: &ExtCoeffs<D>) -> bool {
        a.0 == b.0
    }

    fn zero() -> ExtCoeffs<D> {
        ExtCoeffs::default()
    }

    fn one() -> ExtCoeffs<D> {
        Self::from_u64(1)
    }

    fn from_u64(x: u64) -> ExtCoeffs<D> {
        let mut coeffs = [0; D];
        coeffs[0] = MlDsaField::from_u64(x);
        ExtCoeffs(coeffs)
    }

    fn from_base_type(x: ExtCoeffs<D>) -> ExtCoeffs<D> {
        ExtCoeffs(x.0.map(MlDsaField::from_base_type))
    }
}

/// The element as the integer `Σ c_i·q^i`, in 32 big-endian bytes.
fn to_packed_be<const D: usize>(a: &ExtCoeffs<D>) -> Vec<u8> {
    const { assert!(D <= 11, "q^D must fit 32 bytes") };
    // Little-endian 64-bit limbs; Horner from the top coefficient.
    let mut limbs = [0u64; PACKED_BYTES / 8];
    for &c in a.0.iter().rev() {
        let mut carry = c as u128;
        for limb in limbs.iter_mut() {
            let v = *limb as u128 * Q as u128 + carry;
            *limb = v as u64;
            carry = v >> 64;
        }
    }
    limbs.iter().rev().flat_map(|limb| limb.to_be_bytes()).collect()
}

/// Inverse of [`to_packed_be`]. Any 32 bytes decode: the integer is reduced
/// mod `q^D`. For `D = 11`, `2^256 ≈ 8.09·q^11`, so each element has 8 or 9
/// preimages and a hash gives a challenge at most `9/8.09` times as likely
/// as a uniform one: the DZK loses less than 0.2 bits of its 253.
fn from_packed_be<const D: usize>(bytes: &[u8]) -> Result<ExtCoeffs<D>, ByteConversionError> {
    let bytes: [u8; PACKED_BYTES] = bytes.try_into().map_err(|_| ByteConversionError::FromBEBytesError)?;
    let mut limbs: [u64; PACKED_BYTES / 8] =
        std::array::from_fn(|k| u64::from_be_bytes(bytes[PACKED_BYTES - 8 * (k + 1)..PACKED_BYTES - 8 * k].try_into().unwrap()));
    // Each division by q leaves the next coefficient as the remainder.
    Ok(ExtCoeffs(std::array::from_fn(|_| {
        let mut rem = 0u128;
        for limb in limbs.iter_mut().rev() {
            let cur = (rem << 64) | *limb as u128;
            *limb = (cur / Q as u128) as u64;
            rem = cur % Q as u128;
        }
        rem as u32
    })))
}

/// lambdaworks's byte conversion, which `IsField` requires of every
/// `BaseType`; the protocol's serialization (`ProtocolField::to_bytes_*`)
/// writes the same 32 bytes.
impl<const D: usize> LambdaByteConversion for ExtCoeffs<D> {
    fn to_bytes_be(&self) -> Vec<u8> {
        to_packed_be(self)
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        let mut bytes = to_packed_be(self);
        bytes.reverse();
        bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        from_packed_be(bytes)
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut be = bytes.to_vec();
        be.reverse();
        from_packed_be(&be).map_err(|_| ByteConversionError::FromLEBytesError)
    }
}
