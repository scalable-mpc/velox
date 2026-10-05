// Mersenne-127 prime field, p = 2^127 − 1, on the lambdaworks IsField trait.
//
// The arithmetic is the Mersenne-61 field's one word wider: an element is one
// `u128`, a product is 254 bits wide and folds back with 2^127 ≡ 1 (mod p).
// Unlike Mersenne-61, every operation leaves its result fully reduced in
// `[0, p)`, so equality and the representative are the word itself.

use lambdaworks_math::{
    errors::{ByteConversionError, CreationError},
    field::{
        errors::FieldError,
        traits::{IsField, IsPrimeField},
    },
    traits::ByteConversion as LambdaByteConversion,
};

pub const MERSENNE_127_PRIME_FIELD_ORDER: u128 = (1u128 << 127) - 1;

/// An element of the field: the integer it represents, always in `[0, p)`.
///
/// A newtype rather than a bare `u128` because lambdaworks requires its own
/// `ByteConversion` on every `BaseType` and implements it for `u64` but not
/// for `u128`; the orphan rule lets this crate implement it only on a local
/// type.
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct M127(pub u128);

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Mersenne127Field;

impl Mersenne127Field {
    /// Reduces any `u128` into `[0, p)`: fold the top bit back in
    /// (`2^127 ≡ 1`), which leaves at most `p + 1`, then subtract `p` once.
    #[inline(always)]
    fn reduce(x: u128) -> M127 {
        let folded = (x & MERSENNE_127_PRIME_FIELD_ORDER) + (x >> 127);
        if folded >= MERSENNE_127_PRIME_FIELD_ORDER {
            M127(folded - MERSENNE_127_PRIME_FIELD_ORDER)
        } else {
            M127(folded)
        }
    }

    /// Reduces the 254-bit product `a1b1·2^128 + mid·2^64 + a0b0` of two
    /// elements split into 64-bit halves (`a = a1·2^64 + a0`): split it at
    /// bit 127 as `H·2^127 + L`, and `H + L` is the product mod p.
    #[inline(always)]
    fn fold_product(a0b0: u128, mid: u128, a1b1: u128) -> M127 {
        let (lo, carry) = a0b0.overflowing_add(mid << 64);
        let hi = a1b1 + (mid >> 64) + carry as u128;
        // The product is below 2^254, so `hi < 2^126` and the shift is safe.
        let high_part = (hi << 1) | (lo >> 127);
        Self::reduce((lo & MERSENNE_127_PRIME_FIELD_ORDER) + high_part)
    }

    /// `x^(2^k)`: `k` squarings.
    #[inline(always)]
    fn square_times(x: &M127, k: u32) -> M127 {
        (0..k).fold(*x, |acc, _| Self::square(&acc))
    }
}

impl IsField for Mersenne127Field {
    type BaseType = M127;

    /// Both operands are below `2^127`, so the sum fits a `u128`.
    #[inline(always)]
    fn add(a: &M127, b: &M127) -> M127 {
        Self::reduce(a.0 + b.0)
    }

    /// The 254-bit product from four 64×64-bit products, then the fold.
    #[inline(always)]
    fn mul(a: &M127, b: &M127) -> M127 {
        let (a0, a1) = (a.0 as u64 as u128, a.0 >> 64);
        let (b0, b1) = (b.0 as u64 as u128, b.0 >> 64);
        // a1, b1 < 2^63, so each cross product is below 2^127 and their sum
        // fits a `u128`.
        Self::fold_product(a0 * b0, a0 * b1 + a1 * b0, a1 * b1)
    }

    /// Three 64×64-bit products instead of `mul`'s four: the cross product
    /// `a0·a1` is computed once and doubled (`< 2^128`, since `a1 < 2^63`).
    #[inline(always)]
    fn square(a: &M127) -> M127 {
        let (a0, a1) = (a.0 as u64 as u128, a.0 >> 64);
        Self::fold_product(a0 * a0, (a0 * a1) << 1, a1 * a1)
    }

    #[inline(always)]
    fn sub(a: &M127, b: &M127) -> M127 {
        Self::reduce(a.0 + (MERSENNE_127_PRIME_FIELD_ORDER - b.0))
    }

    #[inline(always)]
    fn neg(a: &M127) -> M127 {
        if a.0 == 0 {
            M127(0)
        } else {
            M127(MERSENNE_127_PRIME_FIELD_ORDER - a.0)
        }
    }

    /// Fermat, `a^(p−2)`, by an addition chain: 126 squarings and 10
    /// multiplications, where square-and-multiply on `p − 2 = 2^127 − 3`
    /// (125 one bits) takes 126 and 125.
    ///
    /// With `e_k = a^(2^k − 1)`, `e_{j+k} = e_j^(2^k) · e_k`. The chain builds
    /// `e_125` through `e_2, e_3, e_5, e_10, e_20, e_40, e_80, e_120`, and
    /// `p − 2 = (2^125 − 1)·4 + 1` gives `a^(p−2) = e_125^4 · a`.
    fn inv(a: &M127) -> Result<M127, FieldError> {
        if a.0 == 0 {
            return Err(FieldError::InvZeroError);
        }
        let e2 = Self::mul(&Self::square(a), a);
        let e3 = Self::mul(&Self::square(&e2), a);
        let e5 = Self::mul(&Self::square_times(&e3, 2), &e2);
        let e10 = Self::mul(&Self::square_times(&e5, 5), &e5);
        let e20 = Self::mul(&Self::square_times(&e10, 10), &e10);
        let e40 = Self::mul(&Self::square_times(&e20, 20), &e20);
        let e80 = Self::mul(&Self::square_times(&e40, 40), &e40);
        let e120 = Self::mul(&Self::square_times(&e80, 40), &e40);
        let e125 = Self::mul(&Self::square_times(&e120, 5), &e5);
        Ok(Self::mul(&Self::square_times(&e125, 2), a))
    }

    fn div(a: &M127, b: &M127) -> Result<M127, FieldError> {
        let b_inv = Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(Self::mul(a, &b_inv))
    }

    #[inline(always)]
    fn eq(a: &M127, b: &M127) -> bool {
        a.0 == b.0
    }

    #[inline(always)]
    fn zero() -> M127 {
        M127(0)
    }

    #[inline(always)]
    fn one() -> M127 {
        M127(1)
    }

    /// Every `u64` is already below `p`.
    #[inline(always)]
    fn from_u64(x: u64) -> M127 {
        M127(x as u128)
    }

    #[inline(always)]
    fn from_base_type(x: M127) -> M127 {
        Self::reduce(x.0)
    }
}

impl IsPrimeField for Mersenne127Field {
    type RepresentativeType = u128;

    fn representative(x: &M127) -> u128 {
        x.0
    }

    fn field_bit_size() -> usize {
        127
    }

    fn from_hex(hex_string: &str) -> Result<M127, CreationError> {
        let digits = hex_string.strip_prefix("0x").unwrap_or(hex_string);
        u128::from_str_radix(digits, 16)
            .map(Self::reduce)
            .map_err(|_| CreationError::InvalidHexString)
    }

    fn to_hex(x: &M127) -> String {
        format!("{:X}", x.0)
    }
}

/// lambdaworks's byte conversion of the raw word, which `IsField` requires of
/// every `BaseType`; the protocol's serialization (`ProtocolField::to_bytes_*`)
/// writes the same 16 bytes.
impl LambdaByteConversion for M127 {
    fn to_bytes_be(&self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; 16] = bytes.try_into().map_err(|_| ByteConversionError::FromBEBytesError)?;
        Ok(Mersenne127Field::reduce(u128::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let bytes: [u8; 16] = bytes.try_into().map_err(|_| ByteConversionError::FromLEBytesError)?;
        Ok(Mersenne127Field::reduce(u128::from_le_bytes(bytes)))
    }
}
