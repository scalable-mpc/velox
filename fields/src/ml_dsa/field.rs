// The ML-DSA prime field, q = 2^23 − 2^13 + 1 = 8380417 (FIPS 204), on the
// lambdaworks IsField trait.
//
// q is the modulus of the ML-DSA ring, so a sharing over this field is a
// sharing of a ring coefficient: A·y, the NTT and the bound checks run on the
// shares with no change of modulus. An element is one `u32`, always fully
// reduced into `[0, q)`. A product is below 2^46 and reduces with one `%` by
// a constant, which the compiler turns into a multiply and a shift.

use lambdaworks_math::{
    errors::CreationError,
    field::{
        errors::FieldError,
        traits::{IsField, IsPrimeField},
    },
};

pub const ML_DSA_PRIME_FIELD_ORDER: u32 = 8_380_417;

const Q: u32 = ML_DSA_PRIME_FIELD_ORDER;

/// `x^e mod q`, for the constants of the field and its extensions.
pub(crate) const fn pow_mod(x: u32, mut e: u64) -> u32 {
    let (mut base, mut acc) = (x as u64 % Q as u64, 1u64);
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * base % Q as u64;
        }
        base = base * base % Q as u64;
        e >>= 1;
    }
    acc as u32
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct MlDsaField;

impl IsField for MlDsaField {
    type BaseType = u32;

    /// Both operands are below `q < 2^23`, so the sum fits a `u32`.
    #[inline(always)]
    fn add(a: &u32, b: &u32) -> u32 {
        let sum = a + b;
        if sum >= Q {
            sum - Q
        } else {
            sum
        }
    }

    #[inline(always)]
    fn mul(a: &u32, b: &u32) -> u32 {
        ((*a as u64 * *b as u64) % Q as u64) as u32
    }

    #[inline(always)]
    fn sub(a: &u32, b: &u32) -> u32 {
        if a >= b {
            a - b
        } else {
            a + Q - b
        }
    }

    #[inline(always)]
    fn neg(a: &u32) -> u32 {
        if *a == 0 {
            0
        } else {
            Q - a
        }
    }

    /// Fermat, `a^(q−2)`.
    fn inv(a: &u32) -> Result<u32, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(a, Q - 2))
    }

    fn div(a: &u32, b: &u32) -> Result<u32, FieldError> {
        let b_inv = Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(Self::mul(a, &b_inv))
    }

    #[inline(always)]
    fn eq(a: &u32, b: &u32) -> bool {
        a == b
    }

    #[inline(always)]
    fn zero() -> u32 {
        0
    }

    #[inline(always)]
    fn one() -> u32 {
        1
    }

    #[inline(always)]
    fn from_u64(x: u64) -> u32 {
        (x % Q as u64) as u32
    }

    /// Reduces a raw word, so `FieldElement::new` accepts any `u32`.
    #[inline(always)]
    fn from_base_type(x: u32) -> u32 {
        x % Q
    }
}

impl IsPrimeField for MlDsaField {
    type RepresentativeType = u32;

    fn representative(x: &u32) -> u32 {
        *x
    }

    fn field_bit_size() -> usize {
        23
    }

    fn from_hex(hex_string: &str) -> Result<u32, CreationError> {
        let digits = hex_string.strip_prefix("0x").unwrap_or(hex_string);
        u32::from_str_radix(digits, 16).map(|x| x % Q).map_err(|_| CreationError::InvalidHexString)
    }

    fn to_hex(x: &u32) -> String {
        format!("{:X}", x)
    }
}
