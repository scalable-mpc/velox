// Degree-2 extension of Mersenne-127: Fp[i] / (i^2 + 1).
//
// p ≡ 3 (mod 4), so −1 is a quadratic non-residue and x^2 + 1 is irreducible.
// The extension has p^2 ≈ 2^254 elements, wide enough for the ACSS DZK
// challenges (at least 240 bits). The arithmetic is the Mersenne-61 Fp2's.

use lambdaworks_math::field::{
    element::FieldElement,
    errors::FieldError,
    traits::{IsField, IsSubFieldOf},
};

use super::field::Mersenne127Field;

type FpE = FieldElement<Mersenne127Field>;
pub type Fp2E = FieldElement<Mersenne127Degree2ExtensionField>;

/// Elements are `[a, b] = a + b·i`. `Copy`, like the base field, so
/// elements pass by value as the base field's do.
#[derive(Clone, Copy, Debug)]
pub struct Mersenne127Degree2ExtensionField;

impl IsField for Mersenne127Degree2ExtensionField {
    type BaseType = [FpE; 2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// `(a0 + a1·i)(b0 + b1·i) = (a0·b0 − a1·b1) + (a0·b1 + a1·b0)·i`, with
    /// the imaginary part by Karatsuba.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 - a1b1, z - a0b0 - a1b1]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        [(a0 + a1) * (a0 - a1), (a0 * a1).double()]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// `(a0 + a1·i)^(−1) = (a0 − a1·i) / (a0^2 + a1^2)`.
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].square() + a[1].square()).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Mersenne127Degree2ExtensionField> for Mersenne127Field {
    fn add(
        a: &Self::BaseType,
        b: &<Mersenne127Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne127Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) + b[0], b[1]]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Mersenne127Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne127Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) - b[0], -b[1]]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Mersenne127Degree2ExtensionField as IsField>::BaseType,
    ) -> <Mersenne127Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) * b[0], FpE::from(a) * b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Mersenne127Degree2ExtensionField as IsField>::BaseType,
    ) -> Result<<Mersenne127Degree2ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Mersenne127Degree2ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Mersenne127Degree2ExtensionField>>::mul(a, &b_inv))
    }

    fn embed(a: Self::BaseType) -> <Mersenne127Degree2ExtensionField as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    fn to_subfield_vec(
        b: <Mersenne127Degree2ExtensionField as IsField>::BaseType,
    ) -> Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}
