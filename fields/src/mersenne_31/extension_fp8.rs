// Ported from scalable_mpc/fields/src/mersenne_31/extension_fp8.rs (arithmetic
// and arithmetic tests; the serialization impls/tests are left for the
// ProtocolField integration since this crate's ByteConversion surface differs).
// Degree-8 extension of Mersenne31 field
//
// Uses lambdaworks' Mersenne31 base field and degree-4 extension,
// then adds a degree-2 extension on top using k^2 = 1 - 2i + ji
//
// Extension tower:
// - Fp = Mersenne31 (from lambdaworks)
// - Fp2 = Fp[i] / (i^2 + 1) (from lambdaworks)
// - Fp4 = Fp2[j] / (j^2 - (2+i)) (from lambdaworks)
// - Fp8 = Fp4[k] / (k^2 - (1 - 2i + ji)) (this file)

pub use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::errors::FieldError;
pub use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
pub use lambdaworks_math::field::fields::mersenne31::extensions::{Degree2ExtensionField, Degree4ExtensionField};
pub use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::traits::IsSubFieldOf;
use rand::random;

/// Type aliases for convenience
pub type Fp = FieldElement<Mersenne31Field>;
pub type Fp2 = FieldElement<Degree2ExtensionField>;
pub type Fp4 = FieldElement<Degree4ExtensionField>;
pub type Fp8 = FieldElement<Degree8ExtensionField>;

// ============================================================================
// Degree-8 Extension: Fp8 = Fp4[k] / (k^2 - (1 - 2i + ji))
// Elements are represented as [c0, c1] = c0 + c1*k where c0, c1 are in Fp4
// The non-residue (1 - 2i + ji) is a full Fp4 element
// ============================================================================

/// A field element in the degree-8 extension of Mersenne31
/// Represented as c0 + c1*k where k^2 = 1 - 2i + ji
#[derive(Clone, Debug)]
pub struct Degree8ExtensionField;

/// Multiply an Fp4 element by the non-residue (1 - 2i + ji) used in Fp8 extension
/// The non-residue is a full Fp4 element (not just an embedded Fp2)
impl Degree8ExtensionField {
    fn mul_fp4_by_nonresidue(a: &Fp4) -> Fp4 {
        // (1 - 2i + ji) as Fp4 element: [[1, -2], [0, 1]]
        let nonresidue = Fp4::new([
            Fp2::new([Fp::from(1u64), -Fp::from(2u64)]),
            Fp2::new([Fp::zero(), Fp::one()]),
        ]);
        a * &nonresidue
    }

    pub const fn const_from_coefficients(coeffs: [u32; 8]) -> Fp8 {
        Fp8::const_from_raw(
            [
                Fp4::const_from_raw(
                    [
                        Fp2::const_from_raw(
                            [
                                Fp::const_from_raw(coeffs[0]),
                                Fp::const_from_raw(coeffs[1])
                            ]
                        ), 
                        Fp2::const_from_raw(
                            [
                                Fp::const_from_raw(coeffs[2]),
                                Fp::const_from_raw(coeffs[3])
                            ]
                        )
                    ]
                ),
                Fp4::const_from_raw(
                    [
                        Fp2::const_from_raw(
                            [
                                Fp::const_from_raw(coeffs[4]),
                                Fp::const_from_raw(coeffs[5])
                            ]
                        ), 
                        Fp2::const_from_raw(
                            [
                                Fp::const_from_raw(coeffs[6]),
                                Fp::const_from_raw(coeffs[7])
                            ]
                        )
                    ]
                ),
            ]
        )
    }

    pub fn const_from_fe(elems: &[Fp]) -> Fp8 {
        Fp8::const_from_raw(
            [
                Fp4::const_from_raw(
                    [
                        Fp2::const_from_raw(
                            [
                                elems[0],
                                elems[1]
                            ]
                        ), 
                        Fp2::const_from_raw(
                            [
                                elems[2],
                                elems[3]
                            ]
                        )
                    ]
                ),
                Fp4::const_from_raw(
                    [
                        Fp2::const_from_raw(
                            [
                                elems[4],
                                elems[5]
                            ]
                        ), 
                        Fp2::const_from_raw(
                            [
                                elems[6],
                                elems[7]
                            ]
                        )
                    ]
                ),
            ]
        )
    }

    pub fn rand_fe() -> Fp8{
        Fp8::new([
                    Fp4::new([
                        Fp2::new([Fp::from(random::<u32>() as u64), Fp::from(random::<u32>() as u64)]),
                        Fp2::new([Fp::from(random::<u32>() as u64), Fp::from(random::<u32>() as u64)]),
                    ]),
                    Fp4::new([
                        Fp2::new([Fp::from(random::<u32>() as u64), Fp::from(random::<u32>() as u64)]),
                        Fp2::new([Fp::from(random::<u32>() as u64), Fp::from(random::<u32>() as u64)]),
                    ]),
                ])
    }
}


impl IsField for Degree8ExtensionField{
    type BaseType = [Fp4;2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1]* &b[1];
        [
            &a0b0 + &Degree8ExtensionField::mul_fp4_by_nonresidue(&a1b1),
            (&a[0] + &a[1]) * (&b[0]+&b[1]) - &a0b0 - &a1b1,
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_square = &a[0].square();
        let a1_square = &a[1].square();
        [
            a0_square + &Degree8ExtensionField::mul_fp4_by_nonresidue(&a1_square),
            (&a[0] +&a[1]).square() - a0_square - a1_square
        ]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, lambdaworks_math::field::errors::FieldError> {
        let norm = &a[0].square() - Degree8ExtensionField::mul_fp4_by_nonresidue(&a[1].square());
        if norm == Fp4::zero() {
            return Err(FieldError::InvZeroError);
        }
        let inv_norm =
            (a[0].square() - Degree8ExtensionField::mul_fp4_by_nonresidue(&a[1].square())).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * &inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, lambdaworks_math::field::errors::FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn one() -> Self::BaseType {
        [Fp4::one(), Fp4::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [Fp4::from(x), Fp4::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}


impl IsSubFieldOf<Degree8ExtensionField> for Mersenne31Field {
    fn add(
        a: &Self::BaseType,
        b: &<Degree8ExtensionField as IsField>::BaseType,
    ) -> <Degree8ExtensionField as IsField>::BaseType {
        [Fp::from(a) + &b[0], b[1].clone()]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree8ExtensionField as IsField>::BaseType,
    ) -> <Degree8ExtensionField as IsField>::BaseType {
        [Fp::from(a) - &b[0], -b[1].clone()]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Degree8ExtensionField as IsField>::BaseType,
    ) -> <Degree8ExtensionField as IsField>::BaseType {
        [Fp::from(a) * &b[0], Fp::from(a) * &b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree8ExtensionField as IsField>::BaseType,
    ) -> Result<<Degree8ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree8ExtensionField::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsSubFieldOf<Degree8ExtensionField>>::mul(
            a, &b_inv,
        ))
    }

    fn embed(a: Self::BaseType) -> <Degree8ExtensionField as IsField>::BaseType {
        [
            Fp4::from_raw(<Self as IsSubFieldOf<Degree4ExtensionField>>::embed(a)), 
            Fp4::zero()
        ]
    }

    
    fn to_subfield_vec(b: <Degree8ExtensionField as IsField>::BaseType) -> Vec<Self::BaseType> {
        let mut result = Vec::new();
        for fp4e in b {
            let fp2e_0 = &fp4e.value()[0];
            result.push(fp2e_0.value()[0].to_raw());
            result.push(fp2e_0.value()[1].to_raw());
            let fp2e_1 = &fp4e.value()[1];
            result.push(fp2e_1.value()[0].to_raw());
            result.push(fp2e_1.value()[1].to_raw());
        }
        result
    }
}
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Degree8ExtensionField::const_from_coefficients([10, 20, 30, 40, 50, 60, 70, 80]);
        let c = a + b;
        let expected = Degree8ExtensionField::const_from_coefficients([11, 22, 33, 44, 55, 66, 77, 88]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub() {
        let a = Degree8ExtensionField::const_from_coefficients([100, 200, 300, 400, 500, 600, 700, 800]);
        let b = Degree8ExtensionField::const_from_coefficients([10, 20, 30, 40, 50, 60, 70, 80]);
        let c = a - b;
        let expected = Degree8ExtensionField::const_from_coefficients([90, 180, 270, 360, 450, 540, 630, 720]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_by_zero() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let result = a * Fp8::zero();
        assert!(result== Fp8::zero());
    }

    #[test]
    fn test_mul_by_one() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let result = a.clone() * Fp8::one();
        assert_eq!(result, a);
    }

    #[test]
    fn test_square_equals_mul() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let square = a.square();
        let mul = &a * &a;
        assert_eq!(square, mul);
    }

    #[test]
    fn test_inv() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp8::one());
    }

    #[test]
    fn test_inv_random() {
        let a = Degree8ExtensionField::const_from_coefficients([
            123456789, 987654321, 111222333, 444555666,
            777888999, 112233445, 556677889, 998877665,
        ]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp8::one());
    }

    #[test]
    fn test_neg() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let neg_a = -a.clone();
        let sum = a + neg_a;
        assert!(sum == Fp8::zero());
    }

    #[test]
    fn test_pow() {
        let a = Degree8ExtensionField::const_from_coefficients([2, 1, 0, 0, 1, 0, 0, 0]);
        let a_cubed = a.pow(3u32);
        let a_cubed_manual = &(&a * &a) * &a;
        assert_eq!(a_cubed, a_cubed_manual);
    }

    #[test]
    fn test_pow_higher() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 1, 1, 1, 1, 1, 1, 1]);
        let a_pow_5 = a.pow(5u32);
        let a_pow_5_manual = &(&(&(&a * &a) * &a) * &a) * &a;
        assert_eq!(a_pow_5, a_pow_5_manual);
    }

    

    #[test]
    fn test_mul_associativity() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Degree8ExtensionField::const_from_coefficients([9, 10, 11, 12, 13, 14, 15, 16]);
        let c = Degree8ExtensionField::const_from_coefficients([17, 18, 19, 20, 21, 22, 23, 24]);
        let ab_c = &(&a * &b) * &c;
        let a_bc = &a * &(&b * &c);
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_mul_commutativity() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Degree8ExtensionField::const_from_coefficients([9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn test_distributivity() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Degree8ExtensionField::const_from_coefficients([9, 10, 11, 12, 13, 14, 15, 16]);
        let c = Degree8ExtensionField::const_from_coefficients([17, 18, 19, 20, 21, 22, 23, 24]);
        let left = &a * &(&b + &c);
        let right = &(&a * &b) + &(&a * &c);
        assert_eq!(left, right);
    }

    #[test]
    fn test_div() {
        let a = Degree8ExtensionField::const_from_coefficients([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Degree8ExtensionField::const_from_coefficients([9, 10, 11, 12, 13, 14, 15, 16]);
        let quotient = (a.clone() / b.clone()).unwrap();
        let product = quotient * b;
        assert_eq!(product, a);
    }

    #[test]
    fn embed_fp_with_fp8() {
        let a = Fp::from(3);
        let a_extension = Fp8::from(3);
        assert_eq!(a.to_extension::<Degree8ExtensionField>(), a_extension);
    }

    #[test]
    fn add_fp_and_fp8() {
        let a = Fp::from(3);
        let a_extension = Fp8::from(3);
        let b = Fp8::from(2);
        assert_eq!(a + &b, a_extension + b);
    }

    #[test]
    fn mul_fp_by_fp8() {
        let a = Fp::from(30000000000);
        let a_extension = a.to_extension::<Degree8ExtensionField>();
        let b = Fp8::new([
            Fp4::new(
                [
                    Fp2::new(
                        [
                            Fp::from(0),
                            Fp::from(1)
                        ]
                    ), 
                    Fp2::new(
                        [
                            Fp::from(2),
                            Fp::from(3)
                        ]
                    )
                ]
            ),
            Fp4::new(
                [
                    Fp2::new(
                        [
                            Fp::from(4),
                            Fp::from(5)
                        ]
                    ), 
                    Fp2::new(
                        [
                            Fp::from(6),
                            Fp::from(7)
                        ]
                    )
                ]
            ),
        ]);
        assert_eq!(a * &b, a_extension * b);
    }


    // -------------------------------------------------------------------------
    // ByteConversion (serialization) tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_1_minus_2i_plus_ji_is_nonsquare_in_fp4() {
        // Verify that α = (1 - 2i) + j*i is NOT a square in Fp4
        // This proves x^2 - α is irreducible over Fp4, valid for an FP8 extension
        //
        // Uses Euler criterion: α^((p^4-1)/2) = -1 iff α is a non-square
        // Factor the exponent to fit in u64:
        //   (p^4-1)/2 = (p-1) * (p+1) * (p^2+1)/2

        fn fp4_pow(base: &Fp4, mut exp: u64) -> Fp4 {
            let mut result = Fp4::one();
            let mut b = base.clone();
            while exp > 0 {
                if exp & 1 == 1 {
                    result = &result * &b;
                }
                b = b.square();
                exp >>= 1;
            }
            result
        }

        let p: u64 = (1u64 << 31) - 1;

        // α = (1 - 2i) + j*(i) in Fp4
        let alpha = Fp4::new([
            Fp2::new([Fp::from(1u64), -Fp::from(2u64)]),  // 1 - 2i
            Fp2::new([Fp::from(0u64), Fp::from(1u64)]),    // i  (j-coefficient)
        ]);

        // Compute α^((p^4-1)/2) in three chained exponentiations
        let r = fp4_pow(&alpha, p - 1);
        let r = fp4_pow(&r, p + 1);
        let r = fp4_pow(&r, (p * p + 1) / 2);

        let neg_one = -Fp4::one();
        assert_eq!(r, neg_one,
            "x^2 - (1 - 2i + ji) should be irreducible over Fp4");
    }
}
