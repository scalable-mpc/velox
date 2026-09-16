//! Mersenne-31 tower: lambdaworks base/Fp2/Fp4 plus the Fp8 layer from scalable_mpc.
pub mod extension_fp8;
pub use extension_fp8::*;

mod protocol_impl;
mod ser;
pub mod sqrt;

use crate::mersenne_prime::MersennePrimeField;

/// The base field is lambdaworks's; the Mersenne-prime facts are added here
/// and its `ProtocolField` impl lives in `protocol_impl`.
impl MersennePrimeField for Mersenne31Field {
    const BITS: usize = 31;

    fn to_canonical_u64(elem: &lambdaworks_math::field::element::FieldElement<Self>) -> u64 {
        <Self as lambdaworks_math::field::traits::IsPrimeField>::representative(elem.value()) as u64
    }
}
