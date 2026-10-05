//! Mersenne-31 tower: lambdaworks base/Fp2/Fp4 plus the Fp8 layer from scalable_mpc.
pub mod extension_fp8;
pub use extension_fp8::*;

mod mpc_impl;
mod ser;
pub mod sqrt;

use crate::mersenne_prime::MersennePrimeField;

/// The base field is lambdaworks's; the Mersenne-prime facts are added here
/// and its `ProtocolField` impl lives in `mpc_impl`.
impl MersennePrimeField for Mersenne31Field {
    const BITS: usize = 31;

    fn to_canonical_u128(elem: &lambdaworks_math::field::element::FieldElement<Self>) -> u128 {
        <Self as lambdaworks_math::field::traits::IsPrimeField>::representative(elem.value()) as u128
    }
}
