mod types;
pub use types::*;

pub mod byte_conv;
pub use byte_conv::ByteConversion;

pub mod protocol_field;
pub use protocol_field::{FieldSer, ProtocolField};

pub mod mersenne_prime;
/// Fields with modulus `2^ℓ − 1`; what the comparison/truncation layer is
/// generic over. Implemented for the Mersenne-61 and Mersenne-31 base fields.
pub use mersenne_prime::MersennePrimeField;

pub mod mersenne_61;
/// The 61-bit base field. Its `ProtocolField::Ext` is the degree-4 extension,
/// so DZK proofs over it run at ~2^-244 while shares stay 8 bytes.
pub use mersenne_61::Mersenne61Field;

pub mod mersenne_31;
/// The 31-bit base field and its degree-8 extension (`Ext`, ~2^-248). Half
/// the carry tree of Mersenne-61 for comparisons; see `mersenne_31::protocol_impl`
/// for the verification-soundness caveat of sharing over 31 bits.
pub use mersenne_31::{Degree8ExtensionField as Mersenne31Degree8ExtensionField, Mersenne31Field};

pub mod prime_fields;
pub use prime_fields::{BLS12381ScalarField, BN254Field, Stark252Field};

pub mod poly;
pub use poly::*;

pub mod mul;
pub use mul::*;

pub mod par;
pub use par::rayon_async;

#[cfg(target_arch = "x86_64")]
pub mod simd;

#[cfg(feature = "gpu")]
pub mod gpu_ffi;
#[cfg(feature = "gpu")]
pub use gpu_ffi::gpu_matrix_matrix_multiply;