mod types;
pub use types::*;

pub mod byte_conv;
pub use byte_conv::ByteConversion;

pub mod protocol_field;
pub use protocol_field::{FieldSer, ProtocolField};

pub mod mersenne_61;
/// The 61-bit base field. Its `ProtocolField::Ext` is the degree-4 extension,
/// so DZK proofs over it run at ~2^-244 while shares stay 8 bytes.
pub use mersenne_61::Mersenne61Field;

pub mod prime_fields;
pub use prime_fields::{BN254Field, Stark252Field};

pub mod poly;
pub use poly::*;

pub mod mul;
pub use mul::*;

pub mod par;
pub use par::rayon_async;

#[cfg(feature = "gpu")]
pub mod gpu_ffi;
#[cfg(feature = "gpu")]
pub use gpu_ffi::gpu_matrix_matrix_multiply;