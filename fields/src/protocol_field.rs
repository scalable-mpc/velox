//! The field abstraction the protocol is generic over.
//!
//! Everything in the engine — preprocessing, multiplication, verification,
//! routing, ACSS/Sh2t — is written against [`ProtocolField`] rather than a
//! concrete field type, so swapping the finite field is a type parameter at
//! `node/src/main.rs` instead of an edit-and-recompile of the whole tree.
//!
//! # Why a trait and not a bare `IsField` bound
//!
//! lambdaworks's [`IsField`] covers the arithmetic and nothing else. The
//! protocol additionally needs six capabilities that are genuinely
//! field-specific and have no generic implementation:
//!
//! | Capability | Used by |
//! |---|---|
//! | uniform sampling | polynomial dealing, ACSS, application inputs |
//! | PRF-seeded sampling | `sample_polynomials_from_prf`, share interpolation |
//! | square roots | `rand_bit` (random-bit generation squares and re-roots) |
//! | byte serialization | every wire message (the wire format stays `Vec<u8>`) |
//! | ASCII input encoding | `mpc::input` reading party inputs from a file |
//! | GPU GEMM dispatch | the optional CUDA path, which is layout-specific |
//!
//! Serialization is exposed as trait methods rather than a
//! `where FieldElement<Self>: ByteConversion` bound on purpose: a where-clause
//! on the trait definition does not propagate as an implied bound, so every one
//! of the several hundred `F: ProtocolField` sites in the workspace would have
//! had to restate it. `Self::BaseType: Send + Sync` *is* expressed as a
//! supertrait associated-type bound, because that one does propagate — and it
//! is what lets `FieldElement<F>` cross Rayon's `par_iter` boundaries.

use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsField},
};
use rand_chacha::ChaCha20Rng;

/// A finite field the MPC engine can run over.
///
/// Implement this for a lambdaworks field to make the whole protocol available
/// over it; see `mersenne_61::Mersenne61Degree4ExtensionField` for the reference
/// implementation and [`crate::DefaultField`] for the one the node currently
/// selects.
pub trait ProtocolField: IsField<BaseType: Send + Sync> + Send + Sync + Sized + 'static {
    /// Width of one serialized element in bytes. Every `to_bytes_*` returns
    /// exactly this many bytes, and every `from_bytes_*` expects them.
    const SER_BYTES: usize;

    /// Largest ASCII payload one element carries losslessly under
    /// [`encode_ascii`](ProtocolField::encode_ascii).
    const MAX_INPUT_PAYLOAD: usize;

    /// Uniformly random element, drawn from the OS/thread RNG.
    fn rand() -> FieldElement<Self>;

    /// Element drawn from a caller-supplied deterministic RNG. Used wherever
    /// two parties must derive the *same* value from a shared seed — PRF-based
    /// polynomial sampling and share interpolation.
    fn from_rng(rng: &mut ChaCha20Rng) -> FieldElement<Self>;

    /// Both square roots of `elem`, or `None` if it is a quadratic non-residue.
    /// `sqrt(0)` is `Some((0, 0))`.
    fn sqrt(elem: &FieldElement<Self>) -> Option<(FieldElement<Self>, FieldElement<Self>)>;

    /// Serialize to exactly [`SER_BYTES`](ProtocolField::SER_BYTES) big-endian bytes.
    fn to_bytes_be(elem: &FieldElement<Self>) -> Vec<u8>;

    /// Serialize to exactly [`SER_BYTES`](ProtocolField::SER_BYTES) little-endian bytes.
    fn to_bytes_le(elem: &FieldElement<Self>) -> Vec<u8>;

    /// Inverse of [`to_bytes_be`](ProtocolField::to_bytes_be).
    fn from_bytes_be(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError>;

    /// Inverse of [`to_bytes_le`](ProtocolField::to_bytes_le).
    fn from_bytes_le(bytes: &[u8]) -> Result<FieldElement<Self>, ByteConversionError>;

    /// Pack a text line into a single element, losslessly, or `None` if it is
    /// longer than [`MAX_INPUT_PAYLOAD`](ProtocolField::MAX_INPUT_PAYLOAD).
    ///
    /// Implementations must avoid any modular fold that would make the
    /// round-trip lossy — see the Mersenne-61 impl for how it dodges the
    /// reduction by leaving each limb's high byte clear.
    fn encode_ascii(input: &str) -> Option<FieldElement<Self>>;

    /// Optional GPU-accelerated batched GEMM.
    ///
    /// Returning `None` (the default) means "no GPU path for this field" and
    /// callers fall back to the Rayon CPU kernel. Only fields whose in-memory
    /// layout matches a compiled CUDA kernel can override this.
    fn try_gpu_gemm(
        _matrix: &[Vec<FieldElement<Self>>],
        _vectors: &[Vec<FieldElement<Self>>],
        _row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        None
    }
}

/// Method-syntax sugar over [`ProtocolField`]'s serialization.
///
/// Generic code writes `elem.ser_be()` rather than `F::to_bytes_be(&elem)`,
/// which keeps the call sites reading the way they did when they were pinned to
/// a concrete element type. Deliberately *not* named `to_bytes_be`: the local
/// [`ByteConversion`](crate::ByteConversion) trait already owns that name for
/// the concrete Mersenne types, and a blanket impl under the same name would be
/// ambiguous wherever both traits are in scope — and coherence rejects blanket-
/// impling `ByteConversion` itself while the per-type impls exist.
pub trait FieldSer {
    fn ser_be(&self) -> Vec<u8>;
    fn ser_le(&self) -> Vec<u8>;
}

impl<F: ProtocolField> FieldSer for FieldElement<F> {
    fn ser_be(&self) -> Vec<u8> {
        F::to_bytes_be(self)
    }

    fn ser_le(&self) -> Vec<u8> {
        F::to_bytes_le(self)
    }
}
