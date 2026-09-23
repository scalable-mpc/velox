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
//! | ASCII input encoding/decoding | `mpc::input` reading party inputs, and the output layer printing them back |
//! | GPU GEMM dispatch | the optional CUDA path, which is layout-specific |
//! | SIMD GEMM dispatch | the AVX2 path, which needs a lane type per field |
//! | lifting to a wider field | DZK proofs and verification coins, whose soundness is bounded by field size |
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

    /// Inverse of [`encode_ascii`](ProtocolField::encode_ascii): recover the
    /// text an element carries, with the encode-time left padding stripped.
    ///
    /// This is the half the output layer runs, and it must mirror
    /// `encode_ascii` exactly — the two live on the same trait so a new field
    /// cannot implement one and inherit the other's byte layout by accident.
    fn decode_ascii(elem: &FieldElement<Self>) -> String;

    // -- Soundness lift -----------------------------------------------------
    //
    // Fiat-Shamir challenges and the random linear combinations they weight are
    // only as sound as the field they live in: a check over a 61-bit field is
    // 61-bit sound however many times it is repeated. When the sharing field is
    // too small for that, the challenge and the combination move to an
    // extension of it, while the shares themselves stay small.

    /// Field the soundness-critical challenges live in.
    ///
    /// An *extension* of `Self`, so a sharing over `Self` lifts into it without
    /// disturbing its degree. For a field that is already wide enough this is
    /// `Self`, which makes every lift below the identity and costs nothing.
    ///
    /// The `Ext = Self::Ext` bound terminates the tower: an extension is its own
    /// extension, so `F::Ext::Ext` is `F::Ext` and the recursion bottoms out.
    ///
    /// # Scope
    ///
    /// The lift covers the ACSS DZK proof. The multiplication-verification
    /// phase — the delinearization coin and the compression levels it feeds —
    /// deliberately stays in `Self`: its challenge is a reconstructed sharing,
    /// so lifting it would make the whole compression pipeline `Ext`-valued and
    /// require `Ext`-valued preprocessing and multiplication. That phase
    /// instead *assumes* `Self` is large enough for statistical security on its
    /// own. Choosing a `Self` small enough to need this lift therefore leaves
    /// verification bounded by `|Self|`, not by `|Ext|`.
    type Ext: ProtocolField<Ext = Self::Ext>;

    /// How many `Self` elements pack into one [`Ext`](ProtocolField::Ext)
    /// element — the degree of the extension, and `1` when `Ext = Self`.
    const CONV_RATIO: usize;

    /// Pack up to [`CONV_RATIO`](ProtocolField::CONV_RATIO) elements into one
    /// `Ext` element, using them as its coefficients over `Self`. A short chunk
    /// is zero-padded.
    ///
    /// This is injective, which is all the soundness argument needs, and it is
    /// coefficient-wise, which is why it preserves degree: if each `s_j(x)` is a
    /// degree-`t` polynomial over `Self`, then `lift([s_0(x), .., s_k(x)])` is a
    /// degree-`t` polynomial over `Ext`. Packing rather than embedding is what
    /// makes the linear combination `CONV_RATIO` times cheaper.
    fn lift(chunk: &[FieldElement<Self>]) -> FieldElement<Self::Ext>;

    /// Embed a single element into `Ext` as a field homomorphism.
    ///
    /// Named `embed_ext` rather than `embed` because lambdaworks's
    /// `IsSubFieldOf` already owns `embed`, and the two resolve ambiguously
    /// wherever both traits are in scope.
    ///
    /// Unlike [`lift`](ProtocolField::lift) this preserves arithmetic, so it is
    /// what evaluation points and other scalars need. `embed_ext(x) + embed_ext(y)
    /// == embed_ext(x + y)`; `lift` makes no such promise.
    fn embed_ext(elem: &FieldElement<Self>) -> FieldElement<Self::Ext>;

    // -- Mersenne capability ------------------------------------------------
    //
    // The comparison and truncation protocols need the modulus to be a
    // Mersenne prime `2^ℓ − 1` (see `MersennePrimeField`). Code that is
    // generic over every protocol field — the Planner — asks through these two
    // items whether it has one, and refuses those operations when it does not,
    // instead of carrying the bound on everything it touches.

    /// `Some(ℓ)` when the modulus is the Mersenne prime `2^ℓ − 1`, `None`
    /// otherwise. The extension fields over a Mersenne prime are `None`:
    /// their elements are not integers mod `p`.
    const MERSENNE_BITS: Option<usize> = None;

    /// The integer in `[0, p)` that `elem` represents, when the field is a
    /// Mersenne prime field; `None` otherwise. Agrees with
    /// `MersennePrimeField::to_canonical_u64` wherever both exist.
    fn mersenne_canonical(_elem: &FieldElement<Self>) -> Option<u64> {
        None
    }

    /// Optional GPU-accelerated batched GEMM.
    ///
    /// Returning `None` (the default) means "no GPU path for this field" and
    /// callers fall back to the next kernel in line. Only fields whose
    /// in-memory layout matches a compiled CUDA kernel can override this.
    fn try_gpu_gemm(
        _matrix: &[Vec<FieldElement<Self>>],
        _vectors: &[Vec<FieldElement<Self>>],
        _row_major: bool,
    ) -> Option<Vec<Vec<FieldElement<Self>>>> {
        None
    }

    /// Optional SIMD (AVX2) batched GEMM.
    ///
    /// Returning `None` (the default) means "no vector kernel for this field"
    /// and `matrix_matrix_multiply` runs the scalar loop. A field overrides
    /// this by handing the call to `crate::simd::gemm`, which itself falls
    /// back to the scalar path when the CPU lacks AVX2 or `VELOX_SIMD=off`
    /// is set — so an override never makes the field *require* SIMD. The
    /// overrides are `#[cfg(target_arch = "x86_64")]`; on other targets the
    /// default applies.
    fn try_simd_gemm(
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
