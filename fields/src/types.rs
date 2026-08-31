use lambdaworks_math::field::element::FieldElement;

use crate::mersenne_61::Mersenne61Degree4ExtensionField;
use crate::protocol_field::ProtocolField;

/// The field the node selects by default: the degree-4 extension of the
/// Mersenne-61 prime field — 4 × u64 little-endian limbs, ~244-bit soundness,
/// 32 bytes on the wire.
///
/// The engine itself is generic over any [`ProtocolField`]; this alias is only
/// the concrete choice, and `node/src/main.rs` is the single place that binds
/// it. An application needing a different field implements [`ProtocolField`]
/// for it and swaps the alias there — no other edit, no full recompile of the
/// protocol against a new static type.
pub type DefaultField = Mersenne61Degree4ExtensionField;

/// Element of [`DefaultField`].
///
/// Retained for the crate's own concrete call sites (benches, GPU FFI, tests).
/// Protocol code should be generic over `F: ProtocolField` and name
/// `FieldElement<F>` instead.
pub type LargeField = FieldElement<DefaultField>;

/// Serialized form of a field element. Field-agnostic by construction — the
/// wire format is bytes, which is why `ProtMsg` and the network layer need no
/// type parameter. Width is `F::SER_BYTES` for whichever `F` is in play.
pub type LargeFieldSer = Vec<u8>;

/// Roots-of-unity stub used by share-point selection. Mersenne61 has no FFT
/// support in lambdaworks (no MontgomeryBackend) — this just hands back the
/// non-FFT party-id-as-field-element points, mirroring the previous behaviour
/// in the `!use_fft` branch.
pub fn gen_roots_of_unity<F: ProtocolField>(n: usize) -> Vec<FieldElement<F>> {
    (1..n + 1)
        .into_iter()
        .map(|x| FieldElement::<F>::from(x as u64))
        .collect()
}

/// Per-share triple emitted by the AVSS layer. Serialized form, so it carries
/// no field type parameter.
pub type AvssShare = (Vec<LargeFieldSer>, LargeFieldSer, LargeFieldSer);
