//! SIMD (AVX2) kernels for the batched GEMM `poly::matrix_matrix_multiply`.
//!
//! Compiled only on `x86_64`; everything else in the crate is unaware of this
//! module except the `ProtocolField::try_simd_gemm` overrides in
//! `mersenne_61::protocol_impl`, which are gated the same way.
//!
//! # What is here
//!
//! | file | contents |
//! |---|---|
//! | [`m61_avx2`] | 4 × u64 lanes over Mersenne-61 and its Fp2/Fp4 tower |
//! | [`m31_avx2`] | 8 × u32 lanes over Mersenne-31 and its Fp2/Fp4/Fp8 tower |
//! | [`mod@gemm`] | the lane-generic GEMM kernel, the field ↔ limb packing, and the scalar reference |
//!
//! `docs/simd-m61-avx2.md` walks through the Mersenne-61 arithmetic in detail.
//!
//! # Backend selection and fallback
//!
//! [`gemm`](fn@gemm) is the single entry point. It picks a [`Backend`] once per
//! process via [`backend`]:
//!
//! 1. `VELOX_SIMD=off` (also `0`, `scalar`) in the environment forces the
//!    scalar path — for A/B runs of the protocol with everything else equal.
//! 2. Otherwise AVX2 is used if `is_x86_feature_detected!("avx2")` says the
//!    CPU has it, and the scalar path if not.
//!
//! On a non-`x86_64` target this module does not exist and the
//! `try_simd_gemm` overrides are compiled out, so the dispatcher's default
//! `None` sends every call to `matrix_matrix_multiply_cpu`. The scalar path
//! is therefore always reachable and is the reference every kernel is tested
//! against (`tests/simd_gemm.rs`).

pub mod gemm;
pub mod m31_avx2;
pub mod m61_avx2;

use std::sync::OnceLock;

use lambdaworks_math::field::element::FieldElement;

pub use gemm::{scalar_gemm, SimdField};

/// Which implementation [`gemm`](fn@gemm) runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Lane-wise kernel using 256-bit AVX2 registers.
    Avx2,
    /// The per-element loop of `poly::matrix_matrix_multiply_cpu`.
    Scalar,
}

/// The backend this process uses, decided once and cached.
///
/// Honors `VELOX_SIMD` (see the module docs), then CPU feature detection.
pub fn backend() -> Backend {
    static CHOICE: OnceLock<Backend> = OnceLock::new();
    *CHOICE.get_or_init(|| {
        if let Ok(v) = std::env::var("VELOX_SIMD") {
            if matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "scalar" | "false") {
                log::info!("VELOX_SIMD={v}: SIMD GEMM disabled, using the scalar path");
                return Backend::Scalar;
            }
        }
        if is_x86_feature_detected!("avx2") {
            Backend::Avx2
        } else {
            log::info!("AVX2 not detected: SIMD GEMM falls back to the scalar path");
            Backend::Scalar
        }
    })
}

/// Batched GEMM with the process-wide [`backend`].
///
/// Same contract as `poly::matrix_matrix_multiply_cpu`: `matrix` is R×C,
/// `vectors` is K vectors of length C, output is `M · Vᵀ` (R×K if
/// `row_major`, else K×R). Mismatched lengths log an error and return an
/// empty result, as the scalar path does.
pub fn gemm<F>(
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>>
where
    F: SimdField,
    FieldElement<F>: Clone + Send + Sync,
{
    gemm_with(backend(), matrix, vectors, row_major)
}

/// [`gemm`](fn@gemm) with an explicit backend. `Backend::Avx2` on a CPU
/// without AVX2 silently runs the scalar path instead of faulting, so this is
/// safe to call with either variant anywhere.
pub fn gemm_with<F>(
    backend: Backend,
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>>
where
    F: SimdField,
    FieldElement<F>: Clone + Send + Sync,
{
    match backend {
        Backend::Avx2 if is_x86_feature_detected!("avx2") => {
            gemm::gemm_avx2(matrix, vectors, row_major)
        }
        _ => scalar_gemm(matrix, vectors, row_major),
    }
}
