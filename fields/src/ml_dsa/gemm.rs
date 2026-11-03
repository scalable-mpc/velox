// Batched GEMM over F_q with lazy reduction: the `try_simd_gemm` kernel of
// `MlDsaField`.
//
// A product of two elements is below 2^46, so a `u64` holds the sum of 2^18
// products without overflow. Each output is a dot product that sums its
// products in a `u64` and reduces once for every 2^18 terms, where the scalar
// path reduces after every product. The inner loop is a plain widening
// multiply-accumulate, which the compiler vectorizes for the target's
// baseline: NEON on aarch64, SSE2 on x86_64.

use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::*;

use super::field::{MlDsaField, ML_DSA_PRIME_FIELD_ORDER};

type FqE = FieldElement<MlDsaField>;

/// Products a `u64` can sum: each is below `q^2 < 2^46`.
const TERMS_PER_REDUCTION: usize = 1 << 18;

/// `Σ a_l·b_l mod q`, reduced once per [`TERMS_PER_REDUCTION`] terms. Reads
/// the stored words in place: a copy into `u32` buffers costs more than the
/// products when the rows are short, as in the dealer's GEMM (6 columns).
fn lazy_dot(a: &[FqE], b: &[FqE]) -> FqE {
    let q = ML_DSA_PRIME_FIELD_ORDER as u64;
    let mut sum = 0u64;
    for (x, y) in a.chunks(TERMS_PER_REDUCTION).zip(b.chunks(TERMS_PER_REDUCTION)) {
        sum += x.iter().zip(y).map(|(u, v)| *u.value() as u64 * *v.value() as u64).sum::<u64>() % q;
    }
    FqE::from_raw((sum % q) as u32)
}

/// Same contract as `poly::matrix_matrix_multiply_cpu`: `matrix` is R×C,
/// `vectors` is K vectors of length C, and the output is `M · Vᵀ`, R×K if
/// `row_major`, else K×R. `None` when the lengths do not match, so the
/// scalar path reports the error.
pub(super) fn lazy_gemm(matrix: &[Vec<FqE>], vectors: &[Vec<FqE>], row_major: bool) -> Option<Vec<Vec<FqE>>> {
    let cols = matrix.first()?.len();
    if vectors.is_empty() || matrix.iter().chain(vectors).any(|row| row.len() != cols) {
        return None;
    }

    // The outer loop runs over the output's rows, so each task writes one
    // output row.
    let (outer, inner) = if row_major { (matrix, vectors) } else { (vectors, matrix) };
    Some(outer.par_iter().map(|a| inner.iter().map(|b| lazy_dot(a, b)).collect()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::matrix_matrix_multiply_cpu;

    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<FqE>> {
        let mut state = seed | 1;
        (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        FqE::from(state % ML_DSA_PRIME_FIELD_ORDER as u64)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn agrees_with_the_scalar_path() {
        for (rows, cols, k) in [(1, 1, 1), (4, 6, 1000), (16, 6, 37), (64, 128, 64), (7, 1023, 5)] {
            let (m, v) = (random_matrix(rows, cols, 3), random_matrix(k, cols, 5));
            for row_major in [true, false] {
                assert_eq!(
                    lazy_gemm(&m, &v, row_major).unwrap(),
                    matrix_matrix_multiply_cpu(&m, &v, row_major),
                    "{rows}x{cols}, {k} vectors, row_major {row_major}"
                );
            }
        }
    }

    #[test]
    fn sums_past_one_reduction_window() {
        // All entries q − 1: each product is the largest possible, and the
        // sum crosses the 2^18-term window.
        let top = FqE::from((ML_DSA_PRIME_FIELD_ORDER - 1) as u64);
        let cols = TERMS_PER_REDUCTION + 5;
        let row = vec![vec![top.clone(); cols]];
        assert_eq!(lazy_gemm(&row, &row, true).unwrap(), matrix_matrix_multiply_cpu(&row, &row, true));
    }

    #[test]
    fn mismatched_lengths_go_to_the_scalar_path() {
        let (m, v) = (random_matrix(2, 3, 1), random_matrix(2, 4, 2));
        assert!(lazy_gemm(&m, &v, true).is_none());
        assert!(lazy_gemm(&[], &v, true).is_none());
    }
}
