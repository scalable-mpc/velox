use std::{collections::HashMap, ops::{Mul, Sub, Add}};

use crypto::hash::do_hash;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::Polynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use types::Replica;

use crate::ProtocolField;

pub fn sample_polynomials_from_prf<F: ProtocolField>(
    secrets: Vec<FieldElement<F>>, 
    sec_key_map: HashMap<Replica, Vec<u8>>, 
    degree: usize,
    is_nonce: bool,
    nonce: u8
)-> Vec<Vec<FieldElement<F>>>{
    let tot_evaluations = secrets.len();
    let mut evaluations = Vec::new();
    for secret in secrets{
        evaluations.push(vec![secret]);
    }
    for i in 0..degree{
        let mut sec_key = sec_key_map.get(&(i as Replica)).unwrap().clone();
        if is_nonce{
            sec_key.push(nonce);
        }
        let samples = pseudorandom_lf(&sec_key, tot_evaluations);
        for (i,sample) in samples.into_iter().enumerate() {
            evaluations[i].push(sample);
        }
    }
    evaluations
}

pub async fn generate_evaluation_points<F: ProtocolField>(
    evaluations_prf: Vec<Vec<FieldElement<F>>>,
    degree: usize,
    shares_total: usize,
) -> (Vec<Vec<FieldElement<F>>>,
    Vec<Polynomial<FieldElement<F>>>
){
    // Routed through `matrix_matrix_multiply` so the GPU dispatcher picks it up
    // under `--features gpu`. Mirrors `generate_evaluation_points_opt` — see that
    // function for the layout commentary.

    let mut evaluation_points = Vec::new();
    evaluation_points.push(FieldElement::<F>::from(0u64));
    for i in 0..degree {
        evaluation_points.push(FieldElement::<F>::from((i + 1) as u64));
    }

    let inverse_vandermonde_mat = inverse_vandermonde(vandermonde_matrix(evaluation_points.clone()));
    let coeffs_mat = matrix_matrix_multiply(&inverse_vandermonde_mat, &evaluations_prf, false);

    let share_points: Vec<FieldElement<F>> = (1..=shares_total)
        .map(|i| FieldElement::<F>::from(i as u64))
        .collect();
    let share_powers = powers_matrix(&share_points, degree + 1);
    let evaluations_full = matrix_matrix_multiply(&share_powers, &coeffs_mat, false);

    let coefficients: Vec<Polynomial<FieldElement<F>>> = coeffs_mat
        .par_iter()
        .map(|row| Polynomial::new(row))
        .collect();

    (evaluations_full, coefficients)
}

pub async fn generate_evaluation_points_opt<F: ProtocolField>(
    evaluations_prf: Vec<Vec<FieldElement<F>>>,
    degree: usize,
    shares_total: usize,
) -> (Vec<Vec<FieldElement<F>>>,
    Vec<Polynomial<FieldElement<F>>>
){

    // The first evaluation is always at 0
    let mut evaluation_points = Vec::new();
    evaluation_points.push(FieldElement::<F>::from(0u64));
    for i in 0..degree{
        evaluation_points.push(FieldElement::<F>::from((i + 1) as u64));
    }

    // Generate vandermonde matrix
    let vandermonde = vandermonde_matrix(evaluation_points.clone());
    let inverse_vandermonde_mat = inverse_vandermonde(vandermonde);

    // Two-GEMM Lagrange (mirrors async_mpc/mpc/protocol/verification/compress_tup.rs:191):
    //   Step 1: coeffs_mat = inv_vandermonde · evaluations_prf  →  num_polys × (degree+1).
    //   Step 2: evals_mat  = powers_matrix(share_points, degree+1) · coeffs_mat
    //                       → num_polys × shares_total.
    // `BatchedLagrange` bench: at protocol-realistic n=16, t=5, GEMM is ~1.18× faster
    // than the prior per-poly mat-vec + per-point Horner once num_polys ≥ 1024.
    let coeffs_mat = matrix_matrix_multiply(&inverse_vandermonde_mat, &evaluations_prf, false);

    let share_points: Vec<FieldElement<F>> = (1..=shares_total)
        .map(|i| FieldElement::<F>::from(i as u64))
        .collect();
    let share_powers = powers_matrix(&share_points, degree + 1);
    let evaluations_full = matrix_matrix_multiply(&share_powers, &coeffs_mat, false);

    // The return type wants `Vec<Polynomial<FieldElement<F>>>`; reconstruct via Polynomial::new
    // (which trims trailing zeros — semantically identical to the prior path).
    let coefficients: Vec<Polynomial<FieldElement<F>>> = coeffs_mat
        .par_iter()
        .map(|row| Polynomial::new(row))
        .collect();

    (evaluations_full, coefficients)
}

pub async fn generate_evaluation_points_fft<F: ProtocolField>(
    secrets: Vec<FieldElement<F>>,
    degree_poly: usize,
    shares_total: usize,
)-> (Vec<Vec<FieldElement<F>>>, 
    Vec<Polynomial<FieldElement<F>>>
){
    // For FFT evaluations, first sample coefficients of polynomial and then interpolate all n points
    let coefficients: Vec<Vec<FieldElement<F>>> = secrets.into_par_iter().map(|secret| {
        let mut coeffs_single_poly = Vec::new();
        coeffs_single_poly.push(secret);
        for _ in 0..degree_poly{
            coeffs_single_poly.push(rand_field_element());
        }
        return Polynomial::new(&coeffs_single_poly).coefficients;
    }).collect();

    return generate_evaluation_points_opt(coefficients, degree_poly, shares_total).await;
}

// pub async fn generate_evaluation_points_fft<F: ProtocolField>(
//     secrets: Vec<FieldElement<F>>,
//     degree_poly: usize,
//     shares_total: usize,
// )-> (Vec<Vec<FieldElement<F>>>, 
//     Vec<Polynomial<FieldElement<F>>>
// ){
//     // For FFT evaluations, first sample coefficients of polynomial and then interpolate all n points
//     let coefficients: Vec<Polynomial<FieldElement<F>>> = secrets.into_par_iter().map(|secret| {
//         let mut coeffs_single_poly = Vec::new();
//         coeffs_single_poly.push(secret);
//         for _ in 0..degree_poly{
//             coeffs_single_poly.push(rand_field_element());
//         }
//         return Polynomial::new(&coeffs_single_poly);
//     }).collect();

//     let evaluations = coefficients.par_iter().map(|poly_coeffs|{
//         let poly_evaluations_fft = Polynomial::evaluate_fft::<MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>>(poly_coeffs, 1, Some(shares_total)).unwrap();
//         poly_evaluations_fft
//     }).collect();
//     (evaluations, coefficients)
// }

pub fn pseudorandom_lf<F: ProtocolField>(rng_seed: &[u8], num: usize) -> Vec<FieldElement<F>> {
    let mut rng = ChaCha20Rng::from_seed(do_hash(rng_seed));
    // Deterministic in `rng_seed`: two parties holding the same seed derive the
    // same elements, which is what the PRF-based sharing scheme relies on.
    (0..num).map(|_| F::from_rng(&mut rng)).collect()
}

/// Uniformly random element. Thin alias for `F::rand()`, kept as a free
/// function because it reads better at the call sites that deal polynomials.
pub fn rand_field_element<F: ProtocolField>() -> FieldElement<F> {
    F::rand()
}


pub fn interpolate_shares<F: ProtocolField>( mut secret_key: Vec<u8>, num_shares: usize, is_nonce: bool, padding: u8) -> Vec<FieldElement<F>>{
    if is_nonce{
        secret_key.push(padding);
    }
    let prf_values = pseudorandom_lf(&secret_key, num_shares);
    prf_values
}

pub fn check_if_all_points_lie_on_degree_x_polynomial<F: ProtocolField>(eval_points: Vec<FieldElement<F>>, polys_vector: Vec<Vec<FieldElement<F>>>, degree: usize) -> (bool,Option<Vec<Polynomial<FieldElement<F>>>>){
    //log::info!("Checking evaluations on points :{:?}, eval_points: {:?}", eval_points, polys_vector);
    let inverse_vandermonde_mat = inverse_vandermonde(vandermonde_matrix(eval_points[0..degree].to_vec()));

    // Two-GEMM Lagrange: recover coefficients from the first `degree` evaluations, then
    // batch-evaluate every recovered polynomial at the remaining `eval_points[degree..]`
    // for consistency-check against the supplied shares. Same idiom as
    // async_mpc/degree_verification/interpolation.rs:645 + 488/509.
    let prefix_vecs: Vec<Vec<FieldElement<F>>> = polys_vector
        .par_iter()
        .map(|points| points[0..degree].to_vec())
        .collect();
    let coeffs_mat = matrix_matrix_multiply(&inverse_vandermonde_mat, &prefix_vecs, false);

    let verify_points: Vec<FieldElement<F>> = eval_points[degree..].to_vec();
    let verify_evals = if verify_points.is_empty() {
        // Nothing to check past the first `degree` points — every poly passes by definition.
        vec![Vec::new(); coeffs_mat.len()]
    } else {
        let verify_powers = powers_matrix(&verify_points, degree);
        matrix_matrix_multiply(&verify_powers, &coeffs_mat, false)
    };

    // For each polynomial: does its recovered form match the supplied `points[degree..]`?
    let polys: Vec<Option<Polynomial<FieldElement<F>>>> = coeffs_mat
        .par_iter()
        .zip(polys_vector.par_iter())
        .zip(verify_evals.par_iter())
        .map(|((coeffs, points), evals)| {
            let expected = &points[degree..];
            if evals
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| got == want)
            {
                Some(Polynomial::new(coeffs))
            } else {
                None
            }
        })
        .collect();

    let all_polys_positive = polys.par_iter().all(|poly| poly.is_some());
    if all_polys_positive {
        let polys_vec = polys.into_iter().map(|x| x.unwrap()).collect();
        (true, Some(polys_vec))
    } else {
        (false, None)
    }
}


/// Constructs the Vandermonde matrix for a given set of x-values.
pub fn vandermonde_matrix<F: ProtocolField>(x_values: Vec<FieldElement<F>>) -> Vec<Vec<FieldElement<F>>> {
    let n = x_values.len();
    let mut matrix = vec![vec![FieldElement::<F>::zero(); n]; n];

    for (row, x) in x_values.iter().enumerate() {
        let mut value = FieldElement::<F>::one();
        for col in 0..n {
            matrix[row][col] = value.clone();
            value = value.mul(x);
        }
    }

    matrix
}

/// Computes the inverse of a Vandermonde matrix modulo prime using Gaussian elimination.
pub fn inverse_vandermonde<F: ProtocolField>(matrix: Vec<Vec<FieldElement<F>>>) -> Vec<Vec<FieldElement<F>>> {
    let n = matrix.len();
    let mut augmented = matrix.clone();

    // Extend the matrix with an identity matrix on the right
    for i in 0..n {
        augmented[i].extend((0..n).map(|j| if i == j { FieldElement::<F>::one() } else { FieldElement::<F>::zero() }));
    }

    // Perform Gaussian elimination
    for col in 0..n {
        // Normalize pivot row
        let inv = &augmented[col][col].inv().unwrap();
        for k in col..2 * n {
            augmented[col][k] = augmented[col][k].clone().mul(inv);
        }

        // Eliminate other rows
        for row in 0..n {
            if row != col {
                let factor = augmented[row][col].clone();
                for k in col..2 * n {
                    augmented[row][k] = augmented[row][k].clone().sub(factor.clone().mul(augmented[col][k].clone()));
                }
            }
        }
    }

    // Extract the right half as the inverse
    augmented
        .into_iter()
        .map(|row| row[n..2 * n].to_vec())
        .collect()
}

pub fn matrix_vector_multiply<F: ProtocolField>(
    matrix: &Vec<Vec<FieldElement<F>>>,
    vector: &Vec<FieldElement<F>>,
) -> Vec<FieldElement<F>> {
    matrix
        .par_iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(FieldElement::<F>::zero(), |sum, (a, b)| sum.add(a.mul(b)))
        })
        .collect()
}

/// CPU-only batched matrix-matrix multiply over `FieldElement<F>` (Rayon-parallel).
///
/// `matrix` is treated as an (R × C) matrix (R = `matrix.len()`, C = `matrix[0].len()`).
/// `vectors` is a slice of K vectors each of length C — i.e. the right operand viewed as
/// a (K × C) row-major buffer whose rows are the *columns* of the right matrix.
/// The output is the (R × K) product `M · Vᵀ`:
///   - row_major = true  → `output[r][i] = Σ_l matrix[r][l] * vectors[i][l]` (shape R × K).
///   - row_major = false → the same product transposed (shape K × R), useful when callers
///     want `output[i][r]` (per-vector outputs grouped first), matching async_mpc's layout.
///
/// Layout is identical to `async_mpc/fields/src/poly.rs::matrix_matrix_multiply_cpu` so the
/// two projects' benchmarks are directly comparable.
///
/// Dispatcher: with `--features gpu`, routes large enough inputs to the CUDA kernel
/// (see `gpu_ffi::gpu_matrix_matrix_multiply`); otherwise (or for small inputs that
/// would be dominated by PCIe upload overhead) calls `matrix_matrix_multiply_cpu`
/// directly. The CPU path stays the canonical reference and is always callable.
pub fn matrix_matrix_multiply<F: ProtocolField>(
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>> {
    // Mirrors async_mpc's small-input bailout. Below this work threshold the
    // device upload cost dominates the actual arithmetic.
    const GPU_THRESHOLD: usize = 50_000;
    let rows = matrix.len();
    let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
    let work = rows.saturating_mul(cols).saturating_mul(vectors.len());
    if work >= GPU_THRESHOLD {
        // `None` for every field without a compiled kernel, and for the one
        // that has it when the `gpu` feature is off — both fall through to CPU.
        if let Some(out) = F::try_gpu_gemm(matrix, vectors, row_major) {
            return out;
        }
    }
    matrix_matrix_multiply_cpu(matrix, vectors, row_major)
}

/// CPU implementation — Rayon-parallel over rows. Always available regardless of
/// the `gpu` feature; callers needing to force CPU (e.g. the bench's reference
/// path) should call this directly.
pub fn matrix_matrix_multiply_cpu<F: ProtocolField>(
    matrix: &[Vec<FieldElement<F>>],
    vectors: &[Vec<FieldElement<F>>],
    row_major: bool,
) -> Vec<Vec<FieldElement<F>>> {
    let k = vectors.len();
    if k == 0 || matrix.is_empty() {
        return Vec::new();
    }

    let cols = matrix[0].len();
    if vectors.iter().any(|v| v.len() != cols) {
        log::error!(
            "matrix_matrix_multiply_cpu: matrix column count ({}) does not match vector lengths {:?}",
            cols,
            vectors.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        return Vec::new();
    }

    let results: Vec<Vec<FieldElement<F>>> = matrix
        .par_iter()
        .map(|m_row| {
            let mut row_results = vec![FieldElement::<F>::zero(); k];
            for i in 0..k {
                let m_col = &vectors[i];
                let mut sum = FieldElement::<F>::zero();
                for l in 0..m_row.len() {
                    // Reference-reference multiply — matches async_mpc verbatim and avoids
                    // the 32-byte per-iteration clone of m_row[l] the previous form did.
                    sum += &m_row[l] * &m_col[l];
                }
                row_results[i] = sum;
            }
            row_results
        })
        .collect();

    if row_major {
        results
    } else {
        transpose(results)
    }
}

/// Build a (points.len() × degree) "powers" matrix where row `i` is
/// `[1, points[i], points[i]^2, …, points[i]^{degree-1}]`.
///
/// Used wherever many polynomials are evaluated at many points: given coefficient matrix
/// `C` (degree × num_polys, column-major over polys), `powers_matrix · C` yields all
/// evaluations in a single GEMM. This is the Vandermonde restricted to the first `degree`
/// columns, mirroring the async_mpc pattern at `fields/src/poly.rs::lagrange_interpolate_par`.
pub fn powers_matrix<F: ProtocolField>(points: &[FieldElement<F>], degree: usize) -> Vec<Vec<FieldElement<F>>> {
    points
        .par_iter()
        .map(|p| {
            let mut row = Vec::with_capacity(degree);
            let mut power = FieldElement::<F>::one();
            for _ in 0..degree {
                row.push(power.clone());
                power = power.mul(p);
            }
            row
        })
        .collect()
}

/// Transpose a rectangular matrix stored as `Vec<Vec<FieldElement<F>>>`.
pub fn transpose<F: ProtocolField>(matrix: Vec<Vec<FieldElement<F>>>) -> Vec<Vec<FieldElement<F>>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    (0..cols)
        .into_par_iter()
        .map(|j| (0..rows).map(|i| matrix[i][j].clone()).collect())
        .collect()
}
/// Pack polynomials over `F` into polynomials over `F::Ext`, `CONV_RATIO` at a
/// time, by lifting them coefficient-wise.
///
/// Used by the DZK proof, which has to run its random linear combination over
/// the wider field for soundness while the shares stay over `F`. Packing rather
/// than embedding each polynomial separately is what makes the combination
/// `CONV_RATIO` times cheaper.
///
/// Degree is preserved: coefficient `k` of the packed polynomial is
/// `F::lift` of coefficient `k` of each input, so a group of degree-`t`
/// polynomials packs into a single degree-`t` polynomial over `F::Ext`. When
/// `F::Ext = F` and `CONV_RATIO = 1` this is the identity.
pub fn lift_polynomials<F: ProtocolField>(
    polys: &[Polynomial<FieldElement<F>>],
) -> Vec<Polynomial<FieldElement<F::Ext>>> {
    polys
        .chunks(F::CONV_RATIO)
        .map(|chunk| {
            let width = chunk
                .iter()
                .map(|p| p.coefficients.len())
                .max()
                .unwrap_or(0);
            let coeffs: Vec<FieldElement<F::Ext>> = (0..width)
                .map(|k| {
                    // A shorter polynomial contributes zero at this degree.
                    let column: Vec<FieldElement<F>> = chunk
                        .iter()
                        .map(|p| {
                            p.coefficients
                                .get(k)
                                .cloned()
                                .unwrap_or_else(FieldElement::<F>::zero)
                        })
                        .collect();
                    F::lift(&column)
                })
                .collect();
            Polynomial::new(&coeffs)
        })
        .collect()
}

/// Pack a party's shares into `F::Ext` elements, `CONV_RATIO` at a time.
///
/// The evaluation-side counterpart of [`lift_polynomials`]: if `shares[j]` is
/// `poly_j` evaluated at this party's point, then the `i`-th packed share is
/// the `i`-th packed polynomial evaluated at that same point. That
/// correspondence is what lets a verifier check the dealer's combination
/// without ever leaving the wide field.
pub fn lift_shares<F: ProtocolField>(
    shares: &[FieldElement<F>],
) -> Vec<FieldElement<F::Ext>> {
    shares.chunks(F::CONV_RATIO).map(F::lift).collect()
}
