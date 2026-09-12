//! The AVX2 GEMM must agree with the scalar reference on every shape and on
//! non-canonical (raw value == p) inputs, for every field with lanes; the
//! dispatcher must reach it; and the explicit scalar backend must equal the
//! repo's `matrix_matrix_multiply_cpu`.
//!
//! On a host without AVX2 `gemm_with(Backend::Avx2, ..)` runs the scalar path,
//! so these tests still pass there — they just stop testing the kernel.
#![cfg(target_arch = "x86_64")]

use fields::mersenne_31::{
    Degree4ExtensionField as M31Fp4, Degree8ExtensionField as M31Fp8, Mersenne31Field,
};
use fields::mersenne_61::{Mersenne61Degree4ExtensionField, Mersenne61Field, MERSENNE_61_PRIME_FIELD_ORDER};
use fields::poly::{matrix_matrix_multiply, matrix_matrix_multiply_cpu};
use fields::simd::{gemm_with, scalar_gemm, Backend, SimdField};
use lambdaworks_math::field::element::FieldElement;

type Fp = FieldElement<Mersenne61Field>;
type Fp4 = FieldElement<Mersenne61Degree4ExtensionField>;
type G = FieldElement<Mersenne31Field>;

const SHAPES: [(usize, usize, usize); 8] =
    [(1, 1, 1), (3, 5, 3), (3, 5, 4), (3, 5, 7), (3, 5, 8), (6, 6, 1000), (16, 11, 513), (64, 128, 1024)];

fn rand_mat<F: SimdField>(r: usize, c: usize) -> Vec<Vec<FieldElement<F>>> {
    (0..r).map(|_| (0..c).map(|_| F::rand_elem()).collect()).collect()
}

fn check<F: SimdField>()
where
    FieldElement<F>: Clone + Send + Sync + PartialEq + std::fmt::Debug,
{
    for (r, c, k) in SHAPES {
        let m = rand_mat::<F>(r, c);
        let v = rand_mat::<F>(k, c);
        for row_major in [true, false] {
            let want = scalar_gemm(&m, &v, row_major);
            let got = gemm_with(Backend::Avx2, &m, &v, row_major);
            assert_eq!(want, got, "r={r} c={c} k={k} row_major={row_major}");
        }
    }
}

#[test]
fn m61_base() { check::<Mersenne61Field>() }
#[test]
fn m61_fp4() { check::<Mersenne61Degree4ExtensionField>() }
#[test]
fn m31_base() { check::<Mersenne31Field>() }
#[test]
fn m31_fp4() { check::<M31Fp4>() }
#[test]
fn m31_fp8() { check::<M31Fp8>() }

/// `scalar_gemm` is the same loop as the repo's `matrix_matrix_multiply_cpu`,
/// and the explicit scalar backend goes through it.
#[test]
fn scalar_reference_matches_repo_path() {
    let m = rand_mat::<Mersenne61Degree4ExtensionField>(16, 11);
    let v = rand_mat::<Mersenne61Degree4ExtensionField>(100, 11);
    let want = matrix_matrix_multiply_cpu(&m, &v, false);
    assert_eq!(scalar_gemm(&m, &v, false), want);
    assert_eq!(gemm_with(Backend::Scalar, &m, &v, false), want);
}

/// The protocol's entry point (`matrix_matrix_multiply`, which now routes
/// through `try_simd_gemm`) agrees with the scalar path for both M61 fields.
#[test]
fn dispatcher_matches_scalar() {
    for (r, c, k) in [(6, 6, 1000), (16, 11, 513), (64, 128, 300)] {
        let m = rand_mat::<Mersenne61Degree4ExtensionField>(r, c);
        let v = rand_mat::<Mersenne61Degree4ExtensionField>(k, c);
        assert_eq!(matrix_matrix_multiply(&m, &v, true), matrix_matrix_multiply_cpu(&m, &v, true));
        let m = rand_mat::<Mersenne61Field>(r, c);
        let v = rand_mat::<Mersenne61Field>(k, c);
        assert_eq!(matrix_matrix_multiply(&m, &v, false), matrix_matrix_multiply_cpu(&m, &v, false));
    }
}

/// A column-count mismatch is reported the same way by every path: empty output.
#[test]
fn mismatched_shapes_return_empty() {
    let m = rand_mat::<Mersenne61Field>(3, 4);
    let v = rand_mat::<Mersenne61Field>(5, 3);
    assert!(gemm_with(Backend::Avx2, &m, &v, true).is_empty());
    assert!(gemm_with(Backend::Scalar, &m, &v, true).is_empty());
    assert!(matrix_matrix_multiply_cpu(&m, &v, true).is_empty());
}

#[test]
fn empty_inputs() {
    let e: Vec<Vec<Fp>> = Vec::new();
    assert!(gemm_with(Backend::Avx2, &e, &e, true).is_empty());
    let m = rand_mat::<Mersenne61Field>(2, 2);
    assert!(gemm_with(Backend::Avx2, &m, &e, true).is_empty());
}

/// Edge lanes: 0, 1, p-1, and the raw non-canonical p that `a - a` produces.
#[test]
fn m61_edge_values() {
    let p = MERSENNE_61_PRIME_FIELD_ORDER;
    let raw: Vec<Fp> = [0u64, 1, p - 1, p, p - 2, 1 << 60].into_iter().map(Fp::const_from_raw).collect();
    let n = raw.len();
    let m: Vec<Vec<Fp>> = (0..n).map(|i| (0..n).map(|j| raw[(i + j) % n].clone()).collect()).collect();
    let v: Vec<Vec<Fp>> = (0..n * 2).map(|i| (0..n).map(|j| raw[(i * j) % n].clone()).collect()).collect();
    assert_eq!(scalar_gemm(&m, &v, true), gemm_with(Backend::Avx2, &m, &v, true));

    let m4: Vec<Vec<Fp4>> = m
        .iter()
        .map(|row| row.iter().map(|x| Mersenne61Degree4ExtensionField::const_from_fe(&[*x, raw[3], *x, raw[2]])).collect())
        .collect();
    let v4: Vec<Vec<Fp4>> = v
        .iter()
        .map(|row| row.iter().map(|x| Mersenne61Degree4ExtensionField::const_from_fe(&[raw[3], *x, raw[4], *x])).collect())
        .collect();
    assert_eq!(scalar_gemm(&m4, &v4, false), gemm_with(Backend::Avx2, &m4, &v4, false));
}

#[test]
fn m31_edge_values() {
    let p: u32 = (1 << 31) - 1;
    let raw: Vec<G> = [0u32, 1, p - 1, p, p - 2, 1 << 30, 2, p - 3].into_iter().map(G::const_from_raw).collect();
    let n = raw.len();
    let m: Vec<Vec<G>> = (0..n).map(|i| (0..n).map(|j| raw[(i + j) % n].clone()).collect()).collect();
    let v: Vec<Vec<G>> = (0..n * 2 + 1).map(|i| (0..n).map(|j| raw[(i * j) % n].clone()).collect()).collect();
    assert_eq!(scalar_gemm(&m, &v, true), gemm_with(Backend::Avx2, &m, &v, true));

    let lift = |x: &G, k: usize| -> FieldElement<M31Fp8> {
        let mut e = [raw[k % n]; 8];
        e[k % 8] = *x;
        e[(k + 3) % 8] = raw[3]; // a raw p limb somewhere
        M31Fp8::const_from_fe(&e)
    };
    let m8: Vec<Vec<_>> = m.iter().enumerate().map(|(i, row)| row.iter().map(|x| lift(x, i)).collect()).collect();
    let v8: Vec<Vec<_>> = v.iter().enumerate().map(|(i, row)| row.iter().map(|x| lift(x, i + 1)).collect()).collect();
    assert_eq!(scalar_gemm(&m8, &v8, false), gemm_with(Backend::Avx2, &m8, &v8, false));
}
