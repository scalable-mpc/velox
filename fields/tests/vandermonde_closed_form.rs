//! `inverse_vandermonde_from_points` replaces O(n^3) Gaussian elimination with
//! the O(n^2) Lagrange closed form, and `lagrange_coefficients_at_zero` skips
//! the matrix entirely where only the secret is wanted. Both must agree with
//! the elimination path exactly — these pin that.

use fields::{
    Mersenne61Field, interpolate_at_zero, inverse_vandermonde, inverse_vandermonde_from_points,
    lagrange_coefficients_at_zero, matrix_vector_multiply, vandermonde_matrix,
};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::Polynomial;

type F = Mersenne61Field;

fn pts(n: usize, stride: u64, offset: u64) -> Vec<FieldElement<F>> {
    (0..n)
        .map(|i| FieldElement::<F>::from(offset + stride * i as u64))
        .collect()
}

#[test]
fn closed_form_matches_gaussian_elimination() {
    for n in 1..14 {
        for (stride, offset) in [(1u64, 1u64), (7, 3), (13, 1000), (1, 0)] {
            let points = pts(n, stride, offset);
            let old = inverse_vandermonde(vandermonde_matrix(points.clone()));
            let new = inverse_vandermonde_from_points(&points);
            assert_eq!(old, new, "n={n} stride={stride} offset={offset}");
        }
    }
}

#[test]
fn closed_form_actually_inverts() {
    let n = 9;
    let points = pts(n, 5, 2);
    let v = vandermonde_matrix(points.clone());
    let inv = inverse_vandermonde_from_points(&points);

    // inv * v == identity
    for r in 0..n {
        for c in 0..n {
            let entry = (0..n).fold(FieldElement::<F>::zero(), |acc, k| acc + &inv[r][k] * &v[k][c]);
            let want = if r == c {
                FieldElement::<F>::one()
            } else {
                FieldElement::<F>::zero()
            };
            assert_eq!(entry, want, "({r},{c})");
        }
    }
}

#[test]
fn interpolation_round_trips_through_the_closed_form() {
    // A known polynomial, sampled at n points, must come back coefficient-wise.
    let coeffs: Vec<FieldElement<F>> = [4u64, 17, 0, 9, 123]
        .iter()
        .map(|c| FieldElement::<F>::from(*c))
        .collect();
    let poly = Polynomial::new(&coeffs);
    let points = pts(coeffs.len(), 3, 11);
    let values: Vec<FieldElement<F>> = points.iter().map(|p| poly.evaluate(p)).collect();

    let recovered = matrix_vector_multiply(&inverse_vandermonde_from_points(&points), &values);
    assert_eq!(recovered, coeffs);
}

#[test]
fn lagrange_at_zero_matches_the_full_interpolation() {
    for n in 1..14 {
        for (stride, offset) in [(1u64, 1u64), (7, 3), (13, 1000)] {
            let points = pts(n, stride, offset);
            let values: Vec<FieldElement<F>> = (0..n)
                .map(|i| FieldElement::<F>::from((i as u64 + 1) * 31 + 5))
                .collect();

            // What the call sites do today: full inverse, full matvec, read [0].
            let coeffs = matrix_vector_multiply(
                &inverse_vandermonde(vandermonde_matrix(points.clone())),
                &values,
            );
            let expected = Polynomial::new(&coeffs).evaluate(&FieldElement::<F>::zero());

            let lambdas = lagrange_coefficients_at_zero(&points);
            let got = interpolate_at_zero(&lambdas, &values);

            assert_eq!(got, expected, "n={n} stride={stride} offset={offset}");
        }
    }
}

#[test]
fn lagrange_at_zero_handles_a_point_at_zero() {
    // Evaluation point sets include 0 in the PRF paths; the formula divides only
    // by differences, so this must still hold.
    let points = pts(6, 1, 0);
    let values: Vec<FieldElement<F>> = (0..6)
        .map(|i| FieldElement::<F>::from((i as u64) * 97 + 13))
        .collect();
    let coeffs =
        matrix_vector_multiply(&inverse_vandermonde(vandermonde_matrix(points.clone())), &values);
    let expected = Polynomial::new(&coeffs).evaluate(&FieldElement::<F>::zero());
    let got = interpolate_at_zero(&lagrange_coefficients_at_zero(&points), &values);
    assert_eq!(got, expected);
    // With x=0 present the secret is just that party's value.
    assert_eq!(got, values[0]);
}
