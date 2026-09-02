//! The second-set evaluation in `compress_tup::ex_compr` was rewritten from a
//! per-polynomial loop into a single `powers_matrix * coeffs` GEMM. This pins
//! the identity that rewrite relies on.

use fields::{Mersenne61Field, matrix_matrix_multiply, powers_matrix};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::Polynomial;

type F = Mersenne61Field;

#[test]
fn gemm_matches_per_polynomial_evaluation() {
    let num_polys = 7;   // d: inner tuple dimension
    let num_coeffs = 5;  // m: coefficients per polynomial == number of points

    // Coefficient matrix: `num_polys` rows of `num_coeffs` coefficients.
    let coeffs_mat: Vec<Vec<FieldElement<F>>> = (0..num_polys)
        .map(|j| {
            (0..num_coeffs)
                .map(|l| FieldElement::<F>::from((3 * j + 7 * l + 1) as u64))
                .collect()
        })
        .collect();

    let points: Vec<FieldElement<F>> = (1..=num_coeffs)
        .map(|i| FieldElement::<F>::from((100 + i) as u64))
        .collect();

    // Reference: what the old loop produced — row `i` is every polynomial
    // evaluated at point `i`, in polynomial order.
    let polys: Vec<Polynomial<FieldElement<F>>> =
        coeffs_mat.iter().map(|row| Polynomial::new(row)).collect();
    let mut expected = vec![Vec::new(); points.len()];
    for poly in polys.iter() {
        for (i, point) in points.iter().enumerate() {
            expected[i].push(poly.evaluate(point));
        }
    }

    // The GEMM form.
    let got = matrix_matrix_multiply(&powers_matrix(&points, num_coeffs), &coeffs_mat, true);

    assert_eq!(got.len(), expected.len(), "row count");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "row {i}");
    }
}

#[test]
fn gemm_row_layout_is_points_by_polys() {
    // Guards the orientation: rows are points, columns are polynomials.
    let coeffs_mat = vec![
        vec![FieldElement::<F>::from(1u64), FieldElement::<F>::from(0u64)], // p0(x) = 1
        vec![FieldElement::<F>::from(0u64), FieldElement::<F>::from(1u64)], // p1(x) = x
    ];
    let points = vec![FieldElement::<F>::from(4u64), FieldElement::<F>::from(9u64)];
    let got = matrix_matrix_multiply(&powers_matrix(&points, 2), &coeffs_mat, true);

    assert_eq!(got.len(), 2);
    assert_eq!(got[0], vec![FieldElement::<F>::from(1u64), FieldElement::<F>::from(4u64)]);
    assert_eq!(got[1], vec![FieldElement::<F>::from(1u64), FieldElement::<F>::from(9u64)]);
}
