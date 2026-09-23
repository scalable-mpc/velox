//! Verification of the multiplication tuples and of the revealed values.
//!
//! Both checks run over the field's statistical extension `K =
//! F::StatisticalExt` (see `ProtocolField`), not over `F`: their soundness is
//! bounded by the size of the field their challenges live in, and `F` may be
//! as narrow as 31 bits. The shares themselves stay in `F`. Verification
//! embeds them into `K`, draws its coins and masks in `K` from `d =
//! F::STATISTICAL_DEGREE` sharings over `F` each, and computes in `K`.
//!
//! Everything that crosses the network, or goes through the base field's
//! linear algebra, goes as coefficients over `F`: interpolating, evaluating
//! and degree-checking are `F`-linear, so they act on each coefficient on its
//! own, and the multiplication protocol computes a `K` inner product as `d`
//! inner products over `F`. When `K = F` every helper below is the identity.

use std::collections::VecDeque;

use fields::{poly::check_if_all_points_lie_on_degree_x_polynomial, LargeFieldSer, ProtocolField};
use lambdaworks_math::field::element::FieldElement;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSlice};

mod compress_tup;

mod ex_compr_state;

mod common_coin;

mod delinearize;

mod reveal_check;

mod verf_state;
pub use verf_state::VerificationState;

/// An element of the field the verification checks run in.
pub type StatisticalElement<F> = FieldElement<<F as ProtocolField>::StatisticalExt>;

/// A uniformly random sharing over `K`, built from `d` random sharings over
/// `F` taken from `pool`, one per coefficient. `None` when the pool is short.
fn pop_statistical_sharing<F: ProtocolField>(pool: &mut VecDeque<FieldElement<F>>) -> Option<StatisticalElement<F>> {
    if pool.len() < F::STATISTICAL_DEGREE {
        return None;
    }
    let coeffs: Vec<FieldElement<F>> = pool.drain(..F::STATISTICAL_DEGREE).collect();
    Some(F::from_statistical_coeffs(&coeffs))
}

/// Each row over `K` as `d` rows over `F`, one per coefficient: row `r`
/// becomes rows `r·d .. r·d + d`.
fn split_into_coeff_rows<F: ProtocolField>(rows: &[Vec<StatisticalElement<F>>]) -> Vec<Vec<FieldElement<F>>> {
    rows.par_iter()
        .flat_map_iter(|row| {
            (0..F::STATISTICAL_DEGREE).map(move |k| row.iter().map(|elem| F::statistical_coeff(elem, k)).collect())
        })
        .collect()
}

/// The inverse of [`split_into_coeff_rows`]: every `d` consecutive rows over
/// `F` as one row over `K`.
fn join_coeff_rows<F: ProtocolField>(rows: &[Vec<FieldElement<F>>]) -> Vec<Vec<StatisticalElement<F>>> {
    rows.par_chunks(F::STATISTICAL_DEGREE)
        .map(|group| {
            let mut coeffs = Vec::with_capacity(group.len());
            (0..group[0].len())
                .map(|i| {
                    coeffs.clear();
                    coeffs.extend(group.iter().map(|row| row[i].clone()));
                    F::from_statistical_coeffs(&coeffs)
                })
                .collect()
        })
        .collect()
}

/// Every `d` consecutive elements of `F` as one element of `K`.
fn join_coeffs<F: ProtocolField>(coeffs: &[FieldElement<F>]) -> Vec<StatisticalElement<F>> {
    coeffs.chunks(F::STATISTICAL_DEGREE).map(F::from_statistical_coeffs).collect()
}

/// Inner products over `K`, rewritten as the inner products over `F` the
/// multiplication protocol computes. Multiplying by `y` is `F`-linear, so
/// with `e_j` the `j`-th basis element of `K`,
///
/// ```text
/// coeff_k(Σ_l x_l·y_l) = Σ_{l,j} coeff_j(x_l) · coeff_k(e_j·y_l)
/// ```
///
/// and both factors are local functions of the shares. Inner product `p`
/// becomes the `F` inner products `p·d .. p·d + d`, one per coefficient of the
/// result and each `d` times as long; [`join_coeffs`] reassembles them.
fn inner_products_over_base<F: ProtocolField>(
    x: &[Vec<StatisticalElement<F>>],
    y: &[Vec<StatisticalElement<F>>],
) -> (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<F>>>) {
    let d = F::STATISTICAL_DEGREE;
    let basis: Vec<StatisticalElement<F>> = (0..d)
        .map(|j| {
            let mut unit = vec![FieldElement::<F>::zero(); d];
            unit[j] = FieldElement::one();
            F::from_statistical_coeffs(&unit)
        })
        .collect();
    let per_product: Vec<(Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>)> = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(x_row, y_row)| {
            let x_coeffs: Vec<FieldElement<F>> =
                x_row.iter().flat_map(|elem| (0..d).map(move |j| F::statistical_coeff(elem, j))).collect();
            let mut y_coeffs = vec![Vec::with_capacity(y_row.len() * d); d];
            for y_elem in y_row {
                for (j, e_j) in basis.iter().enumerate() {
                    // `e_0 = 1`: no product needed, which is all of `K = F`.
                    let product = if j == 0 { y_elem.clone() } else { e_j * y_elem };
                    for (k, y_k) in y_coeffs.iter_mut().enumerate() {
                        y_k.push(F::statistical_coeff(&product, k));
                    }
                }
            }
            (x_coeffs, y_coeffs)
        })
        .collect();
    let (mut xs, mut ys) = (Vec::with_capacity(x.len() * d), Vec::with_capacity(x.len() * d));
    for (x_coeffs, y_coeffs) in per_product {
        // Every coefficient of the product reads the same `x` side.
        for _ in 1..d {
            xs.push(x_coeffs.clone());
        }
        xs.push(x_coeffs);
        ys.extend(y_coeffs);
    }
    (xs, ys)
}

/// An element of `K` on the wire: its `d` coefficients, serialized one after
/// the other. When `K = F` this is `F`'s own serialization.
fn ser_statistical<F: ProtocolField>(elem: &StatisticalElement<F>) -> LargeFieldSer {
    (0..F::STATISTICAL_DEGREE).flat_map(|k| F::to_bytes_be(&F::statistical_coeff(elem, k))).collect()
}

/// The inverse of [`ser_statistical`]; `None` for a malformed encoding.
fn deser_statistical<F: ProtocolField>(bytes: &[u8]) -> Option<StatisticalElement<F>> {
    if bytes.len() != F::STATISTICAL_DEGREE * F::SER_BYTES {
        return None;
    }
    let coeffs: Vec<FieldElement<F>> = bytes.chunks(F::SER_BYTES).map(F::from_bytes_be).collect::<Result<_, _>>().ok()?;
    Some(F::from_statistical_coeffs(&coeffs))
}

/// Open sharings over `K` from the shares at `points` (`2t+1` of them): every
/// coefficient must lie on a degree-`t` polynomial. Returns the secrets, or
/// `None` when a check fails.
fn open_statistical_sharings<F: ProtocolField>(
    points: Vec<FieldElement<F>>,
    sharings: &[Vec<StatisticalElement<F>>],
    num_faults: usize,
) -> Option<Vec<StatisticalElement<F>>> {
    let (consistent, polys) =
        check_if_all_points_lie_on_degree_x_polynomial(points, split_into_coeff_rows::<F>(sharings), num_faults + 1);
    if !consistent {
        return None;
    }
    let secrets: Vec<FieldElement<F>> = polys?.iter().map(|poly| poly.evaluate(&FieldElement::zero())).collect();
    Some(join_coeffs::<F>(&secrets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::Mersenne31Field;

    type F = Mersenne31Field;
    type E = FieldElement<F>;

    fn rand_row(len: usize) -> Vec<StatisticalElement<F>> {
        (0..len).map(|_| F::from_statistical_coeffs(&[F::rand(), F::rand()])).collect()
    }

    /// The `F` inner products, reassembled, are the inner products over Fp2.
    #[test]
    fn inner_products_over_base_reassemble_to_the_extension_products() {
        let (x, y) = ((0..3).map(|_| rand_row(5)).collect::<Vec<_>>(), (0..3).map(|_| rand_row(5)).collect::<Vec<_>>());
        let (xs, ys) = inner_products_over_base::<F>(&x, &y);
        let base: Vec<E> = xs.iter().zip(&ys).map(|(a, b)| a.iter().zip(b).map(|(u, v)| u * v).sum()).collect();
        let want: Vec<StatisticalElement<F>> =
            x.iter().zip(&y).map(|(a, b)| a.iter().zip(b).map(|(u, v)| u * v).sum()).collect();
        assert_eq!(join_coeffs::<F>(&base), want);
    }

    #[test]
    fn coefficient_rows_and_serialization_round_trip() {
        let rows = vec![rand_row(4), rand_row(4)];
        assert_eq!(join_coeff_rows::<F>(&split_into_coeff_rows::<F>(&rows)), rows);
        let elem = rows[0][0].clone();
        assert_eq!(deser_statistical::<F>(&ser_statistical::<F>(&elem)), Some(elem));
        assert_eq!(deser_statistical::<F>(&[0u8; 3]), None);
    }
}


