use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use rayon::prelude::*;

/// Below this total work size, sequential iteration beats rayon's dispatch overhead.
const PARALLEL_THRESHOLD: usize = 4096;
/// Minimum multiplications per rayon task. At ~10 ns/mul this is ~40 μs of work,
/// well above rayon's per-task dispatch/steal cost (~1–10 μs), so parallelism pays.
const PARALLEL_CHUNK: usize = 4096;

/// Element-wise multiplication of two equal-length slices.
/// Parallelizes over rayon when the slice is long enough to amortize dispatch cost.
pub fn elem_mul<F>(xs: &[FieldElement<F>], ys: &[FieldElement<F>]) -> Vec<FieldElement<F>>
where
    F: IsField,
    FieldElement<F>: Clone + Send + Sync,
{
    if xs.len() >= PARALLEL_THRESHOLD {
        xs.par_iter()
            .zip(ys.par_iter())
            .with_min_len(PARALLEL_CHUNK)
            .map(|(a, b)| a.clone() * b)
            .collect()
    } else {
        xs.iter()
            .zip(ys.iter())
            .map(|(a, b)| a.clone() * b)
            .collect()
    }
}

/// Compute one inner product per matching pair of inner vectors.
/// Parallelizes the outer dimension when `outer * inner_len` exceeds the threshold,
/// with per-task chunking sized to keep ~PARALLEL_CHUNK field muls per rayon task.
pub fn inner_products<F>(
    xs: &[Vec<FieldElement<F>>],
    ys: &[Vec<FieldElement<F>>],
) -> Vec<FieldElement<F>>
where
    F: IsField,
    FieldElement<F>: Clone + Send + Sync,
{
    let inner_len = xs.first().map(|v| v.len()).unwrap_or(0);
    let total_work = xs.len().saturating_mul(inner_len);
    if inner_len == 0 || total_work < PARALLEL_THRESHOLD {
        return xs.iter()
            .zip(ys.iter())
            .map(|(x_vec, y_vec)| dot(x_vec, y_vec))
            .collect();
    }
    let min_outer = (PARALLEL_CHUNK / inner_len).max(1);
    xs.par_iter()
        .zip(ys.par_iter())
        .with_min_len(min_outer)
        .map(|(x_vec, y_vec)| dot(x_vec, y_vec))
        .collect()
}

/// Sequential inner product of two equal-length slices.
pub fn dot<F>(xs: &[FieldElement<F>], ys: &[FieldElement<F>]) -> FieldElement<F>
where
    F: IsField,
    FieldElement<F>: Clone,
{
    xs.iter()
        .zip(ys.iter())
        .map(|(a, b)| a.clone() * b)
        .fold(FieldElement::zero(), |acc, x| acc + x)
}
