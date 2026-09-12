//! Scalar vs AVX2 GEMM on the shapes the protocol actually issues, for the
//! M61 (base, Fp4) and M31 (base, Fp4, Fp8) fields. Both paths use rayon; run
//! with `RAYON_NUM_THREADS=1` for the per-core kernel speedup:
//!
//!     RAYON_NUM_THREADS=1 cargo bench -p fields --bench simd_gemm
//!
//! Measured numbers and their interpretation live in `docs/simd-m61-avx2.md`.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fields::mersenne_31::{Degree4ExtensionField as M31Fp4, Degree8ExtensionField as M31Fp8, Mersenne31Field};
use fields::mersenne_61::{Mersenne61Degree4ExtensionField, Mersenne61Field};
use fields::simd::{gemm_with, Backend, SimdField};
use lambdaworks_math::field::element::FieldElement;

fn rand_mat<F: SimdField>(r: usize, c: usize) -> Vec<Vec<FieldElement<F>>> {
    (0..r).map(|_| (0..c).map(|_| F::rand_elem()).collect()).collect()
}

// (rows, cols, batch): bench shape, lin_mult party-eval shape, interpolation shape.
const SHAPES: [(usize, usize, usize); 3] = [(64, 128, 1024), (16, 11, 8192), (6, 6, 65536)];

fn run<F>(c: &mut Criterion, name: &str)
where
    F: SimdField,
    FieldElement<F>: Clone + Send + Sync,
{
    let mut g = c.benchmark_group(name);
    g.sample_size(20);
    for (r, cols, k) in SHAPES {
        let m = rand_mat::<F>(r, cols);
        let v = rand_mat::<F>(k, cols);
        let id = format!("{r}x{cols}_b{k}");
        g.throughput(Throughput::Elements((r * cols * k) as u64));
        g.bench_with_input(BenchmarkId::new("scalar", &id), &(&m, &v), |b, (m, v)| {
            b.iter(|| gemm_with(Backend::Scalar, black_box(m), black_box(v), true))
        });
        g.bench_with_input(BenchmarkId::new("avx2", &id), &(&m, &v), |b, (m, v)| {
            b.iter(|| gemm_with(Backend::Avx2, black_box(m), black_box(v), true))
        });
    }
    g.finish();
}

fn bench(c: &mut Criterion) {
    run::<Mersenne61Field>(c, "m61_base");
    run::<Mersenne61Degree4ExtensionField>(c, "m61_fp4");
    run::<Mersenne31Field>(c, "m31_base");
    run::<M31Fp4>(c, "m31_fp4");
    run::<M31Fp8>(c, "m31_fp8");
}

criterion_group!(benches, bench);
criterion_main!(benches);
