//! Mersenne-127 against Mersenne-61, both base fields: the single operations
//! the engine calls per element (`mul`, `square`, `inv`, `sqrt` — the last two
//! once per random bit) and the GEMM on the shapes the protocol issues.
//!
//! GEMM runs twice for M61: `scalar` (`matrix_matrix_multiply_cpu`, the
//! kernel M127 also runs) and `dispatch` (`matrix_matrix_multiply`, which takes
//! M61's AVX2 path when the CPU has it). M127 has no AVX2 path, so its two
//! numbers are the same kernel and only `scalar` is run.
//!
//!     cargo bench -p fields --bench mersenne_127

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fields::{matrix_matrix_multiply, matrix_matrix_multiply_cpu, Mersenne127Field, Mersenne61Field, ProtocolField};
use lambdaworks_math::field::element::FieldElement;

// (rows, cols, batch): the bench shape, the lin_mult party-eval shape and the
// interpolation shape, as in `simd_gemm`.
const SHAPES: [(usize, usize, usize); 3] = [(64, 128, 1024), (16, 11, 8192), (6, 6, 65536)];

fn rand_mat<F: ProtocolField>(rows: usize, cols: usize) -> Vec<Vec<FieldElement<F>>> {
    (0..rows).map(|_| (0..cols).map(|_| F::rand()).collect()).collect()
}

fn element_ops<F: ProtocolField>(c: &mut Criterion, name: &str) {
    let mut g = c.benchmark_group(format!("ElementOps/{name}"));
    let (x, y) = (F::rand(), F::rand());
    let square = &x * &x;
    g.bench_function("mul", |b| b.iter(|| black_box(&x) * black_box(&y)));
    g.bench_function("square", |b| b.iter(|| black_box(&x).square()));
    g.bench_function("inv", |b| b.iter(|| black_box(&x).inv().unwrap()));
    g.bench_function("sqrt", |b| b.iter(|| F::sqrt(black_box(&square)).unwrap()));
    g.finish();
}

fn gemm<F: ProtocolField>(c: &mut Criterion, name: &str, with_dispatch: bool) {
    let mut g = c.benchmark_group(format!("GEMM/{name}"));
    g.sample_size(20);
    for (rows, cols, batch) in SHAPES {
        let m = rand_mat::<F>(rows, cols);
        let v = rand_mat::<F>(batch, cols);
        let id = format!("{rows}x{cols}_b{batch}");
        g.throughput(Throughput::Elements((rows * cols * batch) as u64));
        g.bench_with_input(BenchmarkId::new("scalar", &id), &(&m, &v), |b, (m, v)| {
            b.iter(|| matrix_matrix_multiply_cpu(black_box(m), black_box(v), true))
        });
        if with_dispatch {
            g.bench_with_input(BenchmarkId::new("dispatch", &id), &(&m, &v), |b, (m, v)| {
                b.iter(|| matrix_matrix_multiply(black_box(m), black_box(v), true))
            });
        }
    }
    g.finish();
}

fn benches(c: &mut Criterion) {
    element_ops::<Mersenne61Field>(c, "M61");
    element_ops::<Mersenne127Field>(c, "M127");
    gemm::<Mersenne61Field>(c, "M61", true);
    gemm::<Mersenne127Field>(c, "M127", false);
}

criterion_group!(mersenne_127, benches);
criterion_main!(mersenne_127);
