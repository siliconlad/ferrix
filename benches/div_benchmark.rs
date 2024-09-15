use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferrix::*;

fn div_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Div");

    // Vector / Scalar
    group.bench_function("Vector / Scalar", |b| {
        b.iter_with_setup(|| Vector::<i32, 100>::random(), |v| v / black_box(2));
    });

    // Matrix / Scalar
    group.bench_function("Matrix / Scalar", |b| {
        b.iter_with_setup(|| Matrix::<i32, 100, 100>::random(), |m| m / black_box(2));
    });
}

fn div_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Div Assign");

    // Vector /= Scalar
    group.bench_function("Vector /= Scalar", |b| {
        b.iter_with_setup(|| Vector::<i32, 100>::random(), |mut v| v /= black_box(2));
    });

    // Matrix /= Scalar
    group.bench_function("Matrix /= Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |mut m| m /= black_box(2),
        );
    });
}

criterion_group!(benches, div_benchmark, div_assign_benchmark);
criterion_main!(benches);
