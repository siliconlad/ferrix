use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferrix::*;

fn sub_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sub");

    // Vector - Scalar
    group.bench_function("Vector - Scalar", |b| {
        b.iter_with_setup(|| Vector::<i32, 100>::random(), |v1| v1 - black_box(1));
    });

    // Vector - Vector
    group.bench_function("Vector - Vector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(v1, v2)| v1 - v2,
        );
    });

    // Matrix - Scalar
    group.bench_function("Matrix - Scalar", |b| {
        b.iter_with_setup(|| Matrix::<i32, 100, 100>::random(), |m1| m1 - black_box(1));
    });

    // Matrix - Matrix
    group.bench_function("Matrix - Matrix", |b| {
        b.iter_with_setup(
            || {
                (
                    Matrix::<i32, 100, 100>::random(),
                    Matrix::<i32, 100, 100>::random(),
                )
            },
            |(m1, m2)| m1 - m2,
        );
    });
}

fn sub_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sub Assign");

    // Vector -= Scalar
    group.bench_function("Vector -= Scalar", |b| {
        b.iter_with_setup(|| Vector::<i32, 100>::random(), |mut v1| v1 -= black_box(1));
    });

    // Vector -= Vector
    group.bench_function("Vector -= Vector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(mut v1, v2)| v1 -= v2,
        );
    });

    // Matrix -= Scalar
    group.bench_function("Matrix -= Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |mut m1| m1 -= black_box(1),
        );
    });

    // Matrix -= Matrix
    group.bench_function("Matrix -= Matrix", |b| {
        b.iter_with_setup(
            || {
                (
                    Matrix::<i32, 100, 100>::random(),
                    Matrix::<i32, 100, 100>::random(),
                )
            },
            |(mut m1, m2)| m1 -= m2,
        );
    });
}

criterion_group!(benches, sub_benchmark, sub_assign_benchmark);
criterion_main!(benches);
