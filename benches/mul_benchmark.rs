use criterion::{Criterion, criterion_main, criterion_group, black_box};
use ferrix::*;

fn mul_matrix_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Matrix");

    // RowVector * Vector
    group.bench_function("RowVector * Vector", |b| {
        b.iter_with_setup(
            || (RowVector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(v1, v2)| v1 * v2
        );
    });
    
    // Vector * RowVector
    group.bench_function("Vector * RowVector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), RowVector::<i32, 100>::random()),
            |(v1, v2)| v1 * v2
        );
    });

    // RowVector * Matrix
    group.bench_function("RowVector * Matrix", |b| {
        b.iter_with_setup(
            || (RowVector::<i32, 100>::random(), Matrix::<i32, 100, 100>::random()),
            |(v1, m1)| v1 * m1
        );
    });
    
    // Matrix * Vector
    group.bench_function("Matrix * Vector", |b| {
        b.iter_with_setup(
            || (Matrix::<i32, 100, 100>::random(), Vector::<i32, 100>::random()),
            |(m1, v1)| m1 * v1
        );
    });

    // Matrix * Matrix
    group.bench_function("Matrix * Matrix", |b| {
        b.iter_with_setup(
            || (Matrix::<i32, 100, 100>::random(), Matrix::<i32, 100, 100>::random()),
            |(m1, m2)| m1 * m2
        );
    });
}

fn mul_scalar_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Mul Scalar");

    // Vector * Scalar
    group.bench_function("Vector * Scalar", |b| {
        b.iter_with_setup(
            || Vector::<i32, 100>::random(),
            |v1| v1 * black_box(2)
        );
    });

    // Matrix * Scalar
    group.bench_function("Matrix * Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |m1| m1 * black_box(2)
        );
    });
}

fn mul_scalar_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Scalar Assign");

    // Vector *= Scalar
    group.bench_function("Vector *= Scalar", |b| {
        b.iter_with_setup(
            || Vector::<i32, 100>::random(),
            |mut v1| v1 *= black_box(2)
        );
    });

    // Matrix *= Scalar
    group.bench_function("Matrix *= Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |mut m1| m1 *= black_box(2)
        );
    });
}

criterion_group!(benches, mul_matrix_benchmark, mul_scalar_benchmark, mul_scalar_assign_benchmark);
criterion_main!(benches);
