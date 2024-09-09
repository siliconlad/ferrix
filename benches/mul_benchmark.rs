use criterion::{Criterion, criterion_main, criterion_group, black_box};
use ferrix::*;

fn mul_matrix_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Matrix");

    // RowVector * Vector
    group.bench_function("RowVector * Vector", |b| {
        b.iter(|| {
            let v1 = RowVector::<i32, 100>::random();
            let v2 = Vector::<i32, 100>::random();
            v1 * v2
        });
    });
    
    // Vector * RowVector
    group.bench_function("Vector * RowVector", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::random();
            let v2 = RowVector::<i32, 100>::random();
            v1 * v2
        });
    });

    // RowVector * Matrix
    group.bench_function("RowVector * Matrix", |b| {
        b.iter(|| {
            let v1 = RowVector::<i32, 100>::random();
            let m1 = Matrix::<i32, 100, 100>::random();
            v1 * m1
        });
    });
    
    // Matrix * Vector
    group.bench_function("Matrix * Vector", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::random();
            let v1 = Vector::<i32, 100>::random();
            m1 * v1
        });
    });

    // Matrix * Matrix
    group.bench_function("Matrix * Matrix", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::random();
            let m2 = Matrix::<i32, 100, 100>::random();
            m1 * m2
        });
    });
}

fn mul_scalar_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Mul Scalar");

    // Vector * Scalar
    group.bench_function("Vector * Scalar", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::random();
            v1 * 1
        });
    });

    // Matrix * Scalar
    group.bench_function("Matrix * Scalar", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::random();
            m1 * 1
        });
    });
}

fn mul_scalar_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Scalar Assign");

    // Vector *= Scalar
    group.bench_function("Vector *= Scalar", |b| {
        b.iter(|| {
            let mut v1 = Vector::<i32, 100>::random();
            v1 *= black_box(1)
        });
    });

    // Matrix *= Scalar
    group.bench_function("Matrix *= Scalar", |b| {
        b.iter(|| {
            let mut m1 = Matrix::<i32, 100, 100>::random();
            m1 *= black_box(1)
        });
    });
}

criterion_group!(benches, mul_matrix_benchmark, mul_scalar_benchmark, mul_scalar_assign_benchmark);
criterion_main!(benches);
