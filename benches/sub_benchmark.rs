use criterion::{Criterion, criterion_main, criterion_group, black_box};
use ferrix::*;

fn sub_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Sub");

    // Vector - Scalar
    group.bench_function("Vector - Scalar", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::random();
            v1 - 1
        });
    });

    // Vector - Vector
    group.bench_function("Vector - Vector", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::random();
            let v2 = Vector::<i32, 100>::random();
            v1 - v2
        });
    });
    
    // Matrix - Scalar
    group.bench_function("Matrix - Scalar", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::random();
            m1 - 1
        });
    });
    
    // Matrix - Matrix
    group.bench_function("Matrix - Matrix", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::random();
            let m2 = Matrix::<i32, 100, 100>::random();
            m1 - m2
        });
    });
}

fn sub_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sub Assign");

    // Vector -= Scalar
    group.bench_function("Vector -= Scalar", |b| {
        b.iter(|| {
            let mut v1 = Vector::<i32, 100>::random();
            v1 -= black_box(1)
        });
    });

    // Vector -= Vector
    group.bench_function("Vector -= Vector", |b| {
        b.iter(|| {
            let mut v1 = Vector::<i32, 100>::random();
            let v2 = Vector::<i32, 100>::random();
            v1 -= v2
        });
    });

    // Matrix -= Scalar
    group.bench_function("Matrix -= Scalar", |b| {
        b.iter(|| {
            let mut m1 = Matrix::<i32, 100, 100>::random();
            m1 -= black_box(1)
        });
    });
    
    // Matrix -= Matrix
    group.bench_function("Matrix -= Matrix", |b| {
        b.iter(|| {
            let mut m1 = Matrix::<i32, 100, 100>::random();
            let m2 = Matrix::<i32, 100, 100>::random();
            m1 -= m2
        });
    });
}

criterion_group!(benches, sub_benchmark, sub_assign_benchmark);
criterion_main!(benches);


