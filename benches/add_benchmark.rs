use criterion::{Criterion, criterion_main, criterion_group, black_box};
use ferrix::*;

fn add_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Add");

    // Vector + Scalar
    group.bench_function("Vector + Scalar", |b| {
        b.iter_with_setup(
            || Vector::<i32, 100>::random(),
            |v| v + black_box(1)
        );
    });

    // Vector + Vector
    group.bench_function("Vector + Vector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(v1, v2)| v1 + v2
        );
    });
    
    // Matrix + Scalar
    group.bench_function("Matrix + Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |m| m + black_box(1)
        );
    });
    
    // Matrix + Matrix
    group.bench_function("Matrix + Matrix", |b| {
        b.iter_with_setup(
            || (Matrix::<i32, 100, 100>::random(), Matrix::<i32, 100, 100>::random()),
            |(m1, m2)| m1 + m2
        );
    });
}

fn add_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add Assign");

    // Vector += Scalar
    group.bench_function("Vector += Scalar", |b| {
        b.iter_with_setup(
            || Vector::<i32, 100>::random(),
            |mut v| v += black_box(1)
        );
    });

    // Vector += Vector
    group.bench_function("Vector += Vector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(mut v1, v2)| v1 += v2
        );
    });

    // Matrix += Scalar
    group.bench_function("Matrix += Scalar", |b| {
        b.iter_with_setup(
            || Matrix::<i32, 100, 100>::random(),
            |mut m| m += black_box(1)
        );
    });
    
    // Matrix += Matrix
    group.bench_function("Matrix += Matrix", |b| {
        b.iter_with_setup(
            || (Matrix::<i32, 100, 100>::random(), Matrix::<i32, 100, 100>::random()),
            |(mut m1, m2)| m1 += m2
        );
    });
}

criterion_group!(benches, add_benchmark, add_assign_benchmark);
criterion_main!(benches);

