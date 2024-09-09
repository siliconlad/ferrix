use criterion::{Criterion, criterion_main, criterion_group, black_box};
use ferrix::*;

fn div_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Div");

    // Vector / Scalar
    group.bench_function("Vector / Scalar", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::new([1; 100]);
            v1 / 1
        });
    });

    // Matrix / Scalar
    group.bench_function("Matrix / Scalar", |b| {
        b.iter(|| {
            let m1 = Matrix::<i32, 100, 100>::new([[1; 100]; 100]);
            m1 / 1
        });
    });
}

fn div_assign_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Div Assign");

    // Vector /= Scalar
    group.bench_function("Vector /= Scalar", |b| {
        b.iter(|| {
            let mut v1 = Vector::<i32, 100>::new([1; 100]);
            v1 /= black_box(1)
        });
    });

    // Matrix /= Scalar
    group.bench_function("Matrix /= Scalar", |b| {
        b.iter(|| {
            let mut m1 = Matrix::<i32, 100, 100>::new([[1; 100]; 100]);
            m1 /= black_box(1)
        });
    });
}

criterion_group!(benches, div_benchmark, div_assign_benchmark);
criterion_main!(benches);



