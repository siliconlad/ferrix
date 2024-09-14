use criterion::{Criterion, criterion_main, criterion_group};
use ferrix::*;

fn dot_product_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Dot Product");

    // Vector dot Vector
    group.bench_function("Vector.Vector", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Vector::<i32, 100>::random()),
            |(v1, v2)| v1.dot(v2)
        );
    });

    // Vector dot Matrix
    group.bench_function("Vector.Matrix", |b| {
        b.iter_with_setup(
            || (Vector::<i32, 100>::random(), Matrix::<i32, 100, 1>::random()),
            |(v1, m1)| v1.dot(m1)
        );
    });

    // RowVector dot RowVector
    group.bench_function("RowVector.RowVector", |b| {
        b.iter_with_setup(
            || (RowVector::<i32, 100>::random(), RowVector::<i32, 100>::random()),
            |(v1, v2)| v1.dot(v2)
        );
    });
    
    // RowVector dot Matrix
    group.bench_function("RowVector.Matrix", |b| {
        b.iter_with_setup(
            || (RowVector::<i32, 100>::random(), Matrix::<i32, 1, 100>::random()),
            |(v1, m1)| v1.dot(m1)
        );
    });
}

criterion_group!(benches, dot_product_benchmark);
criterion_main!(benches);
