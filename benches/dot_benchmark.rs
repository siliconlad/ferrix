use criterion::{Criterion, criterion_main, criterion_group};
use ferrix::*;

fn dot_product_benchmark(c: &mut Criterion) { 
    let mut group = c.benchmark_group("Dot Product");

    // Vector dot Vector
    group.bench_function("Vector.Vector", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::new([1; 100]);
            let v2 = Vector::<i32, 100>::new([1; 100]);
            v1.dot(v2)
        });
    });

    // Vector dot Matrix
    group.bench_function("Vector.Matrix", |b| {
        b.iter(|| {
            let v1 = Vector::<i32, 100>::new([1; 100]);
            let m1 = Matrix::<i32, 100, 1>::new([[1]; 100]);
            v1.dot(m1)
        });
    });

    // RowVector dot RowVector
    group.bench_function("RowVector.RowVector", |b| {
        b.iter(|| {
            let v1 = RowVector::<i32, 100>::new([1; 100]);
            let v2 = RowVector::<i32, 100>::new([1; 100]);
            v1.dot(v2)
        });
    });
    
    // RowVector dot Matrix
    group.bench_function("RowVector.Matrix", |b| {
        b.iter(|| {
            let v1 = RowVector::<i32, 100>::new([1; 100]);
            let m1 = Matrix::<i32, 1, 100>::new([[1; 100]]);
            v1.dot(m1)
        });
    });
}

criterion_group!(benches, dot_product_benchmark);
criterion_main!(benches);
