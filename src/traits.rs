use funty::Numeric;
use std::ops::{Index, IndexMut};

pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}

pub trait MatMul<V> {
    type Output;
    fn matmul(self, other: V) -> Self::Output;
}

pub trait MatrixRead<T: Numeric, const R: usize, const C: usize>:
    Index<(usize, usize), Output = T>
{
}
pub trait MatrixWrite<T: Numeric, const R: usize, const C: usize>:
    IndexMut<(usize, usize), Output = T>
{
}
