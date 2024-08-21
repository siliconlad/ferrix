use funty::Numeric;
use std::ops::Index;

pub trait VectorOps<T: Numeric, const N: usize>: Index<usize, Output = T> {}

pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}
