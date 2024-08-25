pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}

pub trait MatMul<V> {
    type Output;
    fn matmul(self, other: V) -> Self::Output;
}
