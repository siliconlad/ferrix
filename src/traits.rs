pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}
