pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}

pub trait IntRandom {
    fn random() -> Self;
}

pub trait FloatRandom {
    fn random() -> Self;
}
