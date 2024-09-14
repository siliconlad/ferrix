use num_traits::{Float, PrimInt, Zero, One};
use std::ops::{Index, IndexMut};
use rand::Rng;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::distributions::uniform::SampleUniform;
use std::fmt;

use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::traits::{DotProduct, IntRandom, FloatRandom};

#[derive(Debug, Clone)]
pub struct RowVector<T, const N: usize> {
    data: [T; N]
}

impl<T: Default, const N: usize> RowVector<T, N> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> RowVector<T, N> {
    #[inline]
    pub fn shape(&self) -> usize {
        N
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn rows(&self) -> usize {
        1
    }

    #[inline]
    pub fn cols(&self) -> usize {
        N
    }
}

impl<T: Default, const N: usize> Default for RowVector<T, N> {
    fn default() -> Self {
        Self { data: std::array::from_fn(|_| T::default()) }
    }
}

impl<T: Copy> RowVector<T, 1> {
    pub fn into(self) -> T {
        self[0]
    }
}

impl<T: Copy, const N: usize> RowVector<T, N> {
    pub fn fill(value: T) -> Self {
        Self { data: [value; N] }
    }
}

impl<T: Copy + Zero, const N: usize> RowVector<T, N> {
    pub fn zeros() -> Self {
        Self::fill(T::zero())
    }
}

impl<T: Copy + One, const N: usize> RowVector<T, N> {
    pub fn ones() -> Self {
        Self::fill(T::one())
    }
}

impl<T: PrimInt, const N: usize> IntRandom for RowVector<T, N>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self { data: std::array::from_fn(|_| rng.gen()) }
    }
}

impl<T: Float + SampleUniform, const N: usize> FloatRandom for RowVector<T, N>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new_inclusive(-T::one(), T::one());
        Self { data: std::array::from_fn(|_| dist.sample(&mut rng)) }
    }
}

impl<T: Copy + Zero, const N: usize> RowVector<T, N> {
    pub fn diag(&self) -> Matrix<T, N, N> {
        let mut m = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            m[(i, i)] = self[i];
        }
        m
    }
}

impl<T, const N: usize> RowVector<T, N> {
    pub fn t(&self) -> VectorView<'_, RowVector<T, N>, T, N, N> {
        VectorView::new(self, 0)
    }

    pub fn t_mut(&mut self) -> VectorViewMut<'_, RowVector<T, N>, T, N, N> {
        VectorViewMut::new(self, 0)
    }
}

impl<T, const N: usize> RowVector<T, N> {
    pub fn view<const M: usize>(&self, start: usize) -> Option<RowVectorView<'_, RowVector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(RowVectorView::new(self, start))
    }

    pub fn view_mut<const M: usize>(&mut self, start: usize) -> Option<RowVectorViewMut<'_, RowVector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(RowVectorViewMut::new(self, start))
    }
}

impl<T: Float, const N: usize> RowVector<T, N> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

////////////////////////////////
//  Equality Implementations  //
////////////////////////////////

// RowVector == RowVector
impl<T: PartialEq, const N: usize> PartialEq for RowVector<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// RowVector == RowVectorView
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize> PartialEq<RowVectorView<'_, V, T, A, N>> for RowVector<T, N> {
    fn eq(&self, other: &RowVectorView<'_, V, T, A, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// RowVector == RowVectorViewMut
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize> PartialEq<RowVectorViewMut<'_, V, T, A, N>> for RowVector<T, N> {
    fn eq(&self, other: &RowVectorViewMut<'_, V, T, A, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<T: Eq, const N: usize> Eq for RowVector<T, N> {}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const N: usize> fmt::Display for RowVector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "RowVector(")?;
        }

        write!(f, "[")?;
        for i in 0..N-1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{}]", self[N-1])?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<T, const N: usize> Index<usize> for RowVector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for RowVector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const N: usize> Index<(usize, usize)> for RowVector<T, N> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.1]
    }
}

impl<T, const N: usize> IndexMut<(usize, usize)> for RowVector<T, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.1]
    }
}

//////////////////////////////////
//  From Trait Implementations  //
//////////////////////////////////

impl<T, const N: usize> From<[T; N]> for RowVector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const N: usize> From<[[T; N]; 1]> for RowVector<T, N> {
    fn from(data: [[T; N]; 1]) -> Self {
        Self { data: std::array::from_fn(|i| data[0][i]) }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize> From<RowVectorView<'_, V, T, A, N>> for RowVector<T, N> {
    fn from(vector: RowVectorView<'_, V, T, A, N>) -> Self {
        Self { data: std::array::from_fn(|i| vector[i]) }
    }
}

impl<V: IndexMut<usize, Output = T>, T: Copy, const A: usize, const N: usize> From<RowVectorViewMut<'_, V, T, A, N>> for RowVector<T, N> {
    fn from(vector: RowVectorViewMut<'_, V, T, A, N>) -> Self {
        Self { data: std::array::from_fn(|i| vector[i]) }
    }
}

impl<T: Copy, const N: usize> From<Matrix<T, 1, N>> for RowVector<T, N> {
    fn from(matrix: Matrix<T, 1, N>) -> Self {
        Self { data: std::array::from_fn(|i| matrix[(0, i)]) }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixView<'_, T, A, B, 1, N>> for RowVector<T, N> {
    fn from(matrix: MatrixView<'_, T, A, B, 1, N>) -> Self {
        Self { data: std::array::from_fn(|i| matrix[(0, i)]) }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixViewMut<'_, T, A, B, 1, N>> for RowVector<T, N> {
    fn from(matrix: MatrixViewMut<'_, T, A, B, 1, N>) -> Self {
        Self { data: std::array::from_fn(|i| matrix[(0, i)]) }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixTransposeView<'_, T, A, B, 1, N>> for RowVector<T, N> {
    fn from(matrix: MatrixTransposeView<'_, T, A, B, 1, N>) -> Self {
        Self { data: std::array::from_fn(|i| matrix[(0, i)]) }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixTransposeViewMut<'_, T, A, B, 1, N>> for RowVector<T, N> {
    fn from(matrix: MatrixTransposeViewMut<'_, T, A, B, 1, N>) -> Self {
        Self { data: std::array::from_fn(|i| matrix[(0, i)]) }
    }
}
