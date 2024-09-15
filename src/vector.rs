use num_traits::{Float, One, PrimInt, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use std::default::Default;
use std::fmt;
use std::ops::{Index, IndexMut};

use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::traits::{DotProduct, FloatRandom, IntRandom};
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;

#[derive(Debug, Clone)]
pub struct Vector<T, const N: usize> {
    data: [T; N],
}
pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;

impl<T: Default, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self {
            data: std::array::from_fn(|_| T::default()),
        }
    }
}

impl<T: Default, const N: usize> Vector<T, N> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> Vector<T, N> {
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
        N
    }

    #[inline]
    pub fn cols(&self) -> usize {
        1
    }
}

impl<T: PrimInt, const N: usize> IntRandom for Vector<T, N>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            data: std::array::from_fn(|_| rng.gen()),
        }
    }
}

impl<T: Float + SampleUniform, const N: usize> FloatRandom for Vector<T, N>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new_inclusive(-T::one(), T::one());
        Self {
            data: std::array::from_fn(|_| dist.sample(&mut rng)),
        }
    }
}

impl<T: Copy> Vector<T, 1> {
    pub fn into(self) -> T {
        self[0]
    }
}

impl<T: Copy, const N: usize> Vector<T, N> {
    pub fn fill(value: T) -> Self {
        Self {
            data: std::array::from_fn(|_| value),
        }
    }
}

impl<T: Copy + Zero, const N: usize> Vector<T, N> {
    pub fn zeros() -> Self {
        Self::fill(T::zero())
    }
}

impl<T: Copy + One, const N: usize> Vector<T, N> {
    pub fn ones() -> Self {
        Self::fill(T::one())
    }
}

impl<T: Copy + Zero, const N: usize> Vector<T, N> {
    pub fn diag(&self) -> Matrix<T, N, N> {
        let mut m = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            m[(i, i)] = self[i];
        }
        m
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn t(&self) -> RowVectorView<'_, Vector<T, N>, T, N, N> {
        RowVectorView::new(self, 0)
    }

    pub fn t_mut(&mut self) -> RowVectorViewMut<'_, Vector<T, N>, T, N, N> {
        RowVectorViewMut::new(self, 0)
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn view<const M: usize>(
        &self,
        start: usize,
    ) -> Option<VectorView<'_, Vector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorView::new(self, start))
    }

    pub fn view_mut<const M: usize>(
        &mut self,
        start: usize,
    ) -> Option<VectorViewMut<'_, Vector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorViewMut::new(self, start))
    }
}

impl<T: Float, const N: usize> Vector<T, N> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

////////////////////////////////
//  Equality Implementations  //
////////////////////////////////

// Vector == Vector
impl<T: PartialEq, const N: usize> PartialEq for Vector<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// Vector == VectorView
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<VectorView<'_, V, T, A, N>> for Vector<T, N>
{
    fn eq(&self, other: &VectorView<'_, V, T, A, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// Vector == VectorViewMut
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<VectorViewMut<'_, V, T, A, N>> for Vector<T, N>
{
    fn eq(&self, other: &VectorViewMut<'_, V, T, A, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<T: Eq, const N: usize> Eq for Vector<T, N> {}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const N: usize> fmt::Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "Vector(")?;
        }

        write!(f, "[")?;
        for i in 0..N - 1 {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "       ")?;
                }
            }
            writeln!(f, "{}", self[i])?;
        }
        if f.alternate() {
            write!(f, "       ")?;
        }

        write!(f, " {}]", self[N - 1])?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const N: usize> Index<(usize, usize)> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<T, const N: usize> IndexMut<(usize, usize)> for Vector<T, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0]
    }
}

//////////////////////////////////
//  From Trait Implementations  //
//////////////////////////////////

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const N: usize> From<[[T; 1]; N]> for Vector<T, N> {
    fn from(data: [[T; 1]; N]) -> Self {
        Self {
            data: std::array::from_fn(|i| data[i][0]),
        }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<VectorView<'_, V, T, A, N>> for Vector<T, N>
{
    fn from(vector: VectorView<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| vector[i]),
        }
    }
}

impl<V: IndexMut<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<VectorViewMut<'_, V, T, A, N>> for Vector<T, N>
{
    fn from(vector: VectorViewMut<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| vector[i]),
        }
    }
}

impl<T: Copy, const N: usize> From<Matrix<T, N, 1>> for Vector<T, N> {
    fn from(matrix: Matrix<T, N, 1>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(i, 0)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixView<'_, T, A, B, N, 1>>
    for Vector<T, N>
{
    fn from(matrix: MatrixView<'_, T, A, B, N, 1>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(i, 0)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixViewMut<'_, T, A, B, N, 1>>
    for Vector<T, N>
{
    fn from(matrix: MatrixViewMut<'_, T, A, B, N, 1>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(i, 0)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize>
    From<MatrixTransposeView<'_, T, A, B, N, 1>> for Vector<T, N>
{
    fn from(matrix: MatrixTransposeView<'_, T, A, B, N, 1>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(i, 0)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize>
    From<MatrixTransposeViewMut<'_, T, A, B, N, 1>> for Vector<T, N>
{
    fn from(matrix: MatrixTransposeViewMut<'_, T, A, B, N, 1>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(i, 0)]),
        }
    }
}
