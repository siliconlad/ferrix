use num_traits::{Float, One, PrimInt, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
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

/// A static row vector type.
#[derive(Debug, Clone)]
pub struct RowVector<T, const N: usize> {
    data: [T; N],
}

impl<T: Default, const N: usize> RowVector<T, N> {
    /// Creates a new [`RowVector`] with default values.
    ///
    /// This method initializes a new [`RowVector`] of size N, where each element is set to its default value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec: RowVector<f64, 3> = RowVector::new();
    /// assert_eq!(vec, RowVector::from([0.0, 0.0, 0.0]));
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> RowVector<T, N> {
    /// Returns the shape of the [`RowVector`].
    ///
    /// The shape is always equal to `N`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec: RowVector<f64, 5> = RowVector::new();
    /// assert_eq!(vec.shape(), 5);
    /// ```
    #[inline]
    pub fn shape(&self) -> usize {
        N
    }

    /// Returns the total number of elements in the [`RowVector`].
    /// 
    /// The total number of elements is always equal to `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec: RowVector<f64, 5> = RowVector::new();
    /// assert_eq!(vec.capacity(), 5);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of rows in the [`RowVector`].
    ///
    /// The number of rows is always `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec: RowVector<f64, 5> = RowVector::new();
    /// assert_eq!(vec.rows(), 1);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        1
    }

    /// Returns the number of columns in the [`RowVector`].
    ///
    /// The number of columns is always equal to `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec: RowVector<f64, 5> = RowVector::new();
    /// assert_eq!(vec.cols(), 5);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        N
    }
}

impl<T: Default, const N: usize> Default for RowVector<T, N> {
    fn default() -> Self {
        Self {
            data: std::array::from_fn(|_| T::default()),
        }
    }
}

impl<T: Copy> RowVector<T, 1> {
    /// Converts a 1-dimensional [`RowVector`] into its scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([42]);
    /// assert_eq!(vec.into(), 42);
    /// ```
    pub fn into(self) -> T {
        self[0]
    }
}

impl<T: Copy, const N: usize> RowVector<T, N> {
    /// Creates a new [`RowVector`] filled with a specified value.
    ///
    /// This method initializes a new [`RowVector`] of size N, where each element is set to the provided value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::<i32, 3>::fill(42);
    /// assert_eq!(vec, RowVector::from([42, 42, 42]));
    /// ```
    pub fn fill(value: T) -> Self {
        Self { data: [value; N] }
    }
}

impl<T: Copy + Zero, const N: usize> RowVector<T, N> {
    /// Creates a new [`RowVector`] filled with zeros.
    ///
    /// This method initializes a new [`RowVector`] of size N, where each element is set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::<f64, 3>::zeros();
    /// assert_eq!(vec, RowVector::from([0.0, 0.0, 0.0]));
    /// ```
    pub fn zeros() -> Self {
        Self::fill(T::zero())
    }
}

impl<T: Copy + One, const N: usize> RowVector<T, N> {
    /// Creates a new [`RowVector`] filled with ones.
    ///
    /// This method initializes a new [`RowVector`] of size N, where each element is set to one.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::<f64, 3>::ones();
    /// assert_eq!(vec, RowVector::from([1.0, 1.0, 1.0]));
    /// ```
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
        Self {
            data: std::array::from_fn(|_| rng.gen()),
        }
    }
}

impl<T: Float + SampleUniform, const N: usize> FloatRandom for RowVector<T, N>
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

impl<T: Copy + Zero, const N: usize> RowVector<T, N> {
    /// Creates a diagonal matrix from the [`RowVector`].
    ///
    /// This method returns a new NxN [`Matrix`] where the diagonal elements are set to the values of the [`RowVector`],
    /// and all other elements are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::{RowVector, Matrix};
    ///
    /// let vec = RowVector::from([1, 2, 3]);
    /// let mat = vec.diag();
    /// assert_eq!(mat, Matrix::from([[1, 0, 0], [0, 2, 0], [0, 0, 3]]));
    /// ```
    pub fn diag(&self) -> Matrix<T, N, N> {
        let mut m = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            m[(i, i)] = self[i];
        }
        m
    }
}

impl<T, const N: usize> RowVector<T, N> {
    /// Returns a transposed view of the [`RowVector`].
    ///
    /// This method returns a [`VectorView`], which is a read-only view of the [`RowVector`] as a column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::{Vector, RowVector};
    ///
    /// let vec = RowVector::from([1, 2, 3]);
    /// let col_view = vec.t();
    /// assert_eq!(col_view, Vector::from([1, 2, 3]));
    /// ```
    pub fn t(&self) -> VectorView<'_, RowVector<T, N>, T, N, N> {
        VectorView::new(self, 0)
    }

    /// Returns a mutable transposed view of the [`RowVector`].
    ///
    /// This method returns a [`VectorViewMut`], which is a mutable view of the [`RowVector`] as a column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let mut vec = RowVector::from([1, 2, 3]);
    /// let mut col_view = vec.t_mut();
    /// col_view[1] = 5;
    /// assert_eq!(vec, RowVector::from([1, 5, 3]));
    /// ```
    pub fn t_mut(&mut self) -> VectorViewMut<'_, RowVector<T, N>, T, N, N> {
        VectorViewMut::new(self, 0)
    }
}

impl<T, const N: usize> RowVector<T, N> {
    /// Returns a view of [`RowVector`].
    ///
    /// This method returns a [`RowVectorView`] of size `M` starting from the given index.
    /// Returns `None` if the requested view is out of bounds or if `M` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view::<3>(1).unwrap();
    /// assert_eq!(view, RowVector::from([2, 3, 4]));
    /// ```
    pub fn view<const M: usize>(
        &self,
        start: usize,
    ) -> Option<RowVectorView<'_, RowVector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(RowVectorView::new(self, start))
    }

    /// Returns a mutable view of [`RowVector`].
    ///
    /// This method returns a [`RowVectorViewMut`] of size `M` starting from the given index.
    /// Returns `None` if the requested view is out of bounds or if `M` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let mut vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let mut view = vec.view_mut::<3>(1).unwrap();
    /// view[1] = 10;
    /// assert_eq!(vec, RowVector::from([1, 2, 10, 4, 5]));
    /// ```
    pub fn view_mut<const M: usize>(
        &mut self,
        start: usize,
    ) -> Option<RowVectorViewMut<'_, RowVector<T, N>, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(RowVectorViewMut::new(self, start))
    }
}

impl<T: Float, const N: usize> RowVector<T, N> {
    /// Calculates the magnitude of the [`RowVector`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([3.0, 4.0]);
    /// assert_eq!(vec.magnitude(), 5.0);
    /// ```
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
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<RowVectorView<'_, V, T, A, N>> for RowVector<T, N>
{
    fn eq(&self, other: &RowVectorView<'_, V, T, A, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// RowVector == RowVectorViewMut
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<RowVectorViewMut<'_, V, T, A, N>> for RowVector<T, N>
{
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
        for i in 0..N - 1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{}]", self[N - 1])?;

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
        Self {
            data: std::array::from_fn(|i| data[0][i]),
        }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<RowVectorView<'_, V, T, A, N>> for RowVector<T, N>
{
    fn from(vector: RowVectorView<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| vector[i]),
        }
    }
}

impl<V: IndexMut<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<RowVectorViewMut<'_, V, T, A, N>> for RowVector<T, N>
{
    fn from(vector: RowVectorViewMut<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| vector[i]),
        }
    }
}

impl<T: Copy, const N: usize> From<Matrix<T, 1, N>> for RowVector<T, N> {
    fn from(matrix: Matrix<T, 1, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(0, i)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixView<'_, T, A, B, 1, N>>
    for RowVector<T, N>
{
    fn from(matrix: MatrixView<'_, T, A, B, 1, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(0, i)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize> From<MatrixViewMut<'_, T, A, B, 1, N>>
    for RowVector<T, N>
{
    fn from(matrix: MatrixViewMut<'_, T, A, B, 1, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(0, i)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize>
    From<MatrixTransposeView<'_, T, A, B, 1, N>> for RowVector<T, N>
{
    fn from(matrix: MatrixTransposeView<'_, T, A, B, 1, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(0, i)]),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const N: usize>
    From<MatrixTransposeViewMut<'_, T, A, B, 1, N>> for RowVector<T, N>
{
    fn from(matrix: MatrixTransposeViewMut<'_, T, A, B, 1, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| matrix[(0, i)]),
        }
    }
}
