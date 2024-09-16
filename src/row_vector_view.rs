use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;

use crate::row_vector::RowVector;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::traits::DotProduct;
use crate::vector_view::VectorView;

/// A row vector view of a [`Vector`](crate::vector::Vector) (transposed view) or a [`RowVector`].
#[derive(Debug)]
pub struct RowVectorView<'a, V, T, const N: usize, const M: usize> {
    data: &'a V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> RowVectorView<'a, V, T, N, M> {
    pub(super) fn new(data: &'a V, start: usize) -> Self {
        Self {
            data,
            start,
            _phantom: PhantomData,
        }
    }

    /// Returns the shape of the [`RowVectorView`].
    ///
    /// The shape is always equal to `M`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view::<3>(1).unwrap();
    /// assert_eq!(view.shape(), 3);
    /// ```
    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    /// Returns the total number of elements in the [`RowVectorView`].
    /// 
    /// The total number of elements is always equal to `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view::<3>(1).unwrap();
    /// assert_eq!(view.capacity(), 3);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        M
    }

    /// Returns the number of rows in the [`RowVectorView`].
    ///
    /// The number of rows is always `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view::<3>(1).unwrap();
    /// assert_eq!(view.rows(), 1);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        1
    }

    /// Returns the number of columns in the [`RowVectorView`].
    ///
    /// The number of columns is always equal to `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view::<3>(1).unwrap();
    /// assert_eq!(view.cols(), 3);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        M
    }

    /// Returns a transposed view of the [`RowVectorView`].
    ///
    /// This method returns a [`VectorView`], which is a read-only view of the [`RowVectorView`] as a column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::{Vector, RowVector};
    ///
    /// let vec = RowVector::from([1, 2, 3, 4, 5]);
    /// let row_view = vec.view::<3>(1).unwrap();
    /// let col_view = row_view.t();
    /// assert_eq!(col_view, Vector::from([2, 3, 4]));
    /// ```
    pub fn t(&'a self) -> VectorView<'a, V, T, N, M> {
        VectorView::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize>
    RowVectorView<'a, V, T, N, M>
{
    /// Calculates the magnitude of the [`RowVectorView`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::RowVector;
    ///
    /// let vec = RowVector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let view = vec.view::<2>(2).unwrap();
    /// assert_eq!(view.magnitude(), 5.0);
    /// ```
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// RowVectorView == RowVector
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<RowVector<T, N>> for RowVectorView<'_, V, T, A, N>
{
    fn eq(&self, other: &RowVector<T, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// RowVectorView == RowVectorView
impl<
        T: PartialEq,
        V1: Index<usize, Output = T>,
        V2: Index<usize, Output = T>,
        const A1: usize,
        const A2: usize,
        const N: usize,
    > PartialEq<RowVectorView<'_, V2, T, A2, N>> for RowVectorView<'_, V1, T, A1, N>
{
    fn eq(&self, other: &RowVectorView<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// VectorView == VectorViewMut
impl<
        T: PartialEq,
        V1: Index<usize, Output = T>,
        V2: Index<usize, Output = T>,
        const A1: usize,
        const A2: usize,
        const N: usize,
    > PartialEq<RowVectorViewMut<'_, V2, T, A2, N>> for RowVectorView<'_, V1, T, A1, N>
{
    fn eq(&self, other: &RowVectorViewMut<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<'a, V: Index<usize, Output = T>, T: Eq, const N: usize, const M: usize> Eq
    for RowVectorView<'a, V, T, N, M>
{
}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T: fmt::Display, const N: usize, const M: usize> fmt::Display
    for RowVectorView<'a, V, T, N, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "RowVectorView(")?;
        }

        write!(f, "[")?;
        for i in 0..M - 1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{}]", self[M - 1])?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize>
    for RowVectorView<'a, V, T, N, M>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)>
    for RowVectorView<'_, V, T, N, M>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.1]
    }
}
