use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A mutable column vector view of a [`Vector`] or a [`RowVector`](crate::row_vector::RowVector) (transposed view).
#[derive(Debug)]
pub struct VectorViewMut<'a, V, T, const N: usize, const M: usize> {
    data: &'a mut V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> VectorViewMut<'a, V, T, N, M> {
    pub(super) fn new(data: &'a mut V, start: usize) -> Self {
        Self {
            data,
            start,
            _phantom: PhantomData,
        }
    }

    /// Returns the shape of the [`VectorViewMut`].
    ///
    /// The shape is always equal to `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view_mut::<3>(1).unwrap();
    /// assert_eq!(view.shape(), 3);
    /// ```
    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    /// Returns the total number of elements in the [`VectorViewMut`].
    ///
    /// The total number of elements is always equal to `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view_mut::<3>(1).unwrap();
    /// assert_eq!(view.capacity(), 3);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        M
    }

    /// Returns the number of rows in the [`VectorViewMut`].
    ///
    /// The number of rows is always equal to `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view_mut::<3>(1).unwrap();
    /// assert_eq!(view.rows(), 3);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        M
    }

    /// Returns the number of columns in the [`VectorViewMut`].
    ///
    /// The number of columns is always `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view_mut::<3>(1).unwrap();
    /// assert_eq!(view.cols(), 1);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        1
    }

    /// Returns a transposed view of the [`VectorViewMut`].
    ///
    /// This method returns a [`RowVectorView`], which is a read-only view of the [`VectorViewMut`] as a row vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::{Vector, RowVector};
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let view = vec.view_mut::<3>(1).unwrap();
    /// let row_view = view.t();
    /// assert_eq!(row_view, RowVector::from([2, 3, 4]));
    /// ```
    pub fn t(&'a self) -> RowVectorView<'a, V, T, N, M> {
        RowVectorView::new(self.data, self.start)
    }

    /// Returns a mutable transposed view of the [`VectorViewMut`].
    ///
    /// This method returns a [`RowVectorViewMut`], which is a mutable view of the [`VectorViewMut`] as a row vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1, 2, 3, 4, 5]);
    /// let mut view = vec.view_mut::<3>(1).unwrap();
    /// let mut row_view = view.t_mut();
    /// row_view[1] = 10;
    /// assert_eq!(vec, Vector::from([1, 2, 10, 4, 5]));
    /// ```
    pub fn t_mut(&'a mut self) -> RowVectorViewMut<'a, V, T, N, M> {
        RowVectorViewMut::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize>
    VectorViewMut<'a, V, T, N, M>
{
    /// Calculates the magnitude (Euclidean norm) of the [`VectorViewMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Vector;
    ///
    /// let mut vec = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let view = vec.view_mut::<2>(2).unwrap();
    /// assert_eq!(view.magnitude(), 5.0);
    /// ```
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// VectorViewMut == Vector
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize>
    PartialEq<Vector<T, N>> for VectorViewMut<'_, V, T, A, N>
{
    fn eq(&self, other: &Vector<T, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// VectorViewMut == VectorView
impl<
        T: PartialEq,
        V1: Index<usize, Output = T>,
        V2: Index<usize, Output = T>,
        const A1: usize,
        const A2: usize,
        const N: usize,
    > PartialEq<VectorView<'_, V2, T, A2, N>> for VectorViewMut<'_, V1, T, A1, N>
{
    fn eq(&self, other: &VectorView<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// VectorViewMut == VectorViewMut
impl<
        T: PartialEq,
        V1: Index<usize, Output = T>,
        V2: Index<usize, Output = T>,
        const A1: usize,
        const A2: usize,
        const N: usize,
    > PartialEq<VectorViewMut<'_, V2, T, A2, N>> for VectorViewMut<'_, V1, T, A1, N>
{
    fn eq(&self, other: &VectorViewMut<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<'a, V: Index<usize, Output = T>, T: Eq, const N: usize, const M: usize> Eq
    for VectorViewMut<'a, V, T, N, M>
{
}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T: fmt::Display, const N: usize, const M: usize> fmt::Display
    for VectorViewMut<'a, V, T, N, M>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "VectorViewMut(")?;
        }

        write!(f, "[")?;
        for i in 0..M - 1 {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "              ")?;
                }
            }
            writeln!(f, "{}", self[i])?;
        }
        if f.alternate() {
            write!(f, "              ")?;
        }

        write!(f, " {}]", self[M - 1])?;

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
    for VectorViewMut<'a, V, T, N, M>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<usize>
    for VectorViewMut<'a, V, T, N, M>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}

impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)>
    for VectorViewMut<'_, V, T, N, M>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.0]
    }
}

impl<V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<(usize, usize)>
    for VectorViewMut<'_, V, T, N, M>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &mut self.data[self.start + index.0]
    }
}
