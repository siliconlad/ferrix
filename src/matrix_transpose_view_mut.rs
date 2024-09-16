use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use std::fmt;
use std::ops::{Index, IndexMut};

/// A static mutable transposed view of a matrix.
#[derive(Debug)]
pub struct MatrixTransposeViewMut<
    'a,
    T,
    const R: usize,
    const C: usize,
    const VR: usize,
    const VC: usize,
> {
    data: &'a mut Matrix<T, R, C>,
    start: (usize, usize), // In terms of the transposed matrix
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    pub(super) fn new(data: &'a mut Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + VR > C || start.1 + VC > R {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    /// Returns the shape of the [`MatrixTransposeViewMut`].
    ///
    /// The shape is always equal to `(VR, VC)`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t_mut();
    /// assert_eq!(transposed.shape(), (3, 2));
    /// ```
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (VR, VC)
    }

    /// Returns the total number of elements in the [`MatrixTransposeViewMut`].
    /// 
    /// The total number of elements is always equal to `VR * VC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t_mut();
    /// assert_eq!(transposed.capacity(), 6);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        VR * VC
    }

    /// Returns the number of rows in the [`MatrixTransposeViewMut`].
    ///
    /// The number of rows is always equal to `VR`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t_mut();
    /// assert_eq!(transposed.rows(), 3);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        VR
    }

    /// Returns the number of columns in the [`MatrixTransposeViewMut`].
    ///
    /// The number of columns is always equal to `VC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t_mut();
    /// assert_eq!(transposed.cols(), 2);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        VC
    }

    /// Returns a transposed view of the [`MatrixTransposeViewMut`].
    ///
    /// This method returns a [`MatrixView`], which is a read-only view of the [`MatrixTransposeViewMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t_mut();
    /// let original = transposed.t();
    /// assert_eq!(original, Matrix::from([[1, 2, 3], [4, 5, 6]]));
    /// ```
    pub fn t(&'a self) -> MatrixView<'a, T, R, C, VC, VR> {
        MatrixView::new(self.data, (self.start.1, self.start.0))
    }

    /// Returns a mutable transposed view of the [`MatrixTransposeViewMut`].
    ///
    /// This method returns a [`MatrixViewMut`], which is a mutable view of the [`MatrixTransposeViewMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let mut transposed = mat.t_mut();
    /// let mut original = transposed.t_mut();
    /// original[(0, 1)] = 10;
    /// assert_eq!(mat, Matrix::from([[1, 10, 3], [4, 5, 6]]));
    /// ```
    pub fn t_mut(&'a mut self) -> MatrixViewMut<'a, T, R, C, VC, VR> {
        MatrixViewMut::new(self.data, (self.start.1, self.start.0))
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// MatrixTransposeViewMut == Matrix
impl<'a, T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<Matrix<T, VR, VC>> for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &Matrix<T, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixTransposeViewMut == MatrixView
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixView<'a, T, A, B, VR, VC>> for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixTransposeViewMut == MatrixViewMut
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixViewMut<'a, T, A, B, VR, VC>>
    for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixTransposeViewMut == MatrixTransposeView
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixTransposeView<'a, T, A, B, VR, VC>>
    for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixTransposeViewMut == MatrixTransposeViewMut
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixTransposeViewMut<'a, T, A, B, VR, VC>>
    for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

impl<'a, T: Eq, const R: usize, const C: usize, const VR: usize, const VC: usize> Eq
    for MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const R: usize, const C: usize, const VR: usize, const VC: usize> fmt::Display
    for MatrixTransposeViewMut<'_, T, R, C, VR, VC>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "MatrixTransposeViewMut(")?;
        }

        write!(f, "[")?;
        for i in 0..VR {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "                       ")?;
                }
            }
            write!(f, "[")?;
            for j in 0..VC {
                write!(f, "{}", self[(i, j)])?;
                if j < VC - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i < VR - 1 {
                writeln!(f)?;
            }
        }
        write!(f, "]")?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeViewMut<'a, T, R, C, VR, VC>
{
    #[inline]
    fn flip(&self, index: (usize, usize)) -> (usize, usize) {
        (index.1, index.0)
    }

    #[inline]
    fn offset(&self, index: (usize, usize)) -> (usize, usize) {
        (index.0 + self.start.0, index.1 + self.start.1)
    }

    #[inline]
    fn validate_index(&self, index: (usize, usize)) -> bool {
        index.0 < VR && index.1 < VC
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize> Index<usize>
    for MatrixTransposeViewMut<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        &self.data[self.flip(self.offset((row_idx, col_idx)))]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize> IndexMut<usize>
    for MatrixTransposeViewMut<'_, T, R, C, VR, VC>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        let index = self.flip(self.offset((row_idx, col_idx)));
        &mut self.data[index]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize> Index<(usize, usize)>
    for MatrixTransposeViewMut<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.flip(self.offset(index))]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize> IndexMut<(usize, usize)>
    for MatrixTransposeViewMut<'_, T, R, C, VR, VC>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        let index = self.flip(self.offset(index));
        &mut self.data[index]
    }
}
