use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::matrix_view_mut::MatrixViewMut;
use std::fmt;
use std::ops::Index;

/// A static view of a matrix.
#[derive(Debug)]
pub struct MatrixView<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize> {
    data: &'a Matrix<T, R, C>,
    start: (usize, usize),
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixView<'a, T, R, C, VR, VC>
{
    pub(super) fn new(data: &'a Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + VR > R || start.1 + VC > C {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }

    /// Returns the shape of the [`MatrixView`].
    ///
    /// The shape is always equal to `(VR, VC)`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<1, 2>((1, 1)).unwrap();
    /// assert_eq!(view.shape(), (1, 2));
    /// ```
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (VR, VC)
    }

    /// Returns the total number of elements in the [`MatrixView`].
    /// 
    /// The total number of elements is always equal to `VR * VC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<2, 2>((1, 1)).unwrap();
    /// assert_eq!(view.capacity(), 4);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        VR * VC
    }

    /// Returns the number of rows in the [`MatrixView`].
    ///
    /// The number of rows is always equal to `VR`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<1, 2>((1, 1)).unwrap();
    /// assert_eq!(view.rows(), 1);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        VR
    }

    /// Returns the number of columns in the [`MatrixView`].
    ///
    /// The number of columns is always equal to `VC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<1, 2>((1, 1)).unwrap();
    /// assert_eq!(view.cols(), 2);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        VC
    }

    /// Returns a transposed view of the [`MatrixView`].
    ///
    /// This method returns a [`MatrixTransposeView`], which is a read-only view of the [`MatrixView`] transposed.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<2, 2>((1, 1)).unwrap();
    /// let transposed = view.t();
    /// assert_eq!(transposed, Matrix::from([[5, 8], [6, 9]]));
    /// ```
    pub fn t(&self) -> MatrixTransposeView<'a, T, R, C, VC, VR> {
        MatrixTransposeView::new(self.data, (self.start.1, self.start.0))
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// MatrixView == Matrix
impl<'a, T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<Matrix<T, VR, VC>> for MatrixView<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &Matrix<T, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixView == MatrixView
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixView<'a, T, A, B, VR, VC>> for MatrixView<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixView == MatrixViewMut
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixViewMut<'a, T, A, B, VR, VC>> for MatrixView<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixView == MatrixTransposeView
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixTransposeView<'a, T, A, B, VR, VC>> for MatrixView<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixView == MatrixTransposeViewMut
impl<
        'a,
        T: PartialEq,
        const A: usize,
        const B: usize,
        const R: usize,
        const C: usize,
        const VR: usize,
        const VC: usize,
    > PartialEq<MatrixTransposeViewMut<'a, T, A, B, VR, VC>> for MatrixView<'a, T, R, C, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

impl<'a, T: Eq, const R: usize, const C: usize, const VR: usize, const VC: usize> Eq
    for MatrixView<'a, T, R, C, VR, VC>
{
}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const R: usize, const C: usize, const VR: usize, const VC: usize> fmt::Display
    for MatrixView<'_, T, R, C, VR, VC>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "MatrixView(")?;
        }

        write!(f, "[")?;
        for i in 0..VR {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "           ")?;
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
    MatrixView<'a, T, R, C, VR, VC>
{
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
    for MatrixView<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        &self.data[self.offset((row_idx, col_idx))]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize> Index<(usize, usize)>
    for MatrixView<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.offset(index)]
    }
}
