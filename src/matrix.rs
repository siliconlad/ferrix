use num_traits::{Float, One, PrimInt, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use std::fmt;
use std::ops::{Index, IndexMut, Neg};

use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::traits::{FloatRandom, IntRandom};
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;

/// A static matrix type.
#[derive(Debug, Clone)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

/// A static 2x2 matrix alias for [`Matrix<T, 2, 2>`].
pub type Matrix2<T> = Matrix<T, 2, 2>;

/// A static 3x3 matrix alias for [`Matrix<T, 3, 3>`].
pub type Matrix3<T> = Matrix<T, 3, 3>;

impl<T: Default, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    fn default() -> Self {
        Self {
            data: std::array::from_fn(|_| std::array::from_fn(|_| T::default())),
        }
    }
}

impl<T: Default, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Creates a new [`Matrix`] with default values.
    ///
    /// This method initializes a new [`Matrix`] of size RxC, where each element is set to its default value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat: Matrix<f64, 2, 3> = Matrix::new();
    /// assert_eq!(mat, Matrix::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]));
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Returns the shape of the [`Matrix`].
    ///
    /// The shape is always equal to `(R, C)`.
    /// 
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat: Matrix<f64, 2, 3> = Matrix::new();
    /// assert_eq!(mat.shape(), (2, 3));
    /// ```
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Returns the total number of elements in the [`Matrix`].
    /// 
    /// The total number of elements is always equal to `R * C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat: Matrix<f64, 2, 3> = Matrix::new();
    /// assert_eq!(mat.capacity(), 6);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        R * C
    }

    /// Returns the number of rows in the [`Matrix`].
    ///
    /// The number of rows is always equal to `R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat: Matrix<f64, 2, 3> = Matrix::new();
    /// assert_eq!(mat.rows(), 2);
    /// ```
    #[inline]
    pub fn rows(&self) -> usize {
        R
    }

    /// Returns the number of columns in the [`Matrix`].
    ///
    /// The number of columns is always equal to `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat: Matrix<f64, 2, 3> = Matrix::new();
    /// assert_eq!(mat.cols(), 3);
    /// ```
    #[inline]
    pub fn cols(&self) -> usize {
        C
    }
}

impl<T: Copy> Matrix<T, 1, 1> {
    /// Converts a 1x1 [`Matrix`] into its scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[42]]);
    /// assert_eq!(mat.into(), 42);
    /// ```
    pub fn into(self) -> T {
        self[(0, 0)]
    }
}

impl<T: Copy + Zero, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Creates a new [`Matrix`] filled with zeros.
    ///
    /// This method initializes a new [`Matrix`] of size RxC, where each element is set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::<f64, 2, 3>::zeros();
    /// assert_eq!(mat, Matrix::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]));
    /// ```
    pub fn zeros() -> Self {
        Self {
            data: [[T::zero(); C]; R],
        }
    }
}

impl<T: Copy + One, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Creates a new [`Matrix`] filled with ones.
    ///
    /// This method initializes a new [`Matrix`] of size RxC, where each element is set to one.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::<f64, 2, 3>::ones();
    /// assert_eq!(mat, Matrix::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
    /// ```
    pub fn ones() -> Self {
        Self {
            data: [[T::one(); C]; R],
        }
    }
}

impl<T: Copy + Zero + One, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Creates a new identity [`Matrix`].
    ///
    /// This method initializes a new [`Matrix`] of size RxC, where diagonal elements are set to one and all others to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::<f64, 3, 3>::eye();
    /// assert_eq!(mat, Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
    /// ```
    pub fn eye() -> Self {
        let mut data = [[T::zero(); C]; R];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::one();
        }
        Self { data }
    }
}

impl<T: Copy, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Creates a new [`Matrix`] filled with a specified value.
    ///
    /// This method initializes a new [`Matrix`] of size RxC, where each element is set to the provided value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::<i32, 2, 3>::fill(42);
    /// assert_eq!(mat, Matrix::from([[42, 42, 42], [42, 42, 42]]));
    /// ```
    pub fn fill(value: T) -> Self {
        Self {
            data: [[value; C]; R],
        }
    }
}

impl<T: PrimInt, const R: usize, const C: usize> IntRandom for Matrix<T, R, C>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            data: std::array::from_fn(|_| std::array::from_fn(|_| rng.gen())),
        }
    }
}

impl<T: Float + SampleUniform, const R: usize, const C: usize> FloatRandom for Matrix<T, R, C>
where
    Standard: Distribution<T>,
{
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new_inclusive(-T::one(), T::one());
        Self {
            data: std::array::from_fn(|_| std::array::from_fn(|_| dist.sample(&mut rng))),
        }
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Returns a transposed view of the [`Matrix`].
    ///
    /// This method returns a [`MatrixTransposeView`], which is a read-only view of the [`Matrix`] transposed.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transposed = mat.t();
    /// assert_eq!(transposed, Matrix::from([[1, 4], [2, 5], [3, 6]]));
    /// ```
    pub fn t(&self) -> MatrixTransposeView<T, R, C, C, R> {
        MatrixTransposeView::new(self, (0, 0))
    }

    /// Returns a mutable transposed view of the [`Matrix`].
    ///
    /// This method returns a [`MatrixTransposeViewMut`], which is a mutable view of the [`Matrix`] transposed.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let mut transposed = mat.t_mut();
    /// transposed[(1, 0)] = 10;
    /// assert_eq!(mat, Matrix::from([[1, 10, 3], [4, 5, 6]]));
    /// ```
    pub fn t_mut(&mut self) -> MatrixTransposeViewMut<T, R, C, C, R> {
        MatrixTransposeViewMut::new(self, (0, 0))
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Returns a view of [`Matrix`].
    ///
    /// This method returns a [`MatrixView`] of size `VR x VC` starting from the given index.
    /// Returns `None` if the requested view is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let view = mat.view::<2, 2>((1, 1)).unwrap();
    /// assert_eq!(view, Matrix::from([[5, 6], [8, 9]]));
    /// ```
    pub fn view<const VR: usize, const VC: usize>(
        &self,
        start: (usize, usize),
    ) -> Option<MatrixView<T, R, C, VR, VC>> {
        if start.0 + VR > R || start.1 + VC > C {
            return None;
        }
        Some(MatrixView::new(self, start))
    }

    /// Returns a mutable view of [`Matrix`].
    ///
    /// This method returns a [`MatrixViewMut`] of size `VR x VC` starting from the given index.
    /// Returns `None` if the requested view is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    ///
    /// let mut mat = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// let mut view = mat.view_mut::<2, 2>((1, 1)).unwrap();
    /// view[(0, 1)] = 10;
    /// assert_eq!(mat, Matrix::from([[1, 2, 3], [4, 5, 10], [7, 8, 9]]));
    /// ```
    pub fn view_mut<const VR: usize, const VC: usize>(
        &mut self,
        start: (usize, usize),
    ) -> Option<MatrixViewMut<T, R, C, VR, VC>> {
        if start.0 + VR > R || start.1 + VC > C {
            return None;
        }
        Some(MatrixViewMut::new(self, start))
    }
}

impl<T: Float + Neg<Output = T>> Matrix<T, 2, 2> {
    /// Creates a 2D rotation [`Matrix`].
    ///
    /// This method initializes a new 2x2 [`Matrix`] representing a 2D rotation by the given angle (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    /// use std::f64::consts::PI;
    ///
    /// let rot = Matrix::<f64, 2, 2>::rot(PI / 2.0);
    /// ```
    pub fn rot(angle: T) -> Self {
        let data = [
            [T::cos(angle), -T::sin(angle)],
            [T::sin(angle),  T::cos(angle)],
        ];
        Self { data }
    }
}

impl<T: Float + Neg<Output = T>> Matrix<T, 3, 3> {
    /// Creates a 3D rotation [`Matrix`] around the X-axis.
    ///
    /// This method initializes a new 3x3 [`Matrix`] representing a 3D rotation around the X-axis by the given angle (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    /// use std::f64::consts::PI;
    ///
    /// let rot = Matrix::<f64, 3, 3>::rotx(PI / 2.0);
    /// ```
    pub fn rotx(angle: T) -> Self {
        let data = [
            [T::one(),  T::zero(),      T::zero()],
            [T::zero(), T::cos(angle), -T::sin(angle)],
            [T::zero(), T::sin(angle),  T::cos(angle)],
        ];
        Self { data }
    }

    /// Creates a 3D rotation [`Matrix`] around the Y-axis.
    ///
    /// This method initializes a new 3x3 [`Matrix`] representing a 3D rotation around the Y-axis by the given angle (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    /// use std::f64::consts::PI;
    ///
    /// let rot = Matrix::<f64, 3, 3>::roty(PI / 2.0);
    /// ```
    pub fn roty(angle: T) -> Self {
        let data = [
            [ T::cos(angle), T::zero(), T::sin(angle)],
            [ T::zero(),     T::one(),  T::zero()],
            [-T::sin(angle), T::zero(), T::cos(angle)],
        ];
        Self { data }
    }

    /// Creates a 3D rotation [`Matrix`] around the Z-axis.
    ///
    /// This method initializes a new 3x3 [`Matrix`] representing a 3D rotation around the Z-axis by the given angle (in radians).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrix::Matrix;
    /// use std::f64::consts::PI;
    ///
    /// let rot = Matrix::<f64, 3, 3>::rotz(PI / 2.0);
    /// ```
    pub fn rotz(angle: T) -> Self {
        let data = [
            [T::cos(angle), -T::sin(angle), T::zero()],
            [T::sin(angle),  T::cos(angle), T::zero()],
            [T::zero(),      T::zero(),     T::one()],
        ];
        Self { data }
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// Matrix == Matrix
impl<T: PartialEq, const R: usize, const C: usize> PartialEq for Matrix<T, R, C> {
    fn eq(&self, other: &Self) -> bool {
        (0..R).all(|i| (0..C).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// Matrix == MatrixView
impl<T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<MatrixView<'_, T, R, C, VR, VC>> for Matrix<T, VR, VC>
{
    fn eq(&self, other: &MatrixView<'_, T, R, C, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// Matrix == MatrixViewMut
impl<T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<MatrixViewMut<'_, T, R, C, VR, VC>> for Matrix<T, VR, VC>
{
    fn eq(&self, other: &MatrixViewMut<'_, T, R, C, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// Matrix == MatrixTransposeView
impl<T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<MatrixTransposeView<'_, T, R, C, VR, VC>> for Matrix<T, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeView<'_, T, R, C, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// Matrix == MatrixTransposeViewMut
impl<T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize>
    PartialEq<MatrixTransposeViewMut<'_, T, R, C, VR, VC>> for Matrix<T, VR, VC>
{
    fn eq(&self, other: &MatrixTransposeViewMut<'_, T, R, C, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

impl<T: Eq, const R: usize, const C: usize> Eq for Matrix<T, R, C> {}

/////////////////////////////////////
/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "Matrix(")?;
        }

        write!(f, "[")?;
        for i in 0..R {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "       ")?;
                }
            }
            write!(f, "[")?;
            for j in 0..C {
                write!(f, "{}", self[(i, j)])?;
                if j < C - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i < R - 1 {
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

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / C;
        let col_idx = index % C;
        &self.data[row_idx][col_idx]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / C;
        let col_idx = index % C;
        &mut self.data[row_idx][col_idx]
    }
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= R || index.1 >= C {
            panic!("Index out of bounds");
        }
        &self.data[index.0][index.1]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 >= R || index.1 >= C {
            panic!("Index out of bounds");
        }
        &mut self.data[index.0][index.1]
    }
}

//////////////////////////////////
//  From Trait Implementations  //
//////////////////////////////////

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C> {
    fn from(data: [[T; C]; R]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const C: usize> From<Vector<T, C>> for Matrix<T, C, 1> {
    fn from(vector: Vector<T, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| [vector[i]]),
        }
    }
}

impl<T: Copy, const C: usize> From<RowVector<T, C>> for Matrix<T, 1, C> {
    fn from(vector: RowVector<T, C>) -> Self {
        Self {
            data: [std::array::from_fn(|i| vector[i])],
        }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<VectorView<'_, V, T, A, N>> for Matrix<T, N, 1>
{
    fn from(view: VectorView<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| [view[i]]),
        }
    }
}

impl<V: IndexMut<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<VectorViewMut<'_, V, T, A, N>> for Matrix<T, N, 1>
{
    fn from(view: VectorViewMut<'_, V, T, A, N>) -> Self {
        Self {
            data: std::array::from_fn(|i| [view[i]]),
        }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<RowVectorView<'_, V, T, A, N>> for Matrix<T, 1, N>
{
    fn from(view: RowVectorView<'_, V, T, A, N>) -> Self {
        Self {
            data: [std::array::from_fn(|i| view[i]); 1],
        }
    }
}

impl<V: Index<usize, Output = T>, T: Copy, const A: usize, const N: usize>
    From<RowVectorViewMut<'_, V, T, A, N>> for Matrix<T, 1, N>
{
    fn from(view: RowVectorViewMut<'_, V, T, A, N>) -> Self {
        Self {
            data: [std::array::from_fn(|i| view[i]); 1],
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const R: usize, const C: usize>
    From<MatrixView<'_, T, A, B, R, C>> for Matrix<T, R, C>
{
    fn from(view: MatrixView<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const R: usize, const C: usize>
    From<MatrixViewMut<'_, T, A, B, R, C>> for Matrix<T, R, C>
{
    fn from(view: MatrixViewMut<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const R: usize, const C: usize>
    From<MatrixTransposeView<'_, T, A, B, R, C>> for Matrix<T, R, C>
{
    fn from(view: MatrixTransposeView<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Copy, const A: usize, const B: usize, const R: usize, const C: usize>
    From<MatrixTransposeViewMut<'_, T, A, B, R, C>> for Matrix<T, R, C>
{
    fn from(view: MatrixTransposeViewMut<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}
