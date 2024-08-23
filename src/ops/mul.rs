use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Mul;

macro_rules! impl_mul {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Mul<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn mul(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Mul<T> for $lhs {
            type Output = Vector<T, N>;

            fn mul(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] * scalar))
            }
        }
    };
}

macro_rules! impl_mm_mul {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Mul<$rhs> for $lhs {
            type Output = Matrix<T, M, N>;

            fn mul(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] * other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Mul<T> for $lhs {
            type Output = Matrix<T, M, N>;

            fn mul(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] * scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_mul_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Mul<$rhs>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn mul(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] * other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Mul<T>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn mul(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] * scalar)
                }))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_mul!(Vector<T, N>);
impl_mul!(&Vector<T, N>);

impl_mul!(Vector<T, N>, Vector<T, N>);
impl_mul!(Vector<T, N>, &Vector<T, N>);
impl_mul!(&Vector<T, N>, Vector<T, N>);
impl_mul!(&Vector<T, N>, &Vector<T, N>);

impl_mul!(Vector<T, N>, VectorView<'_, T, N>);
impl_mul!(Vector<T, N>, &VectorView<'_, T, N>);
impl_mul!(&Vector<T, N>, VectorView<'_, T, N>);
impl_mul!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_mul!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_mul!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_mul!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_mul!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_mul!(VectorView<'_, T, N>);
impl_mul!(&VectorView<'_, T, N>);

impl_mul!(VectorView<'_, T, N>, Vector<T, N>);
impl_mul!(VectorView<'_, T, N>, &Vector<T, N>);
impl_mul!(&VectorView<'_, T, N>, Vector<T, N>);
impl_mul!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_mul!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_mul!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_mul!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_mul!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_mul!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_mul!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_mul!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_mul!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_mul!(VectorViewMut<'_, T, N>);
impl_mul!(&VectorViewMut<'_, T, N>);

impl_mul!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_mul!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_mul!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_mul!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_mul!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_mul!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_mul!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_mul!(Matrix<T, M, N>);
impl_mm_mul!(&Matrix<T, M, N>);

impl_mm_mul!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_mul!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_mul!(&Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_mul!(&Matrix<T, M, N>, &Matrix<T, M, N>);

impl_mm_mul_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);

impl_mm_mul_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_mul_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_mul_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

//////////////////
//  MatrixView  //
//////////////////

// Scalar
impl_mm_mul_view!(MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&MatrixView<'_, T, A, B, M, N>);

impl_mm_mul_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_mul_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_mul_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_mul_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

// Scalar
impl_mm_mul_view!(MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_view!(&MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_mul_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_mul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>);

impl_mm_mul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_mul() {
        // Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);

        // &Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mul() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView * VectorView
        let result = &view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * &VectorView
        let result = view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test &VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &Vector
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let result = &view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test scalar multiplication
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));
    }

    #[test]
    fn test_vector_view_mut_mul() {
        // VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);

        // &VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_mul() {
        // Matrix * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &Matrix * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = m1 * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // Matrix * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = m1 * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // &Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = &m1 * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // &Matrix * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = &m1 * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // Matrix * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = m1 * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // Matrix * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = m1 * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // &Matrix * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // &Matrix * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 14], [18, 32]]));

        // Matrix * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = m1 * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // &Matrix * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = &m1 * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_view_mul() {
        // MatrixView * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 * view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 * &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 * view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 * &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = view * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // &MatrixView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_view_mut_mul() {
        // MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 * view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 * &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 * view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 * &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = view_mut * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // &MatrixViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view_mut * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_transpose_view_mul() {
        // MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = t_view * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = t_view * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = &t_view * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = &t_view * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 * t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 * &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 * t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 * &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view * t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view * &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = t_view * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // &MatrixTransposeView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = &t_view * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_mul() {
        // MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut * &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view_mut * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view_mut * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view_mut * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view_mut * &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view_mut * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view_mut * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view_mut * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view_mut * &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut * t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut * &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 * t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 * &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 * t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // &MatrixTransposeViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 * &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = t_view_mut * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // &MatrixTransposeViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = &t_view_mut * scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }
}
