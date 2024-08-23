use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Sub;

macro_rules! impl_sub {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Sub<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn sub(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Sub<T> for $lhs {
            type Output = Vector<T, N>;

            fn sub(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
            }
        }
    };
}

macro_rules! impl_mm_sub {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Sub<$rhs> for $lhs {
            type Output = Matrix<T, M, N>;

            fn sub(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] - other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Sub<T> for $lhs {
            type Output = Matrix<T, M, N>;

            fn sub(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] - scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_sub_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Sub<$rhs>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn sub(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] - other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Sub<T>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn sub(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] - scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_sub_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<
                T: Numeric + From<u8>,
                const A: usize,
                const B: usize,
                const C: usize,
                const D: usize,
                const M: usize,
                const N: usize,
            > Sub<$rhs> for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn sub(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] - other[(i, j)])
                }))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_sub!(Vector<T, N>);
impl_sub!(&Vector<T, N>);

impl_sub!(Vector<T, N>, Vector<T, N>);
impl_sub!(Vector<T, N>, &Vector<T, N>);
impl_sub!(&Vector<T, N>, Vector<T, N>);
impl_sub!(&Vector<T, N>, &Vector<T, N>);

impl_sub!(Vector<T, N>, VectorView<'_, T, N>);
impl_sub!(Vector<T, N>, &VectorView<'_, T, N>);
impl_sub!(&Vector<T, N>, VectorView<'_, T, N>);
impl_sub!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_sub!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_sub!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_sub!(VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>);

impl_sub!(VectorView<'_, T, N>, Vector<T, N>);
impl_sub!(VectorView<'_, T, N>, &Vector<T, N>);
impl_sub!(&VectorView<'_, T, N>, Vector<T, N>);
impl_sub!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_sub!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_sub!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_sub!(VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>);

impl_sub!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_sub!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_sub!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_sub!(Matrix<T, M, N>);
impl_mm_sub!(&Matrix<T, M, N>);

impl_mm_sub!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_sub!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_sub!(&Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_sub!(&Matrix<T, M, N>, &Matrix<T, M, N>);

impl_mm_sub_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_sub_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);

impl_mm_sub_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_sub_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_sub_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_sub_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

//////////////////
//  MatrixView  //
//////////////////

// Scalar
impl_mm_sub_view!(MatrixView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&MatrixView<'_, T, A, B, M, N>);

impl_mm_sub_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_sub_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_sub_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_sub_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

// Scalar
impl_mm_sub_view!(MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_sub_view!(&MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_sub_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_sub_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_sub_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>);

impl_mm_sub_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_sub_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_sub_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_sub() {
        // Vector - Vector (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &Vector (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - Vector (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &Vector (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorView (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &VectorView (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - VectorView (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &VectorView (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorViewMut (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &VectorViewMut (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - VectorViewMut (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &VectorViewMut (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - Scalar (non-reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // &Vector - Scalar (reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_sub() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView - VectorView
        let result = &view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - &VectorView
        let result = view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test &VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - &Vector
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let result = &view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test scalar subtraction
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_vector_view_mut_sub() {
        // VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - Vector (non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &Vector (non-reference and reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - Vector (reference and non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &Vector (both references, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_sub() {
        // Matrix - Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // Matrix - &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &Matrix - Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &Matrix - &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // Matrix - MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 - &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // &Matrix - MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // &Matrix - &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 - &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 - &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // &Matrix - MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // &Matrix - &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 - &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m2.t();
        let result = m1 - t_view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // Matrix - &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m2.t();
        let result = m1 - &t_view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // &Matrix - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m2.t();
        let result = &m1 - t_view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // &Matrix - &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m2.t();
        let result = &m1 - &t_view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // Matrix - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view_mut = m2.t_mut();
        let result = m1 - t_view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // Matrix - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view_mut = m2.t_mut();
        let result = m1 - &t_view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // &Matrix - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 - t_view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // &Matrix - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 - &t_view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -5, -6], [-2, -3, -4]])
        );

        // Matrix - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = m1 - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // &Matrix - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = &m1 - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));
    }

    #[test]
    fn test_matrix_view_sub() {
        // MatrixView - Matrix
        let m1 = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 1));
        let result = view - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-3, -3], [-1, -1]]));

        // MatrixView - &Matrix
        let m1 = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((1, 1));
        let result = view - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[0, 0], [1, 1]]));

        // &MatrixView - Matrix
        let m1 =
            Matrix::<i32, 3, 5>::new([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 2));
        let result = &view - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-2, -2], [1, 1]]));

        // &MatrixView - &Matrix
        let m1 =
            Matrix::<i32, 5, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((2, 1));
        let result = &view - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[3, 3], [4, 4]]));

        // MatrixView - MatrixView
        let m1 = Matrix::<i32, 4, 4>::new([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view1 = m1.view::<2, 2>((1, 1));
        let view2 = m2.view::<2, 2>((0, 1));
        let result = view1 - view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[0, 0], [1, 1]]));

        // MatrixView - &MatrixView
        let m1 =
            Matrix::<i32, 3, 5>::new([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]);
        let m2 = Matrix::<i32, 4, 4>::new([
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]);
        let view1 = m1.view::<2, 2>((0, 2));
        let view2 = m2.view::<2, 2>((1, 1));
        let result = view1 - &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-7, -7], [-6, -6]]));

        // &MatrixView - MatrixView
        let m1 =
            Matrix::<i32, 5, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]);
        let m2 = Matrix::<i32, 3, 4>::new([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let view1 = m1.view::<2, 2>((2, 1));
        let view2 = m2.view::<2, 2>((0, 1));
        let result = &view1 - view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [1, 1]]));

        // &MatrixView - &MatrixView
        let m1 = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let m2 =
            Matrix::<i32, 3, 5>::new([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]);
        let view1 = m1.view::<2, 2>((1, 1));
        let view2 = m2.view::<2, 2>((0, 2));
        let result = &view1 - &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-2, -2], [-4, -4]]));

        // MatrixView - MatrixViewMut
        let m1 = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut m2 = Matrix::<i32, 4, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]);
        let view = m1.view::<2, 2>((0, 1));
        let view_mut = m2.view_mut::<2, 2>((1, 1));
        let result = view - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-7, -7], [-6, -6]]));

        // MatrixView - &MatrixViewMut
        let m1 = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let mut m2 =
            Matrix::<i32, 3, 5>::new([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]);
        let view = m1.view::<2, 2>((1, 1));
        let view_mut = m2.view_mut::<2, 2>((0, 2));
        let result = view - &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-2, -2], [-4, -4]]));

        // &MatrixView - MatrixViewMut
        let m1 =
            Matrix::<i32, 3, 5>::new([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]);
        let mut m2 = Matrix::<i32, 4, 4>::new([
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]);
        let view = m1.view::<2, 2>((0, 2));
        let view_mut = m2.view_mut::<2, 2>((1, 1));
        let result = &view - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-7, -7], [-6, -6]]));

        // &MatrixView - &MatrixViewMut
        let m1 =
            Matrix::<i32, 5, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]);
        let mut m2 = Matrix::<i32, 3, 4>::new([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let view = m1.view::<2, 2>((2, 1));
        let view_mut = m2.view_mut::<2, 2>((0, 1));
        let result = &view - &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [1, 1]]));

        // MatrixView - MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixView - &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixView - MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixView - &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixView - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixView - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = view - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // &MatrixView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));
    }

    #[test]
    fn test_matrix_view_mut_sub() {
        // MatrixViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut - &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut - &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut - &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 - view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 - &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 - view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 - &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - &MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - &MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixViewMut - &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = view_mut - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // &MatrixViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view_mut - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));
    }

    #[test]
    fn test_matrix_transpose_view_sub() {
        // MatrixTransposeView - Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let result = t_view - m2;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - &Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let result = t_view - &m2;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let result = &t_view - m2;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - &Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let result = &t_view - &m2;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<3, 2>((0, 0));
        let result = t_view - view;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - &MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<3, 2>((0, 0));
        let result = t_view - &view;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<3, 2>((0, 0));
        let result = &t_view - view;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - &MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<3, 2>((0, 0));
        let result = &t_view - &view;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<3, 2>((0, 0));
        let result = t_view - view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - &MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<3, 2>((0, 0));
        let result = t_view - &view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<3, 2>((0, 0));
        let result = &t_view - view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // &MatrixTransposeView - &MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<3, 2>((0, 0));
        let result = &t_view - &view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 3, 2>::new([[-4, -2], [-5, -3], [-6, -4]])
        );

        // MatrixTransposeView - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 - t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 - &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeView - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 - t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeView - &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 - &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view - t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeView - &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view - &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = t_view - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // &MatrixTransposeView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = &t_view - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_sub() {
        // MatrixTransposeViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut - m2;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut - &m2;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut - m2;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut - &m2;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view_mut - view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view_mut - &view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view_mut - view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view_mut - &view;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view_mut - view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view_mut - &view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view_mut - view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // &MatrixTransposeViewMut - &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view_mut - &view_mut;
        assert_eq!(
            result,
            Matrix::<i32, 2, 3>::new([[-4, -3, -2], [-6, -5, -4]])
        );

        // MatrixTransposeViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut - t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeViewMut - &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut - &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 - t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 - &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 - t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // &MatrixTransposeViewMut - &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 - &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = t_view_mut - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // &MatrixTransposeViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = &t_view_mut - scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));
    }
}
