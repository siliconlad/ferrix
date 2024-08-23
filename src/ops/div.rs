use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Div;

macro_rules! impl_div {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Div<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn div(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Div<T> for $lhs {
            type Output = Vector<T, N>;

            fn div(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
            }
        }
    };
}

macro_rules! impl_mm_div {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Div<$rhs> for $lhs {
            type Output = Matrix<T, M, N>;

            fn div(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] / other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Div<T> for $lhs {
            type Output = Matrix<T, M, N>;

            fn div(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] / scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_div_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Div<$rhs>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn div(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] / other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Div<T>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn div(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] / scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_div_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<
                T: Numeric + From<u8>,
                const A: usize,
                const B: usize,
                const C: usize,
                const D: usize,
                const M: usize,
                const N: usize,
            > Div<$rhs> for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn div(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] / other[(i, j)])
                }))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_div!(Vector<T, N>);
impl_div!(&Vector<T, N>);

impl_div!(Vector<T, N>, Vector<T, N>);
impl_div!(Vector<T, N>, &Vector<T, N>);
impl_div!(&Vector<T, N>, Vector<T, N>);
impl_div!(&Vector<T, N>, &Vector<T, N>);

impl_div!(Vector<T, N>, VectorView<'_, T, N>);
impl_div!(Vector<T, N>, &VectorView<'_, T, N>);
impl_div!(&Vector<T, N>, VectorView<'_, T, N>);
impl_div!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_div!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_div!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_div!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_div!(VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>);

impl_div!(VectorView<'_, T, N>, Vector<T, N>);
impl_div!(VectorView<'_, T, N>, &Vector<T, N>);
impl_div!(&VectorView<'_, T, N>, Vector<T, N>);
impl_div!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_div!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_div!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_div!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_div!(VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>);

impl_div!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_div!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_div!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_div!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_div!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_div!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_div!(Matrix<T, M, N>);
impl_mm_div!(&Matrix<T, M, N>);

impl_mm_div!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_div!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_div!(&Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_div!(&Matrix<T, M, N>, &Matrix<T, M, N>);

impl_mm_div_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_div_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);

impl_mm_div_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_div_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_div_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

//////////////////
//  MatrixView  //
//////////////////

// Scalar
impl_mm_div_view!(MatrixView<'_, T, A, B, M, N>);
impl_mm_div_view!(&MatrixView<'_, T, A, B, M, N>);

impl_mm_div_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_view!(&MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(&MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_div_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_div_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_div_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_view!(&MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

// Scalar
impl_mm_div_view!(MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_view!(&MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_div_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_view!(&MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_div_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>);

impl_mm_div_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, C, D, M, N>
);

impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, C, D, M, N>
);
impl_mm_div_view_view!(
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
    fn test_vector_div() {
        // Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 / v2;
        assert!((result[0] - 10.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 10.0).abs() < f64::EPSILON);

        // Vector / Scalar
        let v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v / 0.5;
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);
        assert!((result[2] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_div() {
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let v2 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView / VectorView
        let result = &view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorView
        let result = view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / &Vector
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let result = &view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test scalar division
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));
    }

    #[test]
    fn test_vector_view_mut_div() {
        // VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 10.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut / Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() / 0.5;
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);
        assert!((result[2] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_div() {
        // Matrix / Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 / &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 / &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // &Matrix / &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view = m2.t();
        let result = m1 / t_view;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // Matrix / &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view = m2.t();
        let result = m1 / &t_view;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // &Matrix / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view = m2.t();
        let result = &m1 / t_view;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // &Matrix / &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view = m2.t();
        let result = &m1 / &t_view;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // Matrix / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view_mut = m2.t_mut();
        let result = m1 / t_view_mut;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // Matrix / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view_mut = m2.t_mut();
        let result = m1 / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // &Matrix / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 / t_view_mut;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // &Matrix / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // Matrix / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let scalar = 5;
        let result = m1 / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // &Matrix / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let scalar = 5;
        let result = &m1 / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_view_div() {
        // MatrixView / Matrix
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / &Matrix
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / Matrix
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / &Matrix
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 / view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / &MatrixView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 / &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / MatrixView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 / view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / &MatrixView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 / &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / &MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / &MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixView / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = view / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // &MatrixView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_view_mut_div() {
        // MatrixViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / &Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / &Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut / &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / &MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut / &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / &MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut / &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 / view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 / &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 / view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 / &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / &MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / &MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // &MatrixViewMut / &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = view_mut / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // &MatrixViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view_mut / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_transpose_view_div() {
        // MatrixTransposeView / Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let result = t_view / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let result = t_view / &m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let result = &t_view / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let result = &t_view / &m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view / view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view / &view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view / view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view / &view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // &MatrixTransposeView / &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 / t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // MatrixTransposeView / &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 / &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // &MatrixTransposeView / MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 / t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // &MatrixTransposeView / &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 / &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // MatrixTransposeView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // MatrixTransposeView / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // &MatrixTransposeView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view / t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // &MatrixTransposeView / &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view / &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // MatrixTransposeView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = t_view / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &MatrixTransposeView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = &t_view / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div() {
        // MatrixTransposeViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut / &m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut / &m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view_mut / view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = t_view_mut / &view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view_mut / view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 3>((0, 0));
        let result = &t_view_mut / &view;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view_mut / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = t_view_mut / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view_mut / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // &MatrixTransposeViewMut / &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 3>((0, 0));
        let result = &t_view_mut / &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // MatrixTransposeViewMut / &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // &MatrixTransposeViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut / t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // &MatrixTransposeViewMut / &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut / &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // MatrixTransposeViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 / t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // MatrixTransposeViewMut / &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 / &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // &MatrixTransposeViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 / t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // &MatrixTransposeViewMut / &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 / &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // MatrixTransposeViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 50]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = t_view_mut / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 10]]));

        // &MatrixTransposeViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 50]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = &t_view_mut / scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 10]]));
    }
}
