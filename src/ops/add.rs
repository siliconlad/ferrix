use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Add;

macro_rules! impl_vv_add {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Add<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn add(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Add<T> for $lhs {
            type Output = Vector<T, N>;

            fn add(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] + scalar))
            }
        }
    };
}

macro_rules! impl_mm_add {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Add<$rhs> for $lhs {
            type Output = Matrix<T, M, N>;

            fn add(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] + other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> Add<T> for $lhs {
            type Output = Matrix<T, M, N>;

            fn add(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] + scalar)
                }))
            }
        }
    };
}

macro_rules! impl_mm_add_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Add<$rhs>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn add(self, other: $rhs) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] + other[(i, j)])
                }))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize> Add<T>
            for $lhs
        {
            type Output = Matrix<T, M, N>;

            fn add(self, scalar: T) -> Self::Output {
                Matrix::<T, M, N>::new(std::array::from_fn(|i| {
                    std::array::from_fn(|j| self[(i, j)] + scalar)
                }))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_vv_add!(Vector<T, N>);
impl_vv_add!(&Vector<T, N>);

impl_vv_add!(Vector<T, N>, Vector<T, N>);
impl_vv_add!(Vector<T, N>, &Vector<T, N>);
impl_vv_add!(&Vector<T, N>, Vector<T, N>);
impl_vv_add!(&Vector<T, N>, &Vector<T, N>);

impl_vv_add!(Vector<T, N>, VectorView<'_, T, N>);
impl_vv_add!(Vector<T, N>, &VectorView<'_, T, N>);
impl_vv_add!(&Vector<T, N>, VectorView<'_, T, N>);
impl_vv_add!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_vv_add!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_vv_add!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_vv_add!(VectorView<'_, T, N>);
impl_vv_add!(&VectorView<'_, T, N>);

impl_vv_add!(VectorView<'_, T, N>, Vector<T, N>);
impl_vv_add!(VectorView<'_, T, N>, &Vector<T, N>);
impl_vv_add!(&VectorView<'_, T, N>, Vector<T, N>);
impl_vv_add!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_vv_add!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_vv_add!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_vv_add!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_vv_add!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_vv_add!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_vv_add!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_vv_add!(VectorViewMut<'_, T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>);

impl_vv_add!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_vv_add!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_vv_add!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_vv_add!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_vv_add!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_vv_add!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_add!(Matrix<T, M, N>);
impl_mm_add!(&Matrix<T, M, N>);

impl_mm_add!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_add!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_add!(&Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_add!(&Matrix<T, M, N>, &Matrix<T, M, N>);

impl_mm_add_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_add_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);

impl_mm_add_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_add_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_add_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_add_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

//////////////////
//  MatrixView  //
//////////////////

// Scalar
impl_mm_add_view!(MatrixView<'_, T, A, B, M, N>);
impl_mm_add_view!(&MatrixView<'_, T, A, B, M, N>);

impl_mm_add_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_add_view!(&MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(&MatrixView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_add_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_add_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&MatrixViewMut<'_, T, A, B, M, N>);

impl_mm_add_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_add_view!(&MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

// Scalar
impl_mm_add_view!(MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_add_view!(&MatrixTransposeView<'_, T, A, B, M, N>);

impl_mm_add_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_add_view!(&MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeView<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_add_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_add_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>);

impl_mm_add_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_add_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_add_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);

impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);

impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
    &MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_add_view!(
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
    fn test_vector_add() {
        // Vector + Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Reference addition for f64
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Mixed reference and non-reference addition for f64
        let result = v1 + &v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v1 + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &Vector + VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &Vector + &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &Vector + VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &Vector + &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Reference addition for f64 scalar
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);

        let result = v + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_add() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView + VectorView
        let result = &view1 + &view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + &VectorView
        let result = view1 + &view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test &VectorView + VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 + view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test VectorView + &Vector
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + &vec;
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test &VectorView + Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test &VectorView + &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let result = &view1 + &vec;
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test VectorView + VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 + view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test VectorView + &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 + &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test &VectorView + VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 + view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test &VectorView + &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 + &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test scalar addition
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + 10;
        assert_eq!(result, Vector::<i32, 3>::new([12, 13, 14]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + 10;
        assert_eq!(result, Vector::<i32, 3>::new([12, 13, 14]));
    }

    #[test]
    fn test_vector_view_mut_add() {
        // VectorViewMut + VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Reference addition for f64
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Mixed reference and non-reference addition for f64
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &VectorViewMut + VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &VectorViewMut + &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &VectorViewMut + Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // &VectorViewMut + &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);

        // &VectorViewMut + Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v.view_mut::<3>(0).unwrap() + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_add() {
        // Matrix + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = &m1 + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = m1 + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m2.view::<2, 2>((0, 0));
        let result = &m1 + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = m1 + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &Matrix + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &m1 + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // Matrix + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = m1 + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // Matrix + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = m1 + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // &Matrix + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = &m1 + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // &Matrix + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m2.t();
        let result = &m1 + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // Matrix + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = m1 + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // Matrix + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = m1 + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // &Matrix + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // &Matrix + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m2.t_mut();
        let result = &m1 + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 9], [9, 12]]));

        // Matrix + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = m1 + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &Matrix + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let scalar = 5;
        let result = &m1 + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_view_add() {
        // MatrixView + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = view + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let result = &view + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 + view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = view1 + &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 + view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0));
        let view2 = m2.view::<2, 2>((0, 0));
        let result = &view1 + &view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = view + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &view + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixView + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixView + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = view + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &MatrixView + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_view_mut_add() {
        // MatrixViewMut + Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = view_mut + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let result = &view_mut + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = view_mut + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let view = m2.view::<2, 2>((0, 0));
        let result = &view_mut + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 + view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = view_mut1 + &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 + view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0));
        let view_mut2 = m2.view_mut::<2, 2>((0, 0));
        let result = &view_mut1 + &view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = view_mut + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view = m2.t();
        let result = &view_mut + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = view_mut + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixViewMut + &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let t_view_mut = m2.t_mut();
        let result = &view_mut + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixViewMut + Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = view_mut + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &MatrixViewMut + Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0));
        let scalar = 5;
        let result = &view_mut + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_transpose_view_add() {
        // MatrixTransposeView + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = t_view + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = t_view + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = &t_view + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + &Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let result = &t_view + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + &MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + &MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view = m1.t();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 + t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = t_view1 + &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 + t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = &t_view1 + &t_view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = t_view + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view + t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeView + &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view = m1.t();
        let t_view_mut = m2.t_mut();
        let result = &t_view + &t_view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeView + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = t_view + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &MatrixTransposeView + Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view = m1.t();
        let scalar = 5;
        let result = &t_view + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_add() {
        // MatrixTransposeViewMut + Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + &Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = &t_view_mut + &m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view_mut + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = t_view_mut + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view_mut + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + &MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view = m2.view::<2, 2>((0, 0));
        let result = &t_view_mut + &view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view_mut + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = t_view_mut + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view_mut + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut = m2.view_mut::<2, 2>((0, 0));
        let result = &t_view_mut + &view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = t_view_mut + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut + t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut = m1.t_mut();
        let t_view = m2.t();
        let result = &t_view_mut + &t_view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 + t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = t_view_mut1 + &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 + t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // &MatrixTransposeViewMut + &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = &t_view_mut1 + &t_view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]));

        // MatrixTransposeViewMut + Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = t_view_mut + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // &MatrixTransposeViewMut + Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let t_view_mut = m1.t_mut();
        let scalar = 5;
        let result = &t_view_mut + scalar;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));
    }
}
