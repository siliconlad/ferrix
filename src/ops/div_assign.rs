use funty::Numeric;
use std::ops::DivAssign;

use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;
use crate::matrix::RowVector;
use crate::matrix_view::RowVectorView;
use crate::matrix_view_mut::RowVectorViewMut;
use crate::matrix_transpose_view::RowVectorTransposeView;
use crate::matrix_transpose_view_mut::RowVectorTransposeViewMut;

macro_rules! impl_vv_div_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> DivAssign<$rhs> for $lhs {
            fn div_assign(&mut self, other: $rhs) {
                (0..N).for_each(|i| self[i] /= other[i]);
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> DivAssign<T> for $lhs {
            fn div_assign(&mut self, scalar: T) {
                (0..N).for_each(|i| self[i] /= scalar);
            }
        }
    };
}

macro_rules! impl_vv_div_assign_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize, const M: usize> DivAssign<$rhs> for $lhs {
            fn div_assign(&mut self, other: $rhs) {
                (0..N).for_each(|i| self[i] /= other[i]);
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize, const M: usize> DivAssign<T> for $lhs {
            fn div_assign(&mut self, scalar: T) {
                (0..N).for_each(|i| self[i] /= scalar);
            }
        }
    };
}

macro_rules! impl_vv_div_assign_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const N: usize, const M: usize> DivAssign<$rhs> for $lhs {
            fn div_assign(&mut self, other: $rhs) {
                (0..N).for_each(|i| self[i] /= other[i]);
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const N: usize, const M: usize> DivAssign<T> for $lhs {
            fn div_assign(&mut self, scalar: T) {
                (0..N).for_each(|i| self[i] /= scalar);
            }
        }
    };
}

macro_rules! impl_mm_div_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> DivAssign<$rhs> for $lhs {
            fn div_assign(&mut self, other: $rhs) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] /= other[(i, j)]));
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> DivAssign<T> for $lhs {
            fn div_assign(&mut self, scalar: T) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] /= scalar));
            }
        }
    };
}

macro_rules! impl_mm_div_assign_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize>
            DivAssign<$rhs> for $lhs
        {
            fn div_assign(&mut self, other: $rhs) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] /= other[(i, j)]));
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize>
            DivAssign<T> for $lhs
        {
            fn div_assign(&mut self, scalar: T) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] /= scalar));
            }
        }
    };
}

macro_rules! impl_mm_div_assign_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<
                T: Numeric + From<u8>,
                const A: usize,
                const B: usize,
                const C: usize,
                const D: usize,
                const M: usize,
                const N: usize,
            > DivAssign<$rhs> for $lhs
        {
            fn div_assign(&mut self, other: $rhs) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] /= other[(i, j)]));
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

impl_vv_div_assign!(Vector<T, N>); // Scalar
impl_vv_div_assign!(Vector<T, N>, Vector<T, N>);
impl_vv_div_assign!(Vector<T, N>, &Vector<T, N>);
impl_vv_div_assign_view!(Vector<T, M>, VectorView<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, &VectorView<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, VectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, &VectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, RowVectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_vv_div_assign_view!(Vector<T, M>, &RowVectorTransposeViewMut<'_, T, N, M>);


/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_div_assign_view!(VectorViewMut<'_, T, N, M>); // Scalar
impl_vv_div_assign_view!(VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_vv_div_assign_view!(VectorViewMut<'_, T, N, M>, &Vector<T, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, &VectorView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorViewMut<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_vv_div_assign_view!(VectorTransposeViewMut<'_, T, N, M>);  // Scalar
impl_vv_div_assign_view!(VectorTransposeViewMut<'_, T, N, M>, RowVector<T, M>);
impl_vv_div_assign_view!(VectorTransposeViewMut<'_, T, N, M>, &RowVector<T, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, &RowVectorView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

/////////////////
//  RowVector  //
/////////////////

impl_vv_div_assign_view!(RowVector<T, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view!(RowVector<T, M>, &VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view!(RowVector<T, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_vv_div_assign_view!(RowVector<T, M>, &VectorTransposeViewMut<'_, T, N, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_vv_div_assign_view_view!(RowVectorViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(RowVectorViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_vv_div_assign_view_view!(RowVectorViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_vv_div_assign_view_view!(RowVectorViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

/////////////////////////////////
//  RowVectorTransposeViewMut  //
/////////////////////////////////

impl_vv_div_assign_view!(RowVectorTransposeViewMut<'_, T, N, M>, Vector<T, M>);
impl_vv_div_assign_view!(RowVectorTransposeViewMut<'_, T, N, M>, &Vector<T, M>);
impl_vv_div_assign_view_view!(RowVectorTransposeViewMut<'_, T, N, M>, VectorView<'_, T, A, M>);
impl_vv_div_assign_view_view!(RowVectorTransposeViewMut<'_, T, N, M>, &VectorView<'_, T, A, M>);
impl_vv_div_assign_view_view!(RowVectorTransposeViewMut<'_, T, N, M>, VectorViewMut<'_, T, A, M>);
impl_vv_div_assign_view_view!(RowVectorTransposeViewMut<'_, T, N, M>, &VectorViewMut<'_, T, A, M>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_div_assign!(Matrix<T, M, N>);
impl_mm_div_assign!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_div_assign!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_div_assign_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, M, N>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_div_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_div_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_div_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);
impl_mm_div_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, M, N>);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_assign() {
        // Vector /= Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2;
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // Floating-point versions
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2;
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // Vector /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v /= 0.5;
        assert!((v[0] - 2.0).abs() < f64::EPSILON);
        assert!((v[1] - 4.0).abs() < f64::EPSILON);
        assert!((v[2] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_div_assign() {
        // VectorViewMut /= Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2;
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut /= &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2;
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut /= VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut /= &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut /= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // &VectorViewMut /= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < f64::EPSILON);
        assert!((v1[1] - 10.0).abs() < f64::EPSILON);
        assert!((v1[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 0.5;
        assert!((v[0] - 2.0).abs() < f64::EPSILON);
        assert!((v[1] - 4.0).abs() < f64::EPSILON);
        assert!((v[2] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_div_assign() {
        // Matrix /= Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= &m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 3, 3>::new([[2.0, 4.0, 0.0], [5.0, 8.0, 0.0], [0.0, 0.0, 1.0]]);
        m1 /= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 3, 3>::new([[2.0, 4.0, 0.0], [5.0, 8.0, 0.0], [0.0, 0.0, 1.0]]);
        m1 /= &m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 3, 3>::new([[2.0, 4.0, 0.0], [5.0, 8.0, 0.0], [0.0, 0.0, 1.0]]);
        m1 /= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 3, 3>::new([[2.0, 4.0, 0.0], [5.0, 8.0, 0.0], [0.0, 0.0, 1.0]]);
        m1 /= &m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixTransposeView
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 5.0, 10.0], [4.0, 8.0, 12.0]]);
        m1 /= m2.t();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 5.0], [6.0, 5.0], [5.0, 5.0]])
        );

        // Matrix /= &MatrixTransposeView
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 5.0, 10.0], [4.0, 8.0, 12.0]]);
        m1 /= &m2.t();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 5.0], [6.0, 5.0], [5.0, 5.0]])
        );

        // Matrix /= MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]);
        let mut m2 = Matrix::<f64, 2, 3>::new([[2.0, 5.0, 10.0], [4.0, 8.0, 12.0]]);
        m1 /= m2.t_mut();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 5.0], [6.0, 5.0], [5.0, 5.0]])
        );

        // Matrix /= &MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]);
        let mut m2 = Matrix::<f64, 2, 3>::new([[2.0, 5.0, 10.0], [4.0, 8.0, 12.0]]);
        m1 /= &m2.t_mut();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 5.0], [6.0, 5.0], [5.0, 5.0]])
        );

        // Matrix /= Scalar
        let mut m = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        m /= 2.0;
        assert_eq!(m, Matrix::<f64, 2, 2>::new([[5.0, 10.0], [15.0, 20.0]]));
    }

    #[test]
    fn test_matrix_view_mut_div_assign() {
        // MatrixViewMut /= Matrix
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2;
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= &Matrix
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= &m2;
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= MatrixView
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= &MatrixView
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= &m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= MatrixViewMut
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= &MatrixViewMut
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= &m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= MatrixTransposeView
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.t();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= &MatrixTransposeView
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= &m2.t();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= MatrixTransposeViewMut
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.t_mut();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= &MatrixTransposeViewMut
        let mut m1 =
            Matrix::<f64, 3, 3>::new([[10.0, 20.0, 1.0], [30.0, 40.0, 1.0], [1.0, 1.0, 1.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= &m2.t_mut();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 3>::new([[5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
        );

        // MatrixViewMut /= Scalar
        let mut m = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view /= 2.0;
        assert_eq!(m, Matrix::<f64, 2, 2>::new([[5.0, 10.0], [15.0, 20.0]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div_assign() {
        // MatrixTransposeViewMut /= Matrix
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= m2;
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= &Matrix
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= &m2;
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= MatrixView
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= &MatrixView
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= &m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= MatrixViewMut
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let mut m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= &MatrixViewMut
        let mut m1 = Matrix::<f64, 3, 2>::new([[10.0, 30.0], [20.0, 40.0], [50.0, 60.0]]);
        let mut m2 = Matrix::<f64, 2, 3>::new([[2.0, 4.0, 5.0], [5.0, 8.0, 10.0]]);
        let mut view = m1.t_mut();
        view /= &m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(
            m1,
            Matrix::<f64, 3, 2>::new([[5.0, 6.0], [5.0, 5.0], [10.0, 6.0]])
        );

        // MatrixTransposeViewMut /= MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= &MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= &m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= &MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= &m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= Scalar
        let mut m = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let mut view = m.t_mut();
        view /= 2.0;
        assert_eq!(m, Matrix::<f64, 2, 2>::new([[5.0, 15.0], [10.0, 20.0]]));
    }
}
