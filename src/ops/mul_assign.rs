use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::MulAssign;

macro_rules! impl_vv_mul_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> MulAssign<$rhs> for $lhs {
            fn mul_assign(&mut self, other: $rhs) {
                (0..N).for_each(|i| self[i] *= other[i]);
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> MulAssign<T> for $lhs {
            fn mul_assign(&mut self, scalar: T) {
                (0..N).for_each(|i| self[i] *= scalar);
            }
        }
    };
}

macro_rules! impl_mm_mul_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> MulAssign<$rhs> for $lhs {
            fn mul_assign(&mut self, other: $rhs) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] *= other[(i, j)]));
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const M: usize, const N: usize> MulAssign<T> for $lhs {
            fn mul_assign(&mut self, scalar: T) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] *= scalar));
            }
        }
    };
}

macro_rules! impl_mm_mul_assign_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize>
            MulAssign<$rhs> for $lhs
        {
            fn mul_assign(&mut self, other: $rhs) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] *= other[(i, j)]));
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const A: usize, const B: usize, const M: usize, const N: usize>
            MulAssign<T> for $lhs
        {
            fn mul_assign(&mut self, scalar: T) {
                (0..M).for_each(|i| (0..N).for_each(|j| self[(i, j)] *= scalar));
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

impl_vv_mul_assign!(Vector<T, N>); // Scalar
impl_vv_mul_assign!(Vector<T, N>, Vector<T, N>);
impl_vv_mul_assign!(Vector<T, N>, &Vector<T, N>);
impl_vv_mul_assign!(Vector<T, N>, VectorView<'_, T, N>);
impl_vv_mul_assign!(Vector<T, N>, &VectorView<'_, T, N>);
impl_vv_mul_assign!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_vv_mul_assign!(Vector<T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_mul_assign!(VectorViewMut<'_, T, N>); // Scalar
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_vv_mul_assign!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_mul_assign!(Matrix<T, M, N>);
impl_mm_mul_assign!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_mul_assign!(Matrix<T, M, N>, &Matrix<T, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, M, N>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_mul_assign_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_mul_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_mul_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_mul_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, M, N>);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeView<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    MatrixTransposeViewMut<'_, T, A, B, M, N>
);
impl_mm_mul_assign_view!(
    MatrixTransposeViewMut<'_, T, A, B, M, N>,
    &MatrixTransposeViewMut<'_, T, A, B, M, N>
);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_mul_assign() {
        // Vector *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // Vector *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v *= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.0).abs() < f64::EPSILON);
        assert!((v[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_mul_assign() {
        // VectorViewMut *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // VectorViewMut *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.0).abs() < f64::EPSILON);
        assert!((v[2] - 1.5).abs() < f64::EPSILON);
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
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= &m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        m1 /= &m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        m1 /= m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        m1 /= &m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        m1 /= m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= &MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        m1 /= &m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // Matrix /= Scalar
        let mut m = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        m /= 2.0;
        assert_eq!(m, Matrix::<f64, 2, 2>::new([[5.0, 10.0], [15.0, 20.0]]));
    }

    #[test]
    fn test_matrix_view_mut_div_assign() {
        // MatrixViewMut /= Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= &Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= &m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= &MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= &m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= &MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= &m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= &MatrixTransposeView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= &m2.t();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= &MatrixTransposeViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 5.0], [4.0, 8.0]]);
        let mut view = m1.view_mut::<2, 2>((0, 0));
        view /= &m2.t_mut();
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 5.0], [6.0, 5.0]]));

        // MatrixViewMut /= Scalar
        let mut m = Matrix::<f64, 2, 2>::new([[10.0, 20.0], [30.0, 40.0]]);
        let mut view = m.view_mut::<2, 2>((0, 0));
        view /= 2.0;
        assert_eq!(m, Matrix::<f64, 2, 2>::new([[5.0, 10.0], [15.0, 20.0]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div_assign() {
        // MatrixTransposeViewMut /= Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= &Matrix
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= &m2;
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= &MatrixView
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= &m2.view::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

        // MatrixTransposeViewMut /= &MatrixViewMut
        let mut m1 = Matrix::<f64, 2, 2>::new([[10.0, 30.0], [20.0, 40.0]]);
        let mut m2 = Matrix::<f64, 2, 2>::new([[2.0, 4.0], [5.0, 8.0]]);
        let mut view = m1.t_mut();
        view /= &m2.view_mut::<2, 2>((0, 0));
        assert_eq!(m1, Matrix::<f64, 2, 2>::new([[5.0, 6.0], [5.0, 5.0]]));

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
