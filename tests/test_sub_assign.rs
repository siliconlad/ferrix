#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_sub_assign() {
        // Vector -= Scalar
        let mut v = Vector::<i32, 3>::new([5, 6, 7]);
        v -= 2;
        assert_eq!(v, Vector::<i32, 3>::new([3, 4, 5]));

        // Vector -= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([0, 1, 2]);
        v1 -= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([1, 1, 1]));

        // Vector -= Vector
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= VectorView (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.t_mut();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));
    }

    #[test]
    fn test_vector_view_mut_sub_assign() {
        // VectorViewMut -= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view -= 2;
        assert_eq!(v, Vector::<i32, 3>::new([-1, 0, 1]));

        // VectorViewMut -= Vector
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= VectorView (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.t_mut();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));
    }

    #[test]
    fn test_row_vector_sub_assign() {
        // RowVector -= Scalar
        let mut v = RowVector::<i32, 3>::new([1, 2, 3]);
        v -= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([-1, 0, 1]));

        // RowVector -= RowVector
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([0, 1, 2]);
        v1 -= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([1, 1, 1]));

        // RowVector -= RowVector
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.view::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 -= v2.t_mut();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));
    }

    #[test]
    fn test_row_vector_view_mut_sub_assign() {
        // RowVectorViewMut -= Scalar
        let mut v = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view -= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([-1, 0, 1]));
        
        // RowVectorViewMut -= RowVector
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.t_mut();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));
    }

    #[test]
    fn test_matrix_sub_assign() {
        // Matrix -= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]);
        m -= 5;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]));

        // Matrix -= Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        m1 -= m2;
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]));

        // Matrix -= MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 0]]);
        m1 -= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]));

        // Matrix -= MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[6, 8], [10, 12]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 0]]);
        m1 -= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]));

        // Matrix -= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[6, 8], [10, 12], [14, 16]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        m1 -= m2.t();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]));

        // Matrix -= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[6, 8], [10, 12], [14, 16]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        m1 -= m2.t_mut();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]));
    }

    #[test]
    fn test_matrix_view_mut_sub_assign() {
        // MatrixViewMut -= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view -= 5;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]));

        // MatrixViewMut -= Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[6, 8, 0], [10, 12, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view -= m2;
        assert_eq!(m1, Matrix::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]));

        // MatrixViewMut -= MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[6, 8, 0], [10, 12, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view -= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]));

        // MatrixViewMut -= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[6, 8, 0], [10, 12, 0], [0, 0, 0]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view -= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]));

        // MatrixViewMut -= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[6, 8, 0], [10, 12, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view -= m2.t();
        assert_eq!(m1, Matrix::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]));

        // MatrixViewMut -= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[6, 8, 0], [10, 12, 0], [0, 0, 0]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view -= m2.t_mut();
        assert_eq!(m1, Matrix::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_sub_assign() {
        // MatrixTransposeViewMut -= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[6, 8], [7, 9]]);
        let mut view = m.t_mut();
        view -= 5;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]));

        // MatrixTransposeViewMut -= Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[6, 9], [8, 12], [10, 15]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [9, 8, 10]]);
        let mut view = m1.t_mut();
        view -= m2;
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[1, 0], [2, 4], [3, 5]]));

        // MatrixTransposeViewMut -= MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[6, 9], [8, 12], [10, 15]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [9, 8, 10]]);
        let mut view = m1.t_mut();
        view -= m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[1, 0], [2, 4], [3, 5]]));

        // MatrixTransposeViewMut -= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[6, 9], [8, 12], [10, 15]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [9, 8, 10]]);
        let mut view = m1.t_mut();
        view -= m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[1, 0], [2, 4], [3, 5]]));

        // MatrixTransposeViewMut -= MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[6, 9], [8, 12]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [8, 8]]);
        let mut view = m1.t_mut();
        view -= m2.t();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[1, 2], [0, 4]]));

        // MatrixTransposeViewMut -= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[6, 9], [8, 12]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [8, 8]]);
        let mut view = m1.t_mut();
        view -= m2.t_mut();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[1, 2], [0, 4]]));
    }
}
