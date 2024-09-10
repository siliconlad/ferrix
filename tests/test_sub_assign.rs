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

        // Vector -= Matrix (1-column matrix as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        v1 -= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= MatrixView (1-column matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]);
        v1 -= m2.view::<3, 1>((0, 0)).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= MatrixViewMut (1-column matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]);
        v1 -= m2.view_mut::<3, 1>((0, 0)).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= MatrixTransposeView (1-row matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        v1 -= m2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // Vector -= MatrixTransposeViewMut (1-row matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        v1 -= m2.t_mut();
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

        // VectorViewMut -= Matrix (1-column matrix as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let v2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= MatrixView (1-column matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.view::<3, 1>((0, 0)).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= MatrixViewMut (1-column matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.view_mut::<3, 1>((0, 0)).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= MatrixTransposeView (1-row matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([3, 3, 3]));

        // VectorViewMut -= MatrixTransposeViewMut (1-row matrix view as vector)
        let mut v1 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.t_mut();
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

        // RowVector -= Matrix (1-row matrix as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        v1 -= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= MatrixView (1-row matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [0, 0, 0]]);
        v1 -= m2.view::<1, 3>((0, 0)).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= MatrixViewMut (1-row matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [0, 0, 0]]);
        v1 -= m2.view_mut::<1, 3>((0, 0)).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= MatrixTransposeView (1-column matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        v1 -= m2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVector -= MatrixTransposeViewMut (1-column matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        v1 -= m2.t_mut();
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

        // RowVectorViewMut -= Matrix (1-row matrix as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let v2 = Matrix::<i32, 1, 3>::new([[1, 2, 3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= MatrixView (1-row matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [0, 0, 0]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.view::<1, 3>((0, 0)).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= MatrixViewMut (1-row matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [0, 0, 0]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.view_mut::<1, 3>((0, 0)).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= MatrixTransposeView (1-column matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let m2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([3, 3, 3]));

        // RowVectorViewMut -= MatrixTransposeViewMut (1-column matrix view as row vector)
        let mut v1 = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut m2 = Matrix::<i32, 3, 1>::new([[1], [2], [3]]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= m2.t_mut();
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

        // Matrix -= Vector (as column)
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        m -= v;
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));

        // Matrix -= VectorView
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        m -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));

        // Matrix -= VectorViewMut
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let mut v = Vector::<i32, 3>::new([4, 5, 6]);
        m -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));

        // Matrix -= RowVector (as row)
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        m -= v;
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // Matrix -= RowVectorView
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        m -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // Matrix -= RowVectorViewMut
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let mut v = RowVector::<i32, 3>::new([4, 5, 6]);
        m -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));
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

        // MatrixViewMut -= Vector (as column)
        let mut m = Matrix::<i32, 3, 2>::new([[5, 0], [7, 0], [9, 0]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<3, 1>((0, 0)).unwrap();
        view -= v;
        assert_eq!(m, Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]));

        // MatrixViewMut -= VectorView
        let mut m = Matrix::<i32, 3, 2>::new([[5, 0], [7, 0], [9, 0]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<3, 1>((0, 0)).unwrap();
        view -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]));

        // MatrixViewMut -= VectorViewMut
        let mut m = Matrix::<i32, 3, 2>::new([[5, 0], [7, 0], [9, 0]]);
        let mut v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<3, 1>((0, 0)).unwrap();
        view -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 2>::new([[1, 0], [2, 0], [3, 0]]));

        // MatrixViewMut -= RowVector (as row)
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<1, 3>((0, 0)).unwrap();
        view -= v;
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // MatrixViewMut -= RowVectorView
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<1, 3>((0, 0)).unwrap();
        view -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // MatrixViewMut -= RowVectorViewMut
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let mut v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.view_mut::<1, 3>((0, 0)).unwrap();
        view -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));
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

        // MatrixTransposeViewMut -= Vector (as column)
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v;
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // MatrixTransposeViewMut -= VectorView
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // MatrixTransposeViewMut -= VectorViewMut
        let mut m = Matrix::<i32, 1, 3>::new([[5, 7, 9]]);
        let mut v = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 1, 3>::new([[1, 2, 3]]));

        // MatrixTransposeViewMut -= RowVector (as row)
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v;
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));

        // MatrixTransposeViewMut -= RowVectorView
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v.view::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));

        // MatrixTransposeViewMut -= RowVectorViewMut
        let mut m = Matrix::<i32, 3, 1>::new([[5], [7], [9]]);
        let mut v = RowVector::<i32, 3>::new([4, 5, 6]);
        let mut view = m.t_mut();
        view -= v.view_mut::<3>(0).unwrap();
        assert_eq!(m, Matrix::<i32, 3, 1>::new([[1], [2], [3]]));
    }
}
