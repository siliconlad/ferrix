#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_div_assign() {
        // Vector /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v, Vector::<i32, 3>::new([1, 2, 3]));

        // Vector /= Vector
        let mut v1 = Vector::<i32, 3>::new([2, 2, 9]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 3]));

        // Vector /= VectorView
        let mut v1 = Vector::<i32, 3>::new([2, 2, 9]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 3]));

        // Vector /= VectorView (transposed)
        let mut v1 = Vector::<i32, 3>::new([2, 2, 9]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 3]));

        // Vector /= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([2, 2, 9]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 3]));

        // Vector /= VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([2, 2, 9]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.t_mut();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 3]));
    }

    #[test]
    fn test_vector_view_mut_div_assign() {
        // VectorViewMut /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v, Vector::<i32, 3>::new([1, 2, 3]));

        // VectorViewMut /= Vector
        let mut v1 = Vector::<i32, 3>::new([2, 2, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 2]));

        // VectorViewMut /= VectorView
        let mut v1 = Vector::<i32, 3>::new([2, 2, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 2]));

        // VectorViewMut /= VectorView (transposed)
        let mut v1 = Vector::<i32, 3>::new([2, 2, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 2]));

        // VectorViewMut /= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([2, 2, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 2]));

        // VectorViewMut /= VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([2, 2, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.t_mut();
        assert_eq!(v1, Vector::<i32, 3>::new([2, 1, 2]));
    }

    #[test]
    fn test_row_vector_div_assign() {
        // RowVector /= Scalar
        let mut v = RowVector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([1, 2, 3]));

        // RowVector /= RowVector
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 9]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 3]));

        // RowVector /= RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 9]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.view::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 3]));

        // RowVector /= RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 9]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 3]));

        // RowVector /= RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 9]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 3]));

        // RowVector /= RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 9]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 /= v2.t_mut();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 3]));
    }

    #[test]
    fn test_row_vector_view_mut_div_assign() {
        // RowVectorViewMut /= Scalar
        let mut v = RowVector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([1, 2, 3]));

        // RowVectorViewMut /= RowVector
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2;
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 2]));

        // RowVectorViewMut /= RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 6]);
        let v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 2]));

        // RowVectorViewMut /= RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 6]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.t();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 2]));

        // RowVectorViewMut /= RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 6]);
        let mut v2 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 2]));

        // RowVectorViewMut /= RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([2, 2, 6]);
        let mut v2 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.t_mut();
        assert_eq!(v1, RowVector::<i32, 3>::new([2, 1, 2]));
    }

    #[test]
    fn test_matrix_div_assign() {
        // Matrix /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        m /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // Matrix /= Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 4], [5, 8]]);
        m1 /= m2;
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[5, 5], [6, 5]]));

        // Matrix /= MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[2, 4, 0], [5, 8, 0], [0, 0, 1]]);
        m1 /= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[5, 5], [6, 5]]));

        // Matrix /= MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[2, 4, 0], [5, 8, 0], [0, 0, 1]]);
        m1 /= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[5, 5], [6, 5]]));

        // Matrix /= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[10, 20], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 5, 10], [4, 8, 12]]);
        m1 /= m2.t();
        assert_eq!(m1, Matrix::new([[5, 5], [6, 5], [5, 5]]));

        // Matrix /= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[10, 20], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[2, 5, 10], [4, 8, 12]]);
        m1 /= m2.t_mut();
        assert_eq!(m1, Matrix::new([[5, 5], [6, 5], [5, 5]]));
    }

    #[test]
    fn test_matrix_view_mut_div_assign() {
        // MatrixViewMut /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // MatrixViewMut /= Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 20, 1], [30, 40, 1], [1, 1, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 4], [5, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2;
        assert_eq!(m1, Matrix::new([[5, 5, 1], [6, 5, 1], [1, 1, 1]]));

        // MatrixViewMut /= MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 20, 1], [30, 40, 1], [1, 1, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 4], [5, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[5, 5, 1], [6, 5, 1], [1, 1, 1]]));

        // MatrixViewMut /= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 20, 1], [30, 40, 1], [1, 1, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 4], [5, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[5, 5, 1], [6, 5, 1], [1, 1, 1]]));

        // MatrixViewMut /= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 20, 1], [30, 40, 1], [1, 1, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.t();
        assert_eq!(m1, Matrix::new([[5, 5, 1], [6, 5, 1], [1, 1, 1]]));

        // MatrixViewMut /= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 20, 1], [30, 40, 1], [1, 1, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view /= m2.t_mut();
        assert_eq!(m1, Matrix::new([[5, 5, 1], [6, 5, 1], [1, 1, 1]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div_assign() {
        // MatrixTransposeViewMut /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 30], [20, 40]]);
        let mut view = m.t_mut();
        view /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 15], [10, 20]]));

        // MatrixTransposeViewMut /= Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[10, 30], [20, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 4, 5], [5, 8, 10]]);
        let mut view = m1.t_mut();
        view /= m2;
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[5, 6], [5, 5], [10, 6]]));

        // MatrixTransposeViewMut /= MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[10, 30], [20, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 4, 5], [5, 8, 10]]);
        let mut view = m1.t_mut();
        view /= m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[5, 6], [5, 5], [10, 6]]));

        // MatrixTransposeViewMut /= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[10, 30], [20, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[2, 4, 5], [5, 8, 10]]);
        let mut view = m1.t_mut();
        view /= m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[5, 6], [5, 5], [10, 6]]));

        // MatrixTransposeViewMut /= MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 30], [20, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.t_mut();
        view /= m2.t();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[5, 6], [5, 5]]));

        // MatrixTransposeViewMut /= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 30], [20, 40]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.t_mut();
        view /= m2.t_mut();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[5, 6], [5, 5]]));
    }
}
