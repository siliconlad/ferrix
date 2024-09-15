#[cfg(test)]
mod tests {
    use ferrix::{Matrix, RowVector, Vector};

    #[test]
    fn test_vector_div() {
        // Vector / Scalar
        let v = Vector::<i32, 3>::from([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result, Vector::from([1, 2, 3]));
    }

    #[test]
    fn test_vector_view_div() {
        // Test scalar division
        let v1 = Vector::<f64, 5>::from([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::from([10.0, 15.0, 20.0]));
    }

    #[test]
    fn test_vector_view_mut_div() {
        // VectorViewMut / Scalar
        let mut v = Vector::<i32, 3>::from([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result, Vector::<i32, 3>::from([1, 2, 3]));
    }

    #[test]
    fn test_row_vector_div() {
        // RowVector / Scalar
        let v = RowVector::<i32, 3>::from([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result, RowVector::from([1, 2, 3]));
    }

    #[test]
    fn test_row_vector_view_div() {
        // RowVectorView / Scalar
        let v1 = RowVector::<f64, 5>::from([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, RowVector::<f64, 3>::from([10.0, 15.0, 20.0]));
    }

    #[test]
    fn test_row_vector_view_mut_div() {
        // RowVectorViewMut / Scalar
        let mut v = RowVector::<i32, 3>::from([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result, RowVector::<i32, 3>::from([1, 2, 3]));
    }

    #[test]
    fn test_matrix_div() {
        // Matrix / Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[10, 20], [30, 40]]);
        let result = m1 / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_view_div() {
        // MatrixView / Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[10, 20], [30, 40]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_view_mut_div() {
        // MatrixViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[10, 20], [30, 40]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_transpose_view_div() {
        // MatrixTransposeView / Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[30, 40], [35, 48]]);
        let result = m1.t() / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 7], [8, 9]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div() {
        // MatrixTransposeViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[30, 40], [35, 50]]);
        let result = m1.t_mut() / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 7], [8, 10]]));
    }
}
