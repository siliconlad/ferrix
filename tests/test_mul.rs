#[cfg(test)]
mod tests {
    use ferrix::{Matrix, RowVector, Vector};

    #[test]
    fn test_vector_mul() {
        // Vector * Scalar
        let v = Vector::<i32, 3>::from([1, 2, 3]);
        let result = v * 2;
        assert_eq!(result, Vector::from([2, 4, 6]));
    }

    #[test]
    fn test_vector_view_mul() {
        // VectorView * Scalar
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::from([4, 6, 8]));
    }

    #[test]
    fn test_vector_view_mut_mul() {
        // VectorViewMut * Scalar
        let mut v = Vector::<i32, 3>::from([1, 2, 3]);
        let result = v.view_mut::<3>(0).unwrap() * 2;
        assert_eq!(result, Vector::<i32, 3>::from([2, 4, 6]));
    }

    #[test]
    fn test_row_vector_mul() {
        // RowVector * Scalar
        let v = RowVector::<i32, 3>::from([1, 2, 3]);
        let result = v * 2;
        assert_eq!(result, RowVector::from([2, 4, 6]));
    }

    #[test]
    fn test_row_vector_view_mul() {
        // RowVectorView * Scalar
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 * 2;
        assert_eq!(result, RowVector::<i32, 3>::from([2, 4, 6]));
    }

    #[test]
    fn test_row_vector_view_mut_mul() {
        // RowVectorViewMut * Scalar
        let mut v = RowVector::<i32, 3>::from([1, 2, 3]);
        let result = v.view_mut::<3>(0).unwrap() * 2;
        assert_eq!(result, RowVector::<i32, 3>::from([2, 4, 6]));
    }

    #[test]
    fn test_matrix_mul() {
        // Matrix * Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let result = m1 * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_view_mul() {
        // MatrixView * Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_view_mut_mul() {
        // MatrixViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_transpose_view_mul() {
        // MatrixTransposeView * Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let result = m1.t() * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_mul() {
        // MatrixTransposeViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let result = m1.t_mut() * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[5, 10], [15, 20]]));
    }
}
