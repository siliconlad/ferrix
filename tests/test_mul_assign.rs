#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_mul_assign() {
        // Vector *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v, Vector::<i32, 3>::new([2, 4, 6]));
    }

    #[test]
    fn test_vector_view_mut_mul_assign() {
        // VectorViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v, Vector::<i32, 3>::new([2, 4, 6]));
    }

    #[test]
    fn test_row_vector_mul_assign() {
        // RowVector *= Scalar
        let mut v = RowVector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([2, 4, 6]));
    }

    #[test]
    fn test_row_vector_view_mut_mul_assign() {
        // RowVectorViewMut *= Scalar
        let mut v = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([2, 4, 6]));
    }

    #[test]
    fn test_matrix_mul_assign() {
        // Matrix *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        m *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_view_mut_mul_assign() {
        // MatrixViewMut *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_mul_assign() {
        // MatrixTransposeViewMut *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut view = m.t_mut();
        view *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 6], [4, 8]]));
    }
}
