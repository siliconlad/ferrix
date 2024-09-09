#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_div_assign() {
        // Vector /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v, Vector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_vector_view_mut_div_assign() {
        // VectorViewMut /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v, Vector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_row_vector_div_assign() {
        // RowVector /= Scalar
        let mut v = RowVector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_row_vector_view_mut_div_assign() {
        // RowVectorViewMut /= Scalar
        let mut v = RowVector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v, RowVector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_matrix_div_assign() {
        // Matrix /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        m /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_view_mut_div_assign() {
        // MatrixViewMut /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div_assign() {
        // MatrixTransposeViewMut /= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[10, 30], [20, 40]]);
        let mut view = m.t_mut();
        view /= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[5, 15], [10, 20]]));
    }
}
