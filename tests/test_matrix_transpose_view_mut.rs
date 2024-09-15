#[cfg(test)]
mod tests {
    use ferrix::{Matrix, MatrixTransposeViewMut};

    #[test]
    fn test_matrix_transpose_view_mut_shape() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(view.shape(), (3, 2));
    }

    #[test]
    fn test_matrix_transpose_view_mut_t() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        let view_t = view.t();
        assert_eq!(view_t.shape(), (2, 3));
        assert_eq!(view_t[(0, 0)], 1);
        assert_eq!(view_t[(0, 1)], 2);
        assert_eq!(view_t[(0, 2)], 3);
        assert_eq!(view_t[(1, 0)], 4);
        assert_eq!(view_t[(1, 1)], 5);
        assert_eq!(view_t[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_transpose_view_mut_t_mut() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut view = matrix.t_mut();
        let mut view_t = view.t_mut();
        view_t[(0, 0)] = 10;
        view_t[(1, 1)] = 20;
        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 4);
        assert_eq!(matrix[(1, 1)], 20);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_transpose_view_mut_eq() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4], [2, 5], [3, 6]]);
        let view1 = matrix1.t_mut();
        assert_eq!(view1, matrix2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_ne() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4], [2, 5], [3, 7]]);
        let view1 = matrix1.t_mut();
        assert_ne!(view1, matrix2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_eq_view() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.view::<3, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_ne_view() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 7, 9]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.view::<3, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_eq_view_mut() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_ne_view_mut() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 7, 9]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_eq_transpose_view() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.t();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_ne_transpose_view() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 2, 3], [4, 5, 7]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.t();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_eq_transpose_view_mut() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.t_mut();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_ne_transpose_view_mut() {
        let mut matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 2, 3], [4, 5, 7]]);
        let view1 = matrix1.t_mut();
        let view2 = matrix2.t_mut();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_mut_display() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(format!("{}", view), "[[1, 4]\n [2, 5]\n [3, 6]]");
    }

    #[test]
    fn test_matrix_transpose_view_mut_display_alternate() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(format!("{:#}", view), "MatrixTransposeViewMut([[1, 4]\n                        [2, 5]\n                        [3, 6]], dtype=i32)");
    }

    #[test]
    fn test_matrix_transpose_view_mut_index() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(view[(0, 0)], 1);
        assert_eq!(view[(0, 1)], 4);
        assert_eq!(view[(1, 0)], 2);
        assert_eq!(view[(1, 1)], 5);
        assert_eq!(view[(2, 0)], 3);
        assert_eq!(view[(2, 1)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_transpose_view_mut_index_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        let _ = view[(3, 0)]; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_mut() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut view = matrix.t_mut();
        view[(0, 0)] = 10;
        view[(1, 1)] = 20;
        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 4);
        assert_eq!(matrix[(1, 1)], 20);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_transpose_view_mut_index_mut_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2], [3, 4]]);
        let mut view = matrix.t_mut();
        view[(1, 2)] = 10; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_single() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(view[0], 1);
        assert_eq!(view[1], 4);
        assert_eq!(view[2], 2);
        assert_eq!(view[3], 5);
        assert_eq!(view[4], 3);
        assert_eq!(view[5], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_transpose_view_mut_index_single_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        let _ = view[6]; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_mut_single() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut view = matrix.t_mut();
        view[0] = 10;
        view[1] = 20;
        assert_eq!(matrix[(0, 0)], 10);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 20);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_transpose_view_mut_index_mut_single_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2], [3, 4]]);
        let mut view = matrix.t_mut();
        view[4] = 10; // This should panic
    }

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MatrixTransposeViewMut<'_, i32, 3, 3, 2, 2>>();
    }

    #[test]
    fn test_not_sync() {
        fn assert_not_sync<T: Sync>() {}
        assert_not_sync::<MatrixTransposeViewMut<'_, i32, 3, 3, 2, 2>>();
    }
}
