#[cfg(test)]
mod tests {
    use ferrix::{Matrix, MatrixTransposeView};

    #[test]
    fn test_matrix_transpose_view_shape() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        assert_eq!(view.shape(), (3, 2));
    }

    #[test]
    fn test_matrix_transpose_view_t() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        let view_t = view.t();
        assert_eq!(view_t.shape(), matrix.shape());
        assert_eq!(view_t[(0, 0)], matrix[(0, 0)]);
        assert_eq!(view_t[(0, 1)], matrix[(0, 1)]);
        assert_eq!(view_t[(0, 2)], matrix[(0, 2)]);
        assert_eq!(view_t[(1, 0)], matrix[(1, 0)]);
        assert_eq!(view_t[(1, 1)], matrix[(1, 1)]);
        assert_eq!(view_t[(1, 2)], matrix[(1, 2)]);
    }

    #[test]
    fn test_matrix_transpose_view_eq() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4], [2, 5], [3, 6]]);
        let view1 = matrix1.t();
        assert_eq!(view1, matrix2);
    }

    #[test]
    fn test_matrix_transpose_view_ne() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4], [2, 5], [3, 7]]);
        let view1 = matrix1.t();
        assert_ne!(view1, matrix2);
    }

    #[test]
    fn test_matrix_transpose_view_eq_view() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
        let view1 = matrix1.t();
        let view2 = matrix2.view::<3, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }
        
    #[test]
    fn test_matrix_transpose_view_ne_view() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 7, 9]]);
        let view1 = matrix1.t();
        let view2 = matrix2.view::<3, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_eq_view_mut() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
        let view1 = matrix1.t();
        let view2 = matrix2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }
    
    #[test]
    fn test_matrix_transpose_view_ne_view_mut() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 4, 7], [2, 5, 8], [3, 7, 9]]);
        let view1 = matrix1.t();
        let view2 = matrix2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_eq_transpose_view() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view1 = matrix1.t();
        let view2 = matrix2.t();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_ne_transpose_view() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[1, 2, 3], [4, 5, 7]]);
        let view1 = matrix1.t();
        let view2 = matrix2.t();
        assert_ne!(view1, view2);
    }
    
    #[test]
    fn test_matrix_transpose_view_eq_transpose_view_mut() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view1 = matrix1.t();
        let view2 = matrix2.t_mut();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_ne_transpose_view_mut() {
        let matrix1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut matrix2 = Matrix::from([[1, 2, 3], [4, 5, 7]]);
        let view1 = matrix1.t();
        let view2 = matrix2.t_mut();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_transpose_view_display() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        assert_eq!(format!("{}", view), "[[1, 4]\n [2, 5]\n [3, 6]]");
    }

    #[test]
    fn test_matrix_transpose_view_display_alternate() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        assert_eq!(format!("{:#}", view), "MatrixTransposeView([[1, 4]\n                     [2, 5]\n                     [3, 6]], dtype=i32)");
    }

    #[test]
    fn test_matrix_transpose_view_index() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();

        assert_eq!(view[(0, 0)], matrix[(0, 0)]);
        assert_eq!(view[(0, 1)], matrix[(1, 0)]);
        assert_eq!(view[(1, 0)], matrix[(0, 1)]);
        assert_eq!(view[(1, 1)], matrix[(1, 1)]);
        assert_eq!(view[(2, 0)], matrix[(0, 2)]);
        assert_eq!(view[(2, 1)], matrix[(1, 2)]);

        assert_eq!(view[0], matrix[(0, 0)]);
        assert_eq!(view[1], matrix[(1, 0)]);
        assert_eq!(view[2], matrix[(0, 1)]);
        assert_eq!(view[3], matrix[(1, 1)]);
        assert_eq!(view[4], matrix[(0, 2)]);
        assert_eq!(view[5], matrix[(1, 2)]);
    }

    #[test]
    fn test_matrix_transpose_view_index_offset() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view_offset = matrix.view::<2, 2>((1, 1)).unwrap();
        let view = view_offset.t();
        assert_eq!(view[(0, 0)], matrix[(1, 1)]);
        assert_eq!(view[(0, 1)], matrix[(2, 1)]);
        assert_eq!(view[(1, 0)], matrix[(1, 2)]);
        assert_eq!(view[(1, 1)], matrix[(2, 2)]);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_transpose_view_index_out_of_bounds() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        let _ = view[(0, 2)];
    }

    #[test]
    fn test_matrix_transpose_view_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MatrixTransposeView<'_, i32, 3, 3, 3, 3>>();
    }

    #[test]
    fn test_matrix_transpose_view_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MatrixTransposeView<'_, i32, 3, 3, 3, 3>>();
    }
}
