#[cfg(test)]
mod tests {
    use ferrix::{Matrix, MatrixView};

    #[test]
    fn test_matrix_view_shape() {
        let matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = matrix.view::<2, 3>((0, 1)).unwrap();
        assert_eq!(view.shape(), (2, 3));
    }

    #[test]
    fn test_matrix_view_t() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.view::<2, 3>((0, 0)).unwrap();
        let transposed = view.t();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed[(0, 0)], 1);
        assert_eq!(transposed[(1, 0)], 2);
        assert_eq!(transposed[(2, 0)], 3);
        assert_eq!(transposed[(0, 1)], 4);
        assert_eq!(transposed[(1, 1)], 5);
        assert_eq!(transposed[(2, 1)], 6);
    }

    #[test]
    fn test_matrix_view_eq() {
        let matrix1 = Matrix::from([[3, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[3, 2], [4, 5]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(view1, matrix2);
    }

    #[test]
    fn test_matrix_view_ne() {
        let matrix1 = Matrix::from([[3, 2, 3], [4, 5, 6]]);
        let matrix2 = Matrix::from([[3, 2], [4, 6]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        assert_ne!(view1, matrix2);
    }

    #[test]
    fn test_matrix_view_eq_view() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let matrix2 = Matrix::from([[3, 2, 4], [4, 5, 7]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        let view2 = matrix2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_matrix_view_ne_view() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let matrix2 = Matrix::from([[3, 2, 4], [5, 5, 7]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        let view2 = matrix2.view::<2, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_view_eq_view_mut() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let mut matrix2 = Matrix::from([[3, 2, 4], [4, 5, 7]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        let view2 = matrix2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(view1, view2);
    }
    
    #[test]
    fn test_matrix_view_ne_view_mut() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let mut matrix2 = Matrix::from([[3, 2, 4], [5, 5, 7]]);
        let view1 = matrix1.view::<2, 2>((0, 0)).unwrap();
        let view2 = matrix2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_view_eq_transpose_view() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let matrix2 = Matrix::from([[3, 4], [2, 5], [3, 6]]);
        let view1 = matrix1.view::<2, 3>((0, 0)).unwrap();
        let view2 = matrix2.t();
        assert_eq!(view1, view2);
    }
    
    #[test]
    fn test_matrix_view_ne_transpose_view() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let matrix2 = Matrix::from([[3, 4], [2, 5], [3, 7]]);
        let view1 = matrix1.view::<2, 3>((0, 0)).unwrap();
        let view2 = matrix2.t();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_view_eq_transpose_view_mut() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let mut matrix2 = Matrix::from([[3, 4], [2, 5], [3, 6]]);
        let view1 = matrix1.view::<2, 3>((0, 0)).unwrap();
        let view2 = matrix2.t_mut();
        assert_eq!(view1, view2);
    }
    
    #[test]
    fn test_matrix_view_ne_transpose_view_mut() {
        let matrix1 = Matrix::from([[3, 2, 3, 4], [4, 5, 6, 7]]);
        let mut matrix2 = Matrix::from([[3, 4], [2, 5], [3, 7]]);
        let view1 = matrix1.view::<2, 3>((0, 0)).unwrap();
        let view2 = matrix2.t_mut();
        assert_ne!(view1, view2);
    }

    #[test]
    fn test_matrix_view_display() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(format!("{}", view), "[[1, 2]\n [4, 5]]");
    }

    #[test]
    fn test_matrix_view_display_alternate() {
        let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(format!("{:#}", view), "MatrixView([[1, 2]\n            [4, 5]], dtype=i32)");
    }

    #[test]
    fn test_matrix_view_index() {
        let matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = matrix.view::<2, 3>((1, 1)).unwrap();
        
        // Test tuple indexing
        assert_eq!(view[(0, 0)], 6);
        assert_eq!(view[(0, 1)], 7);
        assert_eq!(view[(0, 2)], 8);
        assert_eq!(view[(1, 0)], 10);
        assert_eq!(view[(1, 1)], 11);
        assert_eq!(view[(1, 2)], 12);

        // Test single index (row-major order)
        assert_eq!(view[0], 6);
        assert_eq!(view[1], 7);
        assert_eq!(view[2], 8);
        assert_eq!(view[3], 10);
        assert_eq!(view[4], 11);
        assert_eq!(view[5], 12);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_view_index_out_of_bounds() {
        let matrix = Matrix::from([[1, 2], [3, 4]]);
        let view = matrix.view::<2, 2>((0, 0)).unwrap();
        let _ = view[(2, 2)]; // This should panic
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_view_index_single_out_of_bounds() {
        let matrix = Matrix::from([[1, 2], [3, 4]]);
        let view = matrix.view::<2, 2>((0, 0)).unwrap();
        let _ = view[4]; // This should panic
    }

    #[test]
    fn test_matrix_view_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MatrixView<'_, i32, 3, 3, 3, 3>>();
    }

    #[test]
    fn test_matrix_view_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MatrixView<'_, i32, 3, 3, 3, 3>>();
    }
}
