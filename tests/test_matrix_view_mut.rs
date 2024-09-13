#[cfg(test)]
mod tests {
    use ferrix::Matrix;

    #[test]
    fn test_matrix_view_mut_shape() {
        let mut matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = matrix.view_mut::<2, 3>((0, 1)).unwrap();
        assert_eq!(view.shape(), (2, 3));
    }

    #[test]
    fn test_matrix_view_mut_t() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.view_mut::<2, 3>((0, 0)).unwrap();
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
    fn test_matrix_view_mut_t_mut() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut view = matrix.view_mut::<2, 3>((0, 0)).unwrap();
        let mut transposed = view.t_mut();
        transposed[(0, 1)] = 10;
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 10);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_view_mut_index() {
        let mut matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = matrix.view_mut::<2, 3>((1, 1)).unwrap();
        
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
        let mut matrix = Matrix::from([[1, 2], [3, 4]]);
        let view = matrix.view_mut::<2, 2>((0, 0)).unwrap();
        let _ = view[(2, 2)]; // This should panic
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_view_mut_index_single_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2], [3, 4]]);
        let view = matrix.view_mut::<2, 2>((0, 0)).unwrap();
        let _ = view[4]; // This should panic
    }

    #[test]
    fn test_matrix_view_mut_index_mut() {
        let mut matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut view = matrix.view_mut::<2, 3>((1, 1)).unwrap();
        
        view[(0, 0)] = 20;
        view[(1, 2)] = 30;

        assert_eq!(matrix[(1, 1)], 20);
        assert_eq!(matrix[(2, 3)], 30);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_view_mut_index_mut_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let mut view = matrix.view_mut::<2, 3>((0, 0)).unwrap();
        view[(2, 2)] = 10; // This should panic
    }

    #[test]
    fn test_matrix_view_mut_index_mut_single() {
        let mut matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut view = matrix.view_mut::<2, 3>((1, 1)).unwrap();
        
        view[1] = 40;
        view[4] = 50;

        assert_eq!(matrix[(1, 2)], 40);
        assert_eq!(matrix[(2, 2)], 50);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_view_mut_index_mut_single_out_of_bounds() {
        let mut matrix = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut view = matrix.view_mut::<2, 3>((1, 1)).unwrap();
        view[10] = 10; // This should panic
    }
}
