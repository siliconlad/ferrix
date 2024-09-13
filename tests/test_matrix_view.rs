#[cfg(test)]
mod tests {
    use ferrix::Matrix;

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

}
