#[cfg(test)]
mod tests {
    use ferrix::Matrix;

    #[test]
    fn test_matrix_transpose_view_mut_shape() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        assert_eq!(view.shape(), (3, 2));
    }

    #[test]
    fn test_matrix_transpose_view_mut_t() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
    fn test_matrix_transpose_view_mut_index() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        let _ = view[(3, 0)]; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_mut() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let mut matrix = Matrix::new([[1, 2], [3, 4]]);
        let mut view = matrix.t_mut();
        view[(1, 2)] = 10; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_single() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t_mut();
        let _ = view[6]; // This should panic
    }

    #[test]
    fn test_matrix_transpose_view_mut_index_mut_single() {
        let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let mut matrix = Matrix::new([[1, 2], [3, 4]]);
        let mut view = matrix.t_mut();
        view[4] = 10; // This should panic
    }
}
