#[cfg(test)]
mod tests {
    use ferrix::Matrix;

    #[test]
    fn test_matrix_transpose_view_shape() {
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        assert_eq!(view.shape(), (3, 2));
    }

    #[test]
    fn test_matrix_transpose_view_t() {
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
    fn test_matrix_transpose_view_index() {
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
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
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
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
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let view = matrix.t();
        let _ = view[(0, 2)];
    }
}
