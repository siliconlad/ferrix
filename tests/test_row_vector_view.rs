#[cfg(test)]
mod tests {
    use ferrix::RowVector;

    #[test]
    fn test_row_vector_view_shape() {
        let v = RowVector::from([1, 2, 3, 4]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.shape(), 3);
    }

    #[test]
    fn test_row_vector_view_capacity() {
        let v = RowVector::from([1, 2, 3, 4]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.capacity(), 3);
    }

    #[test]
    fn test_row_vector_view_t() {
        let v = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let view_from_t = view.t();

        assert_eq!(view_from_t.shape(), 3);
        assert_eq!(view_from_t[0], 2.0);
        assert_eq!(view_from_t[1], 3.0);
        assert_eq!(view_from_t[2], 4.0);
    }

    #[test]
    fn test_row_vector_view_magnitude() {
        let v = RowVector::from([3.0, 4.0, 5.0]);
        let view = v.view::<2>(0).unwrap();
        assert!((view.magnitude() - 5.0) < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_index_usize() {
        let v = RowVector::from([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[0], 2);
        assert_eq!(view[1], 3);
        assert_eq!(view[2], 4);
    }

    #[test]
    #[should_panic]
    fn test_row_vector_view_index_out_of_bounds() {
        let v = RowVector::from([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[5], 6);
    }

    #[test]
    fn test_row_vector_view_index_tuple() {
        let v = RowVector::from([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[(0, 0)], 2);
        assert_eq!(view[(0, 1)], 3);
        assert_eq!(view[(0, 2)], 4);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_row_vector_view_index_tuple_out_of_bounds_row() {
        let v = RowVector::from([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[(1, 0)], 6);
    }

    #[test]
    #[should_panic]
    fn test_row_vector_view_index_tuple_out_of_bounds_col() {
        let v = RowVector::from([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[(0, 4)], 6);
    }
}
