#[cfg(test)]
mod tests {
    use ferrix::Vector;

    #[test]
    fn test_vector_transpose_view_shape() {
        let v = Vector::new([1, 2, 3, 4]);
        let view = v.view::<3>(1).unwrap();
        let t_view = view.t();
        assert_eq!(t_view.shape(), 3);
    }

    #[test]
    fn test_vector_transpose_view_t() {
        let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let t_view = view.t();
        let view_from_t = t_view.t();

        assert_eq!(view_from_t.shape(), 3);
        assert_eq!(view_from_t[0], 2.0);
        assert_eq!(view_from_t[1], 3.0);
        assert_eq!(view_from_t[2], 4.0);
    }

    #[test]
    fn test_vector_transpose_view_magnitude() {
        let v = Vector::new([3.0, 4.0, 5.0]);
        let view = v.view::<2>(0).unwrap();
        let t_view = view.t();
        assert!((t_view.magnitude() - 5.0) < f64::EPSILON);
    }

    #[test]
    fn test_vector_transpose_view_index() {
        let v = Vector::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        let t_view = view.t();
        assert_eq!(t_view[0], 2);
        assert_eq!(t_view[1], 3);
        assert_eq!(t_view[2], 4);
    }
}
