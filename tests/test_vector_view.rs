#[cfg(test)]
mod tests {
    use ferrix::Vector;

    #[test]
    fn test_vector_view_shape() {
        let v = Vector::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.shape(), 3);
    }

    #[test]
    fn test_vector_view_t() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        let view = v.view::<2>(1).unwrap();
        let t_view = view.t();

        assert_eq!(t_view.shape(), 2);
        assert_eq!(t_view[0], 2.0);
        assert_eq!(t_view[1], 3.0);
    }

    #[test]
    fn test_vector_view_magnitude() {
        let v = Vector::new([0.0, 3.0, 4.0]);
        let view = v.view::<2>(1).unwrap();
        assert!((view.magnitude() - 5.0) < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_indexing() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[0], 2);
        assert_eq!(view[1], 3);
        assert_eq!(view[2], 4);
        assert_eq!(view.shape(), 3);
    }
}
