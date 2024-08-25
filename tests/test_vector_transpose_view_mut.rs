#[cfg(test)]
mod tests {
    use ferrix::Vector;

    #[test]
    fn test_vector_transpose_view_mut_shape() {
        let mut v = Vector::new([1, 2, 3, 4]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();
        assert_eq!(t_view_mut.shape(), 3);
    }

    #[test]
    fn test_vector_transpose_view_mut_t() {
        let mut v = Vector::new([1.0, 2.0, 3.0, 4.0]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();
        let view_from_t = t_view_mut.t();

        assert_eq!(view_from_t.shape(), 3);
        assert_eq!(view_from_t[0], 2.0);
        assert_eq!(view_from_t[1], 3.0);
        assert_eq!(view_from_t[2], 4.0);
    }

    #[test]
    fn test_vector_transpose_view_mut_t_mut() {
        let mut v = Vector::new([1.0, 2.0, 3.0, 4.0]);
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            let mut t_view_mut = view_mut.t_mut();
            t_view_mut[0] = 5.0;
            t_view_mut[1] = 6.0;
            t_view_mut[2] = 7.0;
        }
        assert_eq!(v, Vector::new([1.0, 5.0, 6.0, 7.0]));
    }

    #[test]
    fn test_vector_transpose_view_mut_magnitude() {
        let mut v = Vector::new([3.0, 4.0, 5.0]);
        let mut view_mut = v.view_mut::<2>(0).unwrap();
        let t_view_mut = view_mut.t_mut();
        assert!((t_view_mut.magnitude() - 5.0) < f64::EPSILON);
    }

    #[test]
    fn test_vector_transpose_view_mut_index() {
        let mut v = Vector::new([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();
        assert_eq!(t_view_mut[0], 2);
        assert_eq!(t_view_mut[1], 3);
        assert_eq!(t_view_mut[2], 4);
    }

    #[test]
    fn test_vector_transpose_view_mut_index_mut() {
        let mut v = Vector::new([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let mut t_view_mut = view_mut.t_mut();
        assert_eq!(t_view_mut[1], 3);
        t_view_mut[1] = 10;
        assert_eq!(t_view_mut[1], 10);
        assert_eq!(v[2], 10);
    }
}
