#[cfg(test)]
mod tests {
    use ferrix::{Vector, Matrix};

    #[test]
    fn test_default() {
        let v = Vector::<f64, 3>::default();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_zeros() {
        let v = Vector::<f64, 3>::zeros();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_ones() {
        let v = Vector::<f64, 3>::ones();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn test_new() {
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_fill() {
        let v = Vector::<f64, 3>::fill(5.0);
        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 5.0);
    }

    #[test]
    fn test_shape() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(v.shape(), 3);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.shape(), 0);
    }

    #[test]
    fn test_transpose() {
        let v = Vector::new([1, 2, 3]);
        let transposed = v.t();
        assert_eq!(transposed.shape(), 3);
        assert_eq!(transposed[0], 1);
        assert_eq!(transposed[1], 2);
        assert_eq!(transposed[2], 3);
    }

    #[test]
    fn test_t_mut() {
        let mut v = Vector::new([1, 2, 3]);
        let mut transposed = v.t_mut();
        
        assert_eq!(transposed.shape(), 3);
        assert_eq!(transposed[0], 1);
        assert_eq!(transposed[1], 2);
        assert_eq!(transposed[2], 3);

        transposed[1] = 5;

        assert_eq!(v[0], 1);
        assert_eq!(v[1], 5);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_view() {
        let v = Vector::new([1.0, 2.0, 3.0]);

        let view = v.view::<2>(1).unwrap();
        assert_eq!(view[0], 2.0);
        assert_eq!(view[1], 3.0);
        assert_eq!(view.shape(), 2);

        assert!(v.view::<3>(3).is_none());
        assert!(v.view::<6>(0).is_none());

        let empty_v = Vector::<f64, 0>::new([]);
        assert!(empty_v.view::<3>(0).is_none());
        assert!(empty_v.view::<0>(0).is_none());
    }

    #[test]
    fn test_view_mut() {
        let mut v = Vector::new([1.0, 2.0, 3.0]);
        {
            let mut view = v.view_mut::<2>(1).unwrap();
            view[0] = 10.0;
            view[1] = 20.0;
        }
        assert_eq!(v, Vector::new([1.0, 10.0, 20.0]));

        assert!(v.view_mut::<3>(3).is_none());
        assert!(v.view_mut::<6>(0).is_none());
    }

    #[test]
    fn test_magnitude() {
        let v = Vector::<f64, 3>::new([3.0, 4.0, 0.0]);
        assert!((v.magnitude() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_index() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let v = Vector::new([1, 2, 3]);
        let _ = v[3];
    }

    #[test]
    fn test_index_mut() {
        let mut v = Vector::new([1.0, 2.0, 3.0]);
        v[1] = 5.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut v = Vector::new([1, 2, 3]);
        v[3] = 5;
    }

    #[test]
    fn test_from_vector_view() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        let view = v.view::<2>(1).unwrap();
        let new_v: Vector<f64, 2> = Vector::from(view);
        assert_eq!(new_v, Vector::new([2.0, 3.0]));
    }

    #[test]
    fn test_from_vector_view_mut() {
        let mut v = Vector::new([1.0, 2.0, 3.0]);
        let view_mut = v.view_mut::<2>(1).unwrap();
        let new_v: Vector<f64, 2> = Vector::from(view_mut);
        assert_eq!(new_v, Vector::new([2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix() {
        let m = Matrix::new([[1.0], [2.0], [3.0]]);
        let v: Vector<f64, 3> = Vector::from(m);
        assert_eq!(v, Vector::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix_view() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let view = m.view::<3, 1>((0, 0)).unwrap();
        let v: Vector<f64, 3> = Vector::from(view);
        assert_eq!(v, Vector::new([1.0, 3.0, 5.0]));
    }

    #[test]
    fn test_from_matrix_view_mut() {
        let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let view_mut = m.view_mut::<3, 1>((0, 0)).unwrap();
        let v: Vector<f64, 3> = Vector::from(view_mut);
        assert_eq!(v, Vector::new([1.0, 3.0, 5.0]));
    }

    #[test]
    fn test_from_matrix_transpose_view() {
        let m = Matrix::new([[1.0, 2.0, 3.0]]);
        let v: Vector<f64, 3> = Vector::from(m.t());
        assert_eq!(v, Vector::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix_transpose_view_mut() {
        let mut m = Matrix::new([[1.0, 2.0, 3.0]]);
        let v: Vector<f64, 3> = Vector::from(m.t_mut());
        assert_eq!(v, Vector::new([1.0, 2.0, 3.0]));
    }
}
