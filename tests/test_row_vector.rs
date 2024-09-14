#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix, FloatRandom, IntRandom};

    #[test]
    fn test_default() {
        let v = RowVector::<f64, 3>::default();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_zeros() {
        let v = RowVector::<f64, 3>::zeros();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_ones() {
        let v = RowVector::<f64, 3>::ones();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn test_new() {
        let v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_fill() {
        let v = RowVector::<f64, 3>::fill(5.0);
        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 5.0);
    }

    #[test]
    fn test_random_float() {
        let v = RowVector::<f64, 3>::random();
        assert!(-1.0 <= v[0] && v[0] <= 1.0);
        assert!(-1.0 <= v[1] && v[1] <= 1.0);
        assert!(-1.0 <= v[2] && v[2] <= 1.0);
    }

    #[test]
    fn test_random_int() {
        let v = RowVector::<i32, 3>::random();
        assert!(i32::MIN <= v[0] && v[0] <= i32::MAX);
        assert!(i32::MIN <= v[1] && v[1] <= i32::MAX);
        assert!(i32::MIN <= v[2] && v[2] <= i32::MAX);
    }

    #[test]
    fn test_diag() {
        let v = RowVector::from([1, 2, 3]);
        let m = v.diag();
        assert_eq!(m, Matrix::from([[1, 0, 0], [0, 2, 0], [0, 0, 3]]));
    }

    #[test]
    fn test_shape() {
        let v = RowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(v.shape(), 3);

        let empty_v = RowVector::<f64, 0>::from([]);
        assert_eq!(empty_v.shape(), 0);
    }
    
    #[test]
    fn test_capacity() {
        let v = RowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(v.capacity(), 3);

        let empty_v = RowVector::<f64, 0>::from([]);
        assert_eq!(empty_v.capacity(), 0);
    }

    #[test]
    fn test_into() {
        let v = RowVector::from([1.0]);
        let x: f64 = v.into();
        assert_eq!(x, 1.0);
    }

    #[test]
    fn test_transpose() {
        let v = RowVector::from([1, 2, 3]);
        let transposed = v.t();
        assert_eq!(transposed.shape(), 3);
        assert_eq!(transposed[0], 1);
        assert_eq!(transposed[1], 2);
        assert_eq!(transposed[2], 3);
    }

    #[test]
    fn test_t_mut() {
        let mut v = RowVector::from([1, 2, 3]);
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
        let v = RowVector::from([1.0, 2.0, 3.0]);

        let view = v.view::<2>(1).unwrap();
        assert_eq!(view[0], 2.0);
        assert_eq!(view[1], 3.0);
        assert_eq!(view.shape(), 2);

        assert!(v.view::<3>(3).is_none());
        assert!(v.view::<6>(0).is_none());

        let empty_v = RowVector::<f64, 0>::from([]);
        assert!(empty_v.view::<3>(0).is_none());
        assert!(empty_v.view::<0>(0).is_none());
    }

    #[test]
    fn test_view_mut() {
        let mut v = RowVector::from([1.0, 2.0, 3.0]);
        {
            let mut view = v.view_mut::<2>(1).unwrap();
            view[0] = 10.0;
            view[1] = 20.0;
        }
        assert_eq!(v, RowVector::from([1.0, 10.0, 20.0]));

        assert!(v.view_mut::<3>(3).is_none());
        assert!(v.view_mut::<6>(0).is_none());
    }

    #[test]
    fn test_magnitude() {
        let v = RowVector::<f64, 3>::from([3.0, 4.0, 0.0]);
        assert!((v.magnitude() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eq() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_ne() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::from([4.0, 5.0, 6.0]);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_eq_view() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view2 = v2.view::<3>(0).unwrap();
        assert_eq!(v1, view2);
    }

    #[test]
    fn test_ne_view() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view2 = v2.view::<3>(1).unwrap();
        assert_ne!(v1, view2);
    }

    #[test]
    fn test_eq_view_mut() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view2 = v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, view2);
    }

    #[test]
    fn test_ne_view_mut() {
        let v1 = RowVector::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert_ne!(v1, view2);
    }

    #[test]
    fn test_display() {
        let v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        assert_eq!(format!("{}", v), "[1, 2, 3]");
    }

    #[test]
    fn test_display_alternate() {
        let v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        assert_eq!(format!("{:#}", v), "RowVector([1, 2, 3], dtype=f64)");
    }

    #[test]
    fn test_index() {
        let v = RowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let v = RowVector::from([1, 2, 3]);
        let _ = v[3];
    }

    #[test]
    fn test_index_mut() {
        let mut v = RowVector::from([1.0, 2.0, 3.0]);
        v[1] = 5.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut v = RowVector::from([1, 2, 3]);
        v[3] = 5;
    }

    #[test]
    fn test_from_vector_view() {
        let v = RowVector::from([1.0, 2.0, 3.0]);
        let view = v.view::<2>(1).unwrap();
        let new_v: RowVector<f64, 2> = RowVector::from(view);
        assert_eq!(new_v, RowVector::from([2.0, 3.0]));

        let v = Vector::from([1.0, 2.0, 3.0]);
        let new_v: RowVector<f64, 3> = RowVector::from(v.t());
        assert_eq!(new_v, RowVector::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_vector_view_mut() {
        let mut v = RowVector::from([1.0, 2.0, 3.0]);
        let view_mut = v.view_mut::<2>(1).unwrap();
        let new_v: RowVector<f64, 2> = RowVector::from(view_mut);
        assert_eq!(new_v, RowVector::from([2.0, 3.0]));

        let mut v = Vector::from([1.0, 2.0, 3.0]);
        let new_v: RowVector<f64, 3> = RowVector::from(v.t_mut());
        assert_eq!(new_v, RowVector::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix() {
        let m = Matrix::from([[1.0, 2.0, 3.0]]);
        let v: RowVector<f64, 3> = RowVector::from(m);
        assert_eq!(v, RowVector::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix_view() {
        let m = Matrix::from([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]);
        let view = m.view::<1, 3>((1, 0)).unwrap();
        let v: RowVector<f64, 3> = RowVector::from(view);
        assert_eq!(v, RowVector::from([3.0, 4.0, 5.0]));
    }

    #[test]
    fn test_from_matrix_view_mut() {
        let mut m = Matrix::from([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]);
        let view_mut = m.view_mut::<1, 3>((1, 0)).unwrap();
        let v: RowVector<f64, 3> = RowVector::from(view_mut);
        assert_eq!(v, RowVector::from([3.0, 4.0, 5.0]));
    }

    #[test]
    fn test_from_matrix_transpose_view() {
        let m = Matrix::from([[1.0], [2.0], [3.0]]);
        let v: RowVector<f64, 3> = RowVector::from(m.t());
        assert_eq!(v, RowVector::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_matrix_transpose_view_mut() {
        let mut m = Matrix::from([[1.0], [2.0], [3.0]]);
        let v: RowVector<f64, 3> = RowVector::from(m.t_mut());
        assert_eq!(v, RowVector::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_row_vector_send() {
        fn assert_send<T: Send>() {}
        assert_send::<RowVector<i32, 3>>();
    }

    #[test]
    fn test_row_vector_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<RowVector<i32, 3>>();
    }
}

