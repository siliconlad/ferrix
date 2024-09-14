#[cfg(test)]
mod tests {
    use ferrix::RowVector;

    #[test]
    fn test_row_vector_view_mut_shape() {
        let mut v = RowVector::from([1, 2, 3, 4]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();
        assert_eq!(t_view_mut.shape(), 3);
    }

    #[test]
    fn test_row_vector_view_mut_capacity() {
        let mut v = RowVector::from([1, 2, 3, 4]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();
        assert_eq!(t_view_mut.capacity(), 3);
    }

    #[test]
    fn test_row_vector_view_mut_t() {
        let mut v = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let t_view_mut = view_mut.t_mut();

        assert_eq!(t_view_mut.shape(), 3);
        assert_eq!(t_view_mut[0], 2.0);
        assert_eq!(t_view_mut[1], 3.0);
        assert_eq!(t_view_mut[2], 4.0);
    }

    #[test]
    fn test_row_vector_view_mut_t_mut() {
        let mut v = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            let mut t_view_mut = view_mut.t_mut();
            t_view_mut[0] = 5.0;
            t_view_mut[1] = 6.0;
            t_view_mut[2] = 7.0;
        }
        assert_eq!(v, RowVector::from([1.0, 5.0, 6.0, 7.0]));
    }

    #[test]
    fn test_row_vector_view_mut_magnitude() {
        let mut v = RowVector::from([3.0, 4.0, 5.0]);
        let view_mut = v.view_mut::<2>(0).unwrap();
        assert!((view_mut.magnitude() - 5.0) < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_mut_eq() {
        let mut v1 = RowVector::from([2.0, 3.0, 4.0]);
        let v2 = RowVector::from([2.0, 3.0]);
        let view_mut1 = v1.view_mut::<2>(0).unwrap();
        assert_eq!(view_mut1, v2);
    }

    #[test]
    fn test_row_vector_view_mut_ne() {
        let mut v1 = RowVector::from([2.0, 3.0, 4.0]);
        let v2 = RowVector::from([2.0, 4.0]);
        let view_mut1 = v1.view_mut::<2>(0).unwrap();
        assert_ne!(view_mut1, v2);
    }

    #[test]
    fn test_row_vector_view_mut_eq_view() {
        let mut v1 = RowVector::from([2.0, 3.0, 3.0, 4.0]);
        let v2 = RowVector::from([3.0, 2.0, 3.0]);
        let view_mut1 = v1.view_mut::<2>(0).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        assert_eq!(view_mut1, view2);
    }

    #[test]
    fn test_row_vector_view_mut_ne_view() {
        let mut v1 = RowVector::from([2.0, 3.0, 3.0, 4.0]);
        let v2 = RowVector::from([3.0, 2.0, 4.0]);
        let view_mut1 = v1.view_mut::<2>(0).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        assert_ne!(view_mut1, view2);
    }

    #[test]
    fn test_row_vector_view_mut_eq_view_mut() {
        let mut v1 = RowVector::from([2.0, 3.0, 3.0, 4.0]);
        let mut v2 = RowVector::from([3.0, 2.0, 3.0]);
        let view_mut1 = v1.view_mut::<2>(0).unwrap();
        let view_mut2 = v2.view_mut::<2>(1).unwrap();
        assert_eq!(view_mut1, view_mut2);
    }

    #[test]
    fn test_row_vector_view_mut_ne_view_mut() {
        let mut v1 = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let mut v2 = RowVector::from([3.0, 2.0, 1.0]);
        let view_mut1 = v1.view_mut::<2>(1).unwrap();
        let view_mut2 = v2.view_mut::<2>(1).unwrap();
        assert_ne!(view_mut1, view_mut2);
    }

    #[test]
    fn test_row_vector_view_mut_display() {
        let mut v = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view_mut::<3>(1).unwrap();
        assert_eq!(format!("{}", view), "[2, 3, 4]");
    }

    #[test]
    fn test_row_vector_view_display_alternate() {
        let mut v = RowVector::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view_mut::<3>(1).unwrap();
        assert_eq!(format!("{:#}", view), "RowVectorViewMut([2, 3, 4], dtype=f64)");
    }

    #[test]
    fn test_row_vector_view_mut_index() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[0], 2);
        assert_eq!(view_mut[1], 3);
        assert_eq!(view_mut[2], 4);
    }

    #[test]
    #[should_panic]
    fn test_row_vector_view_mut_index_out_of_bounds() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[3], 4);
    }

    #[test]
    fn test_row_vector_view_mut_index_mut() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[1], 3);
        view_mut[1] = 10;
        assert_eq!(view_mut[1], 10);
        assert_eq!(v[2], 10);
    }

    #[test]
    #[should_panic]
    fn test_row_vector_view_mut_index_mut_out_of_bounds() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(0, 3)], 3);
    }

    #[test]
    fn test_row_vector_view_mut_index_tuple() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(0, 0)], 2);
        assert_eq!(view_mut[(0, 1)], 3);
        assert_eq!(view_mut[(0, 2)], 4);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_row_vector_view_mut_index_tuple_out_of_bounds() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(1, 0)], 2);

    }

    #[test]
    fn test_row_vector_view_mut_index_mut_tuple() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(0, 0)], 2);
        view_mut[(0, 0)] = 10;
        assert_eq!(view_mut[(0, 0)], 10);
        assert_eq!(view_mut[(0, 1)], 3);
        assert_eq!(view_mut[(0, 2)], 4);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_row_vector_view_mut_index_mut_tuple_out_of_bounds_row() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(1, 0)], 2);
    }

    #[test]
    #[should_panic]
    fn test_row_vector_view_mut_index_mut_tuple_out_of_bounds_col() {
        let mut v = RowVector::from([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[(0, 4)], 2);
    }
}
