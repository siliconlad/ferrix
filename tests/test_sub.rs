#[cfg(test)]
mod tests {
    use ferrix::{Vector, Matrix};

    #[test]
    fn test_vector_sub() {
        // Vector - Scalar
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // Vector - Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_sub() {
        // VectorView - Scalar
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // VectorView - Vector
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // VectorView - VectorView
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // VectorView - VectorViewMut
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let mut v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));
    }

    #[test]
    fn test_vector_view_mut_sub() {
        // VectorViewMut - Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // VectorViewMut - Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_transpose_view_sub() {
        // VectorTransposeView - Scalar
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let t_view = v1.t();
        let result = t_view - 1;
        assert_eq!(result, Vector::<i32, 3>::new([0, 1, 2]));

        // VectorTransposeView - VectorTransposeView
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.t() - v2.t();
        assert_eq!(result, Vector::<i32, 3>::new([-3, -3, -3]));

        // VectorTransposeView - VectorTransposeViewMut
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.t() - v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::new([-3, -3, -3]));
    }

    #[test]
    fn test_vector_transpose_view_mut_sub() {
        // VectorTransposeViewMut - Scalar
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let result = v1.t_mut() - 1;
        assert_eq!(result, Vector::<i32, 3>::new([0, 1, 2]));

        // VectorTransposeViewMut - VectorTransposeView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.t_mut() - v2.t();
        assert_eq!(result, Vector::<i32, 3>::new([-3, -3, -3]));

        // VectorTransposeViewMut - VectorTransposeViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.t_mut() - v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::new([-3, -3, -3]));
    }

    #[test]
    fn test_matrix_sub() {
        // Matrix - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let result = m1 - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // Matrix - Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // Matrix - MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1 - m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1 - m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-5, -5]]));

        // Matrix - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1 - m2.t();
        assert_eq!(result, Matrix::new([[-4, -5, -6], [-2, -3, -4]]));

        // Matrix - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1 - m2.t_mut();
        assert_eq!(result, Matrix::new([[-4, -5, -6], [-2, -3, -4]]));
    }

    #[test]
    fn test_matrix_view_sub() {
        // MatrixView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // MatrixView - Matrix
        let m1 = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 1)).unwrap();
        let result = view - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-3, -3], [-1, -1]]));

        // MatrixView - MatrixView
        let m1 = Matrix::<i32, 4, 4>::new([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view1 = m1.view::<2, 2>((1, 1)).unwrap();
        let view2 = m2.view::<2, 2>((0, 1)).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[0, 0], [1, 1]]));

        // MatrixView - MatrixViewMut
        let m1 = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut m2 = Matrix::<i32, 4, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]);
        let view = m1.view::<2, 2>((0, 1)).unwrap();
        let view_mut = m2.view_mut::<2, 2>((1, 1)).unwrap();
        let result = view - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-7, -7], [-6, -6]]));

        // MatrixView - MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));
    }

    #[test]
    fn test_matrix_view_mut_sub() {
        // MatrixViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // MatrixViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view_mut - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut1 - view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));
    }

    #[test]
    fn test_matrix_transpose_view_sub() {
        // MatrixTransposeView - Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let result = m1.t() - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // MatrixTransposeView - Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2;
        assert_eq!(result, Matrix::new([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2.view::<3, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::new([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::new([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_sub() {
        // MatrixTransposeViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let result = m1.t_mut() - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -3], [-2, -1]]));

        // MatrixTransposeViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2;
        assert_eq!(result, Matrix::new([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::new([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::new([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[-4, -4], [-4, -4]]));
    }
}
