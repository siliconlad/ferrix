#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_sub() {
        // Vector - Scalar
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // Vector - Vector
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorView
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorView (transposed)
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorViewMut
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorViewMut (transposed)
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - Matrix
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = v - m;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - MatrixView
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = v - m.view::<3, 1>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - MatrixViewMut
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = v - m.view_mut::<3, 1>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - MatrixTransposeView
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = v - m.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - MatrixTransposeViewMut
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = v - m.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_sub() {
        // VectorView - Scalar
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::from([1, 2, 3]));

        // VectorView - Vector
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::from([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::from([-2, 0, 2]));

        // VectorView - VectorView
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::from([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::from([-2, 0, 2]));

        // VectorView - VectorView (transposed)
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let v2 = RowVector::<i32, 3>::from([5, 4, 3]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - v2.t();
        assert_eq!(result, Vector::<i32, 3>::from([-3, -1, 1]));

        // VectorView - VectorViewMut
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let mut v2 = Vector::<i32, 5>::from([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::from([-2, 0, 2]));

        // VectorView - VectorViewMut (transposed)
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let mut v2 = RowVector::<i32, 3>::from([5, 4, 3]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::from([-3, -1, 1]));

        // VectorView - Matrix
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.2], [0.3], [0.4]]);
        let result = view - m;
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorView - MatrixView
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view - m.view::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorView - MatrixViewMut
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view - m.view_mut::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorView - MatrixTransposeView
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view - m.t();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorView - MatrixTransposeViewMut
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let mut m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view - m.t_mut();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_sub() {
        // VectorViewMut - Scalar
        let mut v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // VectorViewMut - Vector
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorViewMut (transposed)
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorViewMut (transposed)
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - Matrix
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.2], [0.3], [0.4]]);
        let result = view_mut - m;
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorViewMut - MatrixView
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view_mut - m.view::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorViewMut - MatrixViewMut
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view_mut - m.view_mut::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorViewMut - MatrixTransposeView
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view_mut - m.t();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);

        // VectorViewMut - MatrixTransposeViewMut
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let mut m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view_mut - m.t_mut();
        assert!((result[0] - 1.8).abs() < f64::EPSILON);
        assert!((result[1] - 2.7).abs() < f64::EPSILON);
        assert!((result[2] - 3.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_sub() {
        // RowVector - Scalar
        let v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // RowVector - RowVector
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - RowVectorView
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - RowVectorView (transposed)
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - RowVectorViewMut
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - RowVectorViewMut (transposed)
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 - v2.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - Matrix
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = rv - m;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - MatrixView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = rv - m.view::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - MatrixViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = rv - m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - MatrixTransposeView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = rv - m.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVector - MatrixTransposeViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = rv - m.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_sub() {
        // RowVectorView - Scalar
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 - 1;
        assert_eq!(result, RowVector::<i32, 3>::from([0, 1, 2]));

        // RowVectorView - RowVector
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(0).unwrap();
        let result = view1 - view2;
        assert_eq!(result, RowVector::<i32, 3>::from([-3, -3, -3]));

        // RowVectorView - RowVectorView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(0).unwrap();
        let result = view1 - view2;
        assert_eq!(result, RowVector::<i32, 3>::from([-3, -3, -3]));

        // RowVectorView - RowVectorView (transposed)
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 - v2.t();
        assert_eq!(result, RowVector::<i32, 3>::from([-3, -3, -3]));

        // RowVectorView - RowVectorViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(0).unwrap();
        let result = view1 - view2;
        assert_eq!(result, RowVector::<i32, 3>::from([-3, -3, -3]));

        // RowVectorView - RowVectorViewMut (transposed)
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 - v2.t_mut();
        assert_eq!(result, RowVector::<i32, 3>::from([-3, -3, -3]));

        // RowVectorView - Matrix
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view = rv.view::<3>(0).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = view - m;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorView - MatrixView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view = rv.view::<3>(0).unwrap();
        let m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = view - m.view::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorView - MatrixViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view = rv.view::<3>(0).unwrap();
        let mut m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = view - m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorView - MatrixTransposeView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view = rv.view::<3>(0).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = view - m.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorView - MatrixTransposeViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view = rv.view::<3>(0).unwrap();
        let mut m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = view - m.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_mut_sub() {
        // RowVectorViewMut - Scalar
        let mut v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // RowVectorViewMut - RowVector
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - RowVectorView
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - RowVectorView (transposed)
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - RowVectorViewMut
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - RowVectorViewMut (transposed)
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - Matrix
        let mut rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view_mut = rv.view_mut::<3>(0).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = view_mut - m;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - MatrixView
        let mut rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view_mut = rv.view_mut::<3>(0).unwrap();
        let m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = view_mut - m.view::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - MatrixViewMut
        let mut rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view_mut = rv.view_mut::<3>(0).unwrap();
        let mut m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = view_mut - m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - MatrixTransposeView
        let mut rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view_mut = rv.view_mut::<3>(0).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = view_mut - m.t();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // RowVectorViewMut - MatrixTransposeViewMut
        let mut rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let view_mut = rv.view_mut::<3>(0).unwrap();
        let mut m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = view_mut - m.t_mut();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_sub() {
        // Matrix - Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let result = m1 - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -3], [-2, -1]]));

        // Matrix - Matrix
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let result = m1 - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // Matrix - MatrixView
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1 - m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-5, -5]]));

        // Matrix - MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1 - m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-5, -5]]));

        // Matrix - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1 - m2.t();
        assert_eq!(result, Matrix::from([[-4, -5, -6], [-2, -3, -4]]));

        // Matrix - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1 - m2.t_mut();
        assert_eq!(result, Matrix::from([[-4, -5, -6], [-2, -3, -4]]));

        // Matrix - Vector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 - v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // Matrix - VectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // Matrix - VectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // Matrix - RowVector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 - v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // Matrix - RowVectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // Matrix - RowVectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));
    }

    #[test]
    fn test_matrix_view_sub() {
        // MatrixView - Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -3], [-2, -1]]));

        // MatrixView - Matrix
        let m1 = Matrix::<i32, 3, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 1)).unwrap();
        let result = view - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-3, -3], [-1, -1]]));

        // MatrixView - MatrixView
        let m1 = Matrix::<i32, 4, 4>::from([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let m2 = Matrix::<i32, 3, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let view1 = m1.view::<2, 2>((1, 1)).unwrap();
        let view2 = m2.view::<2, 2>((0, 1)).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[0, 0], [1, 1]]));

        // MatrixView - MatrixViewMut
        let m1 = Matrix::<i32, 3, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let mut m2 = Matrix::<i32, 4, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]);
        let view = m1.view::<2, 2>((0, 1)).unwrap();
        let view_mut = m2.view_mut::<2, 2>((1, 1)).unwrap();
        let result = view - view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-7, -7], [-6, -6]]));

        // MatrixView - MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixView - Vector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let view = m1.view::<2, 1>((0, 0)).unwrap();
        let result = view - v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixView - VectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let view = m1.view::<2, 1>((0, 0)).unwrap();
        let result = view - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixView - VectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let view = m1.view::<2, 1>((0, 0)).unwrap();
        let result = view - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixView - RowVector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let view = m1.view::<1, 2>((0, 0)).unwrap();
        let result = view - v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixView - RowVectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let view = m1.view::<1, 2>((0, 0)).unwrap();
        let result = view - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixView - RowVectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let view = m1.view::<1, 2>((0, 0)).unwrap();
        let result = view - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));
    }

    #[test]
    fn test_matrix_view_mut_sub() {
        // MatrixViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -3], [-2, -1]]));

        // MatrixViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view_mut - view;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut1 - view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixViewMut - Vector
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<2, 1>((0, 0)).unwrap();
        let result = view_mut - v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixViewMut - VectorView
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<2, 1>((0, 0)).unwrap();
        let result = view_mut - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixViewMut - VectorViewMut
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<2, 1>((0, 0)).unwrap();
        let result = view_mut - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixViewMut - RowVector
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<1, 2>((0, 0)).unwrap();
        let result = view_mut - v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixViewMut - RowVectorView
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<1, 2>((0, 0)).unwrap();
        let result = view_mut - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixViewMut - RowVectorViewMut
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let view_mut = m1.view_mut::<1, 2>((0, 0)).unwrap();
        let result = view_mut - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));
    }

    #[test]
    fn test_matrix_transpose_view_sub() {
        // MatrixTransposeView - Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let result = m1.t() - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -3], [-2, -1]]));

        // MatrixTransposeView - Matrix
        let m1 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2;
        assert_eq!(result, Matrix::from([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixView
        let m1 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2.view::<3, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() - m2.view_mut::<3, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[-4, -2], [-5, -3], [-6, -4]]));

        // MatrixTransposeView - MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t() - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t() - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixTransposeView - Vector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeView - VectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeView - VectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeView - RowVector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixTransposeView - RowVectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixTransposeView - RowVectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_sub() {
        // MatrixTransposeViewMut - Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let result = m1.t_mut() - 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -3], [-2, -1]]));

        // MatrixTransposeViewMut - Matrix
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2;
        assert_eq!(result, Matrix::from([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() - m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[-4, -3, -2], [-6, -5, -4]]));

        // MatrixTransposeViewMut - MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t_mut() - m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t_mut() - m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[-4, -4], [-4, -4]]));

        // MatrixTransposeViewMut - Vector
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeViewMut - VectorView
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeViewMut - VectorViewMut
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[-4], [-4]]));

        // MatrixTransposeViewMut - RowVector
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixTransposeViewMut - RowVectorView
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));

        // MatrixTransposeViewMut - RowVectorViewMut
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() - v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[-4, -4]]));
    }
}
