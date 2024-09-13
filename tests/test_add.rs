#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_add() {
        // Vector + Scalar
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);

        // Vector + Vector
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorView
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorView (transposed)
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorViewMut
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + VectorViewMut (transposed)
        let v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + Matrix
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = v + m;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + MatrixView
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = v + m.view::<3, 1>((0, 0)).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + MatrixViewMut
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = v + m.view_mut::<3, 1>((0, 0)).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + MatrixTransposeView
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = v + m.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // Vector + MatrixTransposeViewMut
        let v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = v + m.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_add() {
        let v1 = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::from([5, 4, 3, 2, 1]);

        // VectorView + Scalar
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + 10;
        assert_eq!(result, Vector::<i32, 3>::from([12, 13, 14]));

        // Test VectorView + Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + Vector::<i32, 3>::from([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::from([3, 4, 5]));

        // VectorView + VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 + view2;
        assert_eq!(result, Vector::<i32, 3>::from([6, 6, 6]));

        // VectorView + VectorView (transposed)
        let view1 = v1.view::<3>(1).unwrap();
        let v3 = RowVector::<i32, 3>::from([1, 1, 1]);
        let result = view1 + v3.t();
        assert_eq!(result, Vector::<i32, 3>::from([3, 4, 5]));

        // VectorView + VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::from([10, 20, 30, 40, 50]);
        let result = view1 + v3.view_mut::<3>(1).unwrap();
        assert_eq!(result, Vector::<i32, 3>::from([22, 33, 44]));

        // VectorView + VectorViewMut (transposed)
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = RowVector::<i32, 3>::from([10, 20, 30]);
        let result = view1 + v3.t_mut();
        assert_eq!(result, Vector::<i32, 3>::from([12, 23, 34]));

        // VectorView + Matrix
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.2], [0.3], [0.4]]);
        let result = view + m;
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorView + MatrixView
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view + m.view::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorView + MatrixViewMut
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view + m.view_mut::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorView + MatrixTransposeView
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view + m.t();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorView + MatrixTransposeViewMut
        let v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        let mut m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view + m.t_mut();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_add() {
        // VectorViewMut + Scalar
        let mut v = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);

        // VectorViewMut + Vector
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + VectorView
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + VectorView (transposed)
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + VectorViewMut
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + VectorViewMut (transposed)
        let mut v1 = Vector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut + Matrix
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 3, 1>::from([[0.2], [0.3], [0.4]]);
        let result = view_mut + m;
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorViewMut + MatrixView
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view_mut + m.view::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorViewMut + MatrixViewMut
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let mut m = Matrix::<f64, 4, 2>::from([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]]);
        let result = view_mut + m.view_mut::<3, 1>((1, 0)).unwrap();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorViewMut + MatrixTransposeView
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view_mut + m.t();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);

        // VectorViewMut + MatrixTransposeViewMut
        let mut v = Vector::<f64, 4>::from([1.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let mut m = Matrix::<f64, 1, 3>::from([[0.2, 0.3, 0.4]]);
        let result = view_mut + m.t_mut();
        assert!((result[0] - 2.2).abs() < f64::EPSILON);
        assert!((result[1] - 3.3).abs() < f64::EPSILON);
        assert!((result[2] - 4.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_add() {
        // RowVector + Scalar
        let v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v + 0.5;
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
        assert!((result[2] - 3.5).abs() < f64::EPSILON);

        // RowVector + RowVector
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);
        
        // RowVector + RowVectorView
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + RowVectorView (transposed)
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + RowVectorViewMut
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + RowVectorViewMut (transposed)
        let v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1 + v2.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + Matrix
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::from([[0.1, 0.2, 0.3]]);
        let result = rv + m;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + MatrixView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = rv + m.view::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + MatrixViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 2, 4>::from([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        let result = rv + m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + MatrixTransposeView
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = rv + m.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);

        // RowVector + MatrixTransposeViewMut
        let rv = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 3, 1>::from([[0.1], [0.2], [0.3]]);
        let result = rv + m.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        assert!((result[2] - 3.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_add() {
        // RowVectorView + Scalar
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let view = v1.view::<2>(1).unwrap();
        let result = view + 10;
        assert_eq!(result, RowVector::<i32, 2>::from([12, 13]));

        // RowVectorView + RowVector
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::from([1, 2, 3]);
        let result = v1.view::<2>(1).unwrap() + v2.view::<2>(1).unwrap();
        assert_eq!(result, RowVector::<i32, 2>::from([4, 6]));

        // RowVectorView + RowVectorView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::from([1, 2, 3]);
        let result = v1.view::<2>(1).unwrap() + v2.view::<2>(1).unwrap();
        assert_eq!(result, RowVector::<i32, 2>::from([4, 6]));

        // RowVectorView + RowVectorView (transposed)
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 2>::from([1, 2]);
        let result = v1.view::<2>(1).unwrap() + v2.t();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + RowVectorViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::from([1, 2, 3]);
        let result = v1.view::<2>(1).unwrap() + v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, RowVector::<i32, 2>::from([4, 6]));

        // RowVectorView + RowVectorViewMut (transposed)
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = Vector::<i32, 2>::from([1, 2]);
        let result = v1.view::<2>(1).unwrap() + v2.t_mut();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + Matrix
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let result = v1.view::<2>(1).unwrap() + m;
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + MatrixView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let result = v1.view::<2>(1).unwrap() + m.view::<1, 2>((0, 0)).unwrap();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + MatrixViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let result = v1.view::<2>(1).unwrap() + m.view_mut::<1, 2>((0, 0)).unwrap();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + MatrixTransposeView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = v1.view::<2>(1).unwrap() + m.t();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));

        // RowVectorView + MatrixTransposeViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = v1.view::<2>(1).unwrap() + m.t_mut();
        assert_eq!(result, RowVector::<i32, 2>::from([3, 5]));
    }

    #[test]
    fn test_row_vector_view_mut_add() {
        // RowVectorViewMut + Scalar
        let mut v = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let result = v.view_mut::<2>(1).unwrap() + 0.5;
        assert!((result[0] - 2.5).abs() < f64::EPSILON);
        assert!((result[1] - 3.5).abs() < f64::EPSILON);

        // RowVectorViewMut + RowVector
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 2>::from([0.1, 0.2]);
        let result = v1.view_mut::<2>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);

        // RowVectorViewMut + RowVectorView
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<2>(0).unwrap() + v2.view::<2>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);

        // RowVectorViewMut + RowVectorViewMut (transposed)
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 2>::from([0.1, 0.2]);
        let result = v1.view_mut::<2>(0).unwrap() + v2.t();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);

        // RowVectorViewMut + RowVectorViewMut
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 3>::from([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<2>(0).unwrap() + v2.view_mut::<2>(0).unwrap();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);

        // RowVectorViewMut + RowVectorViewMut (transposed)
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 2>::from([0.1, 0.2]);
        let result = v1.view_mut::<2>(0).unwrap() + v2.t_mut();
        assert!((result[0] - 1.1).abs() < f64::EPSILON);
        assert!((result[1] - 2.2).abs() < f64::EPSILON);
        
        // RowVectorViewMut + Matrix
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 2>::from([[1.0, 2.0]]);
        let result = v1.view_mut::<2>(0).unwrap() + m;
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);

        // RowVectorViewMut + MatrixView
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 2, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = v1.view_mut::<2>(0).unwrap() + m.view::<1, 2>((0, 0)).unwrap();
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);

        // RowVectorViewMut + MatrixViewMut
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 2, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = v1.view_mut::<2>(0).unwrap() + m.view_mut::<1, 2>((0, 0)).unwrap();
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);

        // RowVectorViewMut + MatrixTransposeView
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 2, 1>::from([[1.0], [2.0]]);
        let result = v1.view_mut::<2>(0).unwrap() + m.t();
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);

        // RowVectorViewMut + MatrixTransposeViewMut
        let mut v1 = RowVector::<f64, 3>::from([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 2, 1>::from([[1.0], [2.0]]);
        let result = v1.view_mut::<2>(0).unwrap() + m.t_mut();
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_add() {
        // Matrix + Scalar
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let result = m1 + 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 7], [8, 9]]));

        // Matrix + Matrix
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let result = m1 + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [10, 12]]));

        // Matrix + MatrixView
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::from([[5, 6, 0], [7, 8, 0], [0, 0, 0]]);
        let result = m1 + m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [10, 12]]));

        // Matrix + MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::from([[5, 6, 0], [7, 8, 0], [0, 0, 0]]);
        let result = m1 + m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [10, 12]]));

        // Matrix + MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1 + m2.t();
        assert_eq!(result, Matrix::from([[6, 10], [9, 13], [12, 16]]));

        // Matrix + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1 + m2.t_mut();
        assert_eq!(result, Matrix::from([[6, 10], [9, 13], [12, 16]]));

        // Matrix + Vector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1;
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // Matrix + VectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // Matrix + VectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // Matrix + RowVector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1;
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // Matrix + RowVectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // Matrix + RowVectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));
    }

    #[test]
    fn test_matrix_view_add() {
        // MatrixView + Scalar
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view + 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 7], [9, 10]]));

        // MatrixView + Matrix
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let result = m1.view::<2, 2>((0, 0)).unwrap() + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [11, 13]]));

        // MatrixView + MatrixView
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [8, 9]]);
        let view1 = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view1 + m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [12, 14]]));

        // MatrixView + MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 6], [8, 9]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view + view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [12, 14]]));

        // MatrixView + MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let result = m1.view::<2, 2>((0, 0)).unwrap() + m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 9], [10, 13]]));

        // MatrixView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let result = m1.view::<2, 2>((0, 0)).unwrap() + m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 9], [10, 13]]));

        // MatrixView + Vector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixView + VectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixView + VectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixView + RowVector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixView + RowVectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixView + RowVectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));
    }

    #[test]
    fn test_matrix_view_mut_add() {
        // MatrixViewMut + Scalar
        let mut m1 = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut + 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 7], [8, 9]]));

        // MatrixViewMut + Matrix
        let mut m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut + m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [11, 13]]));

        // MatrixViewMut + MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view_mut + view;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [11, 13]]));

        // MatrixViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut1 + view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[6, 8], [11, 13]]));

        // MatrixViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let m2 = Matrix::<i32, 3, 2>::from([[5, 7], [6, 8], [9, 10]]);
        let result = m1.view_mut::<2, 3>((0, 0)).unwrap() + m2.t();
        assert_eq!(result, Matrix::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[5, 7], [6, 8], [9, 10]]);
        let result = m1.view_mut::<2, 3>((0, 0)).unwrap() + m2.t_mut();
        assert_eq!(result, Matrix::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixViewMut + Vector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixViewMut + VectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixViewMut + VectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixViewMut + RowVector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixViewMut + RowVectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixViewMut + RowVectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1 + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));
    }

    #[test]
    fn test_matrix_transpose_view_add() {
        // MatrixTransposeView + Scalar
        let m1 = Matrix::from([[1, 3], [2, 4]]);
        let result = m1.t() + 5;
        assert_eq!(result, Matrix::from([[6, 7], [8, 9]]));

        // MatrixTransposeView + Matrix
        let m1 = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::from([[5, 6], [7, 8], [9, 10]]);
        let result = m1.t() + m2;
        assert_eq!(result, Matrix::from([[6, 10], [9, 13], [12, 16]]));

        // MatrixTransposeView + MatrixView
        let m1 = Matrix::from([[1, 3], [2, 4]]);
        let m2 = Matrix::from([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]);
        let result = m1.t() + m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[6, 8], [11, 13]]));

        // MatrixTransposeView + MatrixViewMut
        let m1 = Matrix::from([[1, 3], [2, 4]]);
        let mut m2 = Matrix::from([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]);
        let result = m1.t() + m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[6, 8], [11, 13]]));

        // MatrixTransposeView + MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t() + m2.t();
        assert_eq!(result, Matrix::from([[6, 8], [10, 12]]));

        // MatrixTransposeView + MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::from([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::from([[5, 7], [6, 8]]);
        let result = m1.t() + m2.t_mut();
        assert_eq!(result, Matrix::from([[6, 8], [10, 12]]));

        // MatrixTransposeView + Vector
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeView + VectorView
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeView + VectorViewMut
        let m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeView + RowVector
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixTransposeView + RowVectorView
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixTransposeView + RowVectorViewMut
        let m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t() + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_add() {
        // MatrixTransposeViewMut + Scalar
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let result = m1.t_mut() + 5;
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 7, 10], [8, 9, 11]]));

        // MatrixTransposeViewMut + Matrix
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::from([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() + m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixTransposeViewMut + MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let m2 = Matrix::<i32, 3, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t_mut() + m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixTransposeViewMut + MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 3, 3>::from([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t_mut() + m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixTransposeViewMut + MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::from([[5, 8], [6, 9], [7, 10]]);
        let result = m1.t_mut() + m2.t();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixTransposeViewMut + MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 3], [2, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[5, 8], [6, 9], [7, 10]]);
        let result = m1.t_mut() + m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[6, 8, 12], [11, 13, 16]]));

        // MatrixTransposeViewMut + Vector
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeViewMut + VectorView
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeViewMut + VectorViewMut
        let mut m1 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let mut v1 = Vector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 1>::from([[6], [8]]));

        // MatrixTransposeViewMut + RowVector
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixTransposeViewMut + RowVectorView
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));

        // MatrixTransposeViewMut + RowVectorViewMut
        let mut m1 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let mut v1 = RowVector::<i32, 2>::from([5, 6]);
        let result = m1.t_mut() + v1.view_mut::<2>(0).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 2>::from([[6, 8]]));
    }
}
