#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_div() {
        // Vector / Scalar
        let v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result, Vector::new([1, 2, 3]));

        // Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2;
        assert_eq!(result, Vector::new([2, 2, 3]));

        // Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result, Vector::new([2, 2, 3]));

        // Vector / VectorView (transposed)
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.t();
        assert_eq!(result, Vector::new([2, 2, 3]));

        // Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, Vector::new([2, 2, 3]));

        // Vector / VectorViewMut (transposed)
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.t_mut();
        assert_eq!(result, Vector::new([2, 2, 3]));
    }

    #[test]
    fn test_vector_view_div() {
        // Test scalar division
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));

        // Test VectorView / Vector
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / VectorView
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let v2 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / VectorView (transposed)
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 30.0, 50.0]);
        let v2 = RowVector::<f64, 3>::new([2.0, 4.0, 6.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / v2.t();
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 5.0]));

        // Test VectorView / VectorViewMut
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut v2 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        let result = view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / VectorViewMut (transposed)
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 30.0, 50.0]);
        let mut v2 = RowVector::<f64, 3>::new([2.0, 4.0, 6.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / v2.t_mut();
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 5.0]));
    }

    #[test]
    fn test_vector_view_mut_div() {
        // VectorViewMut / Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result, Vector::<i32, 3>::new([2, 2, 3]));

        // VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result, Vector::<i32, 3>::new([2, 2, 3]));

        // VectorViewMut / VectorView (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.t();
        assert_eq!(result, Vector::<i32, 3>::new([2, 2, 3]));

        // VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, Vector::<i32, 3>::new([2, 2, 3]));

        // VectorViewMut / VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::new([2, 2, 3]));
    }

    #[test]
    fn test_row_vector_div() {
        // RowVector / Scalar
        let v = RowVector::<i32, 3>::new([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result, RowVector::new([1, 2, 3]));

        // RowVector / RowVector
        let v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2;
        assert_eq!(result, RowVector::new([2, 2, 3]));

        // RowVector / RowVectorView
        let v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result, RowVector::new([2, 2, 3]));

        // RowVector / RowVectorView (transposed)
        let v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.t();
        assert_eq!(result, RowVector::new([2, 2, 3]));

        // RowVector / RowVectorViewMut
        let v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, RowVector::new([2, 2, 3]));

        // RowVector / RowVectorViewMut (transposed)
        let v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.t_mut();
        assert_eq!(result, RowVector::new([2, 2, 3]));
    }

    #[test]
    fn test_row_vector_view_div() {
        // RowVectorView / Scalar
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, RowVector::<f64, 3>::new([10.0, 15.0, 20.0]));

        // RowVectorView / RowVector
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / RowVector::<f64, 3>::new([2.0, 5.0, 10.0]);
        assert_eq!(result, RowVector::<f64, 3>::new([10.0, 6.0, 4.0]));

        // RowVectorView / RowVectorView
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let v2 = RowVector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 / view2;
        assert_eq!(result, RowVector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // RowVectorView / RowVectorView (transposed)
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 30.0, 50.0]);
        let v2 = Vector::<f64, 3>::new([2.0, 4.0, 6.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / v2.t();
        assert_eq!(result, RowVector::<f64, 3>::new([10.0, 7.5, 5.0]));

        // RowVectorView / RowVectorViewMut
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut v2 = RowVector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        let result = view1 / view_mut;
        assert_eq!(result, RowVector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // RowVectorView / RowVectorViewMut (transposed)
        let v1 = RowVector::<f64, 5>::new([10.0, 20.0, 30.0, 30.0, 50.0]);
        let mut v2 = Vector::<f64, 3>::new([2.0, 4.0, 6.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / v2.t_mut();
        assert_eq!(result, RowVector::<f64, 3>::new([10.0, 7.5, 5.0]));
    }

    #[test]
    fn test_row_vector_view_mut_div() {
        // RowVectorViewMut / Scalar
        let mut v = RowVector::<i32, 3>::new([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result, RowVector::<i32, 3>::new([1, 2, 3]));

        // RowVectorViewMut / RowVector
        let mut v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result, RowVector::<i32, 3>::new([2, 2, 3]));

        // RowVectorViewMut / RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result, RowVector::<i32, 3>::new([2, 2, 3]));

        // RowVectorViewMut / RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.t();
        assert_eq!(result, RowVector::<i32, 3>::new([2, 2, 3]));

        // RowVectorViewMut / RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = RowVector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, RowVector::<i32, 3>::new([2, 2, 3]));

        // RowVectorViewMut / RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.t_mut();
        assert_eq!(result, RowVector::<i32, 3>::new([2, 2, 3]));
    }

    #[test]
    fn test_matrix_div() {
        // Matrix / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let result = m1 / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // Matrix / Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = m1 / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[10, 12], [30, 40]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = m1 / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [4, 5]]));

        // Matrix / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let result = m1 / m2.t();
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));

        // Matrix / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[10, 12], [30, 40], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 7, 9], [6, 8, 10]]);
        let result = m1 / m2.t_mut();
        assert_eq!(result, Matrix::<i32, 3, 2>::new([[2, 2], [4, 5], [5, 6]]));
    }

    #[test]
    fn test_matrix_view_div() {
        // MatrixView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // MatrixView / Matrix
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view1 = m1.view::<2, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view1 / view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 0], [7, 8, 0], [0, 0, 1]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixTransposeView
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap(); 
        let result = view / m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view / m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));
    }

    #[test]
    fn test_matrix_view_mut_div() {
        // MatrixViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[10, 20], [30, 40]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // MatrixViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut / m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view_mut / view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut1 / view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut / m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));

        // MatrixViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[10, 12, 0], [21, 24, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut / m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[2, 2], [3, 3]]));
    }

    #[test]
    fn test_matrix_transpose_view_div() {
        // MatrixTransposeView / Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let result = m1.t() / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 9]]));

        // MatrixTransposeView / Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t() / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t() / m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[30, 40], [35, 48], [50, 60]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let view_mut = m2.view_mut::<2, 3>((0, 0)).unwrap();
        let result = m1.t() / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 7], [5, 5, 6]]));

        // MatrixTransposeView / MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() / m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));

        // MatrixTransposeView / MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() / m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [5, 6]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_div() {
        // MatrixTransposeViewMut / Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 40], [35, 50]]);
        let result = m1.t_mut() / 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 7], [8, 10]]));

        // MatrixTransposeViewMut / Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let result = m1.t_mut() / m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let result = m1.t_mut() / m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[30, 49], [30, 48], [60, 96]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[5, 6, 10], [7, 8, 16]]);
        let view_mut = m2.view_mut::<2, 3>((0, 0)).unwrap();
        let result = m1.t_mut() / view_mut;
        assert_eq!(result, Matrix::<i32, 2, 3>::new([[6, 5, 6], [7, 6, 6]]));

        // MatrixTransposeViewMut / MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() / m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));

        // MatrixTransposeViewMut / MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[30, 49], [30, 48]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() / m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[6, 5], [7, 6]]));
    }
}
