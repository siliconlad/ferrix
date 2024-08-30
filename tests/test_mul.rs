#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_mul() {
        // Vector * Scalar
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let result = v * 2;
        assert_eq!(result, Vector::new([2, 4, 6]));

        // Vector * Vector
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2;
        assert_eq!(result, Vector::new([4, 10, 18]));

        // Vector * VectorView
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.view::<3>(0).unwrap();
        assert_eq!(result, Vector::new([4, 10, 18]));

        // Vector * VectorView (transposed)
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.t();
        assert_eq!(result, Vector::new([4, 10, 18]));

        // Vector * VectorViewMut
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, Vector::new([4, 10, 18]));

        // Vector * VectorViewMut (transposed)
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.t_mut();
        assert_eq!(result, Vector::new([4, 10, 18]));
    }

    #[test]
    fn test_vector_view_mul() {
        // Test scalar multiplication
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * VectorView
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * Vector
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * VectorView (transposed)
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = RowVector::<i32, 3>::new([5, 4, 3]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * v2.t();
        assert_eq!(result, Vector::<i32, 3>::new([10, 12, 12]));

        // Test VectorView * VectorViewMut
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let mut v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * VectorViewMut (transposed)
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let mut v2 = RowVector::<i32, 3>::new([5, 4, 3]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::new([10, 12, 12]));
    }

    #[test]
    fn test_vector_view_mut_mul() {
        // VectorViewMut * Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let result = v.view_mut::<3>(0).unwrap() * 2;
        assert_eq!(result, Vector::<i32, 3>::new([2, 4, 6]));

        // VectorViewMut * Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut * VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert_eq!(result, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut * VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.t();
        assert_eq!(result, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut * VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut * VectorViewMut (transposed)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.t_mut();
        assert_eq!(result, Vector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_row_vector_mul() {
        // RowVector * Scalar
        let v = RowVector::<i32, 3>::new([1, 2, 3]);
        let result = v * 2;
        assert_eq!(result, RowVector::new([2, 4, 6]));

        // RowVector * RowVector
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2;
        assert_eq!(result, RowVector::new([4, 10, 18]));

        // RowVector * RowVectorView
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.view::<3>(0).unwrap();
        assert_eq!(result, RowVector::new([4, 10, 18]));

        // RowVector * RowVectorView (transposed)
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.t();
        assert_eq!(result, RowVector::new([4, 10, 18]));

        // RowVector * RowVectorViewMut
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, RowVector::new([4, 10, 18]));

        // RowVector * RowVectorViewMut (transposed)
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1 * v2.t_mut();
        assert_eq!(result, RowVector::new([4, 10, 18]));
    }

    #[test]
    fn test_row_vector_view_mul() {
        // RowVectorView * Scalar
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 * 2;
        assert_eq!(result, RowVector::<i32, 3>::new([2, 4, 6]));

        // RowVectorView * RowVector
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 * RowVector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, RowVector::<i32, 3>::new([2, 4, 6]));

        // RowVectorView * RowVectorView
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(0).unwrap();
        let result = view1 * view2;
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorView * RowVectorView (transposed)
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 * v2.t();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorView * RowVectorViewMut
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(0).unwrap();
        let result = view1 * view2;
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorView * RowVectorViewMut (transposed)
        let v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view1 = v1.view::<3>(0).unwrap();
        let result = view1 * v2.t_mut();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_row_vector_view_mut_mul() {
        // RowVectorViewMut * Scalar
        let mut v = RowVector::<i32, 3>::new([1, 2, 3]);
        let result = v.view_mut::<3>(0).unwrap() * 2;
        assert_eq!(result, RowVector::<i32, 3>::new([2, 4, 6]));

        // RowVectorViewMut * RowVector
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2;
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorViewMut * RowVectorView
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorViewMut * RowVectorView (transposed)
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.t();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorViewMut * RowVectorViewMut
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = RowVector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));

        // RowVectorViewMut * RowVectorViewMut (transposed)
        let mut v1 = RowVector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.t_mut();
        assert_eq!(result, RowVector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_matrix_mul() {
        // Matrix * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let result = m1 * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // Matrix * Matrix
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let result = m1 * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // Matrix * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 4>::new([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let result = m1 * m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [27, 40]]));

        // Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 4>::new([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let view_mut = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = m1 * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [27, 40]]));

        // Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1 * m2.t();
        assert_eq!(result, Matrix::new([[5, 14, 27], [24, 40, 60]]));

        // Matrix * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[5, 6], [7, 8], [9, 10]]);
        let result = m1 * m2.t_mut();
        assert_eq!(result, Matrix::new([[5, 14, 27], [24, 40, 60]]));
    }

    #[test]
    fn test_matrix_view_mul() {
        // MatrixView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // MatrixView * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view1 = m1.view::<2, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view * view_mut;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [28, 40]]));

        // MatrixView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view = m1.view::<2, 2>((0, 0)).unwrap();
        let result = view * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [28, 40]]));
    }

    #[test]
    fn test_matrix_view_mut_mul() {
        // MatrixViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut * m2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view = m2.view::<2, 2>((0, 0)).unwrap();
        let result = view_mut * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 6], [7, 8]]);
        let view_mut1 = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut1 * view_mut2;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 0]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let view_mut = m1.view_mut::<2, 2>((0, 0)).unwrap();
        let result = view_mut * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));
    }

    #[test]
    fn test_matrix_transpose_view_mul() {
        // MatrixTransposeView * Scalar
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let result = m1.t() * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t() * m2;
        assert_eq!(result, Matrix::new([[5, 18, 35], [16, 36, 60]]));

        // MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t() * m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [24, 36]]));

        // MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t() * m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [24, 36]]));

        // MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t() * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_mul() {
        // MatrixTransposeViewMut * Scalar
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let result = m1.t_mut() * 5;
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 10], [15, 20]]));

        // MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[5, 6, 7], [8, 9, 10]]);
        let result = m1.t_mut() * m2;
        assert_eq!(result, Matrix::new([[5, 18, 35], [16, 36, 60]]));

        // MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t_mut() * m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [24, 36]]));

        // MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        let result = m1.t_mut() * m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [24, 36]]));

        // MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));

        // MatrixTransposeViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[5, 7], [6, 8]]);
        let result = m1.t_mut() * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[5, 12], [21, 32]]));
    }
}
