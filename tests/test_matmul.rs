#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix};

    #[test]
    fn test_vector_matmul() {
        // Vector * RowVector
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let result = v1 * v2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // Vector * RowVectorView
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v2.view::<2>(1).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // Vector * RowVectorViewMut
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let mut v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v2.view_mut::<2>(1).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // Vector * Matrix
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let m2 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let result = v1 * m2;
        assert_eq!(result, Matrix::from([[1, 2, 3], [2, 4, 6]]));

        // Vector * MatrixView
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = m2.view::<1, 3>((0, 0)).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[1, 2, 3], [2, 4, 6]]));

        // Vector * MatrixViewMut
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let mut m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = m2.view_mut::<1, 3>((0, 0)).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[1, 2, 3], [2, 4, 6]]));

        // Vector * MatrixTransposeView
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let m2 = Matrix::<i32, 3, 1>::from([[1], [2], [3]]);
        let result = v1 * m2.t();
        assert_eq!(result, Matrix::from([[1, 2, 3], [2, 4, 6]]));

        // Vector * MatrixTransposeViewMut
        let v1 = Vector::<i32, 2>::from([1, 2]);
        let mut m2 = Matrix::<i32, 3, 1>::from([[1], [2], [3]]);
        let result = v1 * m2.t_mut();
        assert_eq!(result, Matrix::from([[1, 2, 3], [2, 4, 6]]));
    }

    #[test]
    fn test_vector_view_matmul() {
        // VectorView * RowVector
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * RowVectorView
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * RowVectorViewMut
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let mut v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * Matrix
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let m2 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixView
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.view::<1, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixViewMut
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let mut m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.view_mut::<1, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixTransposeView
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let view = v1.view::<2>(1).unwrap();
        let m2 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = view * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[1, 2], [2, 4]]));

        // VectorView * MatrixTransposeViewMut
        let v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let view = v1.view::<2>(1).unwrap();
        let mut m2 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = view * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[1, 2], [2, 4]]));
    }

    #[test]
    fn test_vector_view_mut_matmul() {
        // VectorView * RowVector
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * RowVectorView
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * RowVectorViewMut
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let mut v2 = RowVector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[4, 5], [8, 10]]));

        // VectorView * Matrix
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let m2 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixView
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.view::<1, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixViewMut
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let mut m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.view_mut::<1, 3>((0, 0)).unwrap();
        assert_eq!(result, Matrix::<i32, 2, 3>::from([[1, 2, 3], [2, 4, 6]]));

        // VectorView * MatrixTransposeView
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let view = v1.view_mut::<2>(1).unwrap();
        let m2 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = view * m2.t();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[1, 2], [2, 4]]));

        // VectorView * MatrixTransposeViewMut
        let mut v1 = Vector::<i32, 3>::from([0, 1, 2]);
        let view = v1.view_mut::<2>(1).unwrap();
        let mut m2 = Matrix::<i32, 2, 1>::from([[1], [2]]);
        let result = view * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 2, 2>::from([[1, 2], [2, 4]]));
    }

    #[test]
    fn test_row_vector_matmul() {
        // RowVector * Vector
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let v2 = Vector::<i32, 2>::from([1, 2]);
        let result = v1 * v2;
        assert_eq!(result, Matrix::from([[14]]));

        // RowVector * VectorView
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v2.view::<2>(1).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[41]]));

        // RowVector * VectorViewMut
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let mut v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v2.view_mut::<2>(1).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[41]]));

        // RowVector * Matrix
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let result = v1 * m2;
        assert_eq!(result, Matrix::from([[24, 33, 42]]));

        // RowVector * MatrixView
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let m2 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view = m2.view::<2, 3>((0, 0)).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[24, 33, 42]]));

        // RowVector * MatrixViewMut
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let mut m2 = Matrix::<i32, 3, 3>::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view = m2.view_mut::<2, 3>((0, 0)).unwrap();
        let result = v1 * view;
        assert_eq!(result, Matrix::from([[24, 33, 42]]));

        // RowVector * MatrixTransposeView
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let m2 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let result = v1 * m2.t();
        assert_eq!(result, Matrix::from([[14]]));

        // RowVector * MatrixTransposeViewMut
        let v1 = RowVector::<i32, 2>::from([4, 5]);
        let mut m2 = Matrix::<i32, 1, 2>::from([[1, 2]]);
        let result = v1 * m2.t_mut();
        assert_eq!(result, Matrix::from([[14]]));
    }

    #[test]
    fn test_row_vector_view_matmul() {
        // RowVectorView * Vector
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorView * VectorView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorView * VectorViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorView * Matrix
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::<i32, 1, 3>::from([[14, 19, 24]]));

        // RowVectorView * MatrixView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 2, 4>::from([[4, 1, 2, 3], [2, 4, 5, 6]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.view::<2, 3>((0, 1)).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 3>::from([[14, 19, 24]]));

        // RowVectorView * MatrixViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[4, 1, 2, 3], [2, 4, 5, 6]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.view_mut::<2, 3>((0, 1)).unwrap();
        assert_eq!(result, Matrix::<i32, 1, 3>::from([[14, 19, 24]]));

        // RowVectorView * MatrixTransposeView
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 3, 2>::from([[1, 5], [2, 6], [3, 7]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.t();
        assert_eq!(result, Matrix::<i32, 1, 3>::from([[17, 22, 27]]));

        // RowVectorView * MatrixTransposeViewMut
        let v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[1, 5], [2, 6], [3, 7]]);
        let view = v1.view::<2>(1).unwrap();
        let result = view * m2.t_mut();
        assert_eq!(result, Matrix::<i32, 1, 3>::from([[17, 22, 27]]));
    }

    #[test]
    fn test_row_vector_view_mut_matmul() {
        // RowVectorViewMut * Vector
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorViewMut * VectorView
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorViewMut * VectorViewMut
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::from([3, 4, 5]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::from([[23]]));

        // RowVectorViewMut * Matrix
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::from([[14, 19, 24]]));

        // RowVectorViewMut * MatrixView
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.view::<2, 3>((0, 1)).unwrap();
        assert_eq!(result, Matrix::from([[22, 27, 32]]));

        // RowVectorViewMut * MatrixViewMut
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.view_mut::<2, 3>((0, 1)).unwrap();
        assert_eq!(result, Matrix::from([[22, 27, 32]]));

        // RowVectorViewMut * MatrixTransposeView
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let m2 = Matrix::<i32, 3, 2>::from([[1, 5], [2, 6], [3, 7]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.t();
        assert_eq!(result, Matrix::from([[17, 22, 27]]));

        // RowVectorViewMut * MatrixTransposeViewMut
        let mut v1 = RowVector::<i32, 3>::from([1, 2, 3]);
        let mut m2 = Matrix::<i32, 3, 2>::from([[1, 5], [2, 6], [3, 7]]);
        let view = v1.view_mut::<2>(1).unwrap();
        let result = view * m2.t_mut();
        assert_eq!(result, Matrix::from([[17, 22, 27]]));
    }

    #[test]
    fn test_matrix_multiplication() {
        // Matrix * Vector
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let result = m1 * v2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[14], [32], [50]]));

        // Matrix * VectorView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view = v2.view::<2>(1).unwrap();
        let result = m1 * view;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // Matrix * VectorViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view = v2.view_mut::<2>(1).unwrap();
        let result = m1 * view;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // Matrix * RowVector
        let m1 = Matrix::<i32, 3, 1>::from([[1], [3], [5]]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let result = m1 * v2;
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[4, 5], [12, 15], [20, 25]]));

        // Matrix * RowVectorView
        let m1 = Matrix::<i32, 3, 1>::from([[1], [3], [5]]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1 * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // Matrix * RowVectorViewMut
        let m1 = Matrix::<i32, 3, 1>::from([[1], [3], [5]]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1 * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // Matrix * Matrix
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);
        let result = m1 * m2;
        assert_eq!(result, Matrix::from([[9, 12, 15], [19, 26, 33], [29, 40, 51]]));

        // Matrix * MatrixView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1 * view;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1 * view_mut;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::from([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view = m2.t();
        let result = m1 * t_view;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_view_multiplication() {
        // MatrixView * Vector
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[14], [32], [50]]));

        // MatrixView * VectorView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        let result = view * view2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixView * VectorViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = view * view2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixView * RowVector
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let view = m1.view::<3, 1>((0, 0)).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[4, 5], [12, 15], [20, 25]]));

        // MatrixView * RowVectorView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view::<3, 1>((0, 0)).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // MatrixView * RowVectorViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view::<3, 1>((0, 0)).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // MatrixView * Matrix
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view * m2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view1 * view_mut;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::from([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view1 * t_view;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_view_mut_multiplication() {
        // MatrixViewMut * Vector
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut * v2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[14], [32], [50]]));

        // MatrixViewMut * VectorView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        let result = view_mut * view2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixViewMut * VectorViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = view_mut * view2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixViewMut * RowVector
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let view = m1.view_mut::<3, 1>((0, 0)).unwrap();
        let result = view * v2;
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[4, 5], [12, 15], [20, 25]]));

        // MatrixViewMut * RowVectorView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view_mut::<3, 1>((0, 0)).unwrap();
        let result = view * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // MatrixViewMut * RowVectorViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let view = m1.view_mut::<3, 1>((0, 0)).unwrap();
        let result = view * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [15, 18], [25, 30]]));

        // MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut * m2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view_mut * view2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view_mut1 * view_mut2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::from([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view_mut * t_view;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_transpose_view_multiplication() {
        // MatrixTransposeView * Vector
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let result = m1.t() * v2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[14], [32], [50]]));

        // MatrixTransposeView * VectorView
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t() * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixTransposeView * VectorViewMut
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t() * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixTransposeView * RowVector
        let m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let result = m1.t() * v2;
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[4, 5], [8, 10], [12, 15]]));

        // MatrixTransposeView * RowVectorView
        let m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t() * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [10, 12], [15, 18]]));

        // MatrixTransposeView * RowVectorViewMut
        let m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t() * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [10, 12], [15, 18]]));

        // MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t() * m2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t() * m2.view::<2, 4>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t() * m2.view_mut::<2, 4>((0, 0)).unwrap();
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 3, 2>::from([[1, 2], [3, 4], [5, 6]]);
        let result = m1.t() * m2.t();
        assert_eq!(result, Matrix::from([[5, 11, 17], [11, 25, 39], [17, 39, 61]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_multiplication() {
        // MatrixTransposeViewMut * Vector
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 2>::from([4, 5]);
        let result = m1.t_mut() * v2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[14], [32], [50]]));

        // MatrixTransposeViewMut * VectorView
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t_mut() * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixTransposeViewMut * VectorViewMut
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let mut v2 = Vector::<i32, 3>::from([4, 5, 6]);
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = m1.t_mut() * view2;
        assert_eq!(result, Matrix::<i32, 3, 1>::from([[17], [39], [61]]));

        // MatrixTransposeViewMut * RowVector
        let mut m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let v2 = RowVector::<i32, 2>::from([4, 5]);
        let result = m1.t_mut() * v2;
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[4, 5], [8, 10], [12, 15]]));

        // MatrixTransposeViewMut * RowVectorView
        let mut m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t_mut() * v2.view::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [10, 12], [15, 18]]));

        // MatrixTransposeViewMut * RowVectorViewMut
        let mut m1 = Matrix::<i32, 1, 3>::from([[1, 2, 3]]);
        let mut v2 = RowVector::<i32, 3>::from([4, 5, 6]);
        let result = m1.t_mut() * v2.view_mut::<2>(1).unwrap();
        assert_eq!(result, Matrix::<i32, 3, 2>::from([[5, 6], [10, 12], [15, 18]]));

        // MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut * m2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1.t_mut() * view2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1.t_mut() * view_mut2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 4, 2>::from([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view2 = m2.t();
        let result = t_view_mut1 * t_view2;
        assert_eq!(result, Matrix::from([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }
}
