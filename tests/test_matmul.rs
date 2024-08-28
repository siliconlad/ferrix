#[cfg(test)]
mod tests {
    use ferrix::{Vector, Matrix};
    use ferrix::MatMul;

    #[test]
    fn test_vector_matmul() {
        // Vector * VectorTransposeView
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = v1.matmul(v2.t());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[4, 5], [8, 10]]));

        // Vector * VectorTransposeViewMut
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let mut v2 = Vector::<i32, 2>::new([4, 5]);
        let result = v1.matmul(v2.t_mut());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[4, 5], [8, 10]]));
    }

    #[test]
    fn test_vector_view_matmul() {
        // VectorView * VectorTransposeView
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view.matmul(v2.t());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[8, 10], [12, 15]]));

        // VectorView * VectorTransposeViewMut
        let v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 2>::new([4, 5]);
        let view = v1.view::<2>(1).unwrap();
        let result = view.matmul(v2.t_mut());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[8, 10], [12, 15]]));
    }

    #[test]
    fn test_vector_view_mut_matmul() {
        // VectorViewMut * VectorTransposeView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let view_mut = v1.view_mut::<2>(1).unwrap();
        let result = view_mut.matmul(v2.t());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[8, 10], [12, 15]]));

        // VectorViewMut * VectorTransposeViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 2>::new([4, 5]);
        let view_mut = v1.view_mut::<2>(1).unwrap();
        let result = view_mut.matmul(v2.t_mut());
        assert_eq!(result, Matrix::<i32, 2, 2>::new([[8, 10], [12, 15]]));
    }

    #[test]
    fn test_vector_transpose_view_matmul() {
        // VectorTransposeView * Vector
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = v1.t().matmul(v2);
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeView * VectorView
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let v2 = Vector::<i32, 3>::new([3, 4, 5]);
        let result = v1.t().matmul(v2.view::<2>(1).unwrap());
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeView * VectorViewMut
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let mut v2 = Vector::<i32, 3>::new([3, 4, 5]);
        let result = v1.t().matmul(v2.view_mut::<2>(1).unwrap());
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeView * Matrix
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let result = v1.t().matmul(m2);
        assert_eq!(result, Matrix::<i32, 1, 3>::new([[9, 12, 15]]));

        // VectorTransposeView * MatrixView
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 2, 4>::new([[4, 1, 2, 3], [2, 4, 5, 6]]);
        let result = v1.t().matmul(m2.view::<2, 3>((0, 1)).unwrap());
        assert_eq!(result, Matrix::<i32, 1, 3>::new([[9, 12, 15]]));

        // VectorTransposeView * MatrixViewMut
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[4, 1, 2, 3], [2, 4, 5, 6]]);
        let result = v1.t().matmul(m2.view_mut::<2, 3>((0, 1)).unwrap());
        assert_eq!(result, Matrix::<i32, 1, 3>::new([[9, 12, 15]]));

        // VectorTransposeView * MatrixTransposeView
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 5], [2, 6], [3, 7]]);
        let result = v1.t().matmul(m2.t());
        assert_eq!(result, Matrix::<i32, 1, 3>::new([[11, 14, 17]]));

        // VectorTransposeView * MatrixTransposeViewMut
        let v1 = Vector::<i32, 2>::new([1, 2]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 5], [2, 6], [3, 7]]);
        let result = v1.t().matmul(m2.t_mut());
        assert_eq!(result, Matrix::<i32, 1, 3>::new([[11, 14, 17]]));
    }

    #[test]
    fn test_vector_transpose_view_mut_matmul() {
        // VectorTransposeViewMut * Vector
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = v1.t_mut().matmul(v2);
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeViewMut * VectorView
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let v2 = Vector::<i32, 3>::new([3, 4, 5]);
        let result = v1.t_mut().matmul(v2.view::<2>(1).unwrap());
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeViewMut * VectorViewMut
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let mut v2 = Vector::<i32, 3>::new([3, 4, 5]);
        let result = v1.t_mut().matmul(v2.view_mut::<2>(1).unwrap());
        assert_eq!(result, Matrix::new([[14]]));

        // VectorTransposeViewMut * Matrix
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let result = v1.t_mut().matmul(m2);
        assert_eq!(result, Matrix::new([[9, 12, 15]]));

        // VectorTransposeViewMut * MatrixView
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = v1.t_mut().matmul(m2.view::<2, 3>((0, 1)).unwrap());
        assert_eq!(result, Matrix::new([[14, 17, 20]]));

        // VectorTransposeViewMut * MatrixViewMut
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = v1.t_mut().matmul(m2.view_mut::<2, 3>((0, 1)).unwrap());
        assert_eq!(result, Matrix::new([[14, 17, 20]]));

        // VectorTransposeViewMut * MatrixTransposeView
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 5], [2, 6], [3, 7]]);
        let result = v1.t_mut().matmul(m2.t());
        assert_eq!(result, Matrix::new([[11, 14, 17]]));

        // VectorTransposeViewMut * MatrixTransposeViewMut
        let mut v1 = Vector::<i32, 2>::new([1, 2]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 5], [2, 6], [3, 7]]);
        let result = v1.t_mut().matmul(m2.t_mut());
        assert_eq!(result, Matrix::new([[11, 14, 17]]));
    }

    #[test]
    fn test_matrix_multiplication() {
        // Matrix * Vector
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = m1.matmul(v2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[14], [32], [50]]));

        // Matrix * VectorView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view = v2.view::<2>(1).unwrap();
        let result = m1.matmul(view);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // Matrix * VectorViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view = v2.view_mut::<2>(1).unwrap();
        let result = m1.matmul(view);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // Matrix * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let result = m1.matmul(m2);
        assert_eq!(result, Matrix::new([[9, 12, 15], [19, 26, 33], [29, 40, 51]]));

        // Matrix * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(view);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(view_mut);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view = m2.t();
        let result = m1.matmul(t_view);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_view_multiplication() {
        // MatrixView * Vector
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view.matmul(v2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[14], [32], [50]]));

        // MatrixView * VectorView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        let result = view.matmul(view2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixView * VectorViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = view.matmul(view2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixView * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view.matmul(m2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(view2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(view_mut);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view1.matmul(t_view);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_view_mut_multiplication() {
        // MatrixViewMut * Vector
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut.matmul(v2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[14], [32], [50]]));

        // MatrixViewMut * VectorView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view::<2>(1).unwrap();
        let result = view_mut.matmul(view2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixViewMut * VectorViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = view_mut.matmul(view2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut.matmul(m2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view_mut.matmul(view2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view_mut1.matmul(view_mut2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view_mut.matmul(t_view);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }

    #[test]
    fn test_matrix_transpose_view_multiplication() {
        // MatrixTransposeView * Vector
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = m1.t().matmul(v2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[14], [32], [50]]));

        // MatrixTransposeView * VectorView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = m1.t().matmul(v2.view::<2>(1).unwrap());
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixTransposeView * VectorViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = m1.t().matmul(v2.view_mut::<2>(1).unwrap());
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t().matmul(m2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t().matmul(m2.view::<2, 4>((0, 0)).unwrap());
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.t().matmul(m2.view_mut::<2, 4>((0, 0)).unwrap());
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let result = m1.t().matmul(m2.t());
        assert_eq!(result, Matrix::new([[5, 11, 17], [11, 25, 39], [17, 39, 61]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_multiplication() {
        // MatrixTransposeViewMut * Vector
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 2>::new([4, 5]);
        let result = m1.t_mut().matmul(v2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[14], [32], [50]]));

        // MatrixTransposeViewMut * VectorView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let result = m1.t_mut().matmul(v2.view::<2>(1).unwrap());
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixTransposeViewMut * VectorViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let view2 = v2.view_mut::<2>(1).unwrap();
        let result = m1.t_mut().matmul(view2);
        assert_eq!(result, Matrix::<i32, 3, 1>::new([[17], [39], [61]]));

        // MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut.matmul(m2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1.t_mut().matmul(view2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1.t_mut().matmul(view_mut2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));

        // MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view2 = m2.t();
        let result = t_view_mut1.matmul(t_view2);
        assert_eq!(result, Matrix::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]));
    }
}
