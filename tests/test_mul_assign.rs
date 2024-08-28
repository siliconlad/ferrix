#[cfg(test)]
mod tests {
    use ferrix::{Vector, Matrix};

    #[test]
    fn test_vector_mul_assign() {
        // Vector *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v, Vector::<i32, 3>::new([2, 4, 6]));

        // Vector *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));

        // Vector *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));

        // Vector *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_vector_view_mut_mul_assign() {
        // VectorViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v, Vector::<i32, 3>::new([2, 4, 6]));

        // VectorViewMut *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2;
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorViewMut *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_vector_transpose_view_mut_mul_assign() {
        // VectorTransposeViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.t_mut();
        view *= 2;
        assert_eq!(v, Vector::<i32, 3>::new([2, 4, 6]));

        // VectorTransposeViewMut *= VectorTransposeView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.t_mut();
        view *= v2.t();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));

        // VectorTransposeViewMut *= VectorTransposeViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.t_mut();
        view *= v2.t_mut();
        assert_eq!(v1, Vector::<i32, 3>::new([4, 10, 18]));
    }

    #[test]
    fn test_matrix_mul_assign() {
        // Matrix *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        m *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // Matrix *= Matrix
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 0], [1, 2]]);
        m1 *= m2;
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[2, 0], [3, 8]]));

        // Matrix *= MatrixView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let m2 = Matrix::<i32, 3, 3>::new([[2, 0, 0], [1, 2, 0], [0, 0, 1]]);
        m1 *= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[2, 0], [3, 8]]));

        // Matrix *= MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut m2 = Matrix::<i32, 3, 3>::new([[2, 0, 0], [1, 2, 0], [0, 0, 1]]);
        m1 *= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[2, 0], [3, 8]]));

        // Matrix *= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 1, 0], [0, 2, 1]]);
        m1 *= m2.t();
        assert_eq!(m1, Matrix::new([[2, 0], [3, 8], [0, 6]]));

        // Matrix *= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[2, 1, 0], [0, 2, 1]]);
        m1 *= m2.t_mut();
        assert_eq!(m1, Matrix::new([[2, 0], [3, 8], [0, 6]]));
    }

    #[test]
    fn test_matrix_view_mut_mul_assign() {
        // MatrixViewMut *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let mut view = m.view_mut::<2, 2>((0, 0)).unwrap();
        view *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 4], [6, 8]]));

        // MatrixViewMut *= Matrix
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 3], [4, 5]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view *= m2;
        assert_eq!(m1, Matrix::new([[2, 6, 0], [12, 20, 0], [0, 0, 1]]));

        // MatrixViewMut *= MatrixView
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 3], [4, 5]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view *= m2.view::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[2, 6, 0], [12, 20, 0], [0, 0, 1]]));

        // MatrixViewMut *= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 3], [4, 5]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view *= m2.view_mut::<2, 2>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::new([[2, 6, 0], [12, 20, 0], [0, 0, 1]]));

        // MatrixViewMut *= MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 4], [3, 5]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view *= m2.t();
        assert_eq!(m1, Matrix::new([[2, 6, 0], [12, 20, 0], [0, 0, 1]]));

        // MatrixViewMut *= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 3>::new([[1, 2, 0], [3, 4, 0], [0, 0, 1]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 4], [3, 5]]);
        let mut view = m1.view_mut::<2, 2>((0, 0)).unwrap();
        view *= m2.t_mut();
        assert_eq!(m1, Matrix::new([[2, 6, 0], [12, 20, 0], [0, 0, 1]]));
    }

    #[test]
    fn test_matrix_transpose_view_mut_mul_assign() {
        // MatrixTransposeViewMut *= Scalar
        let mut m = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut view = m.t_mut();
        view *= 2;
        assert_eq!(m, Matrix::<i32, 2, 2>::new([[2, 6], [4, 8]]));

        // MatrixTransposeViewMut *= Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 3], [2, 4], [6, 9]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 4, 6], [5, 8, 9]]);
        let mut view = m1.t_mut();
        view *= m2;
        assert_eq!(m1, Matrix::new([[2, 15], [8, 32], [36, 81]]));

        // MatrixTransposeViewMut *= MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 3], [2, 4], [6, 9]]);
        let m2 = Matrix::<i32, 2, 3>::new([[2, 4, 6], [5, 8, 9]]);
        let mut view = m1.t_mut();
        view *= m2.view::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[2, 15], [8, 32], [36, 81]]));

        // MatrixTransposeViewMut *= MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 3], [2, 4], [6, 9]]);
        let mut m2 = Matrix::<i32, 2, 3>::new([[2, 4, 6], [5, 8, 9]]);
        let mut view = m1.t_mut();
        view *= m2.view_mut::<2, 3>((0, 0)).unwrap();
        assert_eq!(m1, Matrix::<i32, 3, 2>::new([[2, 15], [8, 32], [36, 81]]));

        // MatrixTransposeViewMut *= MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.t_mut();
        view *= m2.t();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[2, 15], [8, 32]]));

        // MatrixTransposeViewMut *= MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 2>::new([[1, 3], [2, 4]]);
        let mut m2 = Matrix::<i32, 2, 2>::new([[2, 5], [4, 8]]);
        let mut view = m1.t_mut();
        view *= m2.t_mut();
        assert_eq!(m1, Matrix::<i32, 2, 2>::new([[2, 15], [8, 32]]));
    }
}
