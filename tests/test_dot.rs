#[cfg(test)]
mod tests {
    use ferrix::{Vector, RowVector, Matrix, DotProduct};

    #[test]
    fn test_vector_dot() {
        // Vector dot Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // Vector dot VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // Vector dot VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // Vector dot Matrix
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        assert!((v1.dot(m) - 14.0).abs() < f64::EPSILON);

        // Vector dot MatrixView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view = m.view::<3, 1>((0, 0)).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // Vector dot MatrixViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view_mut = m.view_mut::<3, 1>((0, 0)).unwrap();
        assert!((v1.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // Vector dot MatrixTransposeView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        assert!((v1.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // Vector dot MatrixTransposeViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        assert!((v1.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_dot() {
        // VectorView dot Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorView dot VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorView dot VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view_mut) - 20.0).abs() < f64::EPSILON);

        // VectorView dot Matrix
        let v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m) - 14.0).abs() < f64::EPSILON);

        // VectorView dot MatrixView
        let v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view = v1.view::<3>(0).unwrap();
        let view2 = m.view::<3, 1>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // VectorView dot MatrixViewMut
        let v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view = v1.view::<3>(0).unwrap();
        let view_mut = m.view_mut::<3, 1>((0, 0)).unwrap();
        assert!((view.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // VectorView dot MatrixTransposeView
        let v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // VectorView dot MatrixTransposeViewMut
        let v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_dot() {
        // VectorViewMut dot Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot Matrix
        let mut v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m) - 14.0).abs() < f64::EPSILON);

        // VectorViewMut dot MatrixView
        let mut v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        let view2 = m.view::<3, 1>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // VectorViewMut dot MatrixViewMut
        let mut v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        let view2 = m.view_mut::<3, 1>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // VectorViewMut dot MatrixTransposeView
        let mut v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // VectorViewMut dot MatrixTransposeViewMut
        let mut v1 = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_dot() {
        // RowVector dot RowVector
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // RowVector dot RowVectorView
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // RowVector dot RowVectorViewMut
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // RowVector dot Matrix
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        assert!((v1.dot(m) - 14.0).abs() < f64::EPSILON);

        // RowVector dot MatrixView
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = m.view::<1, 3>((0, 0)).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // RowVector dot MatrixViewMut
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // RowVector dot MatrixTransposeView
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        assert!((v1.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // RowVector dot MatrixTransposeViewMut
        let v1 = RowVector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        assert!((v1.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_dot() {
        // RowVectorView dot RowVector
        let v = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = RowVector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // RowVectorView dot RowVectorView
        let v1 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // RowVectorView dot RowVectorViewMut
        let v1 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // RowVectorView dot Matrix
        let v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m) - 14.0).abs() < f64::EPSILON);

        // RowVectorView dot MatrixView
        let v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = v1.view::<3>(0).unwrap();
        let view2 = m.view::<1, 3>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // RowVectorView dot MatrixViewMut
        let v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = v1.view::<3>(0).unwrap();
        let view2 = m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // RowVectorView dot MatrixTransposeView
        let v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // RowVectorView dot MatrixTransposeViewMut
        let v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view::<3>(0).unwrap();
        assert!((view.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_vector_view_mut_dot() {
        // RowVectorViewMut dot RowVector
        let mut v = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = RowVector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot RowVectorView
        let mut v1 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot RowVectorViewMut
        let mut v1 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = RowVector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot Matrix
        let mut v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 3>::new([[1.0, 2.0, 3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m) - 14.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot MatrixView
        let mut v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        let view2 = m.view::<1, 3>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);
        
        // RowVectorViewMut dot MatrixViewMut
        let mut v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 1, 4>::new([[1.0, 2.0, 3.0, 4.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        let view2 = m.view_mut::<1, 3>((0, 0)).unwrap();
        assert!((view.dot(view2) - 14.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot MatrixTransposeView
        let mut v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m.t()) - 14.0).abs() < f64::EPSILON);

        // RowVectorViewMut dot MatrixTransposeViewMut
        let mut v1 = RowVector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut m = Matrix::<f64, 3, 1>::new([[1.0], [2.0], [3.0]]);
        let view = v1.view_mut::<3>(0).unwrap();
        assert!((view.dot(m.t_mut()) - 14.0).abs() < f64::EPSILON);
    }
}
