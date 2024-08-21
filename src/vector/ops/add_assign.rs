use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::AddAssign;

//////////////
//  Vector  //
//////////////

impl<T: Numeric, const N: usize> AddAssign<Vector<T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&Vector<T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<T> for Vector<T, N> {
    fn add_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] += scalar);
    }
}

/////////////////////
//  VectorViewMut  //
/////////////////////

impl<'a, T: Numeric, const N: usize> AddAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn add_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn add_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<T> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] += scalar);
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_add_assign() {
        // Vector += Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2;
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2;
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // Vector += Vector (i32)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 += v2;
        assert_eq!(v1[0], 2);
        assert_eq!(v1[1], 4);
        assert_eq!(v1[2], 6);

        // Vector += Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v += 2;
        assert_eq!(v[0], 3);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 5);

        // Vector += Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v += 0.5;
        assert!((v[0] - 1.5).abs() < f64::EPSILON);
        assert!((v[1] - 2.5).abs() < f64::EPSILON);
        assert!((v[2] - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_add_assign() {
        // VectorViewMut += VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // VectorViewMut += &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // VectorViewMut += VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // VectorViewMut += &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < f64::EPSILON);
        assert!((v1[1] - 4.0).abs() < f64::EPSILON);
        assert!((v1[2] - 6.0).abs() < f64::EPSILON);

        // VectorViewMut += Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2;
        assert!((v1[0] - 1.1).abs() < f64::EPSILON);
        assert!((v1[1] - 2.2).abs() < f64::EPSILON);
        assert!((v1[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut += &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2;
        assert!((v1[0] - 1.1).abs() < f64::EPSILON);
        assert!((v1[1] - 2.2).abs() < f64::EPSILON);
        assert!((v1[2] - 3.3).abs() < f64::EPSILON);

        // VectorViewMut += Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view += 0.5;
        assert!((v[0] - 1.5).abs() < f64::EPSILON);
        assert!((v[1] - 2.5).abs() < f64::EPSILON);
        assert!((v[2] - 3.5).abs() < f64::EPSILON);
    }
}
