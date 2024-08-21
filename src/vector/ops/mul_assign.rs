use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::MulAssign;

//////////////
//  Vector  //
//////////////

impl<T: Numeric, const N: usize> MulAssign<Vector<T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&Vector<T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] *= scalar);
    }
}

/////////////////////
//  VectorViewMut  //
/////////////////////

impl<'a, T: Numeric, const N: usize> MulAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn mul_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn mul_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<T> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] *= scalar);
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_mul_assign() {
        // Vector *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // Vector *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v *= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.0).abs() < f64::EPSILON);
        assert!((v[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_mul_assign() {
        // VectorViewMut *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // VectorViewMut *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.0).abs() < f64::EPSILON);
        assert!((v[2] - 1.5).abs() < f64::EPSILON);
    }
}
