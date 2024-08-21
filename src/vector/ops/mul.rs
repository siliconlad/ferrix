use crate::vector::traits::VectorOps;
use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Mul;

fn v_mul<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] * v2[i]))
}

fn v_mul_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] * scalar))
}

//////////////
//  Vector  //
//////////////

impl<T: Numeric, const N: usize> Mul<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Mul<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(self, scalar)
    }
}

//////////////////
//  VectorView  //
/////////////////

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * scalar))
    }
}

/////////////////////
//  VectorViewMut  //
/////////////////////

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(self, scalar)
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_mul() {
        // Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);

        // &Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mul() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView * VectorView
        let result = &view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * &VectorView
        let result = view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test &VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &Vector
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let result = &view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test scalar multiplication
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));
    }

    #[test]
    fn test_vector_view_mut_mul() {
        // VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // &VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 18.0).abs() < f64::EPSILON);

        // VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);

        // &VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
        assert!((result[2] - 1.5).abs() < f64::EPSILON);
    }
}
