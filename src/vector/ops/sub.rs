use crate::vector::traits::VectorOps;
use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Sub;

fn v_sub<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] - v2[i]))
}

fn v_sub_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] - scalar))
}

impl<T: Numeric, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

///////////////
//  Vector  //
//////////////

impl<T: Numeric, const N: usize> Sub<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Sub<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(self, scalar)
    }
}

//////////////////
//  VectorView  //
//////////////////

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorView<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorView<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorView<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorView<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorViewMut<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorViewMut<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorViewMut<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorViewMut<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
    }
}

/////////////////////
//  VectorViewMut  //
/////////////////////

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(self, scalar)
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_sub() {
        // Vector - Vector (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &Vector (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - Vector (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &Vector (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorView (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &VectorView (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - VectorView (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &VectorView (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - VectorViewMut (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - &VectorViewMut (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - VectorViewMut (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &Vector - &VectorViewMut (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // Vector - Scalar (non-reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);

        // &Vector - Scalar (reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_sub() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView - VectorView
        let result = &view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - &VectorView
        let result = view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test &VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - &Vector
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let result = &view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test scalar subtraction
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_vector_view_mut_sub() {
        // VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - Vector (non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - &Vector (non-reference and reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - Vector (reference and non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // &VectorViewMut - &Vector (both references, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < f64::EPSILON);
        assert!((result[1] - 1.8).abs() < f64::EPSILON);
        assert!((result[2] - 2.7).abs() < f64::EPSILON);

        // VectorViewMut - Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert!((result[1] - 1.5).abs() < f64::EPSILON);
        assert!((result[2] - 2.5).abs() < f64::EPSILON);
    }
}
