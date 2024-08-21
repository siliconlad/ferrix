use crate::vector::traits::VectorOps;
use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Div;

fn v_div<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] / v2[i]))
}

fn v_div_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] / scalar))
}

///////////////
//  Vector  //
//////////////

impl<T: Numeric, const N: usize> Div<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Div<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(self, scalar)
    }
}

//////////////////
//  VectorView  //
//////////////////

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
    }
}

/////////////////////
//  VectorViewMut  //
/////////////////////

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(self, scalar)
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_div() {
        // Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 / v2;
        assert!((result[0] - 10.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 10.0).abs() < f64::EPSILON);

        // Vector / Scalar
        let v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v / 0.5;
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);
        assert!((result[2] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_div() {
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let v2 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView / VectorView
        let result = &view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorView
        let result = view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / &Vector
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let result = &view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test scalar division
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));
    }

    #[test]
    fn test_vector_view_mut_div() {
        // VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 10.0).abs() < f64::EPSILON);
        assert!((result[1] - 10.0).abs() < f64::EPSILON);
        assert!((result[2] - 10.0).abs() < f64::EPSILON);

        // VectorViewMut / Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() / 0.5;
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 4.0).abs() < f64::EPSILON);
        assert!((result[2] - 6.0).abs() < f64::EPSILON);
    }
}
