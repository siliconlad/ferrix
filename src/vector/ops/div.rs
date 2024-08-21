use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Div;

macro_rules! impl_div {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Div<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn div(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Div<T> for $lhs {
            type Output = Vector<T, N>;

            fn div(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_div!(Vector<T, N>);
impl_div!(&Vector<T, N>);

impl_div!(Vector<T, N>, Vector<T, N>);
impl_div!(Vector<T, N>, &Vector<T, N>);
impl_div!(&Vector<T, N>, Vector<T, N>);
impl_div!(&Vector<T, N>, &Vector<T, N>);

impl_div!(Vector<T, N>, VectorView<'_, T, N>);
impl_div!(Vector<T, N>, &VectorView<'_, T, N>);
impl_div!(&Vector<T, N>, VectorView<'_, T, N>);
impl_div!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_div!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_div!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_div!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_div!(VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>);

impl_div!(VectorView<'_, T, N>, Vector<T, N>);
impl_div!(VectorView<'_, T, N>, &Vector<T, N>);
impl_div!(&VectorView<'_, T, N>, Vector<T, N>);
impl_div!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_div!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_div!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_div!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_div!(VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>);

impl_div!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_div!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_div!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_div!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_div!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_div!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_div!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  Unit Tests  //
//////////////////

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
