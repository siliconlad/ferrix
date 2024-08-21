use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::Sub;

macro_rules! impl_sub {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> Sub<$rhs> for $lhs {
            type Output = Vector<T, N>;

            fn sub(self, other: $rhs) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> Sub<T> for $lhs {
            type Output = Vector<T, N>;

            fn sub(self, scalar: T) -> Self::Output {
                Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

// Scalar
impl_sub!(Vector<T, N>);
impl_sub!(&Vector<T, N>);

impl_sub!(Vector<T, N>, Vector<T, N>);
impl_sub!(Vector<T, N>, &Vector<T, N>);
impl_sub!(&Vector<T, N>, Vector<T, N>);
impl_sub!(&Vector<T, N>, &Vector<T, N>);

impl_sub!(Vector<T, N>, VectorView<'_, T, N>);
impl_sub!(Vector<T, N>, &VectorView<'_, T, N>);
impl_sub!(&Vector<T, N>, VectorView<'_, T, N>);
impl_sub!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_sub!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_sub!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

// Scalar
impl_sub!(VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>);

impl_sub!(VectorView<'_, T, N>, Vector<T, N>);
impl_sub!(VectorView<'_, T, N>, &Vector<T, N>);
impl_sub!(&VectorView<'_, T, N>, Vector<T, N>);
impl_sub!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_sub!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_sub!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// Scalar
impl_sub!(VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>);

impl_sub!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_sub!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_sub!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  Unit Tests  //
//////////////////

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
