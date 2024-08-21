use crate::vector::vector_impl::Vector;
use crate::vector::vector_view::VectorView;
use crate::vector::vector_view_mut::VectorViewMut;
use funty::Numeric;
use std::ops::SubAssign;

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, other: $rhs) {
                (0..N).for_each(|i| self[i] -= other[i]);
            }
        }
    };
    ($lhs:ty) => {
        impl<T: Numeric, const N: usize> SubAssign<T> for $lhs {
            fn sub_assign(&mut self, scalar: T) {
                (0..N).for_each(|i| self[i] -= scalar);
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

impl_sub_assign!(Vector<T, N>); // Scalar
impl_sub_assign!(Vector<T, N>, Vector<T, N>);
impl_sub_assign!(Vector<T, N>, &Vector<T, N>);
impl_sub_assign!(Vector<T, N>, VectorView<'_, T, N>);
impl_sub_assign!(Vector<T, N>, &VectorView<'_, T, N>);
impl_sub_assign!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_sub_assign!(Vector<T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_sub_assign!(VectorViewMut<'_, T, N>); // Scalar
impl_sub_assign!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_sub_assign!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_sub_assign!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_sub_assign!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_sub_assign!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_sub_assign!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn test_vector_sub_assign() {
        // Vector -= Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2;
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= &Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2;
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= &VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // Vector -= Vector (i32)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([0, 1, 2]);
        v1 -= v2;
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 1);
        assert_eq!(v1[2], 1);

        // Vector -= Scalar
        let mut v = Vector::<i32, 3>::new([5, 6, 7]);
        v -= 2;
        assert_eq!(v[0], 3);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 5);

        // Vector -= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v -= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.5).abs() < f64::EPSILON);
        assert!((v[2] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_sub_assign() {
        // VectorViewMut -= Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= &Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2;
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= &VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < f64::EPSILON);
        assert!((v1[1] - 3.0).abs() < f64::EPSILON);
        assert!((v1[2] - 3.0).abs() < f64::EPSILON);

        // VectorViewMut -= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view -= 0.5;
        assert!((v[0] - 0.5).abs() < f64::EPSILON);
        assert!((v[1] - 1.5).abs() < f64::EPSILON);
        assert!((v[2] - 2.5).abs() < f64::EPSILON);
    }
}
