use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use funty::Numeric;

macro_rules! impl_dot {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize> DotProduct<$rhs> for $lhs {
            type Output = T;

            fn dot(self, other: $rhs) -> Self::Output {
                (0..N).map(|i| self[i] * other[i]).sum()
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

impl_dot!(Vector<T, N>, Vector<T, N>);
impl_dot!(Vector<T, N>, &Vector<T, N>);
impl_dot!(&Vector<T, N>, Vector<T, N>);
impl_dot!(&Vector<T, N>, &Vector<T, N>);

impl_dot!(Vector<T, N>, VectorView<'_, T, N>);
impl_dot!(Vector<T, N>, &VectorView<'_, T, N>);
impl_dot!(&Vector<T, N>, VectorView<'_, T, N>);
impl_dot!(&Vector<T, N>, &VectorView<'_, T, N>);

impl_dot!(Vector<T, N>, VectorViewMut<'_, T, N>);
impl_dot!(Vector<T, N>, &VectorViewMut<'_, T, N>);
impl_dot!(&Vector<T, N>, VectorViewMut<'_, T, N>);
impl_dot!(&Vector<T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  VectorView  //
//////////////////

impl_dot!(VectorView<'_, T, N>, Vector<T, N>);
impl_dot!(VectorView<'_, T, N>, &Vector<T, N>);
impl_dot!(&VectorView<'_, T, N>, Vector<T, N>);
impl_dot!(&VectorView<'_, T, N>, &Vector<T, N>);

impl_dot!(VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_dot!(VectorView<'_, T, N>, &VectorView<'_, T, N>);
impl_dot!(&VectorView<'_, T, N>, VectorView<'_, T, N>);
impl_dot!(&VectorView<'_, T, N>, &VectorView<'_, T, N>);

impl_dot!(VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_dot!(VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_dot!(&VectorView<'_, T, N>, VectorViewMut<'_, T, N>);
impl_dot!(&VectorView<'_, T, N>, &VectorViewMut<'_, T, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_dot!(VectorViewMut<'_, T, N>, Vector<T, N>);
impl_dot!(VectorViewMut<'_, T, N>, &Vector<T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, Vector<T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, &Vector<T, N>);

impl_dot!(VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_dot!(VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, VectorView<'_, T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, &VectorView<'_, T, N>);

impl_dot!(VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_dot!(VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, VectorViewMut<'_, T, N>);
impl_dot!(&VectorViewMut<'_, T, N>, &VectorViewMut<'_, T, N>);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_dot() {
        // Vector dot Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // Vector dot &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &Vector dot Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &Vector dot &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // Vector dot VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // Vector dot &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(&view) - 14.0).abs() < f64::EPSILON);

        // &Vector dot VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!(((&v1).dot(view) - 14.0).abs() < f64::EPSILON);

        // &Vector dot &VectorView
        let view = v2.view::<3>(0).unwrap();
        assert!(((&v1).dot(&view) - 14.0).abs() < f64::EPSILON);

        // Vector dot VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // Vector dot &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(&view_mut) - 14.0).abs() < f64::EPSILON);

        // &Vector dot VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!(((&v1).dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // &Vector dot &VectorViewMut
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!(((&v1).dot(&view_mut) - 14.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_dot() {
        // VectorView dot Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorView dot &Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &VectorView dot Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &VectorView dot &Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // VectorView dot VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorView dot &VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot &VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);

        // VectorView dot VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view_mut) - 20.0).abs() < f64::EPSILON);

        // VectorView dot &VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(&view_mut) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(view_mut) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot &VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(&view_mut) - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_dot() {
        // VectorViewMut dot Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorViewMut dot &Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &VectorViewMut dot Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot &VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot &VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);
    }
}
