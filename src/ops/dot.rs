use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;
use crate::matrix::RowVector;
use crate::matrix_view::RowVectorView;
use crate::matrix_view_mut::RowVectorViewMut;
use crate::matrix_transpose_view::RowVectorTransposeView;
use crate::matrix_transpose_view_mut::RowVectorTransposeViewMut;
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

macro_rules! impl_dot_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const N: usize, const M: usize> DotProduct<$rhs> for $lhs {
            type Output = T;

            fn dot(self, other: $rhs) -> Self::Output {
                (0..M).map(|i| self[i] * other[i]).sum()
            }
        }
    };
}

macro_rules! impl_dot_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric, const A: usize, const N: usize, const M: usize> DotProduct<$rhs> for $lhs {
            type Output = T;

            fn dot(self, other: $rhs) -> Self::Output {
                (0..M).map(|i| self[i] * other[i]).sum()
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

impl_dot_view!(Vector<T, M>, VectorView<'_, T, N, M>);
impl_dot_view!(Vector<T, M>, &VectorView<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, VectorView<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, &VectorView<'_, T, N, M>);

impl_dot_view!(Vector<T, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view!(Vector<T, M>, &VectorViewMut<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, &VectorViewMut<'_, T, N, M>);

impl_dot_view!(Vector<T, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view!(Vector<T, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, &RowVectorTransposeView<'_, T, N, M>);

impl_dot_view!(Vector<T, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(Vector<T, M>, &RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(&Vector<T, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

//////////////////
//  VectorView  //
//////////////////

impl_dot_view!(VectorView<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(VectorView<'_, T, N, M>, &Vector<T, M>);
impl_dot_view!(&VectorView<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(&VectorView<'_, T, N, M>, &Vector<T, M>);

impl_dot_view_view!(VectorView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, &VectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, &VectorView<'_, T, N, M>);

impl_dot_view_view!(VectorView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);

impl_dot_view_view!(VectorView<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(VectorView<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorView<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_dot_view!(VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(VectorViewMut<'_, T, N, M>, &Vector<T, M>);
impl_dot_view!(&VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(&VectorViewMut<'_, T, N, M>, &Vector<T, M>);

impl_dot_view_view!(VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, &VectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, &VectorView<'_, T, N, M>);

impl_dot_view_view!(VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);

impl_dot_view_view!(VectorViewMut<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(VectorViewMut<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorViewMut<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_dot_view!(VectorTransposeView<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(VectorTransposeView<'_, T, N, M>, &RowVector<T, M>);
impl_dot_view!(&VectorTransposeView<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(&VectorTransposeView<'_, T, N, M>, &RowVector<T, M>);

impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, &RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, &RowVectorView<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeView<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_dot_view!(VectorTransposeViewMut<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(VectorTransposeViewMut<'_, T, N, M>, &RowVector<T, M>);
impl_dot_view!(&VectorTransposeViewMut<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(&VectorTransposeViewMut<'_, T, N, M>, &RowVector<T, M>);

impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, &RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, &RowVectorView<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&VectorTransposeViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

/////////////////
//  RowVector  //
/////////////////

impl_dot!(RowVector<T, N>, RowVector<T, N>);
impl_dot!(RowVector<T, N>, &RowVector<T, N>);
impl_dot!(&RowVector<T, N>, RowVector<T, N>);
impl_dot!(&RowVector<T, N>, &RowVector<T, N>);

impl_dot_view!(RowVector<T, M>, RowVectorView<'_, T, N, M>);
impl_dot_view!(RowVector<T, M>, &RowVectorView<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, RowVectorView<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, &RowVectorView<'_, T, N, M>);

impl_dot_view!(RowVector<T, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view!(RowVector<T, M>, &RowVectorViewMut<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, &RowVectorViewMut<'_, T, N, M>);

impl_dot_view!(RowVector<T, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view!(RowVector<T, M>, &VectorTransposeView<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, &VectorTransposeView<'_, T, N, M>);

impl_dot_view!(RowVector<T, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(RowVector<T, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view!(&RowVector<T, M>, &VectorTransposeViewMut<'_, T, N, M>);

/////////////////////
//  RowVectorView  //
/////////////////////

impl_dot_view!(RowVectorView<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(RowVectorView<'_, T, N, M>, &RowVector<T, M>);
impl_dot_view!(&RowVectorView<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(&RowVectorView<'_, T, N, M>, &RowVector<T, M>);

impl_dot_view_view!(RowVectorView<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, T, A, M>, &RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, &RowVectorView<'_, T, N, M>);

impl_dot_view_view!(RowVectorView<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);

impl_dot_view_view!(RowVectorView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(RowVectorView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorView<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_dot_view!(RowVectorViewMut<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(RowVectorViewMut<'_, T, N, M>, &RowVector<T, M>);
impl_dot_view!(&RowVectorViewMut<'_, T, N, M>, RowVector<T, M>);
impl_dot_view!(&RowVectorViewMut<'_, T, N, M>, &RowVector<T, M>);

impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, &RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, RowVectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, &RowVectorView<'_, T, N, M>);

impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, RowVectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, &RowVectorViewMut<'_, T, N, M>);

impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, &VectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorViewMut<'_, T, A, M>, &VectorTransposeViewMut<'_, T, N, M>);

//////////////////////////////
//  RowVectorTransposeView  //
//////////////////////////////

impl_dot_view!(RowVectorTransposeView<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(RowVectorTransposeView<'_, T, N, M>, &Vector<T, M>);
impl_dot_view!(&RowVectorTransposeView<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(&RowVectorTransposeView<'_, T, N, M>, &Vector<T, M>);

impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, &VectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, &VectorView<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeView<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeView<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

/////////////////////////////////
//  RowVectorTransposeViewMut  //
/////////////////////////////////

impl_dot_view!(RowVectorTransposeViewMut<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(RowVectorTransposeViewMut<'_, T, N, M>, &Vector<T, M>);
impl_dot_view!(&RowVectorTransposeViewMut<'_, T, N, M>, Vector<T, M>);
impl_dot_view!(&RowVectorTransposeViewMut<'_, T, N, M>, &Vector<T, M>);

impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, &VectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, &VectorView<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, &VectorViewMut<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, RowVectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, &RowVectorTransposeView<'_, T, N, M>);

impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(RowVectorTransposeViewMut<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, RowVectorTransposeViewMut<'_, T, N, M>);
impl_dot_view_view!(&RowVectorTransposeViewMut<'_, T, A, M>, &RowVectorTransposeViewMut<'_, T, N, M>);

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
