use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;
use funty::Numeric;

// Generate macros
generate_dot_macros!();

//////////////
//  Vector  //
//////////////

impl_dot!(Vector<T, M>, Vector<T, M>);
impl_dot_view!(Vector<T, M>, VectorView<'_, T, N, M>);
impl_dot_view!(Vector<T, M>, VectorViewMut<'_, T, N, M>);

//////////////////
//  VectorView  //
//////////////////

impl_dot_view!(VectorView<'_, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(VectorView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_dot_view!(VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_dot_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);
