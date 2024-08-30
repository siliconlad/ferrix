use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use std::ops::{Index, Mul};

// Generate macros
generate_dot_macros!();

//////////////
//  Vector  //
//////////////

impl_dot!(Vector<T, M>, Vector<T, M>);
impl_dot_view!(Vector<T, M>, VectorView<'_, V, T, N, M>);
impl_dot_view!(Vector<T, M>, VectorViewMut<'_, V, T, N, M>);

//////////////////
//  VectorView  //
//////////////////

impl_dot_view!(VectorView<'_, V, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorView<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_dot_view_view!(VectorView<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_dot_view!(VectorViewMut<'_, V, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

/////////////////
//  RowVector  //
/////////////////

impl_dot!(RowVector<T, M>, RowVector<T, M>);
impl_dot_view!(RowVector<T, M>, RowVectorView<'_, V, T, N, M>);
impl_dot_view!(RowVector<T, M>, RowVectorViewMut<'_, V, T, N, M>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_dot_view!(RowVectorView<'_, V, T, N, M>, RowVector<T, M>);
impl_dot_view_view!(RowVectorView<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_dot_view!(RowVectorViewMut<'_, V, T, N, M>, RowVector<T, M>);
impl_dot_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);
