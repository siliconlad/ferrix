use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use std::ops::{Index, Mul, Add};
use num_traits::Zero;

// Generate macros
generate_dot_macros!();

//////////////
//  Vector  //
//////////////

impl_dot!(Vector<T, M>, Vector<T, M>);
impl_dot_view!(Vector<T, M>, VectorView<'_, V, T, N, M>);
impl_dot_view!(Vector<T, M>, VectorViewMut<'_, V, T, N, M>);

impl_dot!(Vector<T, M>, Matrix<T, M, 1>);
impl_dot_mat_view!(Vector<T, M>, MatrixView<'_, T, A, B, M, 1>);
impl_dot_mat_view!(Vector<T, M>, MatrixViewMut<'_, T, A, B, M, 1>);
impl_dot_mat_view!(Vector<T, M>, MatrixTransposeView<'_, T, A, B, M, 1>);
impl_dot_mat_view!(Vector<T, M>, MatrixTransposeViewMut<'_, T, A, B, M, 1>);

//////////////////
//  VectorView  //
//////////////////

impl_dot_view!(VectorView<'_, V, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorView<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_dot_view_view!(VectorView<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

impl_dot_view!(VectorView<'_, V, T, N, M>, Matrix<T, M, 1>);
impl_dot_mat_view_view!(VectorView<'_, V, T, N, M>, MatrixView<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorView<'_, V, T, N, M>, MatrixViewMut<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorView<'_, V, T, N, M>, MatrixTransposeView<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorView<'_, V, T, N, M>, MatrixTransposeViewMut<'_, T, A, B, M, 1>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_dot_view!(VectorViewMut<'_, V, T, N, M>, Vector<T, M>);
impl_dot_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_dot_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

impl_dot_view!(VectorViewMut<'_, V, T, N, M>, Matrix<T, M, 1>);
impl_dot_mat_view_view!(VectorViewMut<'_, V, T, N, M>, MatrixView<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorViewMut<'_, V, T, N, M>, MatrixViewMut<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorViewMut<'_, V, T, N, M>, MatrixTransposeView<'_, T, A, B, M, 1>);
impl_dot_mat_view_view!(VectorViewMut<'_, V, T, N, M>, MatrixTransposeViewMut<'_, T, A, B, M, 1>);

/////////////////
//  RowVector  //
/////////////////

impl_dot!(RowVector<T, M>, RowVector<T, M>);
impl_dot_view!(RowVector<T, M>, RowVectorView<'_, V, T, N, M>);
impl_dot_view!(RowVector<T, M>, RowVectorViewMut<'_, V, T, N, M>);

impl_dot!(RowVector<T, M>, Matrix<T, 1, M>);
impl_dot_mat_view!(RowVector<T, M>, MatrixView<'_, T, A, B, 1, M>);
impl_dot_mat_view!(RowVector<T, M>, MatrixViewMut<'_, T, A, B, 1, M>);
impl_dot_mat_view!(RowVector<T, M>, MatrixTransposeView<'_, T, A, B, 1, M>);
impl_dot_mat_view!(RowVector<T, M>, MatrixTransposeViewMut<'_, T, A, B, 1, M>);

/////////////////////
//  RowVectorView  //
/////////////////////

impl_dot_view!(RowVectorView<'_, V, T, N, M>, RowVector<T, M>);
impl_dot_view_view!(RowVectorView<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_dot_view_view!(RowVectorView<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

impl_dot_view!(RowVectorView<'_, V, T, N, M>, Matrix<T, 1, M>);
impl_dot_mat_view_view!(RowVectorView<'_, V, T, N, M>, MatrixView<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorView<'_, V, T, N, M>, MatrixViewMut<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorView<'_, V, T, N, M>, MatrixTransposeView<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorView<'_, V, T, N, M>, MatrixTransposeViewMut<'_, T, A, B, 1, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_dot_view!(RowVectorViewMut<'_, V, T, N, M>, RowVector<T, M>);
impl_dot_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_dot_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

impl_dot_view!(RowVectorViewMut<'_, V, T, N, M>, Matrix<T, 1, M>);
impl_dot_mat_view_view!(RowVectorViewMut<'_, V, T, N, M>, MatrixView<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorViewMut<'_, V, T, N, M>, MatrixViewMut<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorViewMut<'_, V, T, N, M>, MatrixTransposeView<'_, T, A, B, 1, M>);
impl_dot_mat_view_view!(RowVectorViewMut<'_, V, T, N, M>, MatrixTransposeViewMut<'_, T, A, B, 1, M>);
