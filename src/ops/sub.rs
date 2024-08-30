use std::ops::{Sub, Index};
use funty::Numeric;

use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;

// Generate the macros
generate_op_macros!(Sub, sub, -);

//////////////
//  Vector  //
//////////////

impl_vv_op!(Vector<T, N>); // Scalar
impl_vv_op!(Vector<T, N>, Vector<T, N>);
impl_vv_op_view!(Vector<T, M>, VectorView<'_, V, T, N, M>);
impl_vv_op_view!(Vector<T, M>, VectorViewMut<'_, V, T, N, M>);

//////////////////
//  VectorView  //
//////////////////

impl_vv_op_view!(VectorView<'_, V, T, N, M>); // Scalar
impl_vv_op_view!(VectorView<'_, V, T, N, M>, Vector<T, M>);
impl_vv_op_view_view!(VectorView<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_vv_op_view_view!(VectorView<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_op_view!(VectorViewMut<'_, V, T, N, M>); // Scalar
impl_vv_op_view!(VectorViewMut<'_, V, T, N, M>, Vector<T, M>);
impl_vv_op_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_vv_op_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

/////////////////
//  RowVector  //
/////////////////

impl_vv_op_row!(RowVector<T, N>); // Scalar
impl_vv_op_row!(RowVector<T, N>, RowVector<T, N>);
impl_vv_op_view_row!(RowVector<T, M>, RowVectorView<'_, V, T, N, M>);
impl_vv_op_view_row!(RowVector<T, M>, RowVectorViewMut<'_, V, T, N, M>);

/////////////////////
//  RowVectorView  //
/////////////////////

impl_vv_op_view_row!(RowVectorView<'_, V, T, N, M>); // Scalar
impl_vv_op_view_row!(RowVectorView<'_, V, T, N, M>, RowVector<T, M>);
impl_vv_op_view_view_row!(RowVectorView<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_vv_op_view_view_row!(RowVectorView<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_vv_op_view_row!(RowVectorViewMut<'_, V, T, N, M>); // Scalar
impl_vv_op_view_row!(RowVectorViewMut<'_, V, T, N, M>, RowVector<T, M>);
impl_vv_op_view_view_row!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_vv_op_view_view_row!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

//////////////
//  Matrix  //
//////////////

impl_mm_op!(Matrix<T, M, N>); // Scalar
impl_mm_op!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_op_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_op_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_op_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);

//////////////////
//  MatrixView  //
//////////////////

impl_mm_op_view!(MatrixView<'_, T, A, B, M, N>); // Scalar
impl_mm_op_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

impl_mm_op_view!(MatrixViewMut<'_, T, A, B, M, N>); // Scalar
impl_mm_op_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

impl_mm_op_view!(MatrixTransposeView<'_, T, A, B, M, N>); // Scalar
impl_mm_op_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

impl_mm_op_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>); // Scalar
impl_mm_op_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);
