use std::ops::{SubAssign, Index,IndexMut};

use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;

// Generate macros
generate_op_assign_macros!(SubAssign, sub_assign, -=);

//////////////
//  Vector  //
//////////////

impl_vv_op_assign!(Vector<T, N>); // Scalar
impl_vv_op_assign!(Vector<T, N>, Vector<T, N>);
impl_vv_op_assign_view!(Vector<T, M>, VectorView<'_, V, T, N, M>);
impl_vv_op_assign_view!(Vector<T, M>, VectorViewMut<'_, V, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_op_assign_view!(VectorViewMut<'_, V, T, N, M>); // Scalar
impl_vv_op_assign_view!(VectorViewMut<'_, V, T, N, M>, Vector<T, M>);
impl_vv_op_assign_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorView<'_, V2, T, N, M>);
impl_vv_op_assign_view_view!(VectorViewMut<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, N, M>);

/////////////////
//  RowVector  //
/////////////////

impl_vv_op_assign!(RowVector<T, N>); // Scalar
impl_vv_op_assign!(RowVector<T, N>, RowVector<T, N>);
impl_vv_op_assign_view!(RowVector<T, M>, RowVectorView<'_, V, T, N, M>);
impl_vv_op_assign_view!(RowVector<T, M>, RowVectorViewMut<'_, V, T, N, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_vv_op_assign_view!(RowVectorViewMut<'_, V, T, N, M>); // Scalar
impl_vv_op_assign_view!(RowVectorViewMut<'_, V, T, N, M>, RowVector<T, M>);
impl_vv_op_assign_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorView<'_, V2, T, N, M>);
impl_vv_op_assign_view_view!(RowVectorViewMut<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, N, M>);

//////////////
//  Matrix  //
//////////////

impl_mm_op_assign!(Matrix<T, M, N>); // Scalar
impl_mm_op_assign!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

impl_mm_op_assign_view!(MatrixViewMut<'_, T, A, B, M, N>); // Scalar
impl_mm_op_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

impl_mm_op_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>); // Scalar
impl_mm_op_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);
