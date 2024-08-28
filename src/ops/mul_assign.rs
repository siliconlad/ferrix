use funty::Numeric;
use std::ops::MulAssign;

use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;

// Generate macros
generate_op_assign_macros!(MulAssign, mul_assign, *=);

//////////////
//  Vector  //
//////////////

impl_vv_op_assign!(Vector<T, N>); // Scalar
impl_vv_op_assign!(Vector<T, N>, Vector<T, N>);
impl_vv_op_assign_view!(Vector<T, M>, VectorView<'_, T, N, M>);
impl_vv_op_assign_view!(Vector<T, M>, VectorViewMut<'_, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_op_assign_view!(VectorViewMut<'_, T, N, M>); // Scalar
impl_vv_op_assign_view!(VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_vv_op_assign_view_view!(VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_vv_op_assign_view_view!(VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_vv_op_assign_view!(VectorTransposeViewMut<'_, T, N, M>);  // Scalar
impl_vv_op_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_op_assign_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);

//////////////
//  Matrix  //
//////////////

// Scalar
impl_mm_op_assign!(Matrix<T, M, N>);
impl_mm_op_assign!(Matrix<T, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, M, N>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// Scalar
impl_mm_op_assign_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// Scalar
impl_mm_op_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, M, N>);
impl_mm_op_assign_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, M, N>);
