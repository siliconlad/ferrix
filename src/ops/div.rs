use std::ops::Div;
use funty::Numeric;

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

// Generate the macros
generate_op_macros!(Div, div, /);

//////////////
//  Vector  //
//////////////

impl_vv_op!(Vector<T, N>); // Scalar
impl_vv_op!(Vector<T, N>, Vector<T, N>);
impl_vv_op_view!(Vector<T, M>, VectorView<'_, T, N, M>);
impl_vv_op_view!(Vector<T, M>, VectorViewMut<'_, T, N, M>);

//////////////////
//  VectorView  //
//////////////////

impl_vv_op_view!(VectorView<'_, T, N, M>); // Scalar
impl_vv_op_view!(VectorView<'_, T, N, M>, Vector<T, M>);
impl_vv_op_view_view!(VectorView<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_vv_op_view_view!(VectorView<'_, T, A, M>, VectorViewMut<'_, T, N, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vv_op_view!(VectorViewMut<'_, T, N, M>); // Scalar
impl_vv_op_view!(VectorViewMut<'_, T, N, M>, Vector<T, M>);
impl_vv_op_view_view!(VectorViewMut<'_, T, A, M>, VectorView<'_, T, N, M>);
impl_vv_op_view_view!(VectorViewMut<'_, T, A, M>, VectorViewMut<'_, T, N, M>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_vv_op_view!(VectorTransposeView<'_, T, N, M>); // Scalar
impl_vv_op_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_op_view_view!(VectorTransposeView<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_vv_op_view!(VectorTransposeViewMut<'_, T, N, M>); // Scalar
impl_vv_op_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeView<'_, T, N, M>);
impl_vv_op_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, N, M>);

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
