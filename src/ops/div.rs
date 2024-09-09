use std::ops::{Div, Index};

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

// Generate the macros
generate_op_scalar_macros!(Div, div, /);

impl_vv_op!(Vector<T, N>);
impl_vv_op_view!(VectorView<'_, V, T, N, M>);
impl_vv_op_view!(VectorViewMut<'_, V, T, N, M>);

impl_vv_op_row!(RowVector<T, N>);
impl_vv_op_view_row!(RowVectorView<'_, V, T, N, M>);
impl_vv_op_view_row!(RowVectorViewMut<'_, V, T, N, M>);

impl_mm_op!(Matrix<T, M, N>);
impl_mm_op_view!(MatrixView<'_, T, A, B, M, N>);
impl_mm_op_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_view!(MatrixTransposeView<'_, T, A, B, M, N>);
impl_mm_op_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
