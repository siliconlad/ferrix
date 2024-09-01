use std::ops::{MulAssign, IndexMut};

use crate::matrix::Matrix;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::vector::Vector;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view_mut::RowVectorViewMut;

// Generate macros
generate_op_assign_scalar_macros!(MulAssign, mul_assign, *=);

impl_vv_op_assign!(Vector<T, N>);
impl_vv_op_assign_view!(VectorViewMut<'_, V, T, N, M>);

impl_vv_op_assign!(RowVector<T, N>);
impl_vv_op_assign_view!(RowVectorViewMut<'_, V, T, N, M>);

impl_mm_op_assign!(Matrix<T, M, N>);
impl_mm_op_assign_view!(MatrixViewMut<'_, T, A, B, M, N>);
impl_mm_op_assign_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>);
