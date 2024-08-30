use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::traits::MatMul;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;
use std::ops::{Index, Mul, Add};

// Generate macros
generate_matmul_macros!();

//////////////
//  Vector  //
//////////////

// M x 1 * 1 x N -> M x N
impl_vecmul!(Vector<T, M>, RowVector<T, N>);
impl_vecmul_view!(Vector<T, M>, RowVectorView<'_, V, T, A, N>);
impl_vecmul_view!(Vector<T, M>, RowVectorViewMut<'_, V, T, A, N>);

// M x 1 * 1 x N -> M x N
impl_vecmul!(Vector<T, M>, Matrix<T, 1, N>);
impl_vecmul_mat_row_view!(Vector<T, M>, MatrixView<'_, T, A, B, 1, N>);
impl_vecmul_mat_row_view!(Vector<T, M>, MatrixViewMut<'_, T, A, B, 1, N>);
impl_vecmul_mat_row_view!(Vector<T, M>, MatrixTransposeView<'_, T, A, B, 1, N>);
impl_vecmul_mat_row_view!(Vector<T, M>, MatrixTransposeViewMut<'_, T, A, B, 1, N>);

//////////////////
//  VectorView  //
//////////////////

// M x 1 * 1 x N -> M x N
impl_vecmul_view!(VectorView<'_, V, T, A, M>, RowVector<T, N>);
impl_vecmul_view_view!(VectorView<'_, V1, T, A, M>, RowVectorView<'_, V2, T, B, N>);
impl_vecmul_view_view!(VectorView<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, B, N>);

// M x 1 * 1 x N -> M x N
impl_vecmul_view!(VectorView<'_, V, T, A, M>, Matrix<T, 1, N>);
impl_vecmul_mat_row_view_view!(VectorView<'_, V, T, A, M>, MatrixView<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorView<'_, V, T, A, M>, MatrixViewMut<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorView<'_, V, T, A, M>, MatrixTransposeView<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorView<'_, V, T, A, M>, MatrixTransposeViewMut<'_, T, B, C, 1, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

// N x 1 * 1 x M -> N x M
impl_vecmul_view!(VectorViewMut<'_, V, T, A, M>, RowVector<T, N>);
impl_vecmul_view_view!(VectorViewMut<'_, V1, T, A, M>, RowVectorView<'_, V2, T, B, N>);
impl_vecmul_view_view!(VectorViewMut<'_, V1, T, A, M>, RowVectorViewMut<'_, V2, T, B, N>);

// N x 1 * 1 x M -> N x M
impl_vecmul_view!(VectorViewMut<'_, V, T, A, M>, Matrix<T, 1, N>);
impl_vecmul_mat_row_view_view!(VectorViewMut<'_, V, T, A, M>, MatrixView<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorViewMut<'_, V, T, A, M>, MatrixViewMut<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorViewMut<'_, V, T, A, M>, MatrixTransposeView<'_, T, B, C, 1, N>);
impl_vecmul_mat_row_view_view!(VectorViewMut<'_, V, T, A, M>, MatrixTransposeViewMut<'_, T, B, C, 1, N>);

/////////////////
//  RowVector  //
/////////////////

// 1 x N * N x 1 -> 1 x 1
impl_vecmul_scalar!(RowVector<T, M>, Vector<T, M>);
impl_vecmul_scalar_view!(RowVector<T, M>, VectorView<'_, V, T, N, M>);
impl_vecmul_scalar_view!(RowVector<T, M>, VectorViewMut<'_, V, T, N, M>);

// 1 x N * N x M -> 1 x M
impl_vecmul_mat!(RowVector<T, N>, Matrix<T, N, M>);
impl_vecmul_vecmat_view!(RowVector<T, N>, MatrixView<'_, T, A, B, N, M>);
impl_vecmul_vecmat_view!(RowVector<T, N>, MatrixViewMut<'_, T, A, B, N, M>);
impl_vecmul_vecmat_view!(RowVector<T, N>, MatrixTransposeView<'_, T, A, B, N, M>);
impl_vecmul_vecmat_view!(RowVector<T, N>, MatrixTransposeViewMut<'_, T, A, B, N, M>);

/////////////////////
//  RowVectorView  //
/////////////////////

// 1 x N * N x 1 -> 1 x 1
impl_vecmul_scalar_view!(RowVectorView<'_, V, T, N, M>, Vector<T, M>);
impl_vecmul_scalar_view_view!(RowVectorView<'_, V1, T, A, M>, VectorView<'_, V2, T, B, M>);
impl_vecmul_scalar_view_view!(RowVectorView<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, B, M>);

// 1 x N * N x M -> 1 x M
impl_vecmul_mat_view!(RowVectorView<'_, V, T, A, N>, Matrix<T, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorView<'_, V, T, A, N>, MatrixView<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorView<'_, V, T, A, N>, MatrixViewMut<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorView<'_, V, T, A, N>, MatrixTransposeView<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorView<'_, V, T, A, N>, MatrixTransposeViewMut<'_, T, B, C, N, M>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

// 1 x N * N x 1 -> 1 x 1
impl_vecmul_scalar_view!(RowVectorViewMut<'_, V, T, N, M>, Vector<T, M>);
impl_vecmul_scalar_view_view!(RowVectorViewMut<'_, V1, T, A, M>, VectorView<'_, V2, T, B, M>);
impl_vecmul_scalar_view_view!(RowVectorViewMut<'_, V1, T, A, M>, VectorViewMut<'_, V2, T, B, M>);

// 1 x N * N x M -> 1 x M
impl_vecmul_mat_view!(RowVectorViewMut<'_, V, T, A, N>, Matrix<T, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorViewMut<'_, V, T, A, N>, MatrixView<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorViewMut<'_, V, T, A, N>, MatrixViewMut<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorViewMut<'_, V, T, A, N>, MatrixTransposeView<'_, T, B, C, N, M>);
impl_vecmul_vecmat_view_view!(RowVectorViewMut<'_, V, T, A, N>, MatrixTransposeViewMut<'_, T, B, C, N, M>);

//////////////
//  Matrix  //
//////////////

// M x N * N x 1 -> M x 1
impl_matmul_vec!(Matrix<T, M, N>, Vector<T, N>);
impl_matmul_vec_view!(Matrix<T, M, N>, VectorView<'_, V, T, A, N>);
impl_matmul_vec_view!(Matrix<T, M, N>, VectorViewMut<'_, V, T, A, N>);

// M x 1 * 1 x N -> M x N
impl_matmul_matvec!(Matrix<T, M, 1>, RowVector<T, N>);
impl_matmul_matvec_vec_view!(Matrix<T, M, 1>, RowVectorView<'_, V, T, A, N>);
impl_matmul_matvec_vec_view!(Matrix<T, M, 1>, RowVectorViewMut<'_, V, T, A, N>);

// M x N * N x P -> M x P
impl_matmul!(Matrix<T, M, N>, Matrix<T, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, N, P>);

//////////////////
//  MatrixView  //
//////////////////

// M x N * N x 1 -> M x 1
impl_matmul_mat_view!(MatrixView<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixView<'_, T, A, B, M, N>, VectorView<'_, V, T, C, N>);
impl_matmul_mat_view_view!(MatrixView<'_, T, A, B, M, N>, VectorViewMut<'_, V, T, C, N>);

// M x 1 * 1 x N -> M x N
impl_matmul_matvec_mat_view!(MatrixView<'_, T, A, B, M, 1>, RowVector<T, N>);
impl_matmul_matvec_view_view!(MatrixView<'_, T, A, B, M, 1>, RowVectorView<'_, V, T, C, N>);
impl_matmul_matvec_view_view!(MatrixView<'_, T, A, B, M, 1>, RowVectorViewMut<'_, V, T, C, N>);

// M x N * N x P -> M x P
impl_matmul_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

// M x N * N x 1 -> M x 1
impl_matmul_mat_view!(MatrixViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixViewMut<'_, T, A, B, M, N>, VectorView<'_, V, T, C, N>);
impl_matmul_mat_view_view!(MatrixViewMut<'_, T, A, B, M, N>, VectorViewMut<'_, V, T, C, N>);

// M x 1 * 1 x N -> M x N
impl_matmul_matvec_mat_view!(MatrixViewMut<'_, T, A, B, M, 1>, RowVector<T, N>);
impl_matmul_matvec_view_view!(MatrixViewMut<'_, T, A, B, M, 1>, RowVectorView<'_, V, T, C, N>);
impl_matmul_matvec_view_view!(MatrixViewMut<'_, T, A, B, M, 1>, RowVectorViewMut<'_, V, T, C, N>);

// M x N * N x P -> M x P
impl_matmul_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

// M x N * N x 1 -> M x 1
impl_matmul_mat_view!(MatrixTransposeView<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, VectorView<'_, V, T, C, N>);
impl_matmul_mat_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, VectorViewMut<'_, V, T, C, N>);

// M x 1 * 1 x N -> M x N
impl_matmul_matvec_mat_view!(MatrixTransposeView<'_, T, A, B, M, 1>, RowVector<T, N>);
impl_matmul_matvec_view_view!(MatrixTransposeView<'_, T, A, B, M, 1>, RowVectorView<'_, V, T, C, N>);
impl_matmul_matvec_view_view!(MatrixTransposeView<'_, T, A, B, M, 1>, RowVectorViewMut<'_, V, T, C, N>);

// M x N * N x P -> M x P
impl_matmul_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

// M x N * N x 1 -> M x 1
impl_matmul_mat_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, VectorView<'_, V, T, C, N>);
impl_matmul_mat_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, VectorViewMut<'_, V, T, C, N>);

// M x 1 * 1 x N -> M x N
impl_matmul_matvec_mat_view!(MatrixTransposeViewMut<'_, T, A, B, M, 1>, RowVector<T, N>);
impl_matmul_matvec_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, 1>, RowVectorView<'_, V, T, C, N>);
impl_matmul_matvec_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, 1>, RowVectorViewMut<'_, V, T, C, N>);

// M x N * N x P -> M x P
impl_matmul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
