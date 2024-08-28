use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::traits::MatMul;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;
use funty::Numeric;

// Generate macros
generate_matmul_macros!();

//////////////
//  Vector  //
//////////////

// N x 1 * 1 x M -> N x M
impl_vecmul_view!(Vector<T, M>, VectorTransposeView<'_, T, A, N>);
impl_vecmul_view!(Vector<T, M>, VectorTransposeViewMut<'_, T, A, N>);

//////////////////
//  VectorView  //
//////////////////

// N x 1 * 1 x M -> N x M
impl_vecmul_view_view!(VectorView<'_, T, A, M>, VectorTransposeView<'_, T, B, N>);
impl_vecmul_view_view!(VectorView<'_, T, A, M>, VectorTransposeViewMut<'_, T, B, N>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vecmul_view_view!(VectorViewMut<'_, T, A, M>, VectorTransposeView<'_, T, B, N>);
impl_vecmul_view_view!(VectorViewMut<'_, T, A, M>, VectorTransposeViewMut<'_, T, B, N>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_vecmul_scalar_view!(VectorTransposeView<'_, T, N, M>, Vector<T, M>);
impl_vecmul_scalar_view_view!(VectorTransposeView<'_, T, A, M>, VectorView<'_, T, B, M>);
impl_vecmul_scalar_view_view!(VectorTransposeView<'_, T, A, M>, VectorViewMut<'_, T, B, M>);

impl_vecmul_mat_view!(VectorTransposeView<'_, T, A, N>, Matrix<T, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeView<'_, T, A, N>, MatrixView<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeView<'_, T, A, N>, MatrixViewMut<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeView<'_, T, A, N>, MatrixTransposeView<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeView<'_, T, A, N>, MatrixTransposeViewMut<'_, T, B, C, N, M>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_vecmul_scalar_view!(VectorTransposeViewMut<'_, T, N, M>, Vector<T, M>);
impl_vecmul_scalar_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorView<'_, T, B, M>);
impl_vecmul_scalar_view_view!(VectorTransposeViewMut<'_, T, A, M>, VectorViewMut<'_, T, B, M>);

impl_vecmul_mat_view!(VectorTransposeViewMut<'_, T, A, N>, Matrix<T, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeViewMut<'_, T, A, N>, MatrixView<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeViewMut<'_, T, A, N>, MatrixViewMut<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeViewMut<'_, T, A, N>, MatrixTransposeView<'_, T, B, C, N, M>);
impl_vecmul_mat_view_view!(VectorTransposeViewMut<'_, T, A, N>, MatrixTransposeViewMut<'_, T, B, C, N, M>);

//////////////
//  Matrix  //
//////////////

impl_matmul_vec!(Matrix<T, M, N>, Vector<T, N>);
impl_matmul_vec_view!(Matrix<T, M, N>, VectorView<'_, T, A, N>);
impl_matmul_vec_view!(Matrix<T, M, N>, VectorViewMut<'_, T, A, N>);

impl_matmul!(Matrix<T, M, N>, Matrix<T, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, N, P>);

//////////////////
//  MatrixView  //
//////////////////

impl_matmul_mat_view!(MatrixView<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixView<'_, T, A, B, M, N>, VectorView<'_, T, C, N>);
impl_matmul_mat_view_view!(MatrixView<'_, T, A, B, M, N>, VectorViewMut<'_, T, C, N>);

impl_matmul_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

impl_matmul_mat_view!(MatrixViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixViewMut<'_, T, A, B, M, N>, VectorView<'_, T, C, N>);
impl_matmul_mat_view_view!(MatrixViewMut<'_, T, A, B, M, N>, VectorViewMut<'_, T, C, N>);

impl_matmul_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

impl_matmul_mat_view!(MatrixTransposeView<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, VectorView<'_, T, C, N>);
impl_matmul_mat_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, VectorViewMut<'_, T, C, N>);

impl_matmul_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

impl_matmul_mat_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_matmul_mat_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, VectorView<'_, T, C, N>);
impl_matmul_mat_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, VectorViewMut<'_, T, C, N>);

impl_matmul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
