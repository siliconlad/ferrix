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
use crate::matrix::RowVector;
use crate::matrix_view::RowVectorView;
use crate::matrix_view_mut::RowVectorViewMut;
use crate::matrix_transpose_view::RowVectorTransposeView;
use crate::matrix_transpose_view_mut::RowVectorTransposeViewMut;
use funty::Numeric;

macro_rules! impl_vecmul {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric + From<u8>, const M: usize, const N: usize> MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, 1>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, 1>::zeros();
                for i in 0..M {
                    for j in 0..N {
                        let index = i * N + j;
                        result[i] += self[index] * other[index];
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_vecmul_vec_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric + From<u8>, const M: usize, const N: usize, const P: usize> MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, 1>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, 1>::zeros();
                for i in 0..M {
                    for j in 0..P {
                        let index = i * P + j;
                        result[i] += self[index] * other[index];
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_vecmul_mat_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric + From<u8>, const A: usize, const B: usize, const M: usize, const N: usize> MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, 1>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, 1>::zeros();
                for i in 0..M {
                    for j in 0..N {
                        let index = i * N + j;
                        result[i] += self[index] * other[index];
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_vecmul_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric + From<u8>, const A: usize, const B: usize, const M: usize, const N: usize, const P: usize> MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, 1>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, 1>::zeros();
                for i in 0..M {
                    for j in 0..P {
                        let index = i * P + j;
                        result[i] += self[index] * other[index];
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_matmul {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Numeric + From<u8>, const M: usize, const N: usize, const P: usize> MatMul<$rhs>
            for $lhs
        {
            type Output = Matrix<T, M, P>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, P>::zeros();
                for i in 0..M {
                    for j in 0..P {
                        for k in 0..N {
                            result[(i, j)] += self[(i, k)] * other[(k, j)];
                        }
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_matmul_view {
    ($lhs:ty, $rhs:ty) => {
        impl<
                T: Numeric + From<u8>,
                const A: usize,
                const B: usize,
                const M: usize,
                const N: usize,
                const P: usize,
            > MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, P>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, P>::zeros();
                for i in 0..M {
                    for j in 0..P {
                        for k in 0..N {
                            result[(i, j)] += self[(i, k)] * other[(k, j)];
                        }
                    }
                }
                result
            }
        }
    };
}

macro_rules! impl_matmul_view_view {
    ($lhs:ty, $rhs:ty) => {
        impl<
                T: Numeric + From<u8>,
                const A: usize,
                const B: usize,
                const C: usize,
                const D: usize,
                const M: usize,
                const N: usize,
                const P: usize,
            > MatMul<$rhs> for $lhs
        {
            type Output = Matrix<T, M, P>;

            fn matmul(self, other: $rhs) -> Self::Output {
                let mut result = Matrix::<T, M, P>::zeros();
                for i in 0..M {
                    for j in 0..P {
                        for k in 0..N {
                            result[(i, j)] += self[(i, k)] * other[(k, j)];
                        }
                    }
                }
                result
            }
        }
    };
}

//////////////
//  Vector  //
//////////////

impl_vecmul!(Vector<T, N>, Matrix<T, N, M>);
impl_vecmul!(Vector<T, N>, &Matrix<T, N, M>);
impl_vecmul!(&Vector<T, N>, Matrix<T, N, M>);
impl_vecmul!(&Vector<T, N>, &Matrix<T, N, M>);

impl_vecmul!(Vector<T, M>, VectorTransposeView<'_, T, N, M>);
impl_vecmul!(Vector<T, M>, &VectorTransposeView<'_, T, N, M>);
impl_vecmul!(&Vector<T, M>, VectorTransposeView<'_, T, N, M>);
impl_vecmul!(&Vector<T, M>, &VectorTransposeView<'_, T, N, M>);

impl_vecmul!(Vector<T, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_vecmul!(Vector<T, M>, &VectorTransposeViewMut<'_, T, N, M>);
impl_vecmul!(&Vector<T, M>, VectorTransposeViewMut<'_, T, N, M>);
impl_vecmul!(&Vector<T, M>, &VectorTransposeViewMut<'_, T, N, M>);

impl_vecmul_mat_view!(Vector<T, N>, MatrixView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(Vector<T, N>, &MatrixView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, MatrixView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, &MatrixView<'_, T, A, B, N, M>);

impl_vecmul_mat_view!(Vector<T, N>, MatrixViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(Vector<T, N>, &MatrixViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, MatrixViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, &MatrixViewMut<'_, T, A, B, N, M>);

impl_vecmul_mat_view!(Vector<T, N>, MatrixTransposeView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(Vector<T, N>, &MatrixTransposeView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, MatrixTransposeView<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, &MatrixTransposeView<'_, T, A, B, N, M>);

impl_vecmul_mat_view!(Vector<T, N>, MatrixTransposeViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(Vector<T, N>, &MatrixTransposeViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, MatrixTransposeViewMut<'_, T, A, B, N, M>);
impl_vecmul_mat_view!(&Vector<T, N>, &MatrixTransposeViewMut<'_, T, A, B, N, M>);


//////////////////
//  VectorView  //
//////////////////

impl_vecmul_vec_view!(VectorView<'_, T, N, P>, Matrix<T, M, P>);
impl_vecmul_vec_view!(VectorView<'_, T, N, P>, &Matrix<T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, Matrix<T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, &Matrix<T, M, P>);

impl_vecmul_vec_view!(VectorView<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(VectorView<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);

impl_vecmul_vec_view!(VectorView<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(VectorView<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorView<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);

impl_vecmul_view_view!(VectorView<'_, T, N, P>, MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorView<'_, T, N, P>, &MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, &MatrixView<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorView<'_, T, N, P>, MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorView<'_, T, N, P>, &MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, &MatrixViewMut<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorView<'_, T, N, P>, MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorView<'_, T, N, P>, &MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, &MatrixTransposeView<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorView<'_, T, N, P>, MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorView<'_, T, N, P>, &MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorView<'_, T, N, P>, &MatrixTransposeViewMut<'_, T, A, B, P, M>);

/////////////////////
//  VectorViewMut  //
/////////////////////

impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, Matrix<T, M, P>);
impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, &Matrix<T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, Matrix<T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, &Matrix<T, M, P>);

impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);

impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(VectorViewMut<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&VectorViewMut<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);

impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, &MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, MatrixView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, &MatrixView<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, &MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, MatrixViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, &MatrixViewMut<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, &MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, MatrixTransposeView<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, &MatrixTransposeView<'_, T, A, B, P, M>);

impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(VectorViewMut<'_, T, N, P>, &MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, MatrixTransposeViewMut<'_, T, A, B, P, M>);
impl_vecmul_view_view!(&VectorViewMut<'_, T, N, P>, &MatrixTransposeViewMut<'_, T, A, B, P, M>);

///////////////////////////
//  VectorTransposeView  //
///////////////////////////

impl_vecmul!(VectorTransposeView<'_, T, M, N>, Vector<T, N>);
impl_vecmul!(VectorTransposeView<'_, T, M, N>, &Vector<T, N>);
impl_vecmul!(&VectorTransposeView<'_, T, M, N>, Vector<T, N>);
impl_vecmul!(&VectorTransposeView<'_, T, M, N>, &Vector<T, N>);

impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, &VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, &VectorView<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, &VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, &VectorViewMut<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, &RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, &RowVectorTransposeView<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeView<'_, T, M, P>, &RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeView<'_, T, M, P>, &RowVectorTransposeViewMut<'_, T, N, P>);

//////////////////////////////
//  VectorTransposeViewMut  //
//////////////////////////////

impl_vecmul!(VectorTransposeViewMut<'_, T, M, N>, Vector<T, N>);
impl_vecmul!(VectorTransposeViewMut<'_, T, M, N>, &Vector<T, N>);
impl_vecmul!(&VectorTransposeViewMut<'_, T, M, N>, Vector<T, N>);
impl_vecmul!(&VectorTransposeViewMut<'_, T, M, N>, &Vector<T, N>);

impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, &VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, &VectorView<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, &VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, &VectorViewMut<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, &RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, RowVectorTransposeView<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, &RowVectorTransposeView<'_, T, N, P>);

impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(VectorTransposeViewMut<'_, T, M, P>, &RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, RowVectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&VectorTransposeViewMut<'_, T, M, P>, &RowVectorTransposeViewMut<'_, T, N, P>);

/////////////////
//  RowVector  //
/////////////////

impl_vecmul!(RowVector<T, N>, VectorTransposeView<'_, T, M, N>);
impl_vecmul!(RowVector<T, N>, &VectorTransposeView<'_, T, M, N>);
impl_vecmul!(&RowVector<T, N>, VectorTransposeView<'_, T, M, N>);
impl_vecmul!(&RowVector<T, N>, &VectorTransposeView<'_, T, M, N>);

impl_vecmul!(RowVector<T, N>, VectorTransposeViewMut<'_, T, M, N>);
impl_vecmul!(RowVector<T, N>, &VectorTransposeViewMut<'_, T, M, N>);
impl_vecmul!(&RowVector<T, N>, VectorTransposeViewMut<'_, T, M, N>);
impl_vecmul!(&RowVector<T, N>, &VectorTransposeViewMut<'_, T, M, N>);

/////////////////////
//  RowVectorView  //
/////////////////////

impl_vecmul_vec_view!(RowVectorView<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(RowVectorView<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorView<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorView<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);

impl_vecmul_vec_view!(RowVectorView<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(RowVectorView<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorView<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorView<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);

////////////////////////
//  RowVectorViewMut  //
////////////////////////

impl_vecmul_vec_view!(RowVectorViewMut<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(RowVectorViewMut<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorViewMut<'_, T, N, P>, VectorTransposeView<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorViewMut<'_, T, N, P>, &VectorTransposeView<'_, T, M, P>);

impl_vecmul_vec_view!(RowVectorViewMut<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(RowVectorViewMut<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorViewMut<'_, T, N, P>, VectorTransposeViewMut<'_, T, M, P>);
impl_vecmul_vec_view!(&RowVectorViewMut<'_, T, N, P>, &VectorTransposeViewMut<'_, T, M, P>);


//////////////
//  Matrix  //
//////////////

impl_vecmul!(Matrix<T, M, N>, Vector<T, N>);
impl_vecmul!(Matrix<T, M, N>, &Vector<T, N>);
impl_vecmul!(&Matrix<T, M, N>, Vector<T, N>);
impl_vecmul!(&Matrix<T, M, N>, &Vector<T, N>);

impl_vecmul_vec_view!(Matrix<T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(Matrix<T, M, P>, &VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&Matrix<T, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_vec_view!(&Matrix<T, M, P>, &VectorView<'_, T, N, P>);

impl_vecmul_vec_view!(Matrix<T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(Matrix<T, M, P>, &VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&Matrix<T, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_vec_view!(&Matrix<T, M, P>, &VectorViewMut<'_, T, N, P>);

impl_matmul!(Matrix<T, M, N>, Matrix<T, N, P>);
impl_matmul!(Matrix<T, M, N>, &Matrix<T, N, P>);
impl_matmul!(&Matrix<T, M, N>, Matrix<T, N, P>);
impl_matmul!(&Matrix<T, M, N>, &Matrix<T, N, P>);

impl_matmul_view!(Matrix<T, M, N>, MatrixView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, &MatrixView<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, MatrixView<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, &MatrixView<'_, T, A, B, N, P>);

impl_matmul_view!(Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, MatrixViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, &MatrixViewMut<'_, T, A, B, N, P>);

impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, MatrixTransposeView<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, &MatrixTransposeView<'_, T, A, B, N, P>);

impl_matmul_view!(Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, MatrixTransposeViewMut<'_, T, A, B, N, P>);
impl_matmul_view!(&Matrix<T, M, N>, &MatrixTransposeViewMut<'_, T, A, B, N, P>);

//////////////////
//  MatrixView  //
//////////////////

impl_vecmul_mat_view!(MatrixView<'_, T, A, B, M, N>, Vector<T, N>);
impl_vecmul_mat_view!(MatrixView<'_, T, A, B, M, N>, &Vector<T, N>);
impl_vecmul_mat_view!(&MatrixView<'_, T, A, B, M, N>, Vector<T, N>);
impl_vecmul_mat_view!(&MatrixView<'_, T, A, B, M, N>, &Vector<T, N>);

impl_vecmul_view_view!(MatrixView<'_, T, A, B, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_view_view!(MatrixView<'_, T, A, B, M, P>, &VectorView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixView<'_, T, A, B, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixView<'_, T, A, B, M, P>, &VectorView<'_, T, N, P>);

impl_vecmul_view_view!(MatrixView<'_, T, A, B, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(MatrixView<'_, T, A, B, M, P>, &VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixView<'_, T, A, B, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixView<'_, T, A, B, M, P>, &VectorViewMut<'_, T, N, P>);

impl_matmul_view!(MatrixView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(MatrixView<'_, T, A, B, M, N>, &Matrix<T, N, P>);
impl_matmul_view!(&MatrixView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(&MatrixView<'_, T, A, B, M, N>, &Matrix<T, N, P>);

impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixView<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixView<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);

/////////////////////
//  MatrixViewMut  //
/////////////////////

impl_vecmul_mat_view!(MatrixViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_vecmul_mat_view!(MatrixViewMut<'_, T, A, B, M, N>, &Vector<T, N>);
impl_vecmul_mat_view!(&MatrixViewMut<'_, T, A, B, M, N>, Vector<T, N>);
impl_vecmul_mat_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Vector<T, N>);

impl_vecmul_view_view!(MatrixViewMut<'_, T, A, B, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_view_view!(MatrixViewMut<'_, T, A, B, M, P>, &VectorView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixViewMut<'_, T, A, B, M, P>, VectorView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixViewMut<'_, T, A, B, M, P>, &VectorView<'_, T, N, P>);

impl_vecmul_view_view!(MatrixViewMut<'_, T, A, B, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(MatrixViewMut<'_, T, A, B, M, P>, &VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixViewMut<'_, T, A, B, M, P>, VectorViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixViewMut<'_, T, A, B, M, P>, &VectorViewMut<'_, T, N, P>);

impl_matmul_view!(MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, N, P>);
impl_matmul_view!(&MatrixViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(&MatrixViewMut<'_, T, A, B, M, N>, &Matrix<T, N, P>);

impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);

///////////////////////////
//  MatrixTransposeView  //
///////////////////////////

impl_vecmul_view_view!(MatrixTransposeView<'_, T, A, B, M, P>, VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(MatrixTransposeView<'_, T, A, B, M, P>, &VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, P>, VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, P>, &VectorTransposeView<'_, T, N, P>);

impl_vecmul_view_view!(MatrixTransposeView<'_, T, A, B, M, P>, VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(MatrixTransposeView<'_, T, A, B, M, P>, &VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, P>, VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, P>, &VectorTransposeViewMut<'_, T, N, P>);

impl_matmul_view!(MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, N, P>);
impl_matmul_view!(&MatrixTransposeView<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &Matrix<T, N, P>);

impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeView<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeView<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);

//////////////////////////////
//  MatrixTransposeViewMut  //
//////////////////////////////

impl_vecmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, P>, VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, P>, &VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, P>, VectorTransposeView<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, P>, &VectorTransposeView<'_, T, N, P>);

impl_vecmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, P>, VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, P>, &VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, P>, VectorTransposeViewMut<'_, T, N, P>);
impl_vecmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, P>, &VectorTransposeViewMut<'_, T, N, P>);

impl_matmul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, N, P>);
impl_matmul_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, Matrix<T, N, P>);
impl_matmul_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &Matrix<T, N, P>);

impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixViewMut<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeView<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeView<'_, T, C, D, N, P>);

impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, MatrixTransposeViewMut<'_, T, C, D, N, P>);
impl_matmul_view_view!(&MatrixTransposeViewMut<'_, T, A, B, M, N>, &MatrixTransposeViewMut<'_, T, C, D, N, P>);

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        // Matrix * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = m1.matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = (&m1).matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let result = (&m1).matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(&view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&m1).matmul(view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&m1).matmul(&view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = m1.matmul(&view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&m1).matmul(view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&m1).matmul(&view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view = m2.t();
        let result = m1.matmul(t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // Matrix * &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view = m2.t();
        let result = m1.matmul(&t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut = m2.t_mut();
        let result = (&m1).matmul(t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &Matrix * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut = m2.t_mut();
        let result = (&m1).matmul(&t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );
    }

    #[test]
    fn test_matrix_view_multiplication() {
        // MatrixView * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view.matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = view.matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = (&view).matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * &Matrix
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view = m1.view::<3, 2>((0, 0)).unwrap();
        let result = (&view).matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&view1).matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * &MatrixView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&view1).matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view1.matmul(&view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&view1).matmul(view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * &MatrixViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let view_mut = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&view1).matmul(&view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view1.matmul(t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixView * &MatrixTransposeView
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view1.matmul(&t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view_mut = m2.t_mut();
        let result = (&view1).matmul(t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view1 = m1.view::<3, 2>((0, 0)).unwrap();
        let t_view_mut = m2.t_mut();
        let result = (&view1).matmul(&t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );
    }

    #[test]
    fn test_matrix_view_mut_multiplication() {
        // MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut.matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = view_mut.matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = (&view_mut).matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * &Matrix
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let result = (&view_mut).matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view_mut.matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = view_mut.matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&view_mut).matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&view_mut).matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view_mut1.matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = view_mut1.matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&view_mut1).matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let view_mut1 = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&view_mut1).matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view_mut.matmul(t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view = m2.t();
        let result = view_mut.matmul(&t_view);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view_mut = m2.t_mut();
        let result = (&view_mut).matmul(t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let view_mut = m1.view_mut::<3, 2>((0, 0)).unwrap();
        let t_view_mut = m2.t_mut();
        let result = (&view_mut).matmul(&t_view_mut);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );
    }

    #[test]
    fn test_matrix_transpose_view_multiplication() {
        // MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let result = t_view.matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * &Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let result = t_view.matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let result = (&t_view).matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * &Matrix
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let result = (&t_view).matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = t_view.matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * &MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = t_view.matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&t_view).matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * &MatrixView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&t_view).matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = t_view.matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = t_view.matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&t_view).matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeView * &MatrixViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view = m1.t();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&t_view).matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeView * MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = (&t_view1).matmul(&t_view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 3>::new([[5, 11, 17], [11, 25, 39], [17, 39, 61]])
        );

        // MatrixTransposeView * &MatrixTransposeView
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let t_view1 = m1.t();
        let t_view2 = m2.t();
        let result = (&t_view1).matmul(&t_view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 3>::new([[5, 11, 17], [11, 25, 39], [17, 39, 61]])
        );

        // &MatrixTransposeView * MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let t_view1 = m1.t();
        let t_view_mut2 = m2.t_mut();
        let result = (&t_view1).matmul(&t_view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 3>::new([[5, 11, 17], [11, 25, 39], [17, 39, 61]])
        );

        // &MatrixTransposeView * &MatrixTransposeViewMut
        let m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let t_view1 = m1.t();
        let t_view_mut2 = m2.t_mut();
        let result = (&t_view1).matmul(&t_view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 3>::new([[5, 11, 17], [11, 25, 39], [17, 39, 61]])
        );
    }

    #[test]
    fn test_matrix_transpose_view_mut_multiplication() {
        // MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut.matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = t_view_mut.matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * Matrix
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = (&t_view_mut).matmul(m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * &Matrix
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let result = (&t_view_mut).matmul(&m2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = t_view_mut.matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = t_view_mut.matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&t_view_mut).matmul(view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * &MatrixView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view2 = m2.view::<2, 4>((0, 0)).unwrap();
        let result = (&t_view_mut).matmul(&view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = t_view_mut.matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = t_view_mut.matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&t_view_mut).matmul(view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * &MatrixViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 2, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let t_view_mut = m1.t_mut();
        let view_mut2 = m2.view_mut::<2, 4>((0, 0)).unwrap();
        let result = (&t_view_mut).matmul(&view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view2 = m2.t();
        let result = t_view_mut1.matmul(t_view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // MatrixTransposeViewMut * &MatrixTransposeView
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view2 = m2.t();
        let result = t_view_mut1.matmul(&t_view2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = (&t_view_mut1).matmul(t_view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );

        // &MatrixTransposeViewMut * &MatrixTransposeViewMut
        let mut m1 = Matrix::<i32, 2, 3>::new([[1, 3, 5], [2, 4, 6]]);
        let mut m2 = Matrix::<i32, 4, 2>::new([[1, 5], [2, 6], [3, 7], [4, 8]]);
        let t_view_mut1 = m1.t_mut();
        let t_view_mut2 = m2.t_mut();
        let result = (&t_view_mut1).matmul(&t_view_mut2);
        assert_eq!(
            result,
            Matrix::<i32, 3, 4>::new([[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]])
        );
    }
}
