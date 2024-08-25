use funty::{Floating, Numeric};
use std::default::Default;
use std::ops::{Index, IndexMut};

use crate::traits::DotProduct;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;
use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T: Numeric, const N: usize> {
    data: [T; N]
}

impl<T: Numeric + Default, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T: Numeric + From<u8>, const N: usize> Vector<T, N> {
    pub fn zeros() -> Self {
        Self::fill(T::from(0))
    }

    pub fn ones() -> Self {
        Self::fill(T::from(1))
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }

    pub fn fill(value: T) -> Self {
        Self { data: [value; N] }
    }
}

impl<T: Numeric + From<u8>, const N: usize> Vector<T, N> {
    pub fn diag(&self) -> Matrix<T, N, N> {
        let mut m = Matrix::<T, N, N>::zeros();
        for i in 0..N {
            m[(i, i)] = self[i];
        }
        m
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    #[inline]
    pub fn shape(&self) -> usize {
        N
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn t(&self) -> VectorTransposeView<'_, T, N, N> {
        VectorTransposeView::new(self, 0)
    }

    pub fn t_mut(&mut self) -> VectorTransposeViewMut<'_, T, N, N> {
        VectorTransposeViewMut::new(self, 0)
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn view<const M: usize>(&self, start: usize) -> Option<VectorView<'_, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorView::new(self, start))
    }

    pub fn view_mut<const M: usize>(&mut self, start: usize) -> Option<VectorViewMut<'_, T, N, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorViewMut::new(self, start))
    }
}

impl<T: Floating, const N: usize> Vector<T, N> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<T: Numeric, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Numeric, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

//////////////////////////////////
//  From Trait Implementations  //
//////////////////////////////////

impl<T: Numeric, const A: usize, const N: usize> From<VectorView<'_, T, A, N>> for Vector<T, N> {
    fn from(vector: VectorView<'_, T, A, N>) -> Self {
        Vector::new(std::array::from_fn(|i| vector[i]))
    }
}

impl<T: Numeric, const A: usize, const N: usize> From<VectorViewMut<'_, T, A, N>> for Vector<T, N> {
    fn from(vector: VectorViewMut<'_, T, A, N>) -> Self {
        Vector::new(std::array::from_fn(|i| vector[i]))
    }
}

impl<T: Numeric, const N: usize> From<Matrix<T, N, 1>> for Vector<T, N> {
    fn from(matrix: Matrix<T, N, 1>) -> Self {
        Vector::new(std::array::from_fn(|i| matrix[(i, 0)]))
    }
}

impl<T: Numeric, const A: usize, const B: usize, const N: usize> From<MatrixView<'_, T, A, B, N, 1>> for Vector<T, N> {
    fn from(matrix: MatrixView<'_, T, A, B, N, 1>) -> Self {
        Vector::new(std::array::from_fn(|i| matrix[(i, 0)]))
    }
}

impl<T: Numeric, const A: usize, const B: usize, const N: usize> From<MatrixViewMut<'_, T, A, B, N, 1>> for Vector<T, N> {
    fn from(matrix: MatrixViewMut<'_, T, A, B, N, 1>) -> Self {
        Vector::new(std::array::from_fn(|i| matrix[(i, 0)]))
    }
}

impl<T: Numeric, const A: usize, const B: usize, const N: usize> From<MatrixTransposeView<'_, T, A, B, N, 1>> for Vector<T, N> {
    fn from(matrix: MatrixTransposeView<'_, T, A, B, N, 1>) -> Self {
        Vector::new(std::array::from_fn(|i| matrix[(i, 0)]))
    }
}

impl<T: Numeric, const A: usize, const B: usize, const N: usize> From<MatrixTransposeViewMut<'_, T, A, B, N, 1>> for Vector<T, N> {
    fn from(matrix: MatrixTransposeViewMut<'_, T, A, B, N, 1>) -> Self {
        Vector::new(std::array::from_fn(|i| matrix[(i, 0)]))
    }
}
