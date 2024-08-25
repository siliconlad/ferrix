use funty::{Floating, Numeric};
use std::ops::{Index, IndexMut};

use crate::vector::Vector;
use crate::traits::DotProduct;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;

pub struct VectorViewMut<'a, T: Numeric, const N: usize, const M: usize> {
    data: &'a mut Vector<T, N>,
    start: usize,
}

impl<'a, T: Numeric, const N: usize, const M: usize> VectorViewMut<'a, T, N, M> {
    pub(super) fn new(data: &'a mut Vector<T, N>, start: usize) -> Self {
        Self { data, start }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    pub fn t(&'a self) -> VectorTransposeView<'a, T, N, M> {
        VectorTransposeView::new(self.data, self.start)
    }

    pub fn t_mut(&'a mut self) -> VectorTransposeViewMut<'a, T, N, M> {
        VectorTransposeViewMut::new(self.data, self.start)
    }
}

impl<'a, T: Floating, const N: usize, const M: usize> VectorViewMut<'a, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> Index<usize> for VectorViewMut<'a, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> IndexMut<usize> for VectorViewMut<'a, T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}
