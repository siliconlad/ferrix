use funty::{Floating, Numeric};
use std::ops::Index;

use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_transpose_view::VectorTransposeView;

pub struct VectorView<'a, T: Numeric, const N: usize, const M: usize> {
    data: &'a Vector<T, N>,
    start: usize,
}

impl<'a, T: Numeric, const N: usize, const M: usize> VectorView<'a, T, N, M> {
    pub(super) fn new(data: &'a Vector<T, N>, start: usize) -> Self {
        Self { data, start }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    pub fn t(&'a self) -> VectorTransposeView<'a, T, N, M> {
        VectorTransposeView::new(self.data, self.start)
    }
}

impl<'a, T: Floating, const N: usize, const M: usize> VectorView<'a, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> Index<usize> for VectorView<'a, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}
