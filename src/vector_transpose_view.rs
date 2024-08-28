use funty::{Floating, Numeric};
use std::ops::Index;

use crate::vector::Vector;
use crate::vector_view::VectorView;

pub struct VectorTransposeView<'a, T: Numeric, const N: usize, const M: usize> {
    data: &'a Vector<T, N>,
    start: usize,
}

impl<'a, T: Numeric, const N: usize, const M: usize> VectorTransposeView<'a, T, N, M> {
    pub(super) fn new(data: &'a Vector<T, N>, start: usize) -> Self {
        Self { data, start }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        M
    }

    pub fn t(&self) -> VectorView<'a, T, N, M> {
        VectorView::new(self.data, self.start)
    }
}

impl<'a, T: Floating, const N: usize, const M: usize> VectorTransposeView<'a, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.t().magnitude()
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> Index<usize> for VectorTransposeView<'a, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<T: Numeric, const N: usize, const M: usize> Index<(usize, usize)> for VectorTransposeView<'_, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.start + index.1]
    }
}
