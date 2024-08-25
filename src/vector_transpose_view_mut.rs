use funty::{Floating, Numeric};
use std::ops::{Index, IndexMut};

use crate::traits::DotProduct;
use crate::vector::Vector;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;

pub struct VectorTransposeViewMut<'a, T: Numeric, const N: usize, const M: usize> {
    data: &'a mut Vector<T, N>,
    start: usize,
}

impl<'a, T: Numeric, const N: usize, const M: usize> VectorTransposeViewMut<'a, T, N, M> {
    pub(super) fn new(data: &'a mut Vector<T, N>, start: usize) -> Self {
        Self { data, start }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    pub fn t(&'a self) -> VectorView<'a, T, N, M> {
        VectorView::new(self.data, self.start)
    }

    pub fn t_mut(&'a mut self) -> VectorViewMut<'a, T, N, M> {
        VectorViewMut::new(self.data, self.start)
    }
}

impl<'a, T: Floating, const N: usize, const M: usize> VectorTransposeViewMut<'a, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> Index<usize> for VectorTransposeViewMut<'a, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, T: Numeric, const N: usize, const M: usize> IndexMut<usize> for VectorTransposeViewMut<'a, T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}

