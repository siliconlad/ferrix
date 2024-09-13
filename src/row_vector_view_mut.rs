use num_traits::Float;
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;

use crate::traits::DotProduct;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;

#[derive(Debug)]
pub struct RowVectorViewMut<'a, V, T, const N: usize, const M: usize> {
    data: &'a mut V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> RowVectorViewMut<'a, V, T, N, M> {
    pub(super) fn new(data: &'a mut V, start: usize) -> Self {
        Self { data, start, _phantom: PhantomData }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        M
    }

    #[inline]
    pub fn rows(&self) -> usize {
        1
    }

    #[inline]
    pub fn cols(&self) -> usize {
        M
    }

    pub fn t(&'a self) -> VectorView<'a, V, T, N, M> {
        VectorView::new(self.data, self.start)
    }

    pub fn t_mut(&'a mut self) -> VectorViewMut<'a, V, T, N, M> {
        VectorViewMut::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize> RowVectorViewMut<'a, V, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize> for RowVectorViewMut<'a, V, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<usize> for RowVectorViewMut<'a, V, T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}

impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)> for RowVectorViewMut<'_, V, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.1]
    }
}

impl<V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<(usize, usize)> for RowVectorViewMut<'_, V, T, N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 != 0 {
            panic!("Index out of bounds");
        }
        &mut self.data[self.start + index.1]
    }
}

