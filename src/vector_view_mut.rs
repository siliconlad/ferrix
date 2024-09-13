use num_traits::Float;
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;

use crate::traits::DotProduct;
use crate::row_vector_view::RowVectorView;
use crate::row_vector_view_mut::RowVectorViewMut;

pub struct VectorViewMut<'a, V, T, const N: usize, const M: usize> {
    data: &'a mut V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> VectorViewMut<'a, V, T, N, M> {
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
        M
    }

    #[inline]
    pub fn cols(&self) -> usize {
        1
    }

    pub fn t(&'a self) -> RowVectorView<'a, V, T, N, M> {
        RowVectorView::new(self.data, self.start)
    }

    pub fn t_mut(&'a mut self) -> RowVectorViewMut<'a, V, T, N, M> {
        RowVectorViewMut::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize> VectorViewMut<'a, V, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize> for VectorViewMut<'a, V, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<usize> for VectorViewMut<'a, V, T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}

impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)> for VectorViewMut<'_, V, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.0]
    }
}

impl<V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<(usize, usize)> for VectorViewMut<'_, V, T, N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &mut self.data[self.start + index.0]
    }
}
