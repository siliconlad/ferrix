use funty::Floating;
use std::ops::Index;
use std::marker::PhantomData;

use crate::traits::DotProduct;
use crate::row_vector_view::RowVectorView;

pub struct VectorView<'a, V, T, const N: usize, const M: usize> {
    data: &'a V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> VectorView<'a, V, T, N, M> {
    pub(super) fn new(data: &'a V, start: usize) -> Self {
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

    pub fn t(&'a self) -> RowVectorView<'a, V, T, N, M> {
        RowVectorView::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Floating, const N: usize, const M: usize> VectorView<'a, V, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize> for VectorView<'a, V, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}


impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)> for VectorView<'_, V, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.0]
    }
}
