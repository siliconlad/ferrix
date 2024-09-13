use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use std::ops::{Index, IndexMut};

pub struct MatrixTransposeViewMut<
    'a,
    T,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a mut Matrix<T, R, C>,
    start: (usize, usize), // In terms of the transposed matrix
}

impl<'a, T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeViewMut<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a mut Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > C || start.1 + V_C > R {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeViewMut<'a, T, R, C, V_R, V_C>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        V_R * V_C
    }

    #[inline]
    pub fn rows(&self) -> usize {
        V_R
    }

    #[inline]
    pub fn cols(&self) -> usize {
        V_C
    }

    pub fn t(&'a self) -> MatrixView<'a, T, R, C, V_C, V_R> {
        MatrixView::new(self.data, (self.start.1, self.start.0))
    }

    pub fn t_mut(&'a mut self) -> MatrixViewMut<'a, T, R, C, V_C, V_R> {
        MatrixViewMut::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeViewMut<'a, T, R, C, V_R, V_C>
{
    #[inline]
    fn flip(&self, index: (usize, usize)) -> (usize, usize) {
        (index.1, index.0)
    }

    #[inline]
    fn offset(&self, index: (usize, usize)) -> (usize, usize) {
        (index.0 + self.start.0, index.1 + self.start.1)
    }

    #[inline]
    fn validate_index(&self, index: (usize, usize)) -> bool {
        index.0 < V_R && index.1 < V_C
    }
}

impl<T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    Index<usize> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / V_C;
        let col_idx = index % V_C;
        &self.data[self.flip(self.offset((row_idx, col_idx)))]
    }
}

impl<T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    IndexMut<usize> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / V_C;
        let col_idx = index % V_C;
        let index = self.flip(self.offset((row_idx, col_idx)));
        &mut self.data[index]
    }
}

impl<T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    Index<(usize, usize)> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.flip(self.offset(index))]
    }
}

impl<T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    IndexMut<(usize, usize)> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        let index = self.flip(self.offset(index));
        &mut self.data[index]
    }
}
