use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use std::ops::Index;

pub struct MatrixView<
    'a,
    T,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a Matrix<T, R, C>,
    start: (usize, usize),
}

impl<'a, T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixView<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > R || start.1 + V_C > C {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        V_R * V_C
    }

    pub fn t(&self) -> MatrixTransposeView<'a, T, R, C, V_C, V_R> {
        MatrixTransposeView::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixView<'a, T, R, C, V_R, V_C>
{
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
    Index<usize> for MatrixView<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / V_C;
        let col_idx = index % V_C;
        &self.data[self.offset((row_idx, col_idx))]
    }
}

impl<T, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    Index<(usize, usize)> for MatrixView<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.offset(index)]
    }
}
