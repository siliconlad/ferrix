use crate::matrix::Matrix;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct MatrixViewMut<
    'a,
    T,
    const R: usize,
    const C: usize,
    const VR: usize,
    const VC: usize,
> {
    data: &'a mut Matrix<T, R, C>,
    start: (usize, usize),
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixViewMut<'a, T, R, C, VR, VC>
{
    pub(super) fn new(data: &'a mut Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + VR > R || start.1 + VC > C {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixViewMut<'a, T, R, C, VR, VC>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (VR, VC)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        VR * VC
    }

    #[inline]
    pub fn rows(&self) -> usize {
        VR
    }

    #[inline]
    pub fn cols(&self) -> usize {
        VC
    }

    pub fn t(&'a self) -> MatrixTransposeView<'a, T, R, C, VC, VR> {
        MatrixTransposeView::new(self.data, (self.start.1, self.start.0))
    }

    pub fn t_mut(&'a mut self) -> MatrixTransposeViewMut<'a, T, R, C, VC, VR> {
        MatrixTransposeViewMut::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixViewMut<'a, T, R, C, VR, VC>
{
    #[inline]
    fn offset(&self, index: (usize, usize)) -> (usize, usize) {
        (index.0 + self.start.0, index.1 + self.start.1)
    }

    #[inline]
    fn validate_index(&self, index: (usize, usize)) -> bool {
        index.0 < VR && index.1 < VC
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    Index<usize> for MatrixViewMut<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        &self.data[self.offset((row_idx, col_idx))]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    IndexMut<usize> for MatrixViewMut<'_, T, R, C, VR, VC>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        let offset = self.offset((row_idx, col_idx));
        &mut self.data[offset]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    Index<(usize, usize)> for MatrixViewMut<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.offset(index)]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    IndexMut<(usize, usize)> for MatrixViewMut<'_, T, R, C, VR, VC>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        let offset = self.offset(index);
        &mut self.data[offset]
    }
}
