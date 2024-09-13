use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use std::ops::Index;

#[derive(Debug)]
pub struct MatrixTransposeView<
    'a,
    T,
    const R: usize,
    const C: usize,
    const VR: usize,
    const VC: usize,
> {
    data: &'a Matrix<T, R, C>,
    start: (usize, usize), // In terms of the transposed matrix
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeView<'a, T, R, C, VR, VC>
{
    pub(super) fn new(data: &'a Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + VR > C || start.1 + VC > R {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeView<'a, T, R, C, VR, VC>
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

    pub fn t(&self) -> MatrixView<T, R, C, VC, VR> {
        MatrixView::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    MatrixTransposeView<'a, T, R, C, VR, VC>
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
        index.0 < VR && index.1 < VC
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    Index<usize> for MatrixTransposeView<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / VC;
        let col_idx = index % VC;
        &self.data[self.flip(self.offset((row_idx, col_idx)))]
    }
}

impl<T, const R: usize, const C: usize, const VR: usize, const VC: usize>
    Index<(usize, usize)> for MatrixTransposeView<'_, T, R, C, VR, VC>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.flip(self.offset(index))]
    }
}
