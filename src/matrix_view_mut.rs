use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::traits::{MatrixRead, MatrixWrite};
use funty::Numeric;
use std::ops::{Index, IndexMut};

pub struct MatrixViewMut<
    'a,
    T: Numeric,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a mut Matrix<T, R, C>,
    start: (usize, usize),
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixViewMut<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a mut Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > R || start.1 + V_C > C {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixViewMut<'a, T, R, C, V_R, V_C>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    pub fn t(&'a self) -> MatrixTransposeView<'a, T, R, C, V_C, V_R> {
        MatrixTransposeView::new(self.data, (self.start.1, self.start.0))
    }

    pub fn t_mut(&'a mut self) -> MatrixTransposeViewMut<'a, T, R, C, V_C, V_R> {
        MatrixTransposeViewMut::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixViewMut<'a, T, R, C, V_R, V_C>
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

impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixRead<T, V_R, V_C> for MatrixViewMut<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    Index<(usize, usize)> for MatrixViewMut<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.offset(index)]
    }
}

impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixWrite<T, V_R, V_C> for MatrixViewMut<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    IndexMut<(usize, usize)> for MatrixViewMut<'_, T, R, C, V_R, V_C>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        let offset = self.offset(index);
        &mut self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixViewMut::<i32, 2, 3, 1, 2>::new(&mut matrix, (0, 1));
        assert_eq!(view.shape(), (1, 2));
    }

    #[test]
    fn test_t() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixViewMut::<i32, 2, 3, 2, 2>::new(&mut matrix, (0, 1));
        let transposed = view.t();
        assert_eq!(transposed.shape(), (2, 2));
        assert_eq!(transposed[(0, 0)], 2);
        assert_eq!(transposed[(0, 1)], 5);
        assert_eq!(transposed[(1, 0)], 3);
        assert_eq!(transposed[(1, 1)], 6);
    }

    #[test]
    fn test_t_mut() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut view = MatrixViewMut::<i32, 2, 3, 2, 2>::new(&mut matrix, (0, 1));
        let mut transposed = view.t_mut();
        transposed[(0, 1)] = 100;
        transposed[(1, 0)] = 200;
        assert_eq!(matrix[(1, 1)], 100);
        assert_eq!(matrix[(0, 2)], 200);
    }

    #[test]
    fn test_index() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixViewMut::<i32, 2, 3, 1, 2>::new(&mut matrix, (0, 1));
        assert_eq!(view[(0, 0)], 2);
        assert_eq!(view[(0, 1)], 3);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixViewMut::<i32, 2, 3, 1, 2>::new(&mut matrix, (0, 1));
        let _ = view[(1, 0)];
    }

    #[test]
    fn test_index_mut() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut view = MatrixViewMut::<i32, 2, 3, 1, 2>::new(&mut matrix, (0, 1));
        view[(0, 0)] = 100;
        view[(0, 1)] = 200;
        assert_eq!(matrix[(0, 1)], 100);
        assert_eq!(matrix[(0, 2)], 200);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut view = MatrixViewMut::<i32, 2, 3, 1, 2>::new(&mut matrix, (0, 1));
        view[(1, 0)] = 100;
    }
}
