use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::traits::{MatrixRead, MatrixWrite};
use funty::Numeric;
use std::ops::{Index, IndexMut};

pub struct MatrixTransposeViewMut<
    'a,
    T: Numeric,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a mut Matrix<T, R, C>,
    start: (usize, usize), // In terms of the transposed matrix
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeViewMut<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a mut Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > C || start.1 + V_C > R {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeViewMut<'a, T, R, C, V_R, V_C>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    pub fn t(&'a self) -> MatrixView<'a, T, R, C, V_C, V_R> {
        MatrixView::new(self.data, (self.start.1, self.start.0))
    }

    pub fn t_mut(&'a mut self) -> MatrixViewMut<'a, T, R, C, V_C, V_R> {
        MatrixViewMut::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
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

impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixRead<T, V_R, V_C> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
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

impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixWrite<T, V_R, V_C> for MatrixTransposeViewMut<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        assert_eq!(view.shape(), (2, 3));
    }

    #[test]
    fn test_t() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        let original = view.t();
        assert_eq!(original.shape(), (3, 2));
        assert_eq!(original[(0, 0)], 1);
        assert_eq!(original[(0, 1)], 2);
        assert_eq!(original[(1, 0)], 3);
        assert_eq!(original[(1, 1)], 4);
        assert_eq!(original[(2, 0)], 5);
        assert_eq!(original[(2, 1)], 6);
    }

    #[test]
    fn test_t_mut() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        let mut original = view.t_mut();
        original[(1, 1)] = 100;
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 100);
        assert_eq!(matrix[(2, 0)], 5);
        assert_eq!(matrix[(2, 1)], 6);
    }

    #[test]
    fn test_index() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        assert_eq!(view[(0, 0)], 1);
        assert_eq!(view[(0, 1)], 3);
        assert_eq!(view[(0, 2)], 5);
        assert_eq!(view[(1, 0)], 2);
        assert_eq!(view[(1, 1)], 4);
        assert_eq!(view[(1, 2)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        let _ = view[(2, 0)];
    }

    #[test]
    fn test_index_mut() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        view[(0, 2)] = 100;
        view[(1, 0)] = 200;
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 200);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
        assert_eq!(matrix[(2, 0)], 100);
        assert_eq!(matrix[(2, 1)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32, 3, 2>::new([[1, 2], [3, 4], [5, 6]]);
        let mut view = MatrixTransposeViewMut::<i32, 3, 2, 2, 3>::new(&mut matrix, (0, 0));
        view[(2, 0)] = 100;
    }

    #[test]
    fn test_index_with_offset() {
        let mut matrix = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let view = MatrixTransposeViewMut::<i32, 4, 3, 2, 3>::new(&mut matrix, (1, 1));
        assert_eq!(view[(0, 0)], 5);
        assert_eq!(view[(0, 1)], 8);
        assert_eq!(view[(0, 2)], 11);
        assert_eq!(view[(1, 0)], 6);
        assert_eq!(view[(1, 1)], 9);
        assert_eq!(view[(1, 2)], 12);
    }

    #[test]
    fn test_index_mut_with_offset() {
        let mut matrix = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let mut view = MatrixTransposeViewMut::<i32, 4, 3, 2, 3>::new(&mut matrix, (1, 1));
        view[(0, 1)] = 100;
        view[(1, 2)] = 200;
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 4);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(1, 2)], 6);
        assert_eq!(matrix[(2, 0)], 7);
        assert_eq!(matrix[(2, 1)], 100);
        assert_eq!(matrix[(2, 2)], 9);
        assert_eq!(matrix[(3, 0)], 10);
        assert_eq!(matrix[(3, 1)], 11);
        assert_eq!(matrix[(3, 2)], 200);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds_with_offset() {
        let mut matrix = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let view = MatrixTransposeViewMut::<i32, 4, 3, 2, 3>::new(&mut matrix, (1, 1));
        let _ = view[(2, 0)];
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_mut_out_of_bounds_with_offset() {
        let mut matrix = Matrix::<i32, 4, 3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let mut view = MatrixTransposeViewMut::<i32, 4, 3, 2, 3>::new(&mut matrix, (1, 1));
        view[(0, 3)] = 100;
    }
}
