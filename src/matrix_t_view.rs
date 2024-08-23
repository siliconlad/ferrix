use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::traits::MatrixRead;
use funty::Numeric;
use std::ops::Index;

pub struct MatrixTransposeView<
    'a,
    T: Numeric,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a Matrix<T, R, C>,
    start: (usize, usize), // In terms of the transposed matrix
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeView<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > C || start.1 + V_C > R {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeView<'a, T, R, C, V_R, V_C>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    pub fn t(&self) -> MatrixView<T, R, C, V_C, V_R> {
        MatrixView::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixTransposeView<'a, T, R, C, V_R, V_C>
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
    MatrixRead<T, V_R, V_C> for MatrixTransposeView<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    Index<(usize, usize)> for MatrixTransposeView<'_, T, R, C, V_R, V_C>
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if !self.validate_index(index) {
            panic!("Index out of bounds");
        }
        &self.data[self.flip(self.offset(index))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixTransposeView::<i32, 2, 3, 3, 2>::new(&matrix, (0, 0));
        assert_eq!(view.shape(), (3, 2));
    }

    #[test]
    fn test_shape_with_offset() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = MatrixTransposeView::<i32, 3, 4, 3, 2>::new(&matrix, (1, 1));
        assert_eq!(view.shape(), (3, 2));
        assert_eq!(view[(0, 0)], 6);
        assert_eq!(view[(0, 1)], 10);
        assert_eq!(view[(1, 0)], 7);
        assert_eq!(view[(1, 1)], 11);
        assert_eq!(view[(2, 0)], 8);
        assert_eq!(view[(2, 1)], 12);
    }

    #[test]
    fn test_t() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixTransposeView::<i32, 2, 3, 3, 2>::new(&matrix, (0, 0));
        let original = view.t();
        assert_eq!(original.shape(), (2, 3));
        assert_eq!(original[(0, 0)], 1);
        assert_eq!(original[(0, 1)], 2);
        assert_eq!(original[(0, 2)], 3);
        assert_eq!(original[(1, 0)], 4);
        assert_eq!(original[(1, 1)], 5);
        assert_eq!(original[(1, 2)], 6);
    }

    #[test]
    fn test_t_with_offset() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = MatrixTransposeView::<i32, 3, 4, 3, 2>::new(&matrix, (1, 1));
        let original = view.t();
        assert_eq!(original.shape(), (2, 3));
        assert_eq!(original[(0, 0)], 6);
        assert_eq!(original[(0, 1)], 7);
        assert_eq!(original[(0, 2)], 8);
        assert_eq!(original[(1, 0)], 10);
        assert_eq!(original[(1, 1)], 11);
        assert_eq!(original[(1, 2)], 12);
    }

    #[test]
    fn test_index() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixTransposeView::<i32, 2, 3, 3, 2>::new(&matrix, (0, 0));
        assert_eq!(view[(0, 0)], 1);
        assert_eq!(view[(0, 1)], 4);
        assert_eq!(view[(1, 0)], 2);
        assert_eq!(view[(1, 1)], 5);
        assert_eq!(view[(2, 0)], 3);
        assert_eq!(view[(2, 1)], 6);
    }

    #[test]
    fn test_index_with_offset() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = MatrixTransposeView::<i32, 3, 4, 3, 2>::new(&matrix, (1, 1));
        assert_eq!(view[(0, 0)], 6);
        assert_eq!(view[(0, 1)], 10);
        assert_eq!(view[(1, 0)], 7);
        assert_eq!(view[(1, 1)], 11);
        assert_eq!(view[(2, 0)], 8);
        assert_eq!(view[(2, 1)], 12);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixTransposeView::<i32, 2, 3, 3, 2>::new(&matrix, (0, 0));
        let _ = view[(3, 0)];
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds_with_offset() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = MatrixTransposeView::<i32, 3, 4, 3, 2>::new(&matrix, (1, 1));
        let _ = view[(3, 0)];
    }
}
