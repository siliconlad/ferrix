use crate::matrix::Matrix;
use crate::matrix_t_view::MatrixTransposeView;
use crate::traits::MatrixRead;
use funty::Numeric;
use std::ops::Index;

pub struct MatrixView<
    'a,
    T: Numeric,
    const R: usize,
    const C: usize,
    const V_R: usize,
    const V_C: usize,
> {
    data: &'a Matrix<T, R, C>,
    start: (usize, usize),
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixView<'a, T, R, C, V_R, V_C>
{
    pub(super) fn new(data: &'a Matrix<T, R, C>, start: (usize, usize)) -> Self {
        if start.0 + V_R > R || start.1 + V_C > C {
            panic!("View size out of bounds");
        }
        Self { data, start }
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixView<'a, T, R, C, V_R, V_C>
{
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (V_R, V_C)
    }

    pub fn t(&self) -> MatrixTransposeView<'a, T, R, C, V_C, V_R> {
        MatrixTransposeView::new(self.data, (self.start.1, self.start.0))
    }
}

impl<'a, T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
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

impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
    MatrixRead<T, V_R, V_C> for MatrixView<'_, T, R, C, V_R, V_C>
{
}
impl<T: Numeric, const R: usize, const C: usize, const V_R: usize, const V_C: usize>
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixView::<i32, 2, 3, 1, 2>::new(&matrix, (0, 1));
        assert_eq!(view.shape(), (1, 2));
    }

    #[test]
    fn test_t() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixView::<i32, 2, 3, 1, 2>::new(&matrix, (0, 1));
        let transposed = view.t();
        assert_eq!(transposed.shape(), (2, 1));
        assert_eq!(transposed[(0, 0)], 2);
        assert_eq!(transposed[(1, 0)], 3);
    }

    #[test]
    fn test_index() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixView::<i32, 2, 3, 1, 2>::new(&matrix, (0, 1));
        assert_eq!(view[(0, 0)], 2);
        assert_eq!(view[(0, 1)], 3);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let view = MatrixView::<i32, 2, 3, 1, 2>::new(&matrix, (0, 1));
        let _ = view[(1, 0)];
    }
}
