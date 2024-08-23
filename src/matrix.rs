use crate::matrix_t_view::MatrixTransposeView;
use crate::matrix_t_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::traits::{MatrixRead, MatrixWrite};
use funty::Numeric;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T: Numeric, const R: usize, const C: usize> {
    data: [[T; C]; R],
}
impl<T: Numeric, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new(data: [[T; C]; R]) -> Self {
        Self { data }
    }
}

impl<T: Numeric + From<u8>, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn eye() -> Self {
        let mut data = [[T::from(0); C]; R];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::from(1);
        }
        Self { data }
    }

    pub fn zeros() -> Self {
        Self {
            data: [[T::from(0); C]; R],
        }
    }

    pub fn ones() -> Self {
        Self {
            data: [[T::from(1); C]; R],
        }
    }
}

impl<T: Numeric, const R: usize, const C: usize> Matrix<T, R, C> {
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (R, C)
    }
}

impl<T: Numeric, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn t(&self) -> MatrixTransposeView<T, R, C, C, R> {
        MatrixTransposeView::new(self, (0, 0))
    }

    pub fn t_mut(&mut self) -> MatrixTransposeViewMut<T, R, C, C, R> {
        MatrixTransposeViewMut::new(self, (0, 0))
    }
}

impl<T: Numeric, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn view<const V_R: usize, const V_C: usize>(
        &self,
        start: (usize, usize),
    ) -> MatrixView<T, R, C, V_R, V_C> {
        MatrixView::new(self, start)
    }

    pub fn view_mut<const V_R: usize, const V_C: usize>(
        &mut self,
        start: (usize, usize),
    ) -> MatrixViewMut<T, R, C, V_R, V_C> {
        MatrixViewMut::new(self, start)
    }
}

impl<T: Numeric, const R: usize, const C: usize> MatrixRead<T, R, C> for Matrix<T, R, C> {}
impl<T: Numeric, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T: Numeric, const R: usize, const C: usize> MatrixWrite<T, R, C> for Matrix<T, R, C> {}
impl<T: Numeric, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<T: Numeric + fmt::Display, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for &elem in row {
                write!(f, "{} ", elem)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 0)], 4);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_eye() {
        let eye_matrix = Matrix::<i32, 3, 3>::eye();
        assert_eq!(eye_matrix[(0, 0)], 1);
        assert_eq!(eye_matrix[(1, 1)], 1);
        assert_eq!(eye_matrix[(2, 2)], 1);
        assert_eq!(eye_matrix[(0, 1)], 0);
        assert_eq!(eye_matrix[(1, 0)], 0);
        assert_eq!(eye_matrix[(1, 2)], 0);
        assert_eq!(eye_matrix[(2, 1)], 0);

        let eye_matrix_f64 = Matrix::<f64, 2, 2>::eye();
        assert!((eye_matrix_f64[(0, 0)] - 1.0).abs() < f64::EPSILON);
        assert!((eye_matrix_f64[(1, 1)] - 1.0).abs() < f64::EPSILON);
        assert!(eye_matrix_f64[(0, 1)].abs() < f64::EPSILON);
        assert!(eye_matrix_f64[(1, 0)].abs() < f64::EPSILON);
    }

    #[test]
    fn test_zeros() {
        let zeros_matrix = Matrix::<i32, 2, 3>::zeros();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(zeros_matrix[(i, j)], 0);
            }
        }

        let zeros_matrix_f64 = Matrix::<f64, 3, 2>::zeros();
        for i in 0..3 {
            for j in 0..2 {
                assert!(zeros_matrix_f64[(i, j)].abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_ones() {
        let ones_matrix = Matrix::<i32, 3, 2>::ones();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(ones_matrix[(i, j)], 1);
            }
        }

        let ones_matrix_f64 = Matrix::<f64, 2, 3>::ones();
        for i in 0..2 {
            for j in 0..3 {
                assert!((ones_matrix_f64[(i, j)] - 1.0).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_shape() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(matrix.shape(), (2, 3));

        let matrix = Matrix::<f64, 4, 1>::new([[1.0], [2.0], [3.0], [4.0]]);
        assert_eq!(matrix.shape(), (4, 1));
    }

    #[test]
    fn test_t() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let transposed = matrix.t();

        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed[(0, 0)], 1);
        assert_eq!(transposed[(0, 1)], 4);
        assert_eq!(transposed[(1, 0)], 2);
        assert_eq!(transposed[(1, 1)], 5);
        assert_eq!(transposed[(2, 0)], 3);
        assert_eq!(transposed[(2, 1)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_t_out_of_bounds() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let transposed = matrix.t();
        let _ = transposed[(0, 2)];
    }

    #[test]
    fn test_t_mut() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let mut transposed = matrix.t_mut();

        assert_eq!(transposed.shape(), (3, 2));

        // Modify the transposed view
        transposed[(0, 1)] = 10;
        transposed[(2, 0)] = 20;

        // Check if the original matrix is updated
        assert_eq!(matrix[(1, 0)], 10);
        assert_eq!(matrix[(0, 2)], 20);

        // Verify other elements
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 1)], 5);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_t_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let transposed = matrix.t_mut();
        let _ = transposed[(0, 2)];
    }

    #[test]
    fn test_view() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);

        let view = matrix.view::<2, 2>((1, 1));
        assert_eq!(view.shape(), (2, 2));
        assert_eq!(view[(0, 0)], 6);
        assert_eq!(view[(0, 1)], 7);
        assert_eq!(view[(1, 0)], 10);
        assert_eq!(view[(1, 1)], 11);
    }

    #[test]
    #[should_panic(expected = "View size out of bounds")]
    fn test_view_out_of_bounds() {
        let matrix = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let _ = matrix.view::<2, 2>((1, 1));
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_view_index_out_of_bounds() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = matrix.view::<2, 2>((1, 1));
        let _ = view[(2, 3)];
    }

    #[test]
    fn test_view_mut() {
        let mut matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);

        {
            let mut view = matrix.view_mut::<2, 2>((0, 1));
            assert_eq!(view.shape(), (2, 2));
            view[(0, 0)] = 20;
            view[(1, 1)] = 30;
        }

        assert_eq!(matrix[(0, 1)], 20);
        assert_eq!(matrix[(1, 2)], 30);
        assert_eq!(matrix[(0, 0)], 1); // Unchanged
        assert_eq!(matrix[(2, 3)], 12); // Unchanged
    }

    #[test]
    #[should_panic(expected = "View size out of bounds")]
    fn test_view_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32, 2, 2>::new([[1, 2], [3, 4]]);
        let _ = matrix.view_mut::<2, 2>((1, 1));
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_view_mut_index_out_of_bounds() {
        let matrix = Matrix::<i32, 3, 4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let view = matrix.view::<2, 2>((1, 1));
        let _ = view[(2, 3)];
    }

    #[test]
    fn test_index() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 2)], 3);
        assert_eq!(matrix[(1, 1)], 5);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let _ = matrix[(2, 0)];
    }

    #[test]
    fn test_index_mut() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        matrix[(0, 1)] = 10;
        matrix[(1, 2)] = 20;
        assert_eq!(matrix[(0, 1)], 10);
        assert_eq!(matrix[(1, 2)], 20);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        matrix[(0, 3)] = 10;
    }

    #[test]
    fn test_fmt() {
        let matrix = Matrix::<i32, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let formatted = format!("{}", matrix);
        let expected = "1 2 3 \n4 5 6 \n";
        assert_eq!(formatted, expected);
    }
}
