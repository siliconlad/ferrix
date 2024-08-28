use funty::Numeric;
use std::ops::{Index, IndexMut};

use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use crate::matrix_view::MatrixView;
use crate::matrix_view_mut::MatrixViewMut;
use crate::vector_transpose_view::VectorTransposeView;
use crate::vector_transpose_view_mut::VectorTransposeViewMut;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T: Numeric, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Numeric + Default, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    fn default() -> Self {
        Self {
            data: [[T::default(); C]; R],
        }
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
    pub fn new(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    pub fn fill(value: T) -> Self {
        Self {
            data: [[value; C]; R],
        }
    }
}

impl<T: Numeric, const R: usize, const C: usize> Matrix<T, R, C> {
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        R * C
    }
}

impl<T: Numeric> Matrix<T, 1, 1> {
    pub fn into(self) -> T {
        self[(0, 0)]
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
    ) -> Option<MatrixView<T, R, C, V_R, V_C>> {
        if start.0 + V_R > R || start.1 + V_C > C {
            return None;
        }
        Some(MatrixView::new(self, start))
    }

    pub fn view_mut<const V_R: usize, const V_C: usize>(
        &mut self,
        start: (usize, usize),
    ) -> Option<MatrixViewMut<T, R, C, V_R, V_C>> {
        if start.0 + V_R > R || start.1 + V_C > C {
            return None;
        }
        Some(MatrixViewMut::new(self, start))
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<T: Numeric, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / C;
        let col_idx = index % C;
        &self.data[row_idx][col_idx]
    }
}

impl<T: Numeric, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= R * C {
            panic!("Index out of bounds");
        }

        let row_idx = index / C;
        let col_idx = index % C;
        &mut self.data[row_idx][col_idx]
    }
}

impl<T: Numeric, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= R || index.1 >= C {
            panic!("Index out of bounds");
        }
        &self.data[index.0][index.1]
    }
}

impl<T: Numeric, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 >= R || index.1 >= C {
            panic!("Index out of bounds");
        }
        &mut self.data[index.0][index.1]
    }
}

//////////////////////////////////
//  From Trait Implementations  //
//////////////////////////////////

impl<T: Numeric, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C> {
    fn from(data: [[T; C]; R]) -> Self {
        Self { data }
    }
}

impl<T: Numeric, const A: usize, const N: usize> From<VectorTransposeView<'_, T, A, N>> for Matrix<T, 1, N> {
    fn from(view: VectorTransposeView<'_, T, A, N>) -> Self {
        Self {
            data: [std::array::from_fn(|i| view[i]); 1],
        }
    }
}

impl<T: Numeric, const A: usize, const N: usize> From<VectorTransposeViewMut<'_, T, A, N>> for Matrix<T, 1, N> {
    fn from(view: VectorTransposeViewMut<'_, T, A, N>) -> Self {
        Self {
            data: [std::array::from_fn(|i| view[i]); 1],
        }
    }
}

impl<T: Numeric, const A: usize, const B: usize, const R: usize, const C: usize> From<MatrixView<'_, T, A, B, R, C>> for Matrix<T, R, C> {
    fn from(view: MatrixView<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Numeric, const A: usize, const B: usize, const R: usize, const C: usize> From<MatrixViewMut<'_, T, A, B, R, C>> for Matrix<T, R, C> {
    fn from(view: MatrixViewMut<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Numeric, const A: usize, const B: usize, const R: usize, const C: usize> From<MatrixTransposeView<'_, T, A, B, R, C>> for Matrix<T, R, C> {
    fn from(view: MatrixTransposeView<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}

impl<T: Numeric, const A: usize, const B: usize, const R: usize, const C: usize> From<MatrixTransposeViewMut<'_, T, A, B, R, C>> for Matrix<T, R, C> {
    fn from(view: MatrixTransposeViewMut<'_, T, A, B, R, C>) -> Self {
        Self {
            data: std::array::from_fn(|i| std::array::from_fn(|j| view[(i, j)])),
        }
    }
}
