use crate::matrix::Matrix;
use crate::matrix_view::MatrixView;
use crate::matrix_transpose_view::MatrixTransposeView;
use crate::matrix_transpose_view_mut::MatrixTransposeViewMut;
use std::ops::{Index, IndexMut};
use std::fmt;

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

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// MatrixViewMut == Matrix
impl<'a, T: PartialEq, const R: usize, const C: usize, const VR: usize, const VC: usize> PartialEq<Matrix<T, VR, VC>> for MatrixViewMut<'a, T, R, C, VR, VC> {
    fn eq(&self, other: &Matrix<T, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixViewMut == MatrixView
impl<'a, T: PartialEq, const A: usize, const B: usize, const R: usize, const C: usize, const VR: usize, const VC: usize> PartialEq<MatrixView<'a, T, A, B, VR, VC>> for MatrixViewMut<'a, T, R, C, VR, VC> {
    fn eq(&self, other: &MatrixView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixViewMut == MatrixViewMut
impl<'a, T: PartialEq, const A: usize, const B: usize, const R: usize, const C: usize, const VR: usize, const VC: usize> PartialEq<MatrixViewMut<'a, T, A, B, VR, VC>> for MatrixViewMut<'a, T, R, C, VR, VC> {
    fn eq(&self, other: &MatrixViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixViewMut == MatrixTransposeView
impl<'a, T: PartialEq, const A: usize, const B: usize, const R: usize, const C: usize, const VR: usize, const VC: usize> PartialEq<MatrixTransposeView<'a, T, A, B, VR, VC>> for MatrixViewMut<'a, T, R, C, VR, VC> {
    fn eq(&self, other: &MatrixTransposeView<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

// MatrixViewMut == MatrixTransposeViewMut
impl<'a, T: PartialEq, const A: usize, const B: usize, const R: usize, const C: usize, const VR: usize, const VC: usize> PartialEq<MatrixTransposeViewMut<'a, T, A, B, VR, VC>> for MatrixViewMut<'a, T, R, C, VR, VC> {
    fn eq(&self, other: &MatrixTransposeViewMut<'a, T, A, B, VR, VC>) -> bool {
        (0..VR).all(|i| (0..VC).all(|j| self[(i, j)] == other[(i, j)]))
    }
}

impl<'a, T: Eq, const R: usize, const C: usize, const VR: usize, const VC: usize> Eq for MatrixViewMut<'a, T, R, C, VR, VC> {}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<T: fmt::Display, const R: usize, const C: usize, const VR: usize, const VC: usize>
    fmt::Display for MatrixViewMut<'_, T, R, C, VR, VC>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "MatrixViewMut(")?;
        }

        write!(f, "[")?;
        for i in 0..VR {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "              ")?;
                }
            }
            write!(f, "[")?;
            for j in 0..VC {
                write!(f, "{}", self[(i, j)])?;
                if j < VC - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i < VR - 1 {
                writeln!(f)?;
            }
        }
        write!(f, "]")?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

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
