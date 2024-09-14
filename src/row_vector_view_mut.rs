use num_traits::Float;
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;
use std::fmt;
use crate::traits::DotProduct;
use crate::row_vector::RowVector;
use crate::row_vector_view::RowVectorView;
use crate::vector_view::VectorView;
use crate::vector_view_mut::VectorViewMut;

#[derive(Debug)]
pub struct RowVectorViewMut<'a, V, T, const N: usize, const M: usize> {
    data: &'a mut V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> RowVectorViewMut<'a, V, T, N, M> {
    pub(super) fn new(data: &'a mut V, start: usize) -> Self {
        Self { data, start, _phantom: PhantomData }
    }

    #[inline]
    pub fn shape(&self) -> usize {
        M
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        M
    }

    #[inline]
    pub fn rows(&self) -> usize {
        1
    }

    #[inline]
    pub fn cols(&self) -> usize {
        M
    }

    pub fn t(&'a self) -> VectorView<'a, V, T, N, M> {
        VectorView::new(self.data, self.start)
    }

    pub fn t_mut(&'a mut self) -> VectorViewMut<'a, V, T, N, M> {
        VectorViewMut::new(self.data, self.start)
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// RowVectorViewMut == RowVector
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize> PartialEq<RowVector<T, N>> for RowVectorViewMut<'_, V, T, A, N> {
    fn eq(&self, other: &RowVector<T, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// RowVectorViewMut == RowVectorView
impl<T: PartialEq, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A1: usize, const A2: usize, const N: usize> 
PartialEq<RowVectorView<'_, V2, T, A2, N>> for RowVectorViewMut<'_, V1, T, A1, N> {
    fn eq(&self, other: &RowVectorView<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// RowVectorViewMut == RowVectorViewMut
impl<T: PartialEq, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A1: usize, const A2: usize, const N: usize> 
PartialEq<RowVectorViewMut<'_, V2, T, A2, N>> for RowVectorViewMut<'_, V1, T, A1, N> {
    fn eq(&self, other: &RowVectorViewMut<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<'a, V: Index<usize, Output = T>, T: Eq, const N: usize, const M: usize> Eq for RowVectorViewMut<'a, V, T, N, M> {}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T: fmt::Display, const N: usize, const M: usize> fmt::Display for RowVectorViewMut<'a, V, T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "RowVectorViewMut(")?;
        }

        write!(f, "[")?;
        for i in 0..M-1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{}]", self[M-1])?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize> RowVectorViewMut<'a, V, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize> for RowVectorViewMut<'a, V, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}

impl<'a, V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<usize> for RowVectorViewMut<'a, V, T, N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.start + index]
    }
}

impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)> for RowVectorViewMut<'_, V, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.1]
    }
}

impl<V: IndexMut<usize, Output = T>, T, const N: usize, const M: usize> IndexMut<(usize, usize)> for RowVectorViewMut<'_, V, T, N, M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 != 0 {
            panic!("Index out of bounds");
        }
        &mut self.data[self.start + index.1]
    }
}

