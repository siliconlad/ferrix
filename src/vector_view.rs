use num_traits::Float;
use std::ops::Index;
use std::marker::PhantomData;
use std::fmt;
use crate::traits::DotProduct;
use crate::row_vector_view::RowVectorView;
use crate::vector::Vector;
use crate::vector_view_mut::VectorViewMut;

#[derive(Debug, Clone)]
pub struct VectorView<'a, V, T, const N: usize, const M: usize> {
    data: &'a V,
    start: usize,
    _phantom: PhantomData<T>,
}

impl<'a, V, T, const N: usize, const M: usize> VectorView<'a, V, T, N, M> {
    pub(super) fn new(data: &'a V, start: usize) -> Self {
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
        M
    }

    #[inline]
    pub fn cols(&self) -> usize {
        1
    }

    pub fn t(&'a self) -> RowVectorView<'a, V, T, N, M> {
        RowVectorView::new(self.data, self.start)
    }
}

impl<'a, V: Index<usize, Output = T>, T: Float, const N: usize, const M: usize> VectorView<'a, V, T, N, M> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

//////////////////////////////////////
//  Equality Trait Implementations  //
//////////////////////////////////////

// VectorView == Vector
impl<T: PartialEq, const N: usize, V: Index<usize, Output = T>, const A: usize> PartialEq<Vector<T, N>> for VectorView<'_, V, T, A, N> {
    fn eq(&self, other: &Vector<T, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// VectorView == VectorView
impl<T: PartialEq, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A1: usize, const A2: usize, const N: usize> 
PartialEq<VectorView<'_, V2, T, A2, N>> for VectorView<'_, V1, T, A1, N> {
    fn eq(&self, other: &VectorView<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

// VectorView == VectorViewMut
impl<T: PartialEq, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A1: usize, const A2: usize, const N: usize> 
PartialEq<VectorViewMut<'_, V2, T, A2, N>> for VectorView<'_, V1, T, A1, N> {
    fn eq(&self, other: &VectorViewMut<'_, V2, T, A2, N>) -> bool {
        (0..N).all(|i| self[i] == other[i])
    }
}

impl<'a, V: Index<usize, Output = T>, T: Eq, const N: usize, const M: usize> Eq for VectorView<'a, V, T, N, M> {}

/////////////////////////////////////
//  Display Trait Implementations  //
/////////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T: fmt::Display, const N: usize, const M: usize> fmt::Display for VectorView<'a, V, T, N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "VectorView(")?;
        }

        write!(f, "[")?;
        for i in 0..M-1 {
            if i > 0 {
                write!(f, " ")?;
                if f.alternate() {
                    write!(f, "           ")?;
                }
            }
            writeln!(f, "{}", self[i])?;
        }
        if f.alternate() {
            write!(f, "           ")?;
        }

        write!(f, " {}]", self[M-1])?;

        if f.alternate() {
            write!(f, ", dtype={})", std::any::type_name::<T>())?;
        }

        Ok(())
    }
}

///////////////////////////////////
//  Index Trait Implementations  //
///////////////////////////////////

impl<'a, V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<usize> for VectorView<'a, V, T, N, M> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.start + index]
    }
}


impl<V: Index<usize, Output = T>, T, const N: usize, const M: usize> Index<(usize, usize)> for VectorView<'_, V, T, N, M> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.1 != 0 {
            panic!("Index out of bounds");
        }
        &self.data[self.start + index.0]
    }
}
