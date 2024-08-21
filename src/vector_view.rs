use crate::traits::DotProduct;
use crate::vector::Vector;
use conv::prelude::{ConvUtil, ValueFrom};
use funty::{Floating, Integral, Numeric};
use std::collections::HashMap;
use std::fmt;
use std::ops::Index;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorView<'a, T: Numeric, const N: usize> {
    data: &'a [T],
}

impl<'a, T: Numeric, const N: usize> VectorView<'a, T, N> {
    pub(super) fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    pub fn sum(&self) -> T {
        self.data.iter().sum()
    }

    pub fn prod(&self) -> T {
        self.data.iter().product()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.into_iter()
    }

    #[inline]
    pub fn shape(&self) -> usize {
        N
    }

    pub fn copy(&self) -> Vector<T, N> {
        Vector::new(self.data.try_into().unwrap())
    }
}

impl<'a, T: Numeric + ValueFrom<usize>, const N: usize> VectorView<'a, T, N> {
    pub fn mean(&self) -> T {
        self.sum() / N.value_as::<T>().unwrap()
    }

    pub fn var(&self) -> T {
        let mean = self.mean();
        self.data
            .iter()
            .map(|x| (*x - mean) * (*x - mean))
            .sum::<T>()
            / N.value_as::<T>().unwrap()
    }
}

impl<'a, T: Numeric + PartialOrd, const N: usize> VectorView<'a, T, N> {
    pub fn min(&self) -> Option<T> {
        self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    pub fn max(&self) -> Option<T> {
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }
}

impl<'a, T: Integral + Eq, const N: usize> VectorView<'a, T, N> {
    pub fn mode(&self) -> Option<T> {
        self.data
            .iter()
            .fold(HashMap::new(), |mut counts, &x| {
                *counts.entry(x).or_insert(0) += 1;
                counts
            })
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
    }
}

impl<'a, T: Integral + From<u8>, const N: usize> VectorView<'a, T, N> {
    pub fn median(&self) -> Option<T> {
        if N == 0 {
            None
        } else if N == 1 {
            return Some(self.data[0]);
        } else {
            let mut sorted = self.data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if N % 2 == 0 {
                return Some((sorted[N / 2 - 1] + sorted[N / 2]) / T::from(2));
            }
            return Some(sorted[N / 2]);
        }
    }
}

impl<'a, T: Floating, const N: usize> VectorView<'a, T, N> {
    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, T: Numeric, const N: usize> Index<usize> for VectorView<'a, T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for VectorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for &VectorView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T: Numeric + fmt::Display, const N: usize> fmt::Display for VectorView<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorView[")?;
        for (i, item) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", item)?;
        }
        write!(f, "]")
    }
}

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_view_sum() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.sum(), 9);
    }

    #[test]
    fn test_vector_view_prod() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.prod(), 24);
    }

    #[test]
    fn test_vector_view_iter() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        let mut iter = view.iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_vector_view_shape() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.shape(), 3);
    }

    #[test]
    fn test_vector_view_copy() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        let copied = view.copy();
        assert_eq!(copied, Vector::<i32, 3>::new([2, 3, 4]));
    }

    #[test]
    fn test_vector_view_mean() {
        let v_float = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_float = v_float.view::<3>(2).unwrap();
        assert!((view_float.mean() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_var() {
        let v_float = Vector::<f64, 5>::new([1.0, 4.0, 1.0, 1.0, 5.0]);
        let view_float = v_float.view::<3>(1).unwrap();
        assert!((view_float.var() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_min_max() {
        let v = Vector::<i32, 5>::new([3, 1, 4, 1, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.min(), Some(1));
        assert_eq!(view.max(), Some(4));

        let v = Vector::<i32, 5>::new([-3, -1, -4, -1, -5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.min(), Some(-4));
        assert_eq!(view.max(), Some(-1));

        let v = Vector::<f64, 5>::new([3.14, 2.71, 1.41, 0.58, 1.73]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.min(), Some(0.58));
        assert_eq!(view.max(), Some(2.71));

        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<1>(2).unwrap();
        assert_eq!(view.min(), Some(3));
        assert_eq!(view.max(), Some(3));
    }

    #[test]
    fn test_vector_view_mode() {
        let v = Vector::<i32, 7>::new([1, 2, 2, 3, 3, 3, 4]);
        let view = v.view::<5>(1).unwrap();
        assert_eq!(view.mode(), Some(3));
    }

    #[test]
    fn test_vector_view_median() {
        let v = Vector::<i32, 5>::new([1, 7, 5, 3, 9]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view.median(), Some(5));

        let v = Vector::<i32, 6>::new([1, 2, 3, 4, 5, 6]);
        let view = v.view::<4>(1).unwrap();
        assert_eq!(view.median(), Some(3));
    }

    #[test]
    fn test_vector_view_magnitude() {
        let v = Vector::<f64, 5>::new([1.0, 2.0, 2.0, 3.0, 4.0]);
        let view = v.view::<3>(1).unwrap();
        assert!((view.magnitude() - 4.123105625617661).abs() < f64::EPSILON);

        let v = Vector::<f64, 5>::new([-3.0, -4.0, 0.0, 3.0, 4.0]);
        let view = v.view::<3>(0).unwrap();
        assert!((view.magnitude() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_indexing() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(view[0], 2);
        assert_eq!(view[1], 3);
        assert_eq!(view[2], 4);
    }

    #[test]
    fn test_vector_view_into_iter() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();

        // Test non-reference into_iter
        let mut iter = view.into_iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_vector_view_edge_cases() {
        let empty_v = Vector::<i32, 0>::new([]);
        assert!(empty_v.view::<1>(0).is_none());

        let single_v = Vector::<i32, 1>::new([42]);
        assert!(single_v.view::<2>(0).is_none());
        assert_eq!(single_v.view::<1>(0).unwrap().sum(), 42);

        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert!(v.view::<3>(1).is_none());
        assert!(v.view::<0>(0).is_none());
    }

    #[test]
    fn test_vector_view_display() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view = v.view::<3>(1).unwrap();
        assert_eq!(format!("{}", view), "VectorView[2 3 4]");

        let v = Vector::<f64, 4>::new([1.1, 2.2, 3.3, 4.4]);
        let view = v.view::<2>(2).unwrap();
        assert_eq!(format!("{}", view), "VectorView[3.3 4.4]");

        let v = Vector::<i32, 1>::new([42]);
        let view = v.view::<1>(0).unwrap();
        assert_eq!(format!("{}", view), "VectorView[42]");
    }
}
