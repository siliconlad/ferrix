use crate::vector::traits::DotProduct;
use crate::vector::vector_impl::Vector;
use conv::prelude::{ConvUtil, ValueFrom};
use funty::{Floating, Integral, Numeric};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Debug, PartialEq, Eq)]
pub struct VectorViewMut<'a, T: Numeric, const N: usize> {
    data: &'a mut [T],
}

impl<'a, T: Numeric, const N: usize> VectorViewMut<'a, T, N> {
    pub(super) fn new(data: &'a mut [T]) -> Self {
        Self { data }
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
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

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn shape(&self) -> usize {
        N
    }

    pub fn copy(&self) -> Vector<T, N> {
        Vector::new(self.data.to_vec().try_into().unwrap())
    }
}

impl<'a, T: Numeric + ValueFrom<usize>, const N: usize> VectorViewMut<'a, T, N> {
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

impl<'a, T: Numeric + Ord, const N: usize> VectorViewMut<'a, T, N> {
    pub fn min(&self) -> Option<T> {
        self.data.iter().min().copied()
    }

    pub fn max(&self) -> Option<T> {
        self.data.iter().max().copied()
    }
}

impl<'a, T: Integral + Eq, const N: usize> VectorViewMut<'a, T, N> {
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

impl<'a, T: Integral + From<u8>, const N: usize> VectorViewMut<'a, T, N> {
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

impl<'a, T: Integral, const N: usize> VectorViewMut<'a, T, N> {
    pub fn pow(&mut self, n: T) {
        self.data.iter_mut().for_each(|x| *x = x.pow(n.as_u32()));
    }
}

impl<'a, T: Floating, const N: usize> VectorViewMut<'a, T, N> {
    pub fn powf(&mut self, n: T) {
        self.data.iter_mut().for_each(|x| *x = x.powf(n));
    }

    pub fn norm(&mut self) {
        let mag = self.magnitude();
        self.data.iter_mut().for_each(|x| *x /= mag);
    }

    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }
}

impl<'a, T: Numeric, const N: usize> Index<usize> for VectorViewMut<'a, T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: Numeric, const N: usize> IndexMut<usize> for VectorViewMut<'a, T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for VectorViewMut<'a, T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for &'a VectorViewMut<'a, T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for &'a mut VectorViewMut<'a, T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T: Numeric + fmt::Display, const N: usize> fmt::Display for VectorViewMut<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorViewMut{{")?;
        for (i, item) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}", item)?;
        }
        write!(f, "}}")
    }
}

//////////////////
//  Unit Tests  //
//////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_view_mut_fill() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            view_mut.fill(10);
        }
        assert_eq!(v, Vector::<i32, 5>::new([1, 10, 10, 10, 5]));
    }

    #[test]
    fn test_vector_view_mut_sum() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut.sum(), 9);
    }

    #[test]
    fn test_vector_view_mut_prod() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut.prod(), 24);
    }

    #[test]
    fn test_vector_view_mut_iter() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let mut iter = view_mut.iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_vector_view_mut_iter_mut() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            for elem in view_mut.iter_mut() {
                *elem *= 2;
            }
        }
        assert_eq!(v, Vector::<i32, 5>::new([1, 4, 6, 8, 5]));

        let mut view_mut = v.view_mut::<3>(1).unwrap();
        let mut iter_mut = view_mut.iter_mut();
        assert_eq!(iter_mut.next(), Some(&mut 4));
        assert_eq!(iter_mut.next(), Some(&mut 6));
        assert_eq!(iter_mut.next(), Some(&mut 8));
        assert_eq!(iter_mut.next(), None);
    }

    #[test]
    fn test_vector_view_mut_shape() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut.shape(), 3);
    }

    #[test]
    fn test_vector_view_mut_copy() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        let copied = view_mut.copy();
        assert_eq!(copied, Vector::<i32, 3>::new([2, 3, 4]));
    }

    #[test]
    fn test_vector_view_mut_mean() {
        let mut v_float = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_mut_float = v_float.view_mut::<3>(1).unwrap();
        assert!((view_mut_float.mean() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_var() {
        let mut v_float = Vector::<f64, 5>::new([1.0, 4.0, 1.0, 1.0, 5.0]);
        let view_mut_float = v_float.view_mut::<3>(1).unwrap();
        assert!((view_mut_float.var() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_min_max() {
        let mut v = Vector::<i32, 5>::new([3, 1, 4, 1, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut.min(), Some(1));
        assert_eq!(view_mut.max(), Some(4));
    }

    #[test]
    fn test_vector_view_mut_mode() {
        let mut v = Vector::<i32, 7>::new([1, 2, 2, 3, 3, 3, 4]);
        let view_mut = v.view_mut::<5>(1).unwrap();
        assert_eq!(view_mut.mode(), Some(3));
    }

    #[test]
    fn test_vector_view_mut_median() {
        let mut v = Vector::<i32, 5>::new([1, 7, 5, 3, 9]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut.median(), Some(5));
    }

    #[test]
    fn test_vector_view_mut_pow() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        view_mut.pow(2);
        assert_eq!(v, Vector::<i32, 5>::new([1, 4, 9, 16, 5]));
    }

    #[test]
    fn test_vector_view_mut_powf() {
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        view_mut.powf(2.0);
        assert_eq!(v, Vector::<f64, 5>::new([1.0, 4.0, 9.0, 16.0, 5.0]));
    }

    #[test]
    fn test_vector_view_mut_norm() {
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 2.0, 3.0, 4.0]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        view_mut.norm();
        let expected = Vector::<f64, 5>::new([1.0, 0.4851, 0.4851, 0.7276, 4.0]);
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vector_view_mut_magnitude() {
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 2.0, 3.0, 4.0]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert!((view_mut.magnitude() - 4.123105625617661).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_view_mut_indexing() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let mut view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(view_mut[0], 2);
        view_mut[1] = 10;
        assert_eq!(view_mut[1], 10);
        assert_eq!(v[2], 10);
    }

    #[test]
    fn test_vector_view_mut_index_mut() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            *view_mut.index_mut(0) = 10;
            *view_mut.index_mut(1) = 20;
            *view_mut.index_mut(2) = 30;
        }
        assert_eq!(v, Vector::<i32, 5>::new([1, 10, 20, 30, 5]));
    }

    #[test]
    fn test_vector_view_mut_into_iter() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);

        // Test VectorViewMut into_iter
        {
            let view_mut = v.view_mut::<3>(1).unwrap();
            let mut iter = view_mut.into_iter();
            assert_eq!(iter.next(), Some(&2));
            assert_eq!(iter.next(), Some(&3));
            assert_eq!(iter.next(), Some(&4));
            assert_eq!(iter.next(), None);
        }

        // Test &VectorViewMut into_iter
        {
            let view_mut = v.view_mut::<3>(1).unwrap();
            let mut iter = (&view_mut).into_iter();
            assert_eq!(iter.next(), Some(&2));
            assert_eq!(iter.next(), Some(&3));
            assert_eq!(iter.next(), Some(&4));
            assert_eq!(iter.next(), None);
        }

        // Test &mut VectorViewMut into_iter
        {
            let mut view_mut = v.view_mut::<3>(1).unwrap();
            let mut iter = (&mut view_mut).into_iter();
            assert_eq!(iter.next(), Some(&mut 2));
            assert_eq!(iter.next(), Some(&mut 3));
            assert_eq!(iter.next(), Some(&mut 4));
            assert_eq!(iter.next(), None);
        }
    }

    #[test]
    fn test_vector_view_mut_display() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let view_mut = v.view_mut::<3>(1).unwrap();
        assert_eq!(format!("{}", view_mut), "VectorViewMut{2; 3; 4}");

        let mut v = Vector::<f64, 4>::new([1.1, 2.2, 3.3, 4.4]);
        let view_mut = v.view_mut::<2>(2).unwrap();
        assert_eq!(format!("{}", view_mut), "VectorViewMut{3.3; 4.4}");

        let mut v = Vector::<i32, 1>::new([42]);
        let view_mut = v.view_mut::<1>(0).unwrap();
        assert_eq!(format!("{}", view_mut), "VectorViewMut{42}");
    }
}
