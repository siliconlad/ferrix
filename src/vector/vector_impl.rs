use conv::prelude::{ConvUtil, ValueFrom};
use funty::{Floating, Integral, Numeric};
use num_traits::Signed;
use std::collections::HashMap;
use std::default::Default;
use std::ops::{Index, IndexMut};

use super::VectorView;
use super::VectorViewMut;
use crate::vector::traits::{DotProduct, VectorOps};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T: Numeric, const N: usize> {
    data: [T; N],
}

impl<T: Numeric + Default, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }

    pub fn fill(value: T) -> Self {
        Self { data: [value; N] }
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

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.into_iter()
    }

    #[inline]
    pub fn shape(&self) -> usize {
        N
    }

    pub fn view<const M: usize>(&self, start: usize) -> Option<VectorView<'_, T, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorView::new(&self.data[start..start + M]))
    }

    pub fn view_mut<const M: usize>(&mut self, start: usize) -> Option<VectorViewMut<'_, T, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorViewMut::new(&mut self.data[start..start + M]))
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Numeric, const N: usize> VectorOps<T, N> for Vector<T, N> {}

impl<T: Numeric + From<u8>, const N: usize> Vector<T, N> {
    pub fn zeros() -> Self {
        Self::fill(T::from(0))
    }

    pub fn ones() -> Self {
        Self::fill(T::from(1))
    }
}

impl<T: Numeric + ValueFrom<usize>, const N: usize> Vector<T, N> {
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

impl<T: Numeric + PartialOrd, const N: usize> Vector<T, N> {
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

impl<T: Integral + Eq, const N: usize> Vector<T, N> {
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

impl<T: Numeric + PartialOrd + From<u8>, const N: usize> Vector<T, N> {
    pub fn median(&self) -> Option<T> {
        if N == 0 {
            None
        } else if N == 1 {
            return Some(self.data[0]);
        } else {
            let mut sorted = self.data;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if N % 2 == 0 {
                return Some((sorted[N / 2 - 1] + sorted[N / 2]) / T::from(2));
            }
            return Some(sorted[N / 2]);
        }
    }
}

impl<T: Integral, const N: usize> Vector<T, N> {
    pub fn pow(&self, n: T) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i].pow(n.as_u32())),
        }
    }
}

impl<T: Floating, const N: usize> Vector<T, N> {
    pub fn powf(&self, n: T) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i].powf(n)),
        }
    }

    pub fn norm(&self) -> Self {
        let mag = self.magnitude();
        Self {
            data: std::array::from_fn(|i| self.data[i] / mag),
        }
    }

    pub fn magnitude(&self) -> T {
        self.dot(self).sqrt()
    }

    pub fn sqrt(&self) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i].sqrt()),
        }
    }
}

impl<T: Numeric + Signed, const N: usize> Vector<T, N> {
    pub fn abs(&self) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i].abs()),
        }
    }
}

impl<T: Numeric, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Numeric, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Numeric, const N: usize> IntoIterator for Vector<T, N> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for &'a Vector<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Numeric, const N: usize> IntoIterator for &'a mut Vector<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn convert<U: Numeric + From<T>>(&self) -> Vector<U, N> {
        Vector::new(std::array::from_fn(|i| U::from(self[i])))
    }
}

impl<T: Numeric, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self::new(data)
    }
}

impl<T: Numeric> From<T> for Vector<T, 1> {
    fn from(data: T) -> Self {
        Self::new([data])
    }
}

/////////////////////////////
//  Unit Tests for Vector  //
/////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn test_default() {
        let v = Vector::<i32, 3>::default();
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);

        let v = Vector::<f64, 3>::default();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_new() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_fill() {
        let v = Vector::<i32, 3>::fill(5);
        assert_eq!(v[0], 5);
        assert_eq!(v[1], 5);
        assert_eq!(v[2], 5);

        let v = Vector::<f64, 3>::fill(5.0);
        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 5.0);
    }

    #[test]
    fn test_sum() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert_eq!(v.sum(), 6);

        let empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.sum(), 0);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert_eq!(v.sum(), 6.0);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.sum(), 0.0);
    }

    #[test]
    fn test_prod() {
        let v = Vector::<i32, 3>::new([2, 3, 4]);
        assert_eq!(v.prod(), 24);

        let empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.prod(), 1);

        let v = Vector::<f64, 3>::new([2.0, 3.0, 4.0]);
        assert_eq!(v.prod(), 24.0);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.prod(), 1.0);
    }

    #[test]
    fn test_iter() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);

        let empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.iter().next(), None);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1.0));
        assert_eq!(iter.next(), Some(&2.0));
        assert_eq!(iter.next(), Some(&3.0));
        assert_eq!(iter.next(), None);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.iter().next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        for x in v.iter_mut() {
            *x += 1;
        }
        assert_eq!(v, Vector::<i32, 3>::new([2, 3, 4]));

        let mut empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.iter_mut().next(), None);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        for x in v.iter_mut() {
            *x += 1.0;
        }
        assert_eq!(v, Vector::<f64, 3>::new([2.0, 3.0, 4.0]));

        let mut empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.iter_mut().next(), None);
    }

    #[test]
    fn test_shape() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert_eq!(v.shape(), 3);

        let empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.shape(), 0);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert_eq!(v.shape(), 3);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.shape(), 0);
    }

    #[test]
    fn test_view() {
        let v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);

        let view = v.view::<3>(0).unwrap();
        assert_eq!(view[0], 1);
        assert_eq!(view[1], 2);
        assert_eq!(view[2], 3);
        assert_eq!(view.shape(), 3);

        let view = v.view::<2>(3).unwrap();
        assert_eq!(view[0], 4);
        assert_eq!(view[1], 5);
        assert_eq!(view.shape(), 2);

        assert!(v.view::<3>(3).is_none());
        assert!(v.view::<6>(0).is_none());

        let empty_v = Vector::<i32, 0>::new([]);
        assert!(empty_v.view::<3>(0).is_none());
        assert!(empty_v.view::<0>(0).is_none());

        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        let view = v.view::<3>(0).unwrap();
        assert_eq!(view[0], 1.0);
        assert_eq!(view[1], 2.0);
        assert_eq!(view[2], 3.0);
        assert_eq!(view.shape(), 3);

        let view = v.view::<2>(3).unwrap();
        assert_eq!(view[0], 4.0);
        assert_eq!(view[1], 5.0);
        assert_eq!(view.shape(), 2);

        assert!(v.view::<3>(3).is_none());
        assert!(v.view::<6>(0).is_none());

        let empty_v = Vector::<f64, 0>::new([]);
        assert!(empty_v.view::<3>(0).is_none());
        assert!(empty_v.view::<0>(0).is_none());
    }

    #[test]
    fn test_view_mut() {
        let mut v = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        {
            let mut view = v.view_mut::<3>(0).unwrap();
            view[0] = 10;
            view[1] = 20;
            view[2] = 30;
        }
        assert_eq!(v, Vector::<i32, 5>::new([10, 20, 30, 4, 5]));

        assert!(v.view_mut::<3>(3).is_none());
        assert!(v.view_mut::<6>(0).is_none());

        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        {
            let mut view = v.view_mut::<3>(0).unwrap();
            view[0] = 10.0;
            view[1] = 20.0;
            view[2] = 30.0;
        }
        assert_eq!(v, Vector::<f64, 5>::new([10.0, 20.0, 30.0, 4.0, 5.0]));

        assert!(v.view_mut::<3>(3).is_none());
        assert!(v.view_mut::<6>(0).is_none());
    }

    #[test]
    fn test_as_slice() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1, 2, 3]);

        let empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.as_slice(), &[]);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);

        let empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.as_slice(), &[]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let slice = v.as_mut_slice();
        slice[0] = 10;
        slice[1] = 20;
        slice[2] = 30;
        assert_eq!(v, Vector::<i32, 3>::new([10, 20, 30]));

        let mut empty_v = Vector::<i32, 0>::new([]);
        assert_eq!(empty_v.as_mut_slice(), &mut []);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let slice = v.as_mut_slice();
        slice[0] = 10.0;
        slice[1] = 20.0;
        slice[2] = 30.0;
        assert_eq!(v, Vector::<f64, 3>::new([10.0, 20.0, 30.0]));

        let mut empty_v = Vector::<f64, 0>::new([]);
        assert_eq!(empty_v.as_mut_slice(), &mut []);
    }

    #[test]
    fn test_zeros() {
        let v = Vector::<i32, 3>::zeros();
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);

        let v = Vector::<f64, 3>::zeros();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_ones() {
        let v = Vector::<i32, 3>::ones();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 1);

        let v = Vector::<f64, 3>::ones();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn test_mean() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert_eq!(v.mean(), 2);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!((v.mean() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_var() {
        let v = Vector::<i32, 2>::new([3, 1]);
        assert_eq!(v.var(), 1);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!((v.var() - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_max() {
        let v = Vector::<i32, 3>::new([1, 3, 2]);
        assert_eq!(v.min(), Some(1));
        assert_eq!(v.max(), Some(3));

        let v = Vector::<f64, 3>::new([1.0, 3.0, 2.0]);
        assert_eq!(v.min(), Some(1.0));
        assert_eq!(v.max(), Some(3.0));
    }

    #[test]
    fn test_mode() {
        let v = Vector::<i32, 5>::new([1, 2, 2, 3, 2]);
        assert_eq!(v.mode(), Some(2));
    }

    #[test]
    fn test_median() {
        let v = Vector::<i32, 3>::new([2, 1, 3]);
        assert_eq!(v.median(), Some(2));

        let v = Vector::<i32, 4>::new([1, 2, 3, 4]);
        assert_eq!(v.median(), Some(2));

        let v = Vector::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
        assert!((v.median().unwrap() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pow() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let result = v.pow(2);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 4);
        assert_eq!(result[2], 9);
    }

    #[test]
    fn test_powf() {
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.powf(2.0);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 4.0).abs() < EPSILON);
        assert!((result[2] - 9.0).abs() < EPSILON);

        let v = Vector::<f64, 3>::new([1.0, 4.0, 9.0]);
        let result = v.powf(0.5);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 2.0).abs() < EPSILON);
        assert!((result[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_norm() {
        let v = Vector::<f64, 3>::new([3.0, 4.0, 0.0]);
        let result = v.norm();
        assert!((result[0] - 0.6).abs() < EPSILON);
        assert!((result[1] - 0.8).abs() < EPSILON);
        assert!((result[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_magnitude() {
        let v = Vector::<f64, 3>::new([3.0, 4.0, 0.0]);
        assert!((v.magnitude() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_sqrt() {
        let v = Vector::<f64, 3>::new([4.0, 9.0, 16.0]);
        let result = v.sqrt();
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 3.0).abs() < EPSILON);
        assert!((result[2] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_abs() {
        let v = Vector::<i32, 3>::new([-1, 2, -3]);
        let result = v.abs();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);
        assert_eq!(result.shape(), 3);

        let v = Vector::<f64, 3>::new([-1.0, 2.0, -3.0]);
        let result = v.abs();
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 2.0).abs() < EPSILON);
        assert!((result[2] - 3.0).abs() < EPSILON);
        assert_eq!(result.shape(), 3);
    }

    #[test]
    fn test_index() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!((v[0] - 1.0).abs() < EPSILON);
        assert!((v[1] - 2.0).abs() < EPSILON);
        assert!((v[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let _ = v[3];
    }

    #[test]
    fn test_index_mut() {
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v[1] = 5;
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 5);
        assert_eq!(v[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v[1] = 5.0;
        assert!((v[0] - 1.0).abs() < EPSILON);
        assert!((v[1] - 5.0).abs() < EPSILON);
        assert!((v[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds() {
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v[3] = 5;
    }

    #[test]
    fn test_into_iter() {
        // Test owned IntoIterator
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut iter = v.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let collected: Vec<f64> = v.into_iter().collect();
        assert!((collected[0] - 1.0).abs() < EPSILON);
        assert!((collected[1] - 2.0).abs() < EPSILON);
        assert!((collected[2] - 3.0).abs() < EPSILON);

        // Test owned mutable IntoIterator
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut iter = (&mut v).into_iter();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), None);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        for x in &mut v {
            *x *= 2.0;
        }
        assert!((v[0] - 2.0).abs() < EPSILON);
        assert!((v[1] - 4.0).abs() < EPSILON);
        assert!((v[2] - 6.0).abs() < EPSILON);

        // Test reference IntoIterator
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let collected: Vec<&i32> = (&v).into_iter().collect();
        assert_eq!(collected, vec![&1, &2, &3]);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let collected: Vec<&f64> = (&v).into_iter().collect();
        assert!((collected[0] - &1.0).abs() < EPSILON);
        assert!((collected[1] - &2.0).abs() < EPSILON);
        assert!((collected[2] - &3.0).abs() < EPSILON);

        // Test mutable reference IntoIterator
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        for x in &mut v {
            *x += 1;
        }
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 3);
        assert_eq!(v[2], 4);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        for x in &mut v {
            *x += 1.0;
        }
        assert!((v[0] - 2.0).abs() < EPSILON);
        assert!((v[1] - 3.0).abs() < EPSILON);
        assert!((v[2] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_convert() {
        let v = Vector::<i32, 3>::new([1, 2, 3]);
        let result: Vector<f64, 3> = v.convert();
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 2.0).abs() < EPSILON);
        assert!((result[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_from_array() {
        let arr = [1, 2, 3];
        let v = Vector::<i32, 3>::from(arr);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_from_scalar() {
        let v = Vector::<i32, 1>::from(5);
        assert_eq!(v[0], 5);
    }
}
