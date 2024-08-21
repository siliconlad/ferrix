use conv::prelude::{ConvUtil, ValueFrom};
use funty::{Floating, Integral, Numeric};
use num_traits::Signed;
use std::collections::HashMap;
use std::default::Default;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub trait VectorOps<T: Numeric, const N: usize>: Index<usize, Output = T> {}

pub trait DotProduct<V> {
    type Output;
    fn dot(self, other: V) -> Self::Output;
}

fn dot<T: Numeric, const N: usize>(v1: &dyn VectorOps<T, N>, v2: &dyn VectorOps<T, N>) -> T {
    (0..N).map(|i| v1[i] * v2[i]).sum()
}

fn v_add<T: Numeric, const N: usize>(
    lhs: &dyn VectorOps<T, N>,
    rhs: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| lhs[i] + rhs[i]))
}

fn v_add_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] + scalar))
}

fn v_sub<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] - v2[i]))
}

fn v_sub_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] - scalar))
}

fn v_mul<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] * v2[i]))
}

fn v_mul_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] * scalar))
}

fn v_div<T: Numeric, const N: usize>(
    v1: &dyn VectorOps<T, N>,
    v2: &dyn VectorOps<T, N>,
) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v1[i] / v2[i]))
}

fn v_div_scalar<T: Numeric, const N: usize>(v: &dyn VectorOps<T, N>, scalar: T) -> Vector<T, N> {
    Vector::<T, N>::new(std::array::from_fn(|i| v[i] / scalar))
}

//////////////
//  Vector  //
//////////////

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
        Some(VectorView {
            data: &self.data[start..start + M],
        })
    }

    pub fn view_mut<const M: usize>(&mut self, start: usize) -> Option<VectorViewMut<'_, T, M>> {
        if start + M > N || M == 0 {
            return None;
        }
        Some(VectorViewMut {
            data: &mut self.data[start..start + M],
        })
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

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> Add<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<&Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<T: Numeric, const N: usize> Add<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<T: Numeric, const N: usize> Add<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        v_add_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Add<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        v_add_scalar(self, scalar)
    }
}

impl<T: Numeric, const N: usize> AddAssign<Vector<T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&Vector<T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn add_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<T: Numeric, const N: usize> AddAssign<T> for Vector<T, N> {
    fn add_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] += scalar);
    }
}

impl<T: Numeric, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<T: Numeric, const N: usize> Sub<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Sub<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(self, scalar)
    }
}

impl<T: Numeric, const N: usize> SubAssign<Vector<T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<&Vector<T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<T: Numeric, const N: usize> SubAssign<T> for Vector<T, N> {
    fn sub_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] -= scalar);
    }
}

impl<T: Numeric, const N: usize> Mul<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<T: Numeric, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Mul<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(self, scalar)
    }
}

impl<T: Numeric, const N: usize> MulAssign<Vector<T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&Vector<T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn mul_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<T: Numeric, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] *= scalar);
    }
}

impl<T: Numeric, const N: usize> Div<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<T: Numeric, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(&self, scalar)
    }
}

impl<T: Numeric, const N: usize> Div<T> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(self, scalar)
    }
}

impl<T: Numeric, const N: usize> DivAssign<Vector<T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<&Vector<T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<VectorView<'_, T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<&VectorView<'_, T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<&VectorViewMut<'_, T, N>> for Vector<T, N> {
    fn div_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<T: Numeric, const N: usize> DivAssign<T> for Vector<T, N> {
    fn div_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] /= scalar);
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

///////////////////
//  VectorView  //
///////////////////

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorView<'a, T: Numeric, const N: usize> {
    data: &'a [T],
}

impl<'a, T: Numeric, const N: usize> VectorView<'a, T, N> {
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

impl<'a, T: Numeric, const N: usize> VectorOps<T, N> for VectorView<'a, T, N> {}

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

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for &VectorView<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Add<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Add<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] + scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorView<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorView<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorView<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorView<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorViewMut<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorViewMut<'b, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<VectorViewMut<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, 'b, T: Numeric, const N: usize> Sub<&VectorViewMut<'b, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'b, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] - scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] * scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / other[i]))
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for &VectorView<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        Vector::<T, N>::new(std::array::from_fn(|i| self[i] / scalar))
    }
}

//////////////////////
//  VectorViewMut  //
//////////////////////

#[derive(Debug, PartialEq, Eq)]
pub struct VectorViewMut<'a, T: Numeric, const N: usize> {
    data: &'a mut [T],
}

impl<'a, T: Numeric, const N: usize> VectorViewMut<'a, T, N> {
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

impl<'a, T: Numeric, const N: usize> VectorOps<T, N> for VectorViewMut<'a, T, N> {}

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

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<Vector<T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: Vector<T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&Vector<T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &Vector<T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorView<'_, T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorView<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorView<'_, T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorView<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(&self, other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<VectorViewMut<'_, T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, &other)
    }
}

impl<T: Numeric, const N: usize> DotProduct<&VectorViewMut<'_, T, N>> for &VectorViewMut<'_, T, N> {
    type Output = T;

    fn dot(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        dot(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: Vector<T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_add(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Add<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        v_add_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Add<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn add(self, scalar: T) -> Self::Output {
        v_add_scalar(self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn add_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn add_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] += other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> AddAssign<T> for VectorViewMut<'a, T, N> {
    fn add_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] += scalar);
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: Vector<T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &Vector<T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_sub(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Sub<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn sub(self, scalar: T) -> Self::Output {
        v_sub_scalar(self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn sub_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn sub_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn sub_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn sub_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn sub_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn sub_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] -= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> SubAssign<T> for VectorViewMut<'a, T, N> {
    fn sub_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] -= scalar);
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_mul(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Mul<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        v_mul_scalar(self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn mul_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn mul_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] *= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> MulAssign<T> for VectorViewMut<'a, T, N> {
    fn mul_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] *= scalar);
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: Vector<T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&Vector<T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &Vector<T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorView<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorView<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorView<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(&self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, &other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<&VectorViewMut<'_, T, N>> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, other: &VectorViewMut<'_, T, N>) -> Self::Output {
        v_div(self, other)
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(&self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> Div<T> for &VectorViewMut<'a, T, N> {
    type Output = Vector<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        v_div_scalar(self, scalar)
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<&Vector<T, N>> for VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<&VectorView<'_, T, N>> for VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: &VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn div_assign(&mut self, other: VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<&VectorViewMut<'_, T, N>>
    for VectorViewMut<'a, T, N>
{
    fn div_assign(&mut self, other: &VectorViewMut<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<T> for VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, scalar: T) {
        (0..N).for_each(|i| self[i] /= scalar);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<Vector<T, N>> for &mut VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<&Vector<T, N>> for &mut VectorViewMut<'a, T, N> {
    fn div_assign(&mut self, other: &Vector<T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
    }
}

impl<'a, T: Numeric, const N: usize> DivAssign<VectorView<'_, T, N>>
    for &mut VectorViewMut<'a, T, N>
{
    fn div_assign(&mut self, other: VectorView<'_, T, N>) {
        (0..N).for_each(|i| self[i] /= other[i]);
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
    fn test_dot() {
        // Vector dot Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // Vector dot &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &Vector dot Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &Vector dot &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // Vector dot VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(view) - 14.0).abs() < f64::EPSILON);

        // Vector dot &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let view = v2.view::<3>(0).unwrap();
        assert!((v1.dot(&view) - 14.0).abs() < f64::EPSILON);

        // &Vector dot VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!(((&v1).dot(view) - 14.0).abs() < f64::EPSILON);

        // &Vector dot &VectorView
        let view = v2.view::<3>(0).unwrap();
        assert!(((&v1).dot(&view) - 14.0).abs() < f64::EPSILON);

        // Vector dot VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // Vector dot &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!((v1.dot(&view_mut) - 14.0).abs() < f64::EPSILON);

        // &Vector dot VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert!(((&v1).dot(view_mut) - 14.0).abs() < f64::EPSILON);

        // &Vector dot &VectorViewMut
        let view_mut = v2.view_mut::<3>(0).unwrap();
        assert!(((&v1).dot(&view_mut) - 14.0).abs() < f64::EPSILON);
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
    fn test_add() {
        // Vector + Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Reference addition for f64
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Mixed reference and non-reference addition for f64
        let result = v1 + &v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v1 + v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Vector + VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &Vector + VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Vector + &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &Vector + &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Vector + VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &Vector + VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Vector + &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &Vector + &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Reference addition for f64 scalar
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v + 0.5;
        assert!((result[0] - 1.5).abs() < EPSILON);
        assert!((result[1] - 2.5).abs() < EPSILON);
        assert!((result[2] - 3.5).abs() < EPSILON);

        let result = v + 0.5;
        assert!((result[0] - 1.5).abs() < EPSILON);
        assert!((result[1] - 2.5).abs() < EPSILON);
        assert!((result[2] - 3.5).abs() < EPSILON);
    }

    #[test]
    fn test_add_assign() {
        // Vector += Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2;
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2;
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 += &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // Vector += Vector (i32)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([1, 2, 3]);
        v1 += v2;
        assert_eq!(v1[0], 2);
        assert_eq!(v1[1], 4);
        assert_eq!(v1[2], 6);

        // Vector += Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v += 2;
        assert_eq!(v[0], 3);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 5);

        // Vector += Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v += 0.5;
        assert!((v[0] - 1.5).abs() < EPSILON);
        assert!((v[1] - 2.5).abs() < EPSILON);
        assert!((v[2] - 3.5).abs() < EPSILON);
    }

    #[test]
    fn test_sub() {
        // Vector - Vector (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - &Vector (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - Vector (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - &Vector (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - VectorView (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - &VectorView (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - VectorView (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - &VectorView (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - VectorViewMut (non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - &VectorViewMut (reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - VectorViewMut (reference and non-reference, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &Vector - &VectorViewMut (both references, f64)
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1 - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // Vector - Scalar (non-reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v - 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.5).abs() < EPSILON);
        assert!((result[2] - 2.5).abs() < EPSILON);

        // &Vector - Scalar (reference, f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v - 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.5).abs() < EPSILON);
        assert!((result[2] - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_sub_assign() {
        // Vector -= Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2;
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= &Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2;
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= &VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v1 -= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // Vector -= Vector (i32)
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([0, 1, 2]);
        v1 -= v2;
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 1);
        assert_eq!(v1[2], 1);

        // Vector -= Scalar
        let mut v = Vector::<i32, 3>::new([5, 6, 7]);
        v -= 2;
        assert_eq!(v[0], 3);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 5);

        // Vector -= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v -= 0.5;
        assert!((v[0] - 0.5).abs() < EPSILON);
        assert!((v[1] - 1.5).abs() < EPSILON);
        assert!((v[2] - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_mul() {
        // Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * &Vector
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * &VectorView
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &Vector * &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1 * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v * 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 1.5).abs() < EPSILON);

        // &Vector * Scalar (f64)
        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v * 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_mul_assign() {
        const EPSILON: f64 = 1e-10;

        // Vector *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        v1 *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // Vector *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        v *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // Vector *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v *= 0.5;
        assert!((v[0] - 0.5).abs() < EPSILON);
        assert!((v[1] - 1.0).abs() < EPSILON);
        assert!((v[2] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_div() {
        const EPSILON: f64 = 1e-10;

        // Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &Vector
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorView
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // Vector / &VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1 / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / VectorViewMut
        let v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1 / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &Vector / &VectorViewMut
        let v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1 / v2;
        assert!((result[0] - 10.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 10.0).abs() < EPSILON);

        // Vector / Scalar
        let v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v / 0.5;
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 4.0).abs() < EPSILON);
        assert!((result[2] - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_div_assign() {
        // Vector /= Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2;
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // Floating-point versions
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2;
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        v1 /= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // Vector /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        v /= 2;
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        v /= 0.5;
        assert!((v[0] - 2.0).abs() < EPSILON);
        assert!((v[1] - 4.0).abs() < EPSILON);
        assert!((v[2] - 6.0).abs() < EPSILON);
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

    //////////////////
    //  VectorView  //
    //////////////////

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
    fn test_vector_view_dot() {
        // VectorView dot Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorView dot &Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &VectorView dot Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &VectorView dot &Vector
        let v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // VectorView dot VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorView dot &VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot &VectorView
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);

        // VectorView dot VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view_mut) - 20.0).abs() < f64::EPSILON);

        // VectorView dot &VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(&view_mut) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(view_mut) - 20.0).abs() < f64::EPSILON);

        // &VectorView dot &VectorViewMut
        let v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view::<3>(0).unwrap();
        let view_mut = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(&view_mut) - 20.0).abs() < f64::EPSILON);
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
    fn test_vector_view_addition() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView + VectorView
        let result = &view1 + &view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + &VectorView
        let result = view1 + &view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test &VectorView + VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 + view2;
        assert_eq!(result, Vector::<i32, 3>::new([6, 6, 6]));

        // Test VectorView + Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test VectorView + &Vector
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + &vec;
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test &VectorView + Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test &VectorView + &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let result = &view1 + &vec;
        assert_eq!(result, Vector::<i32, 3>::new([3, 4, 5]));

        // Test VectorView + VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 + view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test VectorView + &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 + &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test &VectorView + VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 + view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test &VectorView + &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 + &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([22, 33, 44]));

        // Test scalar addition
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 + 10;
        assert_eq!(result, Vector::<i32, 3>::new([12, 13, 14]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 + 10;
        assert_eq!(result, Vector::<i32, 3>::new([12, 13, 14]));
    }

    #[test]
    fn test_vector_view_subtraction() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView - VectorView
        let result = &view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - &VectorView
        let result = view1 - &view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test &VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 - view2;
        assert_eq!(result, Vector::<i32, 3>::new([-2, 0, 2]));

        // Test VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - &Vector
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - Vector::<i32, 3>::new([1, 1, 1]);
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test &VectorView - &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([1, 1, 1]);
        let result = &view1 - &vec;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        // Test VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 - view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test &VectorView - &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([10, 20, 30, 40, 50]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 - &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([-18, -27, -36]));

        // Test scalar subtraction
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 - 1;
        assert_eq!(result, Vector::<i32, 3>::new([1, 2, 3]));
    }

    #[test]
    fn test_vector_view_multiplication() {
        let v1 = Vector::<i32, 5>::new([1, 2, 3, 4, 5]);
        let v2 = Vector::<i32, 5>::new([5, 4, 3, 2, 1]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView * VectorView
        let result = &view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * &VectorView
        let result = view1 * &view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test &VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 * view2;
        assert_eq!(result, Vector::<i32, 3>::new([8, 9, 8]));

        // Test VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &Vector
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * Vector::<i32, 3>::new([2, 2, 2]);
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<i32, 3>::new([2, 2, 2]);
        let result = &view1 * &vec;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 * view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test &VectorView * &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<i32, 5>::new([2, 2, 2, 2, 2]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 * &view_mut;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        // Test scalar multiplication
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 * 2;
        assert_eq!(result, Vector::<i32, 3>::new([4, 6, 8]));
    }

    #[test]
    fn test_vector_view_division() {
        let v1 = Vector::<f64, 5>::new([10.0, 20.0, 30.0, 40.0, 50.0]);
        let v2 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();

        // Test VectorView / VectorView
        let result = &view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorView
        let result = view1 / &view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / VectorView
        let view1 = v1.view::<3>(1).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        let result = view1 / view2;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / &Vector
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / Vector
        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test &VectorView / &Vector
        let view1 = v1.view::<3>(1).unwrap();
        let vec = Vector::<f64, 3>::new([2.0, 4.0, 10.0]);
        let result = &view1 / &vec;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 7.5, 4.0]));

        // Test VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v3 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v3.view_mut::<3>(1).unwrap();
        let result = &view1 / view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test &VectorView / &VectorViewMut
        let view1 = v1.view::<3>(1).unwrap();
        let mut v4 = Vector::<f64, 5>::new([2.0, 4.0, 6.0, 8.0, 10.0]);
        let view_mut = v4.view_mut::<3>(1).unwrap();
        let result = &view1 / &view_mut;
        assert_eq!(result, Vector::<f64, 3>::new([5.0, 5.0, 5.0]));

        // Test scalar division
        let view1 = v1.view::<3>(1).unwrap();
        let result = view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));

        let view1 = v1.view::<3>(1).unwrap();
        let result = &view1 / 2.0;
        assert_eq!(result, Vector::<f64, 3>::new([10.0, 15.0, 20.0]));
    }

    /////////////////////
    //  VectorViewMut  //
    /////////////////////

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
    fn test_vector_view_mut_dot() {
        // VectorViewMut dot Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(v2) - 32.0).abs() < f64::EPSILON);

        // VectorViewMut dot &Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f64::EPSILON);

        // &VectorViewMut dot Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(v2) - 32.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &Vector
        let mut v = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = v.view_mut::<3>(0).unwrap();
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        assert!(((&v1).dot(&v2) - 32.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot &VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &VectorView
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(view2) - 20.0).abs() < f64::EPSILON);

        // VectorViewMut dot &VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!((view1.dot(&view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(view2) - 20.0).abs() < f64::EPSILON);

        // &VectorViewMut dot &VectorViewMut
        let mut v1 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v2 = Vector::<f64, 5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let view1 = v1.view_mut::<3>(0).unwrap();
        let view2 = v2.view_mut::<3>(1).unwrap();
        assert!(((&view1).dot(&view2) - 20.0).abs() < f64::EPSILON);
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
    fn test_vector_view_mut_add() {
        // VectorViewMut + VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Reference addition for f64
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // Mixed reference and non-reference addition for f64
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // VectorViewMut + VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &VectorViewMut + VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // VectorViewMut + &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &VectorViewMut + &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2.view::<3>(0).unwrap();
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // VectorViewMut + Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &VectorViewMut + Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // VectorViewMut + &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() + &v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // &VectorViewMut + &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() + &v2;
        assert!((result[0] - 1.1).abs() < EPSILON);
        assert!((result[1] - 2.2).abs() < EPSILON);
        assert!((result[2] - 3.3).abs() < EPSILON);

        // VectorViewMut + Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() + 0.5;
        assert!((result[0] - 1.5).abs() < EPSILON);
        assert!((result[1] - 2.5).abs() < EPSILON);
        assert!((result[2] - 3.5).abs() < EPSILON);

        // &VectorViewMut + Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v.view_mut::<3>(0).unwrap() + 0.5;
        assert!((result[0] - 1.5).abs() < EPSILON);
        assert!((result[1] - 2.5).abs() < EPSILON);
        assert!((result[2] - 3.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_add_assign() {
        // VectorViewMut += VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // VectorViewMut += &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // VectorViewMut += VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // VectorViewMut += &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 2.0).abs() < EPSILON);
        assert!((v1[1] - 4.0).abs() < EPSILON);
        assert!((v1[2] - 6.0).abs() < EPSILON);

        // VectorViewMut += Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += v2;
        assert!((v1[0] - 1.1).abs() < EPSILON);
        assert!((v1[1] - 2.2).abs() < EPSILON);
        assert!((v1[2] - 3.3).abs() < EPSILON);

        // VectorViewMut += &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view += &v2;
        assert!((v1[0] - 1.1).abs() < EPSILON);
        assert!((v1[1] - 2.2).abs() < EPSILON);
        assert!((v1[2] - 3.3).abs() < EPSILON);

        // VectorViewMut += Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view += 0.5;
        assert!((v[0] - 1.5).abs() < EPSILON);
        assert!((v[1] - 2.5).abs() < EPSILON);
        assert!((v[2] - 3.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_sub() {
        // VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2.view::<3>(0).unwrap();
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - Vector (non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - &Vector (non-reference and reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - Vector (reference and non-reference, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // &VectorViewMut - &Vector (both references, f64)
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() - &v2;
        assert!((result[0] - 0.9).abs() < EPSILON);
        assert!((result[1] - 1.8).abs() < EPSILON);
        assert!((result[2] - 2.7).abs() < EPSILON);

        // VectorViewMut - Scalar
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() - 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.5).abs() < EPSILON);
        assert!((result[2] - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_sub_assign() {
        // VectorViewMut -= Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2;
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= &Vector
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2;
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= &VectorView
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let mut v2 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view -= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 3.0).abs() < EPSILON);
        assert!((v1[1] - 3.0).abs() < EPSILON);
        assert!((v1[2] - 3.0).abs() < EPSILON);

        // VectorViewMut -= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view -= 0.5;
        assert!((v[0] - 0.5).abs() < EPSILON);
        assert!((v[1] - 1.5).abs() < EPSILON);
        assert!((v[2] - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_mul() {
        // VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2;
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // &VectorViewMut * &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([4.0, 5.0, 6.0]);
        let result = &v1.view_mut::<3>(0).unwrap() * &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 18.0).abs() < EPSILON);

        // VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 1.5).abs() < EPSILON);

        // &VectorViewMut * Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = &v.view_mut::<3>(0).unwrap() * 0.5;
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_mul_assign() {
        const EPSILON: f64 = 1e-10;

        // VectorViewMut *= Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &Vector
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2;
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorView
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([1, 2, 3]);
        let mut v2 = Vector::<i32, 3>::new([4, 5, 6]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view *= &v2.view_mut::<3>(0).unwrap();
        assert_eq!(v1[0], 4);
        assert_eq!(v1[1], 10);
        assert_eq!(v1[2], 18);

        // VectorViewMut *= Scalar
        let mut v = Vector::<i32, 3>::new([1, 2, 3]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 2;
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);

        // VectorViewMut *= Scalar (f64)
        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view *= 0.5;
        assert!((v[0] - 0.5).abs() < EPSILON);
        assert!((v[1] - 1.0).abs() < EPSILON);
        assert!((v[2] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_div() {
        const EPSILON: f64 = 1e-10;

        // VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &Vector
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2;
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorView
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / VectorViewMut
        let mut v1 = Vector::<i32, 3>::new([4, 10, 18]);
        let mut v2 = Vector::<i32, 3>::new([2, 5, 6]);
        let result = &v1.view_mut::<3>(0).unwrap() / v2.view_mut::<3>(0).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        // &VectorViewMut / &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let result = &v1.view_mut::<3>(0).unwrap() / &v2.view_mut::<3>(0).unwrap();
        assert!((result[0] - 10.0).abs() < EPSILON);
        assert!((result[1] - 10.0).abs() < EPSILON);
        assert!((result[2] - 10.0).abs() < EPSILON);

        // VectorViewMut / Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let result = v.view_mut::<3>(0).unwrap() / 2;
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let result = v.view_mut::<3>(0).unwrap() / 0.5;
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 4.0).abs() < EPSILON);
        assert!((result[2] - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_view_mut_div_assign() {
        // VectorViewMut /= Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2;
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // VectorViewMut /= &Vector
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2;
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // VectorViewMut /= VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // VectorViewMut /= &VectorView
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2.view::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // VectorViewMut /= VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // &VectorViewMut /= &VectorViewMut
        let mut v1 = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut v2 = Vector::<f64, 3>::new([0.1, 0.2, 0.3]);
        let mut view = v1.view_mut::<3>(0).unwrap();
        view /= &v2.view_mut::<3>(0).unwrap();
        assert!((v1[0] - 10.0).abs() < EPSILON);
        assert!((v1[1] - 10.0).abs() < EPSILON);
        assert!((v1[2] - 10.0).abs() < EPSILON);

        // VectorViewMut /= Scalar
        let mut v = Vector::<i32, 3>::new([2, 4, 6]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 2;
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);

        let mut v = Vector::<f64, 3>::new([1.0, 2.0, 3.0]);
        let mut view = v.view_mut::<3>(0).unwrap();
        view /= 0.5;
        assert!((v[0] - 2.0).abs() < EPSILON);
        assert!((v[1] - 4.0).abs() < EPSILON);
        assert!((v[2] - 6.0).abs() < EPSILON);
    }
}
