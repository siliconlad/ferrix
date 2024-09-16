/// Trait for the dot product operation.
///
/// The [dot product](https://en.wikipedia.org/wiki/Dot_product) is the sum of the element-wise products of two vectors.
///
/// The `dot` method is only defined for [`Vector`](crate::vector::Vector) and [`RowVector`](crate::row_vector::RowVector).
/// However, the `dot` method accepts as an argument a [`Matrix`](crate::matrix::Matrix) or any of its views,
/// as long it represents a column or row vector respectively. That is, to `dot` with a [`Vector`](crate::vector::Vector)
/// of length `n`, the rhs must be a [`Matrix`](crate::matrix::Matrix) of shape `(n, 1)`.
/// Similarly for [`RowVector`](crate::row_vector::RowVector).
///
/// # Example
///
/// This example shows some of the possible dot product combinations.
///
/// ```
/// use ferrix::DotProduct;
/// use ferrix::{Vector, RowVector, Matrix};
///
/// // Dot product of two vectors
/// let a = Vector::from([1, 2, 3]);
/// let b = Vector::from([4, 5, 6]);
/// assert_eq!(a.dot(b), 32);
///
/// // Alternatively
/// let a = Vector::from([1, 2, 3]);
/// let b = Matrix::from([[4], [5], [6]]);
/// assert_eq!(a.dot(b), 32);
///
/// // Dot product of two row vectors
/// let a = RowVector::from([1, 2, 3]);
/// let b = RowVector::from([4, 5, 6]);
/// assert_eq!(a.dot(b), 32);
///
/// // Similarly
/// let a = RowVector::from([1, 2, 3]);
/// let b = Matrix::from([[4, 5, 6]]);
/// assert_eq!(a.dot(b), 32);
/// ```
pub trait DotProduct<V> {
    type Output;

    /// Computes the dot product of two vectors.
    fn dot(self, other: V) -> Self::Output;
}

/// Trait for integer random number generation.
///
/// Generates a [`Vector`](crate::vector::Vector), [`RowVector`](crate::row_vector::RowVector), or [`Matrix`](crate::matrix::Matrix)
/// filled with random integers uniformly across all values of the integer type.
///
/// # Example
///
/// ```
/// use ferrix::IntRandom;
/// use ferrix::Vector;
///
/// let a = Vector::<i32, 3>::random();
/// ```
pub trait IntRandom {
    /// Will generate random integers uniformly across all values of the type.
    fn random() -> Self;
}

/// Trait for floating-point random number generation.
///
/// Generates a [`Vector`](crate::vector::Vector), [`RowVector`](crate::row_vector::RowVector), or [`Matrix`](crate::matrix::Matrix)
/// filled with random floating-point numbers uniformly in the range `[-1, 1]`.
///
/// # Example
///
/// ```
/// use ferrix::FloatRandom;
/// use ferrix::Vector;
///
/// let a = Vector::<f32, 3>::random();
/// ```
pub trait FloatRandom {
    /// Will generate random floating-point numbers uniformly in the range `[-1, 1]`.
    fn random() -> Self;
}
