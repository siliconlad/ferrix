//! A simple static matrix library for Rust.
//!
//! This crate implements three main types: [`Vector`], [`RowVector`], and [`Matrix`].
//!
//! Alongside, various views are implemented to minimize memory allocation and copying.
//!
//! Common matrix operations are implemented via operator overloading.
//! - Matrix addition: `A + B` and `A += B`
//! - Matrix subtraction: `A - B` and `A -= B`
//! - Matrix multiplication: `A * B`
//!
//! Scalar operations are also supported.
//! - Scalar addition: `A + s` and `A += s`
//! - Scalar subtraction: `A - s` and `A -= s`
//! - Scalar multiplication: `A * s` and `A *= s`
//! - Scalar division: `A / s` and `A /= s`
//!
//! The [`DotProduct`] trait is implemented for [`Vector`] and [`RowVector`].
//!
//! # Example
//!
//! ```
//! use ferrix::*;
//!
//! // Initialize a 3x3 matrix
//! let a = Matrix3::from([
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//! ]);
//!
//! // Initialize a 3x1 column vector
//! let b = Vector3::from([1.0, 2.0, 3.0]);
//!
//! // Perform matrix multiplication
//! let c = a * b;
//! assert_eq!(c, Matrix::from([[14.0], [32.0], [50.0]]));
//!
//! ```

mod vector;
mod vector_view;
mod vector_view_mut;

mod row_vector;
mod row_vector_view;
mod row_vector_view_mut;

mod matrix;
mod matrix_transpose_view;
mod matrix_transpose_view_mut;
mod matrix_view;
mod matrix_view_mut;

mod ops;
mod traits;

pub use self::traits::DotProduct;
pub use self::traits::FloatRandom;
pub use self::traits::IntRandom;

pub use self::vector::Vector;
pub use self::vector::Vector2;
pub use self::vector::Vector3;
pub use self::vector_view::VectorView;
pub use self::vector_view_mut::VectorViewMut;

pub use self::row_vector::RowVector;
pub use self::row_vector_view::RowVectorView;
pub use self::row_vector_view_mut::RowVectorViewMut;

pub use self::matrix::Matrix;
pub use self::matrix::Matrix2;
pub use self::matrix::Matrix3;
pub use self::matrix_transpose_view::MatrixTransposeView;
pub use self::matrix_transpose_view_mut::MatrixTransposeViewMut;
pub use self::matrix_view::MatrixView;
pub use self::matrix_view_mut::MatrixViewMut;
