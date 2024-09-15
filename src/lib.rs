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
