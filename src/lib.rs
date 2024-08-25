mod vector;
mod vector_view;
mod vector_view_mut;
mod vector_transpose_view;
mod vector_transpose_view_mut;
mod matrix;
mod matrix_view;
mod matrix_view_mut;
mod matrix_transpose_view;
mod matrix_transpose_view_mut;
mod ops;
mod traits;

pub use self::traits::DotProduct;
pub use self::traits::MatMul;

pub use self::vector::Vector;
pub use self::vector_view::VectorView;
pub use self::vector_view_mut::VectorViewMut;
pub use self::vector_transpose_view::VectorTransposeView;
pub use self::vector_transpose_view_mut::VectorTransposeViewMut;

pub use self::matrix::RowVector;
pub use self::matrix_view::RowVectorView;
pub use self::matrix_view_mut::RowVectorViewMut;
pub use self::matrix_transpose_view::RowVectorTransposeView;
pub use self::matrix_transpose_view_mut::RowVectorTransposeViewMut;

pub use self::matrix::Matrix;
pub use self::matrix_view::MatrixView;
pub use self::matrix_view_mut::MatrixViewMut;
pub use self::matrix_transpose_view::MatrixTransposeView;
pub use self::matrix_transpose_view_mut::MatrixTransposeViewMut;
