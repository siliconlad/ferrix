mod matrix;
mod matrix_t_view;
mod matrix_t_view_mut;
mod matrix_view;
mod matrix_view_mut;
mod ops;
mod traits;
mod vector;
mod vector_view;
mod vector_view_mut;

pub use self::matrix::Matrix;
pub use self::matrix_t_view::MatrixTransposeView;
pub use self::matrix_t_view_mut::MatrixTransposeViewMut;
pub use self::matrix_view::MatrixView;
pub use self::matrix_view_mut::MatrixViewMut;
pub use self::vector::Vector;
pub use self::vector_view::VectorView;
pub use self::vector_view_mut::VectorViewMut;

pub use self::traits::DotProduct;
pub use self::traits::MatMul;
pub use self::traits::MatrixRead;
pub use self::traits::MatrixWrite;
