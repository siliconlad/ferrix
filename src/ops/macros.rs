#[macro_use]
mod op_macros {
    macro_rules! impl_inner {
        (mat, $lhs:ty, $rhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T, Output=T>, $($generics)*> $trait<$rhs> for $lhs {
                type Output = $output;

                fn $method(self, other: $rhs) -> Self::Output {
                    Self::Output::from(std::array::from_fn(|i| std::array::from_fn(|j| self[(i, j)] $op other[(i, j)])))
                }
            }
        };
        (mat, $lhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T, Output=T>, $($generics)*> $trait<T> for $lhs {
                type Output = $output;

                fn $method(self, scalar: T) -> Self::Output {
                    Self::Output::from(std::array::from_fn(|i| std::array::from_fn(|j| self[(i, j)] $op scalar)))
                }
            }
        };
        (vec, $lhs:ty, $rhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T, Output=T>, $($generics)*> $trait<$rhs> for $lhs {
                type Output = $output;

                fn $method(self, other: $rhs) -> Self::Output {
                    Self::Output::from(std::array::from_fn(|i| self[i] $op other[i]))
                }
            }
        };
        (vec, $lhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T, Output=T>, $($generics)*> $trait<T> for $lhs {
                type Output = $output;

                fn $method(self, scalar: T) -> Self::Output {
                    Self::Output::from(std::array::from_fn(|i| self[i] $op scalar))
                }
            }
        };
    }

    macro_rules! impl_combinations {
        ($type:tt, $lhs:ty, $rhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl_inner!($type, $lhs, $rhs, $trait, $method, $op, $output, $($generics)*);
            impl_inner!($type, &$lhs, $rhs, $trait, $method, $op, $output, $($generics)*);
            impl_inner!($type, $lhs, &$rhs, $trait, $method, $op, $output, $($generics)*);
            impl_inner!($type, &$lhs, &$rhs, $trait, $method, $op, $output, $($generics)*);
        };
        ($type:tt, $lhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl_inner!($type, $lhs, $trait, $method, $op, $output, $($generics)*);
            impl_inner!($type, &$lhs, $trait, $method, $op, $output, $($generics)*);
        };
    }

    macro_rules! generate_op_scalar_macros {
        ($trait:tt, $method:tt, $op:tt) => {
            macro_rules! impl_vv_op {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, N>, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, N>, const N: usize);
                }
            }

            macro_rules! impl_vv_op_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_vv_op_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, N>, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, N>, const N: usize);
                }
            }

            macro_rules! impl_vv_op_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_mm_op {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, N>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mm_op_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }
        };
    }

    macro_rules! generate_op_all_macros {
        ($trait:tt, $method:tt, $op:tt) => {
            generate_op_scalar_macros!($trait, $method, $op);

            macro_rules! impl_vv_op_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, M>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, M>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_vv_op_view_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, M>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, M>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_mm_op_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const C: usize, const D: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const C: usize, const D: usize, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mv_op {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, N, 1>, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, N, 1>, const N: usize);
                }
            }

            macro_rules! impl_mv_op_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, 1, N>, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, 1, N>, const N: usize);
                }
            }

            macro_rules! impl_vm_op_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, M>, const A: usize, const B: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, M>, const A: usize, const B: usize, const M: usize);
                }
            }

            macro_rules! impl_vm_op_mat_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, M>, const A: usize, const B: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, M>, const A: usize, const B: usize, const M: usize);
                }
            }

            macro_rules! impl_vm_op_vec_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_vm_op_vec_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_vm_op_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, Vector<T, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_vm_op_view_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(vec, $lhs, $rhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(vec, $lhs, $trait, $method, $op, RowVector<T, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mv_op_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, 1>, const A: usize, const B: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, 1>, const A: usize, const B: usize, const M: usize);
                }
            }

            macro_rules! impl_mv_op_mat_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, 1, M>, const A: usize, const B: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, 1, M>, const A: usize, const B: usize, const M: usize);
                }
            }

            macro_rules! impl_mv_op_vec_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, 1>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, 1>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mv_op_vec_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, 1, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, 1, M>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mv_op_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, M, 1>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, M, 1>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mv_op_view_view_row {
                ($lhs:ty, $rhs:ty) => {
                    impl_combinations!(mat, $lhs, $rhs, $trait, $method, $op, Matrix<T, 1, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_combinations!(mat, $lhs, $trait, $method, $op, Matrix<T, 1, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }
        };
    }
}

#[macro_use]
mod op_assign_macros {
    macro_rules! impl_assign_inner {
        ($lhs:ty, $rhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T>, $($generics)*> $trait<$rhs> for $lhs {
                fn $method(&mut self, other: $rhs) {
                    (0..self.capacity()).for_each(|i| self[i] $op other[i]);
                }
            }
        };
        ($lhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + $trait<T>, $($generics)*> $trait<T> for $lhs {
                fn $method(&mut self, scalar: T) {
                    (0..self.capacity()).for_each(|i| self[i] $op scalar);
                }
            }
        };
    }

    macro_rules! impl_assign_combinations {
        ($lhs:ty, $rhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl_assign_inner!($lhs, $rhs, $trait, $method, $op, $output, $($generics)*);
            impl_assign_inner!($lhs, &$rhs, $trait, $method, $op, $output, $($generics)*);
        };
        ($lhs:ty, $trait:tt, $method:tt, $op:tt, $output:ty, $($generics:tt)*) => {
            impl_assign_inner!($lhs, $trait, $method, $op, $output, $($generics)*);
        };
    }

    macro_rules! generate_op_assign_scalar_macros {
        ($trait:tt, $method:tt, $op:tt) => {
            macro_rules! impl_vv_op_assign {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, N>, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, N>, const N: usize);
                }
            }

            macro_rules! impl_vv_op_assign_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_mm_op_assign {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Matrix<T, M, N>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_mm_op_assign_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }
        };
    }

    macro_rules! generate_op_assign_all_macros {
        ($trait:tt, $method:tt, $op:tt) => {
            generate_op_assign_scalar_macros!($trait, $method, $op);

            macro_rules! impl_vv_op_assign_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, M>, V1: IndexMut<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, M>, V1: IndexMut<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                }
            }

            macro_rules! impl_mm_op_assign_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const C: usize, const D: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Matrix<T, M, N>, const A: usize, const B: usize, const C: usize, const D: usize, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_vm_op_assign_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, M>, const A: usize, const B: usize, const M: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, M>, const A: usize, const B: usize, const M: usize);
                }
            }

            macro_rules! impl_vm_op_assign_vec_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const M: usize, const N: usize);
                }
            }

            macro_rules! impl_vm_op_assign_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_assign_combinations!($lhs, $rhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
                ($lhs:ty) => {
                    impl_assign_combinations!($lhs, $trait, $method, $op, Vector<T, M>, V: IndexMut<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                }
            }
        };
    }
}

#[macro_use]
mod dot_macros {
    macro_rules! impl_dot_inner {
        ($lhs:ty, $rhs:ty, $($generics:tt)*) => {
            impl<T: Copy + Mul<T, Output = T> + std::iter::Sum<T>, $($generics)*> DotProduct<$rhs> for $lhs {
                type Output = T;

                fn dot(self, other: $rhs) -> Self::Output {
                    (0..M).map(|i| self[i] * other[i]).sum()
                }
            }
        };
    }

    macro_rules! impl_dot_combinations {
        ($lhs:ty, $rhs:ty, $($generics:tt)*) => {
            impl_dot_inner!($lhs, $rhs, $($generics)*);
            impl_dot_inner!(&$lhs, $rhs, $($generics)*);
            impl_dot_inner!($lhs, &$rhs, $($generics)*);
            impl_dot_inner!(&$lhs, &$rhs, $($generics)*);
        };
    }

    macro_rules! generate_dot_macros {
        () => {
            macro_rules! impl_dot {
                ($lhs:ty, $rhs:ty) => {
                    impl_dot_combinations!($lhs, $rhs, const M: usize);
                };
            }

            macro_rules! impl_dot_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_dot_combinations!($lhs, $rhs, V: Index<usize, Output = T>, const N: usize, const M: usize);
                };
            }

            macro_rules! impl_dot_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_dot_combinations!($lhs, $rhs, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const N: usize, const M: usize);
                };
            }

            macro_rules! impl_dot_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_dot_combinations!($lhs, $rhs, const A: usize, const B: usize, const M: usize);
                };
            }

            macro_rules! impl_dot_mat_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_dot_combinations!($lhs, $rhs, V: Index<usize, Output = T>, const A: usize, const B: usize, const N: usize, const M: usize);
                };
            }
        };
    }
}

#[macro_use]
mod matmul_macros {
    macro_rules! impl_matmul_inner {
        (scalar, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + Mul<T, Output = T> + From<u8> + Add<T, Output = T> + From<u8>, $($generics)*> Mul<$rhs> for $lhs
            {
                type Output = $output;

                fn mul(self, other: $rhs) -> Self::Output {
                    let mut result = T::from(0);
                    for i in 0..M {
                        result = result + (self[i] * other[i]);
                    }
                    Self::Output::new([[result]])
                }
            }
        };
        (vecvec, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + Mul<T, Output = T> + From<u8> + Add<T, Output = T> + From<u8>, $($generics)*> Mul<$rhs> for $lhs
            {
                type Output = $output;

                fn mul(self, other: $rhs) -> Self::Output {
                    let mut result = Self::Output::zeros();
                    for i in 0..M {
                        for j in 0..N {
                            result[(i, j)] = result[(i, j)] + (self[i] * other[j]);
                        }
                    }
                    result
                }
            }
        };
        (vecmat, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {

            impl<T: Copy + Mul<T, Output = T> + From<u8> + Add<T, Output = T> + From<u8>, $($generics)*> Mul<$rhs> for $lhs
            {
                type Output = $output;

                fn mul(self, other: $rhs) -> Self::Output {
                    let mut result = Self::Output::zeros();
                    for i in 0..M {
                        for j in 0..N {
                            result[i] = result[i] + (self[j] * other[(j, i)]);
                        }
                    }
                    result
                }
            }
        };
        (matvec, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + Mul<T, Output = T> + From<u8> + Add<T, Output = T> + From<u8>, $($generics)*> Mul<$rhs> for $lhs {
                type Output = $output;

                fn mul(self, other: $rhs) -> Self::Output {
                    let mut result = Self::Output::zeros();
                    for i in 0..M {
                        for j in 0..N {
                            result[(i, 0)] = result[(i, 0)] + (self[(i, j)] * other[j]);
                        }
                    }
                    result
                }
            }
        };
        (matmat, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {
            impl<T: Copy + Mul<T, Output = T> + From<u8> + Add<T, Output = T> + From<u8>, $($generics)*> Mul<$rhs> for $lhs {
                type Output = $output;

                fn mul(self, other: $rhs) -> Self::Output {
                    let mut result = Self::Output::zeros();
                    for i in 0..M {
                        for j in 0..P {
                            for k in 0..N {
                                result[(i, j)] = result[(i, j)] + (self[(i, k)] * other[(k, j)]);
                            }
                        }
                    }
                    result
                }
            }
        };
    }

    macro_rules! impl_matmul_combinations {
        ($type:tt, $lhs:ty, $rhs:ty, $output:ty, $($generics:tt)*) => {
            impl_matmul_inner!($type, $lhs, $rhs, $output, $($generics)*);
            impl_matmul_inner!($type, &$lhs, $rhs, $output, $($generics)*);
            impl_matmul_inner!($type, $lhs, &$rhs, $output, $($generics)*);
            impl_matmul_inner!($type, &$lhs, &$rhs, $output, $($generics)*);
        };
    }

    macro_rules! generate_matmul_macros {
        () => {
            macro_rules! impl_vecmul_scalar {
                ($lhs:ty, $rhs:ty) => {
                   impl_matmul_combinations!(scalar, $lhs, $rhs, Matrix<T, 1, 1>, const M: usize);
                };
            }

            macro_rules! impl_vecmul_scalar_view {
                ($lhs:ty, $rhs:ty) => {
                   impl_matmul_combinations!(scalar, $lhs, $rhs, Matrix<T, 1, 1>, V: Index<usize, Output = T>, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_scalar_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(scalar, $lhs, $rhs, Matrix<T, 1, 1>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize);
                };
            }

            macro_rules! impl_vecmul {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, V: Index<usize, Output = T>, const A: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_mat_row_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, V1: Index<usize, Output = T>, V2: Index<usize, Output = T>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_mat_row_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, V: Index<usize, Output = T>, const A: usize, const B: usize, const C: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_mat {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecmat, $lhs, $rhs, Matrix<T, 1, M>, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecmat, $lhs, $rhs, Matrix<T, 1, M>, V: Index<usize, Output = T>, const A: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_vecmat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecmat, $lhs, $rhs, Matrix<T, 1, M>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_vecmul_vecmat_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecmat, $lhs, $rhs, Matrix<T, 1, M>, V: Index<usize, Output = T>, const A: usize, const B: usize, const C: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_vec {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matvec, $lhs, $rhs, Matrix<T, M, 1>, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_vec_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matvec, $lhs, $rhs, Matrix<T, M, 1>, V: Index<usize, Output = T>, const A: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_matvec {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_matvec_vec_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, V: Index<usize, Output = T>, const A: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_matvec_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_matvec_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(vecvec, $lhs, $rhs, Matrix<T, M, N>, V: Index<usize, Output = T>, const A: usize, const B: usize, const C: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_mat_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matvec, $lhs, $rhs, Matrix<T, M, 1>, const A: usize, const B: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul_mat_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matvec, $lhs, $rhs, Matrix<T, M, 1>, V: Index<usize, Output = T>, const A: usize, const B: usize, const C: usize, const M: usize, const N: usize);
                };
            }

            macro_rules! impl_matmul {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matmat, $lhs, $rhs, Matrix<T, M, P>, const M: usize, const N: usize, const P: usize);
                };
            }

            macro_rules! impl_matmul_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matmat, $lhs, $rhs, Matrix<T, M, P>, const A: usize, const B: usize, const M: usize, const N: usize, const P: usize);
                };
            }

            macro_rules! impl_matmul_view_view {
                ($lhs:ty, $rhs:ty) => {
                    impl_matmul_combinations!(matmat, $lhs, $rhs, Matrix<T, M, P>, const A: usize, const B: usize, const C: usize, const D: usize, const M: usize, const N: usize, const P: usize);
                };
            }
        }
    }
}
