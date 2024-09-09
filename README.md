<div align="center">
<p style="padding-top: 10px; margin-bottom: 0px; font-size: 40px;">Ferrix</p>
<p style="padding-top: 0px;">A simple typed matrix library for Rust.</p>
</div>

# Example

```rust
use ferrix::*;

fn main() {
    // Create a 3x3 matrix
    let matrix = Matrix3::fill(1.0);
    println!("Mat: {}", matrix);

    // Create a 3x1 vector
    let vector = Vector3::new([1.0, 2.0, 3.0]);
    println!("Vec: {}", vector);

    // Perform matrix-vector multiplication
    let result = matrix * vector;
    println!("Result: {}", result);

    // Transpose the matrix
    let transposed = matrix.t();
    println!("Transposed: {}", transposed);

    // Perform element-wise operations
    let scaled = matrix * 2.0;
    let sum = matrix + scaled;
    println!("Result: {}", sum);
}
```

For more comprehensive examples, check out the [examples](./examples).

# Installation

To add Ferrix to your project, install it via cargo:

```bash
cargo add ferrix
```

> This will **not** work until I publish the crate.

# Contributing

If you find any bugs or have any suggestions, please open an issue or submit a pull request.

# Roadmap

- [ ] Add more useful methods to `Vector` and `Matrix`
- [ ] Add dynamic `Vector` and `Matrix` types
- [ ] Add linear algebra library (e.g. `inv`, `det`, `eig`, etc.)
- [ ] Performance optimizations (e.g. SIMD)
- [ ] Add more examples
