<div align="center" id="user-content-toc">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./assets/ferrix-dark.svg">
    <img src="./assets/ferrix-light.svg" alt="Ferrix Logo" width="300">
  </picture>
  <p style="padding-top: 5px;">A simple matrix library for Rust.</p>
</div>

Checkout the write up on how it was built [here](www.siliconlad.com/blog/ferrix).

## Example

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

## Installation

To add Ferrix to your project, install it via cargo:

```bash
cargo add ferrix
```

## Contributing

If you find any bugs, please open an issue or submit a pull request.

If you have any suggestions or questions, please open an issue.

Specifically, I'm looking for help with:
- Creating more examples to illustrate the library's capabilities
    - This would help identify missing features
    - Opportunity for feedback on how to improve the library
- Implementing a linear algebra crate on top of this one
- Aggressively increasing the performance of the library
