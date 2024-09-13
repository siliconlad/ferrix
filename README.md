<div align="center" id="user-content-toc">
  <ul style="list-style: none; margin-left: 0px; padding-left: 0px;">
    <summary>
      <h1>Ferrix</h1>
    </summary>
  </ul>
  <p style="padding-top: 0px;">A simple typed matrix library for Rust.</p>
</div>

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

> This will **not** work until I publish the crate.

## Contributing

If you find any bugs or have any suggestions, please open an issue or submit a pull request.

Specifically, I'm looking for help with:
- Creating more examples to illustrate the library's capabilities
    - This would help identify missing features
    - Opportunity for feedback on how to improve the library
- Implementing a linear algebra crate on top of this one
- Aggressively increasing the performance of the library
