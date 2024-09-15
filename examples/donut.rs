// Based on https://www.a1k0n.net/2011/07/20/donut-math.html
use ferrix::{DotProduct, Matrix, Vector3};
use std::f64::consts::PI;

// Window size
const WIDTH: usize = 80;
const HEIGHT: usize = 30;

// Torus parameters
const R1: f64 = 1.0; // Radius of inner hole
const R2: f64 = 2.0; // Radius of tube
const K1_X: f64 = 37.5;
const K1_Y: f64 = 18.75;
const K2: f64 = 5.0; // Distance to torus

// Render spacing
const THETA_SPACING: f64 = 0.07;
const PHI_SPACING: f64 = 0.02;

fn render_frame(theta_a: f64, theta_b: f64) {
    let mut output = Matrix::<char, HEIGHT, WIDTH>::fill(' ');
    let mut zbuffer = Matrix::<f64, HEIGHT, WIDTH>::fill(0.0);

    let rot_a = Matrix::rotx(theta_a);
    let rot_b = Matrix::rotz(theta_b);

    let mut theta = 0.0;
    while theta < (2.0 * PI) {
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let mut phi = 0.0;
        while phi < (2.0 * PI) {
            let rot_phi = Matrix::roty(phi);

            // Calculate 3D point on donut
            let circle = Vector3::from([cos_theta, sin_theta, 0.0]) * R1;
            let circle = circle + Vector3::from([R2, 0.0, 0.0]);
            let point = &rot_b * &rot_a * &rot_phi * &circle;

            let ooz = 1.0 / (point[2] + K2); // One over z (larger = closer)

            // Project 3D point to 2D
            let xp: usize = ((WIDTH as f64 / 2.0) + (point[0] * ooz * K1_X)) as usize;
            let yp: usize = ((HEIGHT as f64 / 2.0) - (point[1] * ooz * K1_Y)) as usize;

            // Calculate luminance
            let n_point = Vector3::from([cos_theta, sin_theta, 0.0]);
            let normal = &rot_b * &rot_a * &rot_phi * &n_point;
            let luminance: f64 = Vector3::from([0.0, 1.0, -1.0]).dot(&normal);

            // Only render points facing the viewer
            if luminance > 0.0 {
                if ooz > zbuffer[(yp, xp)] {
                    zbuffer[(yp, xp)] = ooz;
                    let l_index = (luminance * 8.0) as usize;
                    output[(yp, xp)] = ".,-~:;=!*#$@".chars().nth(l_index).unwrap();
                }
            }

            // Prepare for next loop
            phi += PHI_SPACING;
        }

        // Prepare for next loop
        theta += THETA_SPACING;
    }

    // Send output to screen
    println!("\x1b[H");
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            print!("{}", output[(row, col)]);
        }
        print!("\n");
    }
}

fn main() {
    let mut m_a = 1.0;
    let mut m_b = 1.0;

    loop {
        render_frame(m_a, m_b);
        m_a = (m_a + 0.07) % (2.0 * PI);
        m_b = (m_b + 0.03) % (2.0 * PI);
    }
}
