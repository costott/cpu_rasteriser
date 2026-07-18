use crate::maths::{Vec3, Vec4};
use std::ops::{Add, Mul, Sub};

/// Represents a 4x4 matrix, stored in row-major order
/// Assumes vectors will be column vectors, so matrix multiplication is done as M * v
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub data: [[f32; 4]; 4],
}
impl Mat4 {
    pub fn new(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn translation_vec(position: Vec3) -> Self {
        Self::translation(position.x, position.y, position.z)
    }

    pub fn scaling(x: f32, y: f32, z: f32) -> Self {
        Self {
            data: [
                [x, 0.0, 0.0, 0.0],
                [0.0, y, 0.0, 0.0],
                [0.0, 0.0, z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn scaling_vec(scale: Vec3) -> Self {
        Self::scaling(scale.x, scale.y, scale.z)
    }

    pub fn reflect_x() -> Self {
        Self {
            data: [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn reflect_y() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn reflect_z() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn shear(xy: f32, xz: f32, yx: f32, yz: f32, zx: f32, zy: f32) -> Self {
        Self {
            data: [
                [1.0, xy, xz, 0.0],
                [yx, 1.0, yz, 0.0],
                [zx, zy, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_x(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, -sin, 0.0],
                [0.0, sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_y(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            data: [
                [cos, 0.0, sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin, 0.0, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_z(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            data: [
                [cos, -sin, 0.0, 0.0],
                [sin, cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_vec(rotation: Vec3) -> Self {
        let rx = Self::rotate_x(rotation.x);
        let ry = Self::rotate_y(rotation.y);
        let rz = Self::rotate_z(rotation.z);

        rz * ry * rx
    }

    /// Inverts the matrix using Gaussian elimination.
    /// Returns the identity matrix if the matrix is singular (non-invertible).
    pub fn inverse(&self) -> Self {
        let mut m = self.data;
        let mut inv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        for col in 0..4 {
            let mut pivot_row = col;
            let mut pivot_value = m[pivot_row][col].abs();

            for row in (col + 1)..4 {
                let value = m[row][col].abs();
                if value > pivot_value {
                    pivot_row = row;
                    pivot_value = value;
                }
            }

            if pivot_value < 1e-6 {
                return Self::identity();
            }

            if pivot_row != col {
                m.swap(col, pivot_row);
                inv.swap(col, pivot_row);
            }

            let pivot = m[col][col];
            for j in 0..4 {
                m[col][j] /= pivot;
                inv[col][j] /= pivot;
            }

            for row in 0..4 {
                if row == col {
                    continue;
                }

                let factor = m[row][col];
                if factor == 0.0 {
                    continue;
                }

                for j in 0..4 {
                    m[row][j] -= factor * m[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }

        Self { data: inv }
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                transposed[i][j] = self.data[j][i];
            }
        }
        Self { data: transposed }
    }
}

impl Add for Mat4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Self { data: result }
    }
}
impl Sub for Mat4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        Self { data: result }
    }
}
impl Mul<f32> for Mat4 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[i][j] * scalar;
            }
        }
        Self { data: result }
    }
}
impl Mul for Mat4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Self { data: result }
    }
}
// Mat4 * Vec4 -> Vec4
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    /// Mat4 * Vec4
    /// # Example
    /// ```
    /// let mat = Mat4::identity();
    /// let vec = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(mat * vec, vec);
    /// ```
    fn mul(self, vec: Vec4) -> Vec4 {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = self.data[i][0] * vec.x
                + self.data[i][1] * vec.y
                + self.data[i][2] * vec.z
                + self.data[i][3] * vec.w;
        }
        Vec4::new(result[0], result[1], result[2], result[3])
    }
}

#[cfg(test)]
mod tests {
    use super::{Mat4, Vec4};
    use std::f32::consts::FRAC_PI_2;

    const EPSILON: f32 = 1e-6;

    fn assert_mat4_approx_eq(actual: Mat4, expected: Mat4) {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (actual.data[i][j] - expected.data[i][j]).abs() < EPSILON,
                    "matrix mismatch at [{i}][{j}]: actual={} expected={}",
                    actual.data[i][j],
                    expected.data[i][j]
                );
            }
        }
    }

    fn assert_vec4_approx_eq(actual: Vec4, expected: Vec4) {
        assert!(
            (actual.x - expected.x).abs() < EPSILON,
            "x mismatch: actual={} expected={}",
            actual.x,
            expected.x
        );
        assert!(
            (actual.y - expected.y).abs() < EPSILON,
            "y mismatch: actual={} expected={}",
            actual.y,
            expected.y
        );
        assert!(
            (actual.z - expected.z).abs() < EPSILON,
            "z mismatch: actual={} expected={}",
            actual.z,
            expected.z
        );
        assert!(
            (actual.w - expected.w).abs() < EPSILON,
            "w mismatch: actual={} expected={}",
            actual.w,
            expected.w
        );
    }

    #[test]
    fn new_stores_data() {
        let data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];

        let mat = Mat4::new(data);

        assert_eq!(mat.data, data);
    }

    #[test]
    fn identity_has_ones_on_diagonal() {
        let id = Mat4::identity();
        let expected = Mat4::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        assert_eq!(id, expected);
    }

    #[test]
    fn translation_moves_point_coordinates() {
        let point = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let translation = Mat4::translation(5.0, -2.0, 4.0);

        let moved = translation * point;

        assert_eq!(moved, Vec4::new(6.0, 0.0, 7.0, 1.0));
    }

    #[test]
    fn scaling_scales_axes() {
        let point = Vec4::new(-2.0, 3.0, 4.0, 1.0);
        let scale = Mat4::scaling(2.0, 3.0, -0.5);

        let scaled = scale * point;

        assert_eq!(scaled, Vec4::new(-4.0, 9.0, -2.0, 1.0));
    }

    #[test]
    fn reflections_flip_expected_axis() {
        let point = Vec4::new(2.0, -3.0, 4.0, 1.0);

        assert_eq!(Mat4::reflect_x() * point, Vec4::new(-2.0, -3.0, 4.0, 1.0));
        assert_eq!(Mat4::reflect_y() * point, Vec4::new(2.0, 3.0, 4.0, 1.0));
        assert_eq!(Mat4::reflect_z() * point, Vec4::new(2.0, -3.0, -4.0, 1.0));
    }

    #[test]
    fn shear_applies_off_diagonal_terms() {
        let shear = Mat4::shear(1.0, 0.0, 0.0, 0.5, 0.0, -1.0);
        let point = Vec4::new(2.0, 3.0, 4.0, 1.0);

        let sheared = shear * point;

        assert_eq!(sheared, Vec4::new(5.0, 5.0, 1.0, 1.0));
    }

    #[test]
    fn rotate_x_by_90_degrees() {
        let v = Vec4::new(0.0, 1.0, 0.0, 1.0);

        let rotated = Mat4::rotate_x(FRAC_PI_2) * v;

        assert_vec4_approx_eq(rotated, Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn rotate_y_by_90_degrees() {
        let v = Vec4::new(0.0, 0.0, 1.0, 1.0);

        let rotated = Mat4::rotate_y(FRAC_PI_2) * v;

        assert_vec4_approx_eq(rotated, Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn rotate_z_by_90_degrees() {
        let v = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let rotated = Mat4::rotate_z(FRAC_PI_2) * v;

        assert_vec4_approx_eq(rotated, Vec4::new(0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn add_and_subtract_work_elementwise() {
        let a = Mat4::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let b = Mat4::new([
            [2.0, 1.0, 0.0, -1.0],
            [3.0, 2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);

        let sum = a + b;
        let diff = a - b;

        assert_eq!(
            sum,
            Mat4::new([
                [3.0, 3.0, 3.0, 3.0],
                [8.0, 8.0, 8.0, 8.0],
                [13.0, 13.0, 13.0, 13.0],
                [18.0, 18.0, 18.0, 18.0],
            ])
        );
        assert_eq!(
            diff,
            Mat4::new([
                [-1.0, 1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0, 8.0],
                [5.0, 7.0, 9.0, 11.0],
                [8.0, 10.0, 12.0, 14.0],
            ])
        );
    }

    #[test]
    fn scalar_multiplication_scales_every_element() {
        let m = Mat4::new([
            [1.0, -2.0, 3.0, -4.0],
            [0.5, -1.5, 2.5, -3.5],
            [9.0, 8.0, 7.0, 6.0],
            [0.0, 1.0, 0.0, 1.0],
        ]);

        let scaled = m * 2.0;

        assert_eq!(
            scaled,
            Mat4::new([
                [2.0, -4.0, 6.0, -8.0],
                [1.0, -3.0, 5.0, -7.0],
                [18.0, 16.0, 14.0, 12.0],
                [0.0, 2.0, 0.0, 2.0],
            ])
        );
    }

    #[test]
    fn matrix_multiplication_composes_transforms() {
        let scale = Mat4::scaling(2.0, 3.0, 4.0);
        let translation = Mat4::translation(5.0, 6.0, 7.0);
        let composed = translation * scale;
        let point = Vec4::new(1.0, 1.0, 1.0, 1.0);

        let transformed = composed * point;

        assert_eq!(transformed, Vec4::new(7.0, 9.0, 11.0, 1.0));
    }

    #[test]
    fn identity_is_multiplicative_neutral() {
        let m = Mat4::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        assert_mat4_approx_eq(Mat4::identity() * m, m);
        assert_mat4_approx_eq(m * Mat4::identity(), m);
    }

    #[test]
    fn inverse_of_identity_is_identity() {
        let inverse = Mat4::identity().inverse();

        assert_mat4_approx_eq(inverse, Mat4::identity());
    }

    #[test]
    fn inverse_of_translation_reverses_translation() {
        let translation = Mat4::translation(2.0, -3.0, 4.0);
        let inverse = translation.inverse();
        let point = Vec4::new(1.0, 2.0, 3.0, 1.0);

        assert_vec4_approx_eq(inverse * (translation * point), point);
    }

    // test transpose
    #[test]
    fn transpose_swaps_rows_and_columns() {
        let m = Mat4::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let transposed = m.transpose();

        assert_eq!(
            transposed,
            Mat4::new([
                [1.0, 5.0, 9.0, 13.0],
                [2.0, 6.0, 10.0, 14.0],
                [3.0, 7.0, 11.0, 15.0],
                [4.0, 8.0, 12.0, 16.0],
            ])
        );
    }
}
