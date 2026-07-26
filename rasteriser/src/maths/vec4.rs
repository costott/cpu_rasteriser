use crate::maths::Vec3;
use std::ops::{Add, Div, Mul, Sub};

/// Represents a 4D (column) vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_vec3(vec3: &Vec3) -> Self {
        Self {
            x: vec3.x,
            y: vec3.y,
            z: vec3.z,
            w: 1.0,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn normalise(&mut self) {
        let length = self.dot(self).sqrt();
        if length > 0.0 {
            *self = *self / length;
        }
    }

    pub fn homogenize(&mut self) {
        if self.w != 0.0 {
            self.x /= self.w;
            self.y /= self.w;
            self.z /= self.w;
            self.w = 1.0;
        }
    }

    pub fn homogenize_to_vec3(&self) -> Vec3 {
        if self.w != 0.0 {
            Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w)
        } else {
            Vec3::new(self.x, self.y, self.z)
        }
    }
}

impl Add for Vec4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}
impl Sub for Vec4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}
impl Mul<f32> for Vec4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}
impl Div<f32> for Vec4 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from(tuple: (f32, f32, f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
            z: tuple.2,
            w: tuple.3,
        }
    }
}
impl From<(usize, usize, usize, usize)> for Vec4 {
    fn from(tuple: (usize, usize, usize, usize)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
            z: tuple.2 as f32,
            w: tuple.3 as f32,
        }
    }
}
impl From<(i32, i32, i32, i32)> for Vec4 {
    fn from(tuple: (i32, i32, i32, i32)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
            z: tuple.2 as f32,
            w: tuple.3 as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Vec4;

    const EPSILON: f32 = 1e-6;

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
    fn new_sets_components() {
        let v = Vec4::new(3.5, -2.0, 7.25, 1.0);
        assert_eq!(v.x, 3.5);
        assert_eq!(v.y, -2.0);
        assert_eq!(v.z, 7.25);
        assert_eq!(v.w, 1.0);
    }

    #[test]
    fn add_combines_components() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(3.0, -4.0, 1.0, 0.5);

        let sum = a + b;

        assert_eq!(sum, Vec4::new(4.0, -2.0, 4.0, 4.5));
    }

    #[test]
    fn sub_subtracts_components() {
        let a = Vec4::new(5.0, 2.0, -1.0, 2.0);
        let b = Vec4::new(3.0, 4.0, 1.0, -1.0);

        let diff = a - b;

        assert_eq!(diff, Vec4::new(2.0, -2.0, -2.0, 3.0));
    }

    #[test]
    fn mul_scales_components() {
        let v = Vec4::new(2.0, -3.0, 4.0, 0.5);

        let scaled = v * 1.5;

        assert_eq!(scaled, Vec4::new(3.0, -4.5, 6.0, 0.75));
    }

    #[test]
    fn div_scales_components() {
        let v = Vec4::new(9.0, -6.0, 3.0, 2.0);

        let scaled = v / 3.0;

        assert_eq!(scaled, Vec4::new(3.0, -2.0, 1.0, 0.6666667));
    }

    #[test]
    fn lerp_at_zero_returns_self() {
        let a = Vec4::new(-2.0, 5.0, 3.0, 1.0);
        let b = Vec4::new(8.0, -1.0, 7.0, 0.0);

        let result = a.lerp(&b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn lerp_at_one_returns_other() {
        let a = Vec4::new(-2.0, 5.0, 3.0, 1.0);
        let b = Vec4::new(8.0, -1.0, 7.0, 0.0);

        let result = a.lerp(&b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn lerp_at_halfway_returns_midpoint() {
        let a = Vec4::new(2.0, 4.0, 6.0, 0.0);
        let b = Vec4::new(6.0, 8.0, 10.0, 2.0);

        let result = a.lerp(&b, 0.5);

        assert_vec4_approx_eq(result, Vec4::new(4.0, 6.0, 8.0, 1.0));
    }

    #[test]
    fn dot_returns_scalar_product() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(4.0, -5.0, 6.0, -2.0);

        let dot = a.dot(&b);

        assert_eq!(dot, 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0 + 4.0 * (-2.0));
    }

    #[test]
    fn normalise_scales_non_zero_vector_to_unit_length() {
        let mut v = Vec4::new(3.0, 4.0, 0.0, 0.0);

        v.normalise();

        assert_vec4_approx_eq(v, Vec4::new(0.6, 0.8, 0.0, 0.0));
    }

    #[test]
    fn normalise_zero_vector_keeps_zero_vector() {
        let mut v = Vec4::new(0.0, 0.0, 0.0, 0.0);

        v.normalise();

        assert_eq!(v, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn from_f32_tuple_creates_vector() {
        let v = Vec4::from((1.25_f32, -9.5_f32, 2.0_f32, 0.5_f32));

        assert_eq!(v, Vec4::new(1.25, -9.5, 2.0, 0.5));
    }

    #[test]
    fn from_usize_tuple_creates_vector() {
        let v = Vec4::from((3_usize, 7_usize, 9_usize, 1_usize));

        assert_eq!(v, Vec4::new(3.0, 7.0, 9.0, 1.0));
    }

    #[test]
    fn from_i32_tuple_creates_vector() {
        let v = Vec4::from((-3_i32, 7_i32, -11_i32, 2_i32));

        assert_eq!(v, Vec4::new(-3.0, 7.0, -11.0, 2.0));
    }

    #[test]
    fn homogenize_divides_by_w_and_sets_w_to_one() {
        let mut v = Vec4::new(2.0, 4.0, 6.0, 2.0);

        v.homogenize();

        assert_vec4_approx_eq(v, Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn homogenize_does_nothing_when_w_is_zero() {
        let mut v = Vec4::new(2.0, 4.0, 6.0, 0.0);

        v.homogenize();

        assert_eq!(v, Vec4::new(2.0, 4.0, 6.0, 0.0));
    }

    #[test]
    fn homogenize_to_vec3_divides_by_w_when_nonzero() {
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);

        let r = v.homogenize_to_vec3();

        assert_eq!(r, crate::maths::Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn homogenize_to_vec3_returns_xyz_when_w_zero() {
        let v = Vec4::new(2.0, 4.0, 6.0, 0.0);

        let r = v.homogenize_to_vec3();

        assert_eq!(r, crate::maths::Vec3::new(2.0, 4.0, 6.0));
    }
}
