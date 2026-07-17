use crate::maths::Vec4;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn normalise(&self) -> Vec3 {
        let length = self.dot(self).sqrt();
        if length > 0.0 { *self / length } else { *self }
    }

    pub fn to_homogenous(&self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, 1.0)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}
impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl From<(f32, f32, f32)> for Vec3 {
    fn from(tuple: (f32, f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
            z: tuple.2,
        }
    }
}
impl From<(usize, usize, usize)> for Vec3 {
    fn from(tuple: (usize, usize, usize)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
            z: tuple.2 as f32,
        }
    }
}
impl From<(i32, i32, i32)> for Vec3 {
    fn from(tuple: (i32, i32, i32)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
            z: tuple.2 as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Vec3;

    const EPSILON: f32 = 1e-6;

    fn assert_vec3_approx_eq(actual: Vec3, expected: Vec3) {
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
    }

    #[test]
    fn new_sets_components() {
        let v = Vec3::new(3.5, -2.0, 7.25);
        assert_eq!(v.x, 3.5);
        assert_eq!(v.y, -2.0);
        assert_eq!(v.z, 7.25);
    }

    #[test]
    fn add_combines_components() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, -4.0, 1.0);

        let sum = a + b;

        assert_eq!(sum, Vec3::new(4.0, -2.0, 4.0));
    }

    #[test]
    fn sub_subtracts_components() {
        let a = Vec3::new(5.0, 2.0, -1.0);
        let b = Vec3::new(3.0, 4.0, 1.0);

        let diff = a - b;

        assert_eq!(diff, Vec3::new(2.0, -2.0, -2.0));
    }

    #[test]
    fn mul_scales_components() {
        let v = Vec3::new(2.0, -3.0, 4.0);

        let scaled = v * 1.5;

        assert_eq!(scaled, Vec3::new(3.0, -4.5, 6.0));
    }

    #[test]
    fn div_scales_components() {
        let v = Vec3::new(9.0, -6.0, 3.0);

        let scaled = v / 3.0;

        assert_eq!(scaled, Vec3::new(3.0, -2.0, 1.0));
    }

    #[test]
    fn lerp_at_zero_returns_self() {
        let a = Vec3::new(-2.0, 5.0, 3.0);
        let b = Vec3::new(8.0, -1.0, 7.0);

        let result = a.lerp(&b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn lerp_at_one_returns_other() {
        let a = Vec3::new(-2.0, 5.0, 3.0);
        let b = Vec3::new(8.0, -1.0, 7.0);

        let result = a.lerp(&b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn lerp_at_halfway_returns_midpoint() {
        let a = Vec3::new(2.0, 4.0, 6.0);
        let b = Vec3::new(6.0, 8.0, 10.0);

        let result = a.lerp(&b, 0.5);

        assert_vec3_approx_eq(result, Vec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn dot_returns_scalar_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);

        let dot = a.dot(&b);

        assert_eq!(dot, 12.0);
    }

    #[test]
    fn cross_of_basis_vectors_matches_right_handed_rule() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);

        let z = x.cross(&y);

        assert_eq!(z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cross_is_anti_commutative() {
        let a = Vec3::new(2.0, -1.0, 3.0);
        let b = Vec3::new(0.0, 4.0, -2.0);

        let ab = a.cross(&b);
        let ba = b.cross(&a);

        assert_vec3_approx_eq(ab, ba * -1.0);
    }

    #[test]
    fn normalise_scales_non_zero_vector_to_unit_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);

        assert_vec3_approx_eq(v.normalise(), Vec3::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn normalise_zero_vector_keeps_zero_vector() {
        let v = Vec3::new(0.0, 0.0, 0.0);

        assert_eq!(v.normalise(), Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn from_f32_tuple_creates_vector() {
        let v = Vec3::from((1.25_f32, -9.5_f32, 2.0_f32));

        assert_eq!(v, Vec3::new(1.25, -9.5, 2.0));
    }

    #[test]
    fn from_usize_tuple_creates_vector() {
        let v = Vec3::from((3_usize, 7_usize, 9_usize));

        assert_eq!(v, Vec3::new(3.0, 7.0, 9.0));
    }

    #[test]
    fn from_i32_tuple_creates_vector() {
        let v = Vec3::from((-3_i32, 7_i32, -11_i32));

        assert_eq!(v, Vec3::new(-3.0, 7.0, -11.0));
    }
}
