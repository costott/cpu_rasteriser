use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}
impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }
}
impl Add for Vec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl From<(f32, f32)> for Vec2 {
    fn from(tuple: (f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
        }
    }
}
impl From<(usize, usize)> for Vec2 {
    fn from(tuple: (usize, usize)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
        }
    }
}
impl From<(i32, i32)> for Vec2 {
    fn from(tuple: (i32, i32)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Vec2;

    const EPSILON: f32 = 1e-6;

    fn assert_vec2_approx_eq(actual: Vec2, expected: Vec2) {
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
    }

    #[test]
    fn new_sets_components() {
        let v = Vec2::new(3.5, -2.0);
        assert_eq!(v.x, 3.5);
        assert_eq!(v.y, -2.0);
    }

    #[test]
    fn add_combines_components() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, -4.0);

        let sum = a + b;

        assert_eq!(sum, Vec2::new(4.0, -2.0));
    }

    #[test]
    fn mul_scales_components() {
        let v = Vec2::new(2.0, -3.0);

        let scaled = v * 1.5;

        assert_eq!(scaled, Vec2::new(3.0, -4.5));
    }

    #[test]
    fn lerp_at_zero_returns_self() {
        let a = Vec2::new(-2.0, 5.0);
        let b = Vec2::new(8.0, -1.0);

        let result = a.lerp(&b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn lerp_at_one_returns_other() {
        let a = Vec2::new(-2.0, 5.0);
        let b = Vec2::new(8.0, -1.0);

        let result = a.lerp(&b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn lerp_at_halfway_returns_midpoint() {
        let a = Vec2::new(2.0, 4.0);
        let b = Vec2::new(6.0, 8.0);

        let result = a.lerp(&b, 0.5);

        assert_vec2_approx_eq(result, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn from_f32_tuple_creates_vector() {
        let v = Vec2::from((1.25_f32, -9.5_f32));

        assert_eq!(v, Vec2::new(1.25, -9.5));
    }

    #[test]
    fn from_usize_tuple_creates_vector() {
        let v = Vec2::from((3_usize, 7_usize));

        assert_eq!(v, Vec2::new(3.0, 7.0));
    }

    #[test]
    fn from_i32_tuple_creates_vector() {
        let v = Vec2::from((-3_i32, 7_i32));

        assert_eq!(v, Vec2::new(-3.0, 7.0));
    }
}
