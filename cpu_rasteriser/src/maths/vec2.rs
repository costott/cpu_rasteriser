use std::ops::{Add, Mul, Sub};

use crate::prelude::*;

/// A 2D vector of `f32`, used for screen-space positions and texture coordinates.
///
/// The operators are component-wise: `a + b`, `a - b`, `a * b` (the per-component product, not
/// a dot product), and `v * s` with `s: f32`. `From` converts `(f32, f32)`, `(usize, usize)`, and
/// `(i32, i32)` tuples, so pixel coordinates convert directly. `Vec2` implements
/// [`Interpolate`](crate::Interpolate), so it can be a varying.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let pixel: Vec2 = (3_usize, 4_usize).into();
/// let centre = pixel + Vec2::new(0.5, 0.5);
///
/// assert_eq!(centre, Vec2::new(3.5, 4.5));
/// assert_eq!(Vec2::new(3.0, 4.0).length(), 5.0);
/// assert_eq!(Vec2::new(2.0, 3.0) * Vec2::new(4.0, 5.0), Vec2::new(8.0, 15.0));
/// ```
#[derive(Debug, Copy, PartialEq, Interpolate)]
pub struct Vec2 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
}
impl Vec2 {
    /// The zero vector, `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// The vector with every component 1, `(1, 1)`.
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    /// Creates a vector from its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let uv = Vec2::new(0.25, 0.75);
    /// assert_eq!((uv.x, uv.y), (0.25, 0.75));
    /// ```
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Linearly interpolates from `self` (at `t = 0`) to `other` (at `t = 1`).
    ///
    /// `t` is not clamped, so values outside `0.0..=1.0` extrapolate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let midpoint = Vec2::ZERO.lerp(Vec2::new(4.0, 2.0), 0.5);
    /// assert_eq!(midpoint, Vec2::new(2.0, 1.0));
    /// ```
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Returns the 2D cross product, `self.x * other.y - self.y * other.x`.
    ///
    /// This is the z component of the 3D cross product of the two vectors extended with z = 0,
    /// and twice the signed area of the triangle they span. It is positive when `other` is
    /// counter-clockwise from `self` with y pointing up, and zero when they are parallel.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let (a, b, c) = (Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0), Vec2::new(0.0, 2.0));
    /// let doubled_area = (b - a).cross(c - a);
    ///
    /// assert_eq!(doubled_area, 8.0);
    /// assert_eq!((c - a).cross(b - a), -8.0);
    /// ```
    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Returns the Euclidean length.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec2::new(-6.0, 8.0).length(), 10.0);
    /// ```
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
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
impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl Mul for Vec2 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
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
    use super::*;

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

        let result = a.lerp(b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn lerp_at_one_returns_other() {
        let a = Vec2::new(-2.0, 5.0);
        let b = Vec2::new(8.0, -1.0);

        let result = a.lerp(b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn lerp_at_halfway_returns_midpoint() {
        let a = Vec2::new(2.0, 4.0);
        let b = Vec2::new(6.0, 8.0);

        let result = a.lerp(b, 0.5);

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
