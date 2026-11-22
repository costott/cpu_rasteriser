use std::ops::{Add, AddAssign, Div, Mul, Sub};

use crate::prelude::*;

/// A 4D vector of `f32`: a homogeneous coordinate, or any four-component value.
///
/// As a homogeneous coordinate it is a column vector transformed by `m * v` (see [`Mat4`]). A
/// point has w = 1 and a direction w = 0; after a projection, clip-space positions have any w and
/// [`Vec4::homogenize`] performs the perspective divide. A [`VertexShader`](crate::VertexShader)
/// returns its clip-space position as a `Vec4`. Colours convert to and from `Vec4` through
/// [`Colour`]'s `From` impls.
///
/// The operators are component-wise over all four components, w included: `a + b`, `a += b`,
/// `a - b`, `v * s`, and `v / s` with `s: f32`. `From` converts four-element tuples of `f32`,
/// `usize`, or `i32`. `Vec4` implements [`Interpolate`](crate::Interpolate), so it can be a
/// varying.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let clip = Vec4::new(2.0, -4.0, 1.0, 2.0);
/// let ndc = clip.homogenize_to_vec3();
/// assert_eq!(ndc, Vec3::new(1.0, -2.0, 0.5));
///
/// let colour: Colour = Vec4::new(1.0, 0.5, 0.0, 1.0).into();
/// assert_eq!(colour, Colour::rgb(1.0, 0.5, 0.0));
/// ```
#[derive(Debug, Copy, PartialEq, Interpolate)]
pub struct Vec4 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
    /// The w component: 1 for a point, 0 for a direction.
    pub w: f32,
}
impl Vec4 {
    /// Creates a vector from its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let clip = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!((clip.x, clip.y, clip.z, clip.w), (1.0, 2.0, 3.0, 4.0));
    /// ```
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Linearly interpolates all four components from `self` (at `t = 0`) to `other` (at
    /// `t = 1`).
    ///
    /// `t` is not clamped, so values outside `0.0..=1.0` extrapolate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let a = Vec4::new(0.0, 0.0, 0.0, 1.0);
    /// let b = Vec4::new(4.0, 2.0, 0.0, 1.0);
    /// assert_eq!(a.lerp(b, 0.25), Vec4::new(1.0, 0.5, 0.0, 1.0));
    /// ```
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Returns the dot product over all four components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(a.dot(Vec4::new(1.0, 1.0, 1.0, 1.0)), 10.0);
    /// ```
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Returns the vector scaled to length 1 over all four components, w included.
    ///
    /// Because w counts towards the length, normalise a homogeneous point's direction as a
    /// [`Vec3`] instead. A zero-length vector is returned unchanged, as the zero vector, rather
    /// than producing NaN; there is no tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec4::new(0.0, 3.0, 0.0, 4.0).normalise(), Vec4::new(0.0, 0.6, 0.0, 0.8));
    /// assert_eq!(Vec4::new(0.0, 0.0, 0.0, 0.0).normalise(), Vec4::new(0.0, 0.0, 0.0, 0.0));
    /// ```
    pub fn normalise(self) -> Self {
        let length = self.dot(self).sqrt();
        if length > 0.0 { self / length } else { self }
    }

    /// Performs the perspective divide: returns `(x/w, y/w, z/w, 1)`.
    ///
    /// A vector with w = 0 (a direction, or a point at infinity) is returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec4::new(2.0, 4.0, 6.0, 2.0).homogenize(), Vec4::new(1.0, 2.0, 3.0, 1.0));
    ///
    /// let direction = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// assert_eq!(direction.homogenize(), direction);
    /// ```
    pub fn homogenize(self) -> Self {
        if self.w != 0.0 {
            Self::new(self.x / self.w, self.y / self.w, self.z / self.w, 1.0)
        } else {
            self
        }
    }

    /// Performs the perspective divide and drops w: returns `(x/w, y/w, z/w)`.
    ///
    /// For a clip-space position this gives normalised device coordinates. A vector with w = 0
    /// gives `(x, y, z)` without dividing.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let clip = Vec4::new(3.0, -3.0, 1.5, 3.0);
    /// assert_eq!(clip.homogenize_to_vec3(), Vec3::new(1.0, -1.0, 0.5));
    ///
    /// let direction = Vec4::new(1.0, 2.0, 3.0, 0.0);
    /// assert_eq!(direction.homogenize_to_vec3(), Vec3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn homogenize_to_vec3(self) -> Vec3 {
        if self.w != 0.0 {
            Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w)
        } else {
            Vec3::new(self.x, self.y, self.z)
        }
    }

    /// Returns the x, y, and z components, dropping w without dividing by it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
    /// assert_eq!(v.xyz(), Vec3::new(2.0, 4.0, 6.0));
    /// assert_eq!(v.xy(), Vec2::new(2.0, 4.0));
    /// ```
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Returns the x and y components, dropping z and w without dividing.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let screen = Vec4::new(320.0, 240.0, 0.5, 1.0);
    /// assert_eq!(screen.xy(), Vec2::new(320.0, 240.0));
    /// ```
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
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
impl AddAssign for Vec4 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
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
    use super::*;

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

        let result = a.lerp(b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn lerp_at_one_returns_other() {
        let a = Vec4::new(-2.0, 5.0, 3.0, 1.0);
        let b = Vec4::new(8.0, -1.0, 7.0, 0.0);

        let result = a.lerp(b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn lerp_at_halfway_returns_midpoint() {
        let a = Vec4::new(2.0, 4.0, 6.0, 0.0);
        let b = Vec4::new(6.0, 8.0, 10.0, 2.0);

        let result = a.lerp(b, 0.5);

        assert_vec4_approx_eq(result, Vec4::new(4.0, 6.0, 8.0, 1.0));
    }

    #[test]
    fn dot_returns_scalar_product() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(4.0, -5.0, 6.0, -2.0);

        let dot = a.dot(b);

        assert_eq!(dot, 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0 + 4.0 * (-2.0));
    }

    #[test]
    fn normalise_scales_non_zero_vector_to_unit_length() {
        let v = Vec4::new(3.0, 4.0, 0.0, 0.0);

        assert_vec4_approx_eq(v.normalise(), Vec4::new(0.6, 0.8, 0.0, 0.0));
    }

    #[test]
    fn normalise_zero_vector_keeps_zero_vector() {
        let v = Vec4::new(0.0, 0.0, 0.0, 0.0);

        assert_eq!(v.normalise(), Vec4::new(0.0, 0.0, 0.0, 0.0));
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
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);

        assert_vec4_approx_eq(v.homogenize(), Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn homogenize_does_nothing_when_w_is_zero() {
        let v = Vec4::new(2.0, 4.0, 6.0, 0.0);

        assert_eq!(v.homogenize(), Vec4::new(2.0, 4.0, 6.0, 0.0));
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
