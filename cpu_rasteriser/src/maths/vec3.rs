use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use wide::f32x8;

use crate::prelude::*;

/// A 3D vector of `f32`, used for positions, directions, and normals.
///
/// Coordinates follow the crate's right-handed convention (see [`Mat4`]). To transform a `Vec3`
/// by a [`Mat4`], extend it to a [`Vec4`] with [`Vec3::to_point4`] or [`Vec3::to_direction4`].
///
/// The operators are component-wise: `a + b`, `a - b`, `a += b`, `a -= b`, `-v`, `v * s`, and
/// `v / s` with `s: f32`. There is no `Vec3 * Vec3`; use [`Vec3::dot`] or [`Vec3::cross`]. `From`
/// converts `(f32, f32, f32)`, `(usize, usize, usize)`, and `(i32, i32, i32)` tuples. `Vec3`
/// implements [`Interpolate`](crate::Interpolate), so it can be a varying.
///
/// # Examples
///
/// Computing a triangle's face normal:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let a = Vec3::new(0.0, 0.0, 0.0);
/// let b = Vec3::new(2.0, 0.0, 0.0);
/// let c = Vec3::new(0.0, 2.0, 0.0);
///
/// let normal = (b - a).cross(c - a).normalise();
/// assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));
/// ```
#[derive(Debug, Copy, PartialEq, Interpolate)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}
impl Vec3 {
    /// The zero vector, `(0, 0, 0)`.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// The vector with every component 1, `(1, 1, 1)`.
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// Creates a vector from its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let position = Vec3::new(1.0, 2.0, 3.0);
    /// assert_eq!((position.x, position.y, position.z), (1.0, 2.0, 3.0));
    /// ```
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns the dot product.
    ///
    /// For unit vectors it is the cosine of the angle between them, as used in diffuse lighting.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let normal = Vec3::new(0.0, 1.0, 0.0);
    /// let to_light = Vec3::new(0.0, 3.0, 4.0).normalise();
    /// let diffuse = normal.dot(to_light).max(0.0);
    ///
    /// assert_eq!(diffuse, 0.6);
    /// ```
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product, perpendicular to both vectors.
    ///
    /// The result follows the right-hand rule (`x × y = z`), and its length is the area of the
    /// parallelogram the two vectors span.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let x = Vec3::new(1.0, 0.0, 0.0);
    /// let y = Vec3::new(0.0, 1.0, 0.0);
    ///
    /// assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    /// assert_eq!(y.cross(x), Vec3::new(0.0, 0.0, -1.0));
    /// assert_eq!((x * 2.0).cross(y * 3.0).length(), 6.0);
    /// ```
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the Euclidean length.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec3::new(2.0, 3.0, 6.0).length(), 7.0);
    /// ```
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns the vector scaled to length 1, in the same direction.
    ///
    /// A zero-length vector is returned unchanged, as the zero vector, rather than producing NaN.
    /// There is no tolerance: any non-zero length is divided by.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec3::new(0.0, 3.0, 4.0).normalise(), Vec3::new(0.0, 0.6, 0.8));
    /// assert_eq!(Vec3::ZERO.normalise(), Vec3::ZERO);
    /// ```
    pub fn normalise(self) -> Vec3 {
        let length = self.length();
        if length > 0.0 { self / length } else { self }
    }

    /// Extends the vector to a homogeneous point, with w = 1.
    ///
    /// A point is moved by a translation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let point = Vec3::new(1.0, 2.0, 3.0).to_point4();
    ///
    /// assert_eq!(point, Vec4::new(1.0, 2.0, 3.0, 1.0));
    /// assert_eq!(Mat4::translation(1.0, 1.0, 1.0) * point, Vec4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    pub fn to_point4(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, 1.0)
    }

    /// Extends the vector to a homogeneous direction, with w = 0.
    ///
    /// A direction is rotated and scaled by a matrix but not translated.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let direction = Vec3::new(0.0, 0.0, -1.0).to_direction4();
    ///
    /// assert_eq!(direction, Vec4::new(0.0, 0.0, -1.0, 0.0));
    /// assert_eq!(Mat4::translation(1.0, 1.0, 1.0) * direction, direction);
    /// ```
    pub fn to_direction4(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, 0.0)
    }

    /// Returns the x and y components, dropping z.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Vec3::new(1.0, 2.0, 3.0).xy(), Vec2::new(1.0, 2.0));
    /// ```
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
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
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
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

/// Eight 3D vectors in struct-of-arrays form, one [`f32x8`] per component.
///
/// Lane `i` of `x`, `y`, and `z` together form the `i`th vector, as in
/// [`ColourSimd`](crate::ColourSimd). It is the [`Vec3`] counterpart for
/// [`FragmentShaderSimd`](crate::FragmentShaderSimd) code, for example to shade eight normals at
/// once.
///
/// The operators work per lane: `a + b`, `a - b`, `-v`, and `v * s` with `s: f32x8` scaling each
/// lane by the matching lane of `s`.
///
/// # Examples
///
/// Diffuse lighting for eight normals against one light direction:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// let normals = Vec3Simd::new(f32x8::splat(0.0), f32x8::splat(2.0), f32x8::splat(0.0));
/// let to_light = Vec3Simd::new(f32x8::splat(0.0), f32x8::splat(1.0), f32x8::splat(0.0));
///
/// let diffuse = normals.normalise().dot(to_light).max(f32x8::splat(0.0));
/// assert!(diffuse.to_array().iter().all(|d| (d - 1.0).abs() < 1e-3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Simd {
    /// The x component of each lane.
    pub x: f32x8,
    /// The y component of each lane.
    pub y: f32x8,
    /// The z component of each lane.
    pub z: f32x8,
}

impl Vec3Simd {
    /// Creates eight vectors from their per-lane components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    ///
    /// let lanes = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    /// let v = Vec3Simd::new(lanes, f32x8::splat(0.0), f32x8::splat(1.0));
    ///
    /// assert_eq!(v.x.to_array()[3], 3.0);
    /// assert_eq!(v.z.to_array()[3], 1.0);
    /// ```
    #[inline(always)]
    pub fn new(x: f32x8, y: f32x8, z: f32x8) -> Self {
        Self { x, y, z }
    }

    /// Returns the dot product of each pair of lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    ///
    /// let a = Vec3Simd::new(f32x8::splat(1.0), f32x8::splat(2.0), f32x8::splat(3.0));
    /// let b = Vec3Simd::new(f32x8::splat(4.0), f32x8::splat(5.0), f32x8::splat(6.0));
    ///
    /// assert_eq!(a.dot(b), f32x8::splat(32.0));
    /// ```
    #[inline(always)]
    pub fn dot(self, other: Self) -> f32x8 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns each lane's vector scaled to approximately length 1.
    ///
    /// The result is close to, but not exactly, unit length: the reciprocal of the length is
    /// approximate, with a relative error of at most 1.5 × 2⁻¹² on x86 targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    ///
    /// let v = Vec3Simd::new(f32x8::splat(0.0), f32x8::splat(3.0), f32x8::splat(4.0)).normalise();
    /// assert!(v.z.to_array().iter().all(|z| (z - 0.8).abs() < 1e-3));
    /// ```
    ///
    /// # Known issues
    ///
    /// - A zero-length lane becomes NaN in every component, where [`Vec3::normalise`] returns the
    ///   zero vector. Masked-out lanes in a SIMD shader, whose varyings are extrapolated, can hit
    ///   this ([#62](https://github.com/costott/cpu_rasteriser/issues/62)).
    #[inline(always)]
    pub fn normalise(self) -> Self {
        let inv_length = self.dot(self).sqrt().recip();

        Self {
            x: self.x * inv_length,
            y: self.y * inv_length,
            z: self.z * inv_length,
        }
    }
}

impl std::ops::Add for Vec3Simd {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3Simd {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Vec3Simd {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<f32x8> for Vec3Simd {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32x8) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn interpolate_at_zero_returns_self() {
        let a = Vec3::new(-2.0, 5.0, 3.0);
        let b = Vec3::new(8.0, -1.0, 7.0);

        let result = a.interpolate(&b, 0.0);

        assert_eq!(result, a);
    }

    #[test]
    fn interpolate_at_one_returns_other() {
        let a = Vec3::new(-2.0, 5.0, 3.0);
        let b = Vec3::new(8.0, -1.0, 7.0);

        let result = a.interpolate(&b, 1.0);

        assert_eq!(result, b);
    }

    #[test]
    fn interpolate_at_halfway_returns_midpoint() {
        let a = Vec3::new(2.0, 4.0, 6.0);
        let b = Vec3::new(6.0, 8.0, 10.0);

        let result = a.interpolate(&b, 0.5);

        assert_vec3_approx_eq(result, Vec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn dot_returns_scalar_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);

        let dot = a.dot(b);

        assert_eq!(dot, 12.0);
    }

    #[test]
    fn cross_of_basis_vectors_matches_right_handed_rule() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);

        let z = x.cross(y);

        assert_eq!(z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cross_is_anti_commutative() {
        let a = Vec3::new(2.0, -1.0, 3.0);
        let b = Vec3::new(0.0, 4.0, -2.0);

        let ab = a.cross(b);
        let ba = b.cross(a);

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
