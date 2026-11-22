use std::ops::{Add, Mul, Sub};

use crate::maths::{Vec3, Vec4};

/// A 4×4 `f32` matrix for affine and projective transforms.
///
/// # Conventions
///
/// - **Layout:** row-major, indexed `data[row][column]`. A translation sits in the last column,
///   `data[0..3][3]`. Use [`Mat4::transpose`] to hand the matrix to an API that expects
///   column-major data.
/// - **Column vectors:** a matrix transforms a vector as `m * v`. A product `a * b` applies `b`
///   first, so a model-view-projection matrix is `projection * view * model`.
/// - **Handedness:** right-handed. Rotations by a positive angle are counter-clockwise when
///   looking from the positive end of the axis towards the origin. Angles are in radians.
/// - **View and projection:** [`Mat4::look_at`] builds a view space looking down −z, and
///   [`Mat4::perspective`] and [`Mat4::orthographic`] are OpenGL-style: they map view space to
///   clip space whose z, after the divide by w, lies in `-1..=1` from the near plane to the far
///   plane.
///
/// # Operators
///
/// - `a + b`, `a - b`: element-wise.
/// - `m * s` with `s: f32`: every element scaled by `s`.
/// - `a * b` with `b: Mat4`: the matrix product, composing the transforms.
/// - `m * v` with `v: Vec4`: the matrix–vector product, transforming the column vector `v`. To
///   transform a [`Vec3`], extend it with [`Vec3::to_point4`] (affected by translation) or
///   [`Vec3::to_direction4`] (not affected).
///
/// # Examples
///
/// The order of a product decides which transform applies first:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let scale = Mat4::scaling(2.0, 2.0, 2.0);
/// let shift = Mat4::translation(1.0, 0.0, 0.0);
/// let point = Vec3::new(1.0, 1.0, 1.0).to_point4();
///
/// // Scale, then translate.
/// assert_eq!(shift * scale * point, Vec4::new(3.0, 2.0, 2.0, 1.0));
/// // Translate, then scale.
/// assert_eq!(scale * shift * point, Vec4::new(4.0, 2.0, 2.0, 1.0));
///
/// // Directions ignore translation.
/// let direction = Vec3::new(1.0, 0.0, 0.0).to_direction4();
/// assert_eq!(shift * direction, direction);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// The elements, indexed `data[row][column]`.
    pub data: [[f32; 4]; 4],
}
impl Mat4 {
    /// Creates a matrix from its elements, given row by row (`data[row][column]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let m = Mat4::new([
    ///     [1.0, 0.0, 0.0, 5.0],
    ///     [0.0, 1.0, 0.0, 6.0],
    ///     [0.0, 0.0, 1.0, 7.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ]);
    ///
    /// assert_eq!(m, Mat4::translation(5.0, 6.0, 7.0));
    /// ```
    pub fn new(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    /// Returns the identity matrix, which leaves every vector unchanged.
    ///
    /// It is the starting point for an object that is not transformed, and the neutral element of
    /// matrix multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let vec = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(Mat4::identity() * vec, vec);
    ///
    /// let m = Mat4::translation(1.0, 2.0, 3.0);
    /// assert_eq!(Mat4::identity() * m, m);
    /// ```
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

    /// Returns a matrix that translates points by `(x, y, z)`.
    ///
    /// Only vectors with a non-zero w are moved: a point (w = 1) moves by `(x, y, z)`, while a
    /// direction (w = 0) is unchanged.
    ///
    /// # Examples
    ///
    /// Placing a model at its position in the world:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let model = Mat4::translation(5.0, -2.0, 4.0);
    /// let vertex = Vec3::new(1.0, 2.0, 3.0).to_point4();
    ///
    /// assert_eq!(model * vertex, Vec4::new(6.0, 0.0, 7.0, 1.0));
    /// ```
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

    /// Returns a matrix that translates points by `position`.
    ///
    /// Equivalent to [`Mat4::translation`] with the vector's components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let spawn_position = Vec3::new(0.0, 1.5, -3.0);
    /// let model = Mat4::translation_vec(spawn_position);
    ///
    /// assert_eq!(model, Mat4::translation(0.0, 1.5, -3.0));
    /// assert_eq!(model * Vec3::ZERO.to_point4(), spawn_position.to_point4());
    /// ```
    pub fn translation_vec(position: Vec3) -> Self {
        Self::translation(position.x, position.y, position.z)
    }

    /// Returns a matrix that scales by `x`, `y`, and `z` along the respective axes, about the
    /// origin.
    ///
    /// A negative factor mirrors that axis; an odd number of negative factors reverses triangle
    /// winding, as the reflections do (see [`VertexShader`](crate::VertexShader)). A zero factor
    /// makes the matrix singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let stretch = Mat4::scaling(2.0, 3.0, -0.5);
    /// let point = Vec4::new(-2.0, 3.0, 4.0, 1.0);
    ///
    /// assert_eq!(stretch * point, Vec4::new(-4.0, 9.0, -2.0, 1.0));
    /// ```
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

    /// Returns a matrix that scales by the components of `scale` along the respective axes.
    ///
    /// Equivalent to [`Mat4::scaling`] with the vector's components.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let size = Vec3::new(4.0, 0.5, 1.0);
    /// let model = Mat4::scaling_vec(size);
    ///
    /// assert_eq!(model, Mat4::scaling(4.0, 0.5, 1.0));
    /// assert_eq!(model * Vec4::new(1.0, 2.0, 3.0, 1.0), Vec4::new(4.0, 1.0, 3.0, 1.0));
    /// ```
    pub fn scaling_vec(scale: Vec3) -> Self {
        Self::scaling(scale.x, scale.y, scale.z)
    }

    /// Returns a matrix that mirrors across the yz plane, negating x.
    ///
    /// Like every reflection, it reverses triangle winding, so front faces become back faces
    /// (see [`VertexShader`](crate::VertexShader)).
    ///
    /// # Examples
    ///
    /// Mirroring a model left to right:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let point = Vec4::new(2.0, -3.0, 4.0, 1.0);
    ///
    /// assert_eq!(Mat4::reflect_x() * point, Vec4::new(-2.0, -3.0, 4.0, 1.0));
    /// ```
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

    /// Returns a matrix that mirrors across the xz plane, negating y.
    ///
    /// Like every reflection, it reverses triangle winding, so front faces become back faces
    /// (see [`VertexShader`](crate::VertexShader)).
    ///
    /// # Examples
    ///
    /// Flipping a model upside down:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let point = Vec4::new(2.0, -3.0, 4.0, 1.0);
    ///
    /// assert_eq!(Mat4::reflect_y() * point, Vec4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
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

    /// Returns a matrix that mirrors across the xy plane, negating z.
    ///
    /// Like every reflection, it reverses triangle winding, so front faces become back faces
    /// (see [`VertexShader`](crate::VertexShader)).
    ///
    /// # Examples
    ///
    /// Turning a model back to front:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let point = Vec4::new(2.0, -3.0, 4.0, 1.0);
    ///
    /// assert_eq!(Mat4::reflect_z() * point, Vec4::new(2.0, -3.0, -4.0, 1.0));
    /// ```
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

    /// Returns a shear matrix, in which each coordinate gains multiples of the other two.
    ///
    /// Each parameter is named after the coordinate it changes followed by the coordinate that is
    /// added to it, so `xy` is how much of `y` is added to `x`. The result is:
    ///
    /// ```text
    /// x' = x + xy·y + xz·z
    /// y' = yx·x + y + yz·z
    /// z' = zx·x + zy·y + z
    /// ```
    ///
    /// # Examples
    ///
    /// Slanting a shape sideways in proportion to its height, as for an italic effect:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let slant = Mat4::shear(0.5, 0.0, 0.0, 0.0, 0.0, 0.0);
    ///
    /// assert_eq!(slant * Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 0.0, 0.0, 1.0));
    /// assert_eq!(slant * Vec4::new(0.0, 2.0, 0.0, 1.0), Vec4::new(1.0, 2.0, 0.0, 1.0));
    /// ```
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

    /// Returns a matrix that rotates by `angle` radians about the x axis.
    ///
    /// A positive angle turns +y towards +z (counter-clockwise looking from +x towards the
    /// origin).
    ///
    /// # Examples
    ///
    /// Tipping a model forwards:
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let rotated = Mat4::rotate_x(FRAC_PI_2) * Vec4::new(0.0, 1.0, 0.0, 1.0);
    ///
    /// assert!(rotated.y.abs() < 1e-6);
    /// assert!((rotated.z - 1.0).abs() < 1e-6);
    /// ```
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

    /// Returns a matrix that rotates by `angle` radians about the y axis.
    ///
    /// A positive angle turns +z towards +x (counter-clockwise looking from +y towards the
    /// origin).
    ///
    /// # Examples
    ///
    /// Turning a model to face right:
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let rotated = Mat4::rotate_y(FRAC_PI_2) * Vec4::new(0.0, 0.0, 1.0, 1.0);
    ///
    /// assert!((rotated.x - 1.0).abs() < 1e-6);
    /// assert!(rotated.z.abs() < 1e-6);
    /// ```
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

    /// Returns a matrix that rotates by `angle` radians about the z axis.
    ///
    /// A positive angle turns +x towards +y (counter-clockwise looking from +z towards the
    /// origin).
    ///
    /// # Examples
    ///
    /// Rolling a model on its side:
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let rotated = Mat4::rotate_z(FRAC_PI_2) * Vec4::new(1.0, 0.0, 0.0, 1.0);
    ///
    /// assert!(rotated.x.abs() < 1e-6);
    /// assert!((rotated.y - 1.0).abs() < 1e-6);
    /// ```
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

    /// Returns a matrix that rotates by the Euler angles in `rotation`, in radians.
    ///
    /// The rotations are about the fixed x, y, and z axes, applied in that order: the result is
    /// `rotate_z(rotation.z) * rotate_y(rotation.y) * rotate_x(rotation.x)`.
    ///
    /// # Examples
    ///
    /// Orienting a model from per-axis angles:
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let rotation = Vec3::new(0.0, FRAC_PI_2, FRAC_PI_2);
    /// let combined = Mat4::rotate_vec(rotation);
    /// let explicit = Mat4::rotate_z(FRAC_PI_2) * Mat4::rotate_y(FRAC_PI_2);
    /// assert_eq!(combined, explicit);
    ///
    /// // +z turns to +x about y, then +x turns to +y about z.
    /// let forward = combined * Vec4::new(0.0, 0.0, 1.0, 0.0);
    /// assert!((forward.y - 1.0).abs() < 1e-6);
    /// assert!(forward.x.abs() < 1e-6 && forward.z.abs() < 1e-6);
    /// ```
    pub fn rotate_vec(rotation: Vec3) -> Self {
        let rx = Self::rotate_x(rotation.x);
        let ry = Self::rotate_y(rotation.y);
        let rz = Self::rotate_z(rotation.z);

        rz * ry * rx
    }

    /// Returns a view matrix for a camera at `eye` looking towards `target`.
    ///
    /// The matrix transforms world space into a right-handed view space with the camera at the
    /// origin, looking down −z, with +x to its right and +y up. `up` gives the world's up
    /// direction; it need not be normalised or perpendicular to the view direction, and the
    /// camera's y axis is the component of `up` perpendicular to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let eye = Vec3::new(0.0, 0.0, 5.0);
    /// let view = Mat4::look_at(eye, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
    ///
    /// // The camera sits at the origin of view space and sees the target straight ahead, on -z.
    /// assert_eq!(view * eye.to_point4(), Vec4::new(0.0, 0.0, 0.0, 1.0));
    /// assert_eq!(view * Vec3::ZERO.to_point4(), Vec4::new(0.0, 0.0, -5.0, 1.0));
    /// ```
    ///
    /// # Known issues
    ///
    /// - If `eye` equals `target`, or `up` is parallel to the view direction, the result is a
    ///   singular matrix and no error is reported
    ///   ([#61](https://github.com/costott/cpu_rasteriser/issues/61)).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let n = (eye - target).normalise();
        let u = up.cross(n).normalise();
        let v = n.cross(u);

        Self::new([
            [u.x, u.y, u.z, -u.dot(eye)],
            [v.x, v.y, v.z, -v.dot(eye)],
            [n.x, n.y, n.z, -n.dot(eye)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Returns an OpenGL-style perspective projection matrix.
    ///
    /// `fov_y` is the vertical field of view in radians, `aspect` is width divided by height, and
    /// `near` and `far` are the positive distances to the clip planes along the view direction.
    ///
    /// It maps the view frustum of a right-handed view space looking down −z (as built by
    /// [`Mat4::look_at`]) to clip space, with `w = -z`. After the divide by w, visible points lie
    /// in `-1..=1` on each axis; z is −1 on the near plane and 1 on the far plane. See
    /// [`VertexShader`](crate::VertexShader) for how the pipeline uses clip space.
    ///
    /// Parameters with `near == far`, `near == 0`, `fov_y == 0`, or `aspect == 0` give non-finite
    /// elements; they are not checked.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let projection = Mat4::perspective(FRAC_PI_2, 16.0 / 9.0, 1.0, 10.0);
    ///
    /// let near = (projection * Vec4::new(0.0, 0.0, -1.0, 1.0)).homogenize_to_vec3();
    /// let far = (projection * Vec4::new(0.0, 0.0, -10.0, 1.0)).homogenize_to_vec3();
    ///
    /// assert!((near.z + 1.0).abs() < 1e-6);
    /// assert!((far.z - 1.0).abs() < 1e-6);
    /// ```
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let top = near * (fov_y / 2.0).tan();
        let bottom = -top;

        let right = top * aspect;
        let left = -right;

        Self::new([
            [
                (2.0 * near) / (right - left),
                0.0,
                (right + left) / (right - left),
                0.0,
            ],
            [
                0.0,
                (2.0 * near) / (top - bottom),
                (top + bottom) / (top - bottom),
                0.0,
            ],
            [
                0.0,
                0.0,
                -(far + near) / (far - near),
                -(2.0 * far * near) / (far - near),
            ],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }

    /// Returns an OpenGL-style orthographic projection matrix.
    ///
    /// It maps the view-space box `left..=right` × `bottom..=top` × `-near..=-far` to `-1..=1` on
    /// each axis: x = `left` maps to −1, y = `bottom` maps to −1, and z is −1 at `z = -near` and
    /// 1 at `z = -far`. w is unchanged, so no perspective divide takes place.
    ///
    /// Swapping `bottom` and `top` makes y point down, which suits drawing in pixel coordinates.
    /// Equal bounds on any axis give non-finite elements; they are not checked.
    ///
    /// # Examples
    ///
    /// A projection for 2D drawing in pixel coordinates with the origin at the top left:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let projection = Mat4::orthographic(0.0, 512.0, 256.0, 0.0, -1.0, 1.0);
    ///
    /// let top_left = projection * Vec4::new(0.0, 0.0, 0.0, 1.0);
    /// let bottom_right = projection * Vec4::new(512.0, 256.0, 0.0, 1.0);
    ///
    /// assert_eq!(top_left, Vec4::new(-1.0, 1.0, 0.0, 1.0));
    /// assert_eq!(bottom_right, Vec4::new(1.0, -1.0, 0.0, 1.0));
    /// ```
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self::new([
            [
                2.0 / (right - left),
                0.0,
                0.0,
                -(right + left) / (right - left),
            ],
            [
                0.0,
                2.0 / (top - bottom),
                0.0,
                -(top + bottom) / (top - bottom),
            ],
            [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Returns the inverse of the matrix.
    ///
    /// The inverse undoes the transform: `m.inverse() * (m * v)` is `v`, up to rounding.
    ///
    /// # Examples
    ///
    /// Getting a camera's world transform back from its view matrix:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let view = Mat4::translation(2.0, -3.0, 4.0);
    /// let point = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// assert_eq!(view.inverse() * (view * point), point);
    /// ```
    ///
    /// # Known issues
    ///
    /// - A singular matrix gives the identity, with no error reported. A matrix is treated as
    ///   singular when a pivot is below an absolute `1e-6`, so a valid matrix with small elements,
    ///   such as `Mat4::scaling(1e-7, 1.0, 1.0)`, "inverts" to the identity too
    ///   ([#61](https://github.com/costott/cpu_rasteriser/issues/61)).
    pub fn inverse(self) -> Self {
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

    /// Returns the transpose, swapping rows and columns.
    ///
    /// The transpose of a pure rotation is its inverse, and the transposed data is the
    /// column-major layout other graphics APIs expect.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let column_major = Mat4::translation(1.0, 2.0, 3.0).transpose();
    /// assert_eq!(column_major.data[3], [1.0, 2.0, 3.0, 1.0]);
    ///
    /// let rotation = Mat4::rotate_y(0.5);
    /// assert_eq!(rotation.transpose(), Mat4::rotate_y(-0.5));
    /// ```
    pub fn transpose(self) -> Self {
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
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

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
    use std::f32::consts::FRAC_PI_2;

    use super::*;

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
