use crate::prelude::*;

/// A vertex in screen space, after the perspective divide and viewport mapping: a corner of a
/// [`Triangle2D`](crate::graphics::triangle::Triangle2D).
///
/// The varyings are stored divided by the clip-space `w`, alongside `1/w`, which is what
/// perspective-correct interpolation needs (see [`Interpolate`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RasterVertex<V>
where
    V: Interpolate,
{
    /// The position in pixels in the render target, with `y` pointing down.
    pub position: Vec2,
    /// The depth: normalised device `z`, as in the [`VertexShader`] contract.
    pub depth: f32,
    /// The reciprocal of the clip-space `w`.
    pub inv_w: f32,
    /// The vertex shader's varyings, multiplied by `inv_w`.
    pub varyings: V,
}
/// Unused constructor, kept for now.
impl<V: Interpolate> RasterVertex<V> {
    /// Creates a vertex from its fields; `varyings` must already be multiplied by `inv_w`.
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub fn new(position: Vec2, depth: f32, inv_w: f32, varyings: V) -> Self {
        Self {
            position,
            depth,
            inv_w,
            varyings,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipVertex<V>
where
    V: Interpolate,
{
    pub position: Vec4,
    pub varyings: V,
}
impl<V: Interpolate> ClipVertex<V> {
    #[expect(
        dead_code,
        reason = "unused internally; kept rather than deleted for now"
    )]
    pub fn new(position: Vec4, varyings: V) -> Self {
        Self { position, varyings }
    }

    pub fn interpolate(&self, other: &Self, t: f32) -> Self
    where
        V: Interpolate,
    {
        Self {
            position: self.position.lerp(other.position, t),
            varyings: self.varyings.interpolate(&other.varyings, t),
        }
    }
}

/// A vertex with the attributes of a typical model vertex: a position, a colour, a normal,
/// and texture coordinates.
///
/// It can serve as a vertex shader's [`Vertex`](crate::VertexShader::Vertex) type, and as the
/// vertex type of a [`Triangle3D`](crate::Triangle3D) whose face normal is wanted. The renderer
/// doesn't interpret the fields; their meaning is up to the shader that reads them.
///
/// See [`Triangle3D::normal`](crate::Triangle3D::normal) for an example.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjVertex {
    /// The position, usually in model space.
    pub position: Vec3,
    /// The vertex colour.
    pub colour: Colour,
    /// The surface normal, usually of unit length.
    pub normal: Vec3,
    /// The texture coordinates.
    pub uv: Vec2,
}
impl ObjVertex {
    /// Creates a vertex with only a position and a colour, with a zero normal and zero texture
    /// coordinates.
    ///
    /// # Examples
    ///
    /// For a vertex shader that reads only positions and colours:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let vertex = ObjVertex::simple(Vec3::new(0.0, 1.0, 0.0), Colour::RED);
    ///
    /// assert_eq!((vertex.normal, vertex.uv), (Vec3::ZERO, Vec2::ZERO));
    /// ```
    pub fn simple(position: Vec3, colour: Colour) -> Self {
        Self {
            position,
            colour,
            normal: Vec3::ZERO,
            uv: Vec2::ZERO,
        }
    }

    /// Creates a vertex from all its attributes.
    ///
    /// # Examples
    ///
    /// The top-right corner of a lit, textured quad facing +z:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let vertex = ObjVertex::new(
    ///     Vec3::new(1.0, 1.0, 0.0),
    ///     Colour::WHITE,
    ///     Vec3::new(0.0, 0.0, 1.0),
    ///     Vec2::new(1.0, 1.0),
    /// );
    ///
    /// assert_eq!(vertex.uv, Vec2::new(1.0, 1.0));
    /// ```
    pub fn new(position: Vec3, colour: Colour, normal: Vec3, uv: Vec2) -> Self {
        Self {
            position,
            colour,
            normal,
            uv,
        }
    }
}
