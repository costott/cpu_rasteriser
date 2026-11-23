use crate::prelude::*;

/// A vertex shader: transforms each vertex of a draw to clip space and computes the varyings the
/// fragment shader receives.
///
/// A vertex shader is paired with a fragment shader in a [`Pipeline`] or [`SimdPipeline`]. It
/// defines three types:
///
/// - [`Vertex`](Self::Vertex), the element type of the vertex slice passed to [`DrawCall::new`];
/// - [`Uniforms`](Self::Uniforms), the data that is the same for every vertex of a draw, such as
///   the model, view, and projection matrices, passed by value to [`RenderPass::draw`] or
///   [`RenderPass::draw_simd`];
/// - [`Varyings`](Self::Varyings), the per-vertex values the rasteriser interpolates across each
///   triangle and passes to the fragment shader, such as colours, normals, and texture coordinates.
///
/// # Contract
///
/// - **When it is called:** during [`RenderPass::finish`], for the vertices the draw's indices
///   refer to. Don't rely on how many times it runs for a vertex, or in what order.
/// - **Output position:** [`shade`](Self::shade) returns the vertex's position in homogeneous clip
///   space. Triangles are clipped to `-w ≤ x, y, z ≤ w` and divided by `w`. The resulting
///   normalised device coordinates map `x` and `y` from `[-1, 1]` to the pass's [`Viewport`], with
///   `y` pointing up, and `z` in `[-1, 1]` is the fragment's depth, where smaller is nearer.
///   Triangles that are counter-clockwise in normalised device coordinates face the front. The
///   projection matrices of [`Mat4`] follow these conventions.
/// - **Output varyings:** the varyings are interpolated with perspective correction, so each
///   fragment receives the value at its position on the triangle's surface.
///
/// # Examples
///
/// A shader that transforms positions by a matrix uniform and passes the vertex colour on. Scaling
/// a quad that would fill the viewport by a half leaves it covering the middle of the target:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// #[derive(Clone)]
/// struct Vertex {
///     position: Vec3,
///     colour: Vec4,
/// }
///
/// struct Transform {
///     model_view_projection: Mat4,
/// }
///
/// #[derive(Interpolate)]
/// struct Varyings {
///     colour: Vec4,
/// }
///
/// struct TransformShader;
///
/// impl VertexShader for TransformShader {
///     type Vertex = Vertex;
///     type Uniforms = Transform;
///     type Varyings = Varyings;
///
///     fn shade(&self, vertex: Vertex, uniforms: &Transform) -> (Vec4, Varyings) {
///         let clip_position = uniforms.model_view_projection * vertex.position.to_point4();
///
///         (clip_position, Varyings { colour: vertex.colour })
///     }
/// }
/// #
/// # struct VertexColour;
/// #
/// # impl FragmentShader<Varyings> for VertexColour {
/// #     type Uniforms = ();
/// #
/// #     fn shade(&self, varyings: Varyings, _: &()) -> Colour {
/// #         Colour::from(varyings.colour)
/// #     }
/// # }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let red = Vec4::new(1.0, 0.0, 0.0, 1.0);
/// let vertices = [
///     Vertex { position: Vec3::new(-1.0, -1.0, 0.0), colour: red },
///     Vertex { position: Vec3::new(1.0, -1.0, 0.0), colour: red },
///     Vertex { position: Vec3::new(1.0, 1.0, 0.0), colour: red },
///     Vertex { position: Vec3::new(-1.0, 1.0, 0.0), colour: red },
/// ];
/// // Two counter-clockwise triangles.
/// let indices = [0, 1, 2, 0, 2, 3];
///
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = Pipeline::new(TransformShader, VertexColour);
///
/// let mut pass = renderer.begin_render_pass(
///     &mut target,
///     RenderPassDescriptor {
///         viewport: Viewport::full(&extent),
///         colour_load_op: LoadOp::Clear(Colour::BLACK),
///         depth_load_op: None,
///     },
/// );
/// pass.draw(
///     &pipeline,
///     DrawCall::new(&vertices, &indices, PrimitiveMode::TRIANGLES, ()),
///     Transform { model_view_projection: Mat4::scaling(0.5, 0.5, 1.0) },
/// );
/// pass.finish();
///
/// let pixels = target.pixels();
/// let pixel = |x: usize, y: usize| pixels[y * extent.width + x];
/// assert_eq!(pixel(1, 1), Colour::RED);
/// assert_eq!(pixel(2, 2), Colour::RED);
/// assert_eq!(pixel(0, 0), Colour::BLACK);
/// assert_eq!(pixel(3, 2), Colour::BLACK);
/// # Ok(())
/// # }
/// ```
pub trait VertexShader: Send + Sync + 'static {
    /// The type of the vertices the shader reads, the element type of a draw's vertex slice.
    ///
    /// The rasteriser may clone vertices before shading them.
    type Vertex: Clone;

    /// The data shared by every vertex of a draw, passed to [`RenderPass::draw`] or
    /// [`RenderPass::draw_simd`].
    type Uniforms: Send + Sync + 'static;

    /// The per-vertex values interpolated across each triangle and passed to the fragment shader.
    ///
    /// A [`SimdPipeline`] also requires them to implement [`SimdInterpolate`].
    type Varyings: Interpolate + Send + Sync + 'static;

    /// Returns the clip-space position of `vertex` and its varyings.
    ///
    /// See the [trait documentation](VertexShader#contract) for the conventions the position
    /// follows.
    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings);
}
