use crate::prelude::*;

/// A fragment shader for a scalar [`Pipeline`]: computes the colour of each pixel a triangle
/// covers.
///
/// `V` is the varying type, the [`Varyings`](VertexShader::Varyings) of the vertex shader it is
/// paired with in a [`Pipeline`]. A shader may implement the trait for several varying types.
///
/// # Contract
///
/// - **Inputs:** [`shade`](Self::shade) receives the vertex shader's varyings, interpolated with
///   perspective correction to the centre of the pixel, and a shared reference to the draw's
///   [`Uniforms`](Self::Uniforms).
/// - **Uniforms** are the data that is the same for every fragment of a draw, such as a material,
///   lights, or a time. They are passed to [`DrawCall::new`] and shared, read-only, by every
///   fragment of that draw.
/// - **When it is called:** during [`RenderPass::finish`], once for each pixel whose centre lies
///   inside a triangle of the draw and inside the pass's viewport. If the pipeline's
///   [`DepthState`] tests depth, fragments that fail the test are discarded before shading.
///   Don't rely on the order in which fragments are shaded; their results reach the target in
///   the order the draws were submitted (see [`RenderPass`]). Fragments may be shaded
///   concurrently, so the shader and its uniforms must be `Send + Sync + 'static`.
/// - **Output:** the fragment's [`Colour`]. It is written to the render target, or combined with
///   the colour already there if the pipeline has a [`BlendState`].
///
/// For the 8-wide path of a [`SimdPipeline`], implement [`FragmentShaderSimd`] instead. The two
/// traits are independent; a shader may implement either or both.
///
/// # Examples
///
/// A shader that darkens the vertex colour by an ambient light level given as a uniform, drawn
/// over a whole render target:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// #[derive(Interpolate)]
/// struct Varyings {
///     colour: Vec4,
/// }
///
/// struct Lighting {
///     ambient: f32,
/// }
///
/// struct AmbientShader;
///
/// impl FragmentShader<Varyings> for AmbientShader {
///     type Uniforms = Lighting;
///
///     fn shade(&self, varyings: Varyings, uniforms: &Lighting) -> Colour {
///         Colour::from(varyings.colour) * uniforms.ambient
///     }
/// }
/// #
/// # #[derive(Clone)]
/// # struct Vertex {
/// #     position: Vec3,
/// #     colour: Vec4,
/// # }
/// #
/// # struct PassThrough;
/// #
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vertex;
/// #     type Uniforms = ();
/// #     type Varyings = Varyings;
/// #
/// #     fn shade(&self, vertex: Vertex, _: &()) -> (Vec4, Varyings) {
/// #         (vertex.position.to_point4(), Varyings { colour: vertex.colour })
/// #     }
/// # }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let orange = Vec4::new(1.0, 0.5, 0.0, 1.0);
/// // One triangle that covers the whole viewport.
/// let vertices = [
///     Vertex { position: Vec3::new(-1.0, -1.0, 0.0), colour: orange },
///     Vertex { position: Vec3::new(3.0, -1.0, 0.0), colour: orange },
///     Vertex { position: Vec3::new(-1.0, 3.0, 0.0), colour: orange },
/// ];
///
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = Pipeline::new(PassThrough, AmbientShader);
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
///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Lighting { ambient: 0.5 }),
///     (),
/// );
/// pass.finish();
///
/// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::new(0.5, 0.25, 0.0, 1.0)));
/// # Ok(())
/// # }
/// ```
///
/// # Known issues
///
/// - A panic in [`shade`](Self::shade) doesn't reach the caller: [`RenderPass::finish`] returns
///   normally, and the region of the target around the pixels it was shading keeps its previous
///   contents, including what other draws in the pass drew there
///   ([#58](https://github.com/costott/cpu_rasteriser/issues/58)).
pub trait FragmentShader<V>: Send + Sync + 'static
where
    V: Interpolate,
{
    /// The data shared by every fragment of a draw, passed to [`DrawCall::new`].
    type Uniforms: Send + Sync + 'static;

    /// Returns the colour of the fragment with the given interpolated `varyings`.
    ///
    /// See the [trait documentation](FragmentShader#contract) for when it is called and what the
    /// colour is used for.
    fn shade(&self, varyings: V, uniforms: &Self::Uniforms) -> Colour;
}

/// A fragment shader for a [`SimdPipeline`]: computes the colours of eight pixels at once.
///
/// `V` is the varying type, the [`Varyings`](VertexShader::Varyings) of the vertex shader it is
/// paired with in a [`SimdPipeline`]; it must implement [`SimdInterpolate`].
/// [`shade_simd`](Self::shade_simd) receives `V::Simd`, which holds the varyings of eight
/// adjacent pixels with one [`f32x8`](crate::wide::f32x8) per component, and returns their
/// colours as a [`ColourSimd`].
///
/// # Contract
///
/// The contract is that of [`FragmentShader`], eight pixels at a time:
///
/// - **When it is called:** during [`RenderPass::finish`], for groups of eight horizontally
///   adjacent pixels of one row, as long as at least one pixel of the group is covered by a
///   triangle of the draw and passes the depth test. Groups aren't aligned to multiples of eight.
/// - **Lanes:** lane `i` of every input and of the returned [`ColourSimd`] belongs to the pixel `i`
///   places to the right of the group's first pixel. Only the colours of lanes whose pixel is
///   covered and passes the depth test are written; the other lanes are masked off.
/// - **Masked lanes still run.** The shader is called on all eight lanes, and a masked lane can
///   hold varyings extrapolated beyond the triangle. It must therefore accept any input without
///   panicking, for example by clamping texture coordinates before using them as indices.
/// - **Uniforms** are passed to [`DrawCall::new`] and shared by every fragment of the draw.
///   Fragments may be shaded concurrently, so [`RenderPass::draw_simd`] requires the shader to be
///   `Send + Sync + 'static`, and the trait requires it of the uniforms.
///
/// The traits are independent; a shader may implement either or both.
///
/// # Examples
///
/// A shader that scales the vertex colour by a brightness uniform. The derive of
/// [`SimdInterpolate`] generates `VaryingsSimd`, whose private fields are readable in this module:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// #[derive(Interpolate, SimdInterpolate)]
/// struct Varyings {
///     colour: Vec4,
/// }
///
/// struct Brightness(f32);
///
/// struct BrightnessShader;
///
/// impl FragmentShaderSimd<Varyings> for BrightnessShader {
///     type Uniforms = Brightness;
///
///     fn shade_simd(&self, varyings: VaryingsSimd, uniforms: &Brightness) -> ColourSimd {
///         let [r, g, b, a] = varyings.colour;
///         let brightness = f32x8::splat(uniforms.0);
///
///         ColourSimd { r: r * brightness, g: g * brightness, b: b * brightness, a }
///     }
/// }
/// #
/// # #[derive(Clone)]
/// # struct Vertex {
/// #     position: Vec3,
/// #     colour: Vec4,
/// # }
/// #
/// # struct PassThrough;
/// #
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vertex;
/// #     type Uniforms = ();
/// #     type Varyings = Varyings;
/// #
/// #     fn shade(&self, vertex: Vertex, _: &()) -> (Vec4, Varyings) {
/// #         (vertex.position.to_point4(), Varyings { colour: vertex.colour })
/// #     }
/// # }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let white = Vec4::new(1.0, 1.0, 1.0, 1.0);
/// // One triangle that covers the whole viewport.
/// let vertices = [
///     Vertex { position: Vec3::new(-1.0, -1.0, 0.0), colour: white },
///     Vertex { position: Vec3::new(3.0, -1.0, 0.0), colour: white },
///     Vertex { position: Vec3::new(-1.0, 3.0, 0.0), colour: white },
/// ];
///
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(16, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = SimdPipeline::new(PassThrough, BrightnessShader);
///
/// let mut pass = renderer.begin_render_pass(
///     &mut target,
///     RenderPassDescriptor {
///         viewport: Viewport::full(&extent),
///         colour_load_op: LoadOp::Clear(Colour::BLACK),
///         depth_load_op: None,
///     },
/// );
/// pass.draw_simd(
///     &pipeline,
///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Brightness(0.5)),
///     (),
/// );
/// pass.finish();
///
/// // SIMD varyings can differ slightly from a scalar pipeline's (see Known issues).
/// assert!(target.pixels().iter().all(|pixel| (pixel.r - 0.5).abs() < 1e-3));
/// # Ok(())
/// # }
/// ```
///
/// # Known issues
///
/// - With [`BlendState`], a [`SimdPipeline`] leaves different alpha in the target from a
///   [`Pipeline`] with an equivalent scalar shader
///   ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
/// - The varyings can differ from those a [`Pipeline`] gives the same pixel by more than expected;
///   the cause hasn't been investigated
///   ([#21](https://github.com/costott/cpu_rasteriser/issues/21)).
/// - A panic in [`shade_simd`](Self::shade_simd) doesn't reach the caller: [`RenderPass::finish`]
///   returns normally, and the region of the target around the pixels it was shading keeps its
///   previous contents, including what other draws in the pass drew there
///   ([#58](https://github.com/costott/cpu_rasteriser/issues/58)).
pub trait FragmentShaderSimd<V>
where
    V: SimdInterpolate,
{
    /// The data shared by every fragment of a draw, passed to [`DrawCall::new`].
    type Uniforms: Send + Sync + 'static;

    /// Returns the colours of the eight fragments whose interpolated varyings are `varyings`, one
    /// lane per fragment.
    ///
    /// See the [trait documentation](FragmentShaderSimd#contract) for when it is called and which
    /// lanes are used.
    fn shade_simd(&self, varyings: V::Simd, uniforms: &Self::Uniforms) -> ColourSimd;
}
