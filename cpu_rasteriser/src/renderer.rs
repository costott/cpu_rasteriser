use std::sync::Arc;

use threadpool::ThreadPool;
use wide::f32x8;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::triangle::Triangle2D;
use crate::prelude::*;

/// The entry point for rendering: begins render passes that draw into a [`RenderTarget`].
///
/// The target belongs to the caller, so one renderer can serve any number of targets. Create the
/// renderer once and keep it for the lifetime of the application.
///
/// Rendering happens in render passes:
///
/// 1. [`Renderer::begin_render_pass`] binds a target and returns a [`RenderPass`].
/// 2. [`RenderPass::draw`] (for a [`Pipeline`]) and [`RenderPass::draw_simd`] (for a
///    [`SimdPipeline`]) queue draw calls.
/// 3. [`RenderPass::finish`] renders the queued draw calls into the target.
///
/// # Examples
///
/// Draws a red triangle over the bottom-left half of a 4×4 target, then reads the pixels back:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// /// Passes clip-space positions straight through.
/// struct PassThrough;
/// impl VertexShader for PassThrough {
///     type Vertex = Vec4;
///     type Uniforms = ();
///     type Varyings = f32;
///
///     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
///         (position, 0.0)
///     }
/// }
///
/// /// Fills every fragment with the draw call's colour.
/// struct FlatColour;
/// impl FragmentShader<f32> for FlatColour {
///     type Uniforms = Colour;
///
///     fn shade(&self, _: f32, colour: &Colour) -> Colour {
///         *colour
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = Pipeline::new(PassThrough, FlatColour);
///
/// let vertices = [
///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
///     Vec4::new(1.0, -1.0, 0.0, 1.0),
///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
/// ];
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
///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Colour::RED),
///     (),
/// );
/// pass.finish();
///
/// let pixels = target.pixels();
/// assert_eq!(pixels[3 * 4], Colour::RED); // bottom-left corner
/// assert_eq!(pixels[3], Colour::BLACK); // top-right corner
/// # Ok(())
/// # }
/// ```
pub struct Renderer {
    thread_pool: ThreadPool,
    tile_pool: TilePool,
}
impl Renderer {
    /// Creates a renderer that makes use of the available CPU parallelism.
    ///
    /// # Errors
    ///
    /// Returns the error from [`std::thread::available_parallelism`] if the amount of available
    /// parallelism can't be determined.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let renderer = Renderer::new()?;
    /// # let _ = renderer;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
            tile_pool: TilePool::default(),
        })
    }

    /// Begins a render pass that draws into `target`.
    ///
    /// The descriptor's load operations set each attachment's starting contents for the pass:
    /// cleared, or the target's existing contents. With a `depth_load_op` of `None` the pass has
    /// no depth buffer, and a depth buffer the target already has is left unchanged.
    ///
    /// The pass borrows the renderer and the target mutably, so only one pass can be active at a
    /// time and the target can't be read until the pass ends. Nothing is written to the target,
    /// clears included, until [`RenderPass::finish`]; dropping the pass without finishing it
    /// leaves the target unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `descriptor.depth_load_op` is `Some` and `target` has no depth buffer (it wasn't
    /// created with [`RenderTarget::with_depth`]).
    ///
    /// # Examples
    ///
    /// Clears a target with a depth buffer and finishes the pass with no draws:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut renderer = Renderer::new()?;
    /// let extent = Extent::new(8, 8);
    /// let mut target = RenderTarget::new(extent).with_depth();
    ///
    /// let pass = renderer.begin_render_pass(
    ///     &mut target,
    ///     RenderPassDescriptor {
    ///         viewport: Viewport::full(&extent),
    ///         colour_load_op: LoadOp::Clear(Colour::BLUE),
    ///         depth_load_op: Some(LoadOp::Clear(1.0)),
    ///     },
    /// );
    /// pass.finish();
    ///
    /// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::BLUE));
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin_render_pass<'renderer, 'target>(
        &'renderer mut self,
        target: &'target mut RenderTarget,
        descriptor: RenderPassDescriptor,
    ) -> RenderPass<'renderer, 'target> {
        let tile_binner = TileBinner::new(
            target,
            &descriptor.colour_load_op,
            &descriptor.depth_load_op,
            &mut self.tile_pool,
        );

        RenderPass {
            renderer: self,
            render_target: target,
            descriptor,
            queued_draws: Vec::new(),
            tile_binner,
        }
    }

    fn render_tiles(&mut self, tile_binner: TileBinner, render_target: &mut RenderTarget) {
        let (tx, rx) = std::sync::mpsc::channel();

        for tile in tile_binner.tiles {
            let tx = tx.clone();

            self.thread_pool.execute(move || {
                tx.send(render_tile(tile)).unwrap();
            });
        }

        drop(tx);

        for result in rx {
            self.merge_tile(result, render_target);
        }

        self.thread_pool.join();
    }

    fn merge_tile(&mut self, result: TileResult, render_target: &mut RenderTarget) {
        let TileResult {
            bounds,
            framebuffer,
            depthbuffer,
        } = result;

        let (x, y) = (bounds.min_x as usize, bounds.min_y as usize);
        let size = framebuffer.extent();

        render_target
            .framebuffer
            .blit_from((x, y), &framebuffer, (0, 0), size);

        if let (Some(dst), Some(src)) = (render_target.depthbuffer.as_mut(), depthbuffer.as_ref()) {
            dst.blit_from((x, y), src, (0, 0), size);
        }

        self.tile_pool.recycle(framebuffer, depthbuffer);
    }
}
impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("threads", &self.thread_pool.max_count())
            .finish_non_exhaustive()
    }
}

fn render_tile(tile: Tile) -> TileResult {
    let mut tile = tile;

    for command in tile.commands {
        command.rasterise(
            &mut tile.framebuffer,
            tile.depthbuffer.as_mut(),
            tile.bounds,
        );
    }

    TileResult {
        bounds: tile.bounds,
        framebuffer: tile.framebuffer,
        depthbuffer: tile.depthbuffer,
    }
}

/// A tile result is the output of rendering a single tile, used for merging into the main framebuffer.
struct TileResult {
    bounds: Rect,
    framebuffer: FrameBuffer,
    depthbuffer: Option<DepthBuffer>,
}

/// The destination of a render pass: a colour buffer and an optional depth buffer.
///
/// Render into a target with [`Renderer::begin_render_pass`], then read the result back with
/// [`RenderTarget::pixels`] or [`RenderTarget::pixels_u32`] (or their allocation-free forms).
/// Pixels are addressed from the top-left corner, row by row. The colour buffer holds
/// [`Colour`]s.
///
/// # Examples
///
/// A target for 3D rendering needs a depth buffer for depth testing:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let target = RenderTarget::new(Extent::new(320, 180)).with_depth();
///
/// assert_eq!(target.extent(), Extent::new(320, 180));
/// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::BLACK));
/// ```
pub struct RenderTarget {
    framebuffer: FrameBuffer,
    depthbuffer: Option<DepthBuffer>,
}
impl RenderTarget {
    /// Creates a target of the given size with a colour buffer and no depth buffer.
    ///
    /// Every pixel starts as [`Colour::BLACK`]. Add a depth buffer with
    /// [`RenderTarget::with_depth`].
    ///
    /// # Examples
    ///
    /// See [`RenderTarget`].
    pub fn new(extent: Extent) -> Self {
        Self {
            framebuffer: FrameBuffer::new(extent.width, extent.height),
            depthbuffer: None,
        }
    }

    /// Adds a depth buffer of the same size, with every depth set to 1.0 (the far plane).
    ///
    /// A target needs a depth buffer for any pass whose
    /// [`RenderPassDescriptor::depth_load_op`] is `Some`.
    ///
    /// # Examples
    ///
    /// See [`RenderTarget`].
    pub fn with_depth(mut self) -> Self {
        self.depthbuffer = Some(DepthBuffer::new(
            self.framebuffer.width(),
            self.framebuffer.height(),
        ));
        self
    }

    /// Returns a copy of the pixels, row by row from the top-left corner.
    ///
    /// The result has `width * height` elements, and the pixel at `(x, y)` is at index
    /// `y * width + x`. This allocates on every call; [`RenderTarget::write_pixels`] reuses a
    /// buffer instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    /// let pixels = target.pixels();
    ///
    /// assert_eq!(pixels.len(), 4 * 2);
    /// assert_eq!(pixels[2 * 4 - 1], Colour::BLACK); // the bottom-right pixel
    /// ```
    pub fn pixels(&self) -> Vec<Colour> {
        self.framebuffer.pixels_compact()
    }

    /// Writes the pixels into `out`, row by row, without allocating.
    ///
    /// Elements of `out` past `width * height` are left untouched. This is the allocation-free
    /// form of [`RenderTarget::pixels`], for reusing one buffer across reads.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `width * height`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    /// let mut pixels = vec![Colour::WHITE; 4 * 2];
    ///
    /// target.write_pixels(&mut pixels);
    ///
    /// assert_eq!(pixels, target.pixels());
    /// ```
    pub fn write_pixels(&self, out: &mut [Colour]) {
        self.framebuffer.write_pixels(out);
    }

    /// Returns a copy of the pixels packed into `u32`s by [`Colour::to_u32`], row by row.
    ///
    /// The result has `width * height` elements, laid out as in [`RenderTarget::pixels`]. This
    /// allocates on every call; [`RenderTarget::write_pixels_u32`] reuses a buffer instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    ///
    /// assert_eq!(target.pixels_u32(), vec![Colour::BLACK.to_u32(); 4 * 2]);
    /// ```
    pub fn pixels_u32(&self) -> Vec<u32> {
        self.framebuffer.pixels_u32()
    }

    /// Writes the pixels into `out`, packed by [`Colour::to_u32`], row by row, without allocating.
    ///
    /// Elements of `out` past `width * height` are left untouched. This is the allocation-free
    /// form of [`RenderTarget::pixels_u32`], for reusing one buffer across reads.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `width * height`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    /// let mut presentation = vec![0; 4 * 2];
    ///
    /// target.write_pixels_u32(&mut presentation);
    ///
    /// assert_eq!(presentation, target.pixels_u32());
    /// ```
    pub fn write_pixels_u32(&self, out: &mut [u32]) {
        self.framebuffer.write_pixels_u32(out);
    }

    /// Returns the width in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(RenderTarget::new(Extent::new(4, 2)).width(), 4);
    /// ```
    pub fn width(&self) -> usize {
        self.framebuffer.width()
    }

    /// Returns the height in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(RenderTarget::new(Extent::new(4, 2)).height(), 2);
    /// ```
    pub fn height(&self) -> usize {
        self.framebuffer.height()
    }

    /// Returns the size in pixels.
    ///
    /// This is the usual source of a viewport covering the whole target, through
    /// [`Viewport::full`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    ///
    /// assert_eq!(Viewport::full(&target.extent()), Viewport::new(0, 0, 4, 2));
    /// ```
    pub fn extent(&self) -> Extent {
        self.framebuffer.extent()
    }

    /// Resizes the colour buffer, and the depth buffer if there is one, to `extent`.
    ///
    /// Existing contents are discarded: afterwards the pixels and depths are unspecified until a
    /// render pass clears or draws over them, so the next pass should use [`LoadOp::Clear`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let mut target = RenderTarget::new(Extent::new(640, 360)).with_depth();
    ///
    /// target.resize(Extent::new(1280, 720));
    ///
    /// assert_eq!(target.extent(), Extent::new(1280, 720));
    /// ```
    pub fn resize(&mut self, extent: Extent) {
        self.framebuffer.resize(extent.width, extent.height);
        if let Some(depthbuffer) = &mut self.depthbuffer {
            depthbuffer.resize(extent.width, extent.height);
        }
    }
}
impl std::fmt::Debug for RenderTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderTarget")
            .field("extent", &self.extent())
            .field("has_depth_buffer", &self.depthbuffer.is_some())
            .finish_non_exhaustive()
    }
}

/// The configuration of a render pass, passed to [`Renderer::begin_render_pass`].
///
/// # Examples
///
/// A typical 3D pass clears colour and depth; [`LoadOp`] shows passes that keep the previous
/// contents.
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let target = RenderTarget::new(Extent::new(640, 360)).with_depth();
///
/// let descriptor = RenderPassDescriptor {
///     viewport: Viewport::full(&target.extent()),
///     colour_load_op: LoadOp::Clear(Colour::BLACK),
///     depth_load_op: Some(LoadOp::Clear(1.0)),
/// };
/// # let _ = descriptor;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPassDescriptor {
    /// The region of the target, in pixels, that normalised device coordinates map onto.
    pub viewport: Viewport,
    /// How the target's colour buffer is initialised at the start of the pass.
    pub colour_load_op: LoadOp<Colour>,
    /// How the target's depth buffer is initialised, or `None` for a pass without depth.
    ///
    /// `Some` requires a target with a depth buffer ([`RenderTarget::with_depth`]). With `None`,
    /// pipelines drawn in the pass must not test or write depth, and the target's depth buffer, if
    /// it has one, is left unchanged. For ordinary depth testing, clear with `LoadOp::Clear(1.0)`,
    /// the far plane (see [`DepthState`]).
    pub depth_load_op: Option<LoadOp<f32>>,
}

/// How a render pass initialises an attachment (the colour or depth buffer) before drawing.
///
/// `T` is the attachment's element type: [`Colour`] for the colour buffer and `f32` for depth.
///
/// # Examples
///
/// Multipass rendering: a second pass with [`LoadOp::Load`] keeps what the first pass produced.
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
///
/// for colour_load_op in [LoadOp::Clear(Colour::GREEN), LoadOp::Load] {
///     let pass = renderer.begin_render_pass(
///         &mut target,
///         RenderPassDescriptor {
///             viewport: Viewport::full(&extent),
///             colour_load_op,
///             depth_load_op: None,
///         },
///     );
///     pass.finish();
/// }
///
/// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::GREEN));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp<T> {
    /// Keeps the attachment's existing contents.
    Load,
    /// Fills the attachment with the given value.
    Clear(T),
}

/// A pair of shaders and the fixed-function state used to draw with them, shading one fragment at
/// a time.
///
/// A pipeline contains:
///
/// - a [`VertexShader`], which transforms each vertex into clip space and produces the varyings
///   interpolated across the triangle;
/// - a [`FragmentShader`], which calculates the colour of each rasterised fragment;
/// - fixed-function state: back-face culling ([`CullingMode`]), blending ([`BlendState`]), and
///   depth testing ([`DepthState`]).
///
/// Create a pipeline once and reuse it across draw calls and render passes; draw with it through
/// [`RenderPass::draw`]. For a shader that shades eight fragments at once, use [`SimdPipeline`].
///
/// # Examples
///
/// A pipeline for opaque 3D geometry, with depth testing and both sides of each triangle drawn:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # struct PassThrough;
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vec4;
/// #     type Uniforms = ();
/// #     type Varyings = f32;
/// #     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
/// #         (position, 0.0)
/// #     }
/// # }
/// # struct FlatColour;
/// # impl FragmentShader<f32> for FlatColour {
/// #     type Uniforms = Colour;
/// #     fn shade(&self, _: f32, colour: &Colour) -> Colour {
/// #         *colour
/// #     }
/// # }
/// let pipeline = Pipeline::new(PassThrough, FlatColour)
///     .with_culling_mode(CullingMode::None)
///     .with_depth_state(DepthState::DEFAULT);
/// # let _ = pipeline;
/// ```
///
/// [`Renderer`] shows a complete draw with the shaders hidden here.
pub struct Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    vertex_shader: VS,
    fragment_shader: Arc<FS>,
    state: PipelineState,
}
impl<VS, FS> Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    /// Creates a pipeline from a vertex shader and a fragment shader.
    ///
    /// The pipeline starts with [`CullingMode::BackFace`], no blending (fragment colours replace
    /// the target's), and [`DepthState::DISABLED`]. The builder methods change these.
    ///
    /// # Examples
    ///
    /// See [`Pipeline`].
    pub fn new(vertex_shader: VS, fragment_shader: FS) -> Self {
        Self {
            vertex_shader,
            fragment_shader: Arc::new(fragment_shader),
            state: PipelineState {
                culling_mode: CullingMode::BackFace,
                blend_state: None,
                depth_state: DepthState::DISABLED,
            },
        }
    }

    /// Sets which triangles are culled by their winding before rasterisation.
    ///
    /// # Examples
    ///
    /// See [`CullingMode`].
    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.state.culling_mode = culling_mode;
        self
    }

    /// Sets how fragment colours are combined with the colours already in the target.
    ///
    /// # Examples
    ///
    /// See [`BlendState`].
    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.state.blend_state = Some(blend_state);
        self
    }

    /// Removes the blend state, so fragment colours replace the colours already in the target.
    ///
    /// This is the default.
    ///
    /// # Examples
    ///
    /// See [`BlendState`].
    pub fn without_blend_state(mut self) -> Self {
        self.state.blend_state = None;
        self
    }

    /// Sets whether fragments are depth tested and whether they write their depth.
    ///
    /// A pipeline that tests or writes depth can only be drawn in a pass with a depth buffer.
    ///
    /// # Examples
    ///
    /// See [`DepthState`].
    pub fn with_depth_state(mut self, depth_state: DepthState) -> Self {
        self.state.depth_state = depth_state;
        self
    }
}
impl<VS, FS> std::fmt::Debug for Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// A pair of shaders and the fixed-function state used to draw with them, shading eight fragments
/// at a time.
///
/// The fragment shader receives eight fragments per call; [`FragmentShaderSimd`] describes how the
/// lanes map to pixels. Otherwise it has the same shader stages and fixed-function state as
/// [`Pipeline`], and the two can be drawn in the same [`RenderPass`]. Draw with it through
/// [`RenderPass::draw_simd`].
///
/// The varyings a SIMD fragment shader receives can differ slightly from the ones a [`Pipeline`]
/// gives the same fragment.
///
/// # SIMD shader requirements
///
/// - The fragment shader implements [`FragmentShaderSimd`]. It doesn't need a scalar
///   [`FragmentShader`] implementation; a shader may implement both to be usable in either
///   pipeline.
/// - The vertex shader's varyings implement [`SimdInterpolate`], usually through
///   `#[derive(Interpolate, SimdInterpolate)]`, which also generates the eight-lane varyings type
///   the fragment shader receives (`VaryingsSimd` for a struct named `Varyings`).
/// - Both the fragment shader and the varyings are `Send + Sync + 'static`, which
///   [`RenderPass::draw_simd`] requires because shading can run on other threads.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// #[derive(Interpolate, SimdInterpolate)]
/// struct Varyings {
///     brightness: f32,
/// }
///
/// struct PassThrough;
/// impl VertexShader for PassThrough {
///     type Vertex = Vec4;
///     type Uniforms = ();
///     type Varyings = Varyings;
///
///     fn shade(&self, position: Vec4, _: &()) -> (Vec4, Varyings) {
///         (position, Varyings { brightness: 1.0 })
///     }
/// }
///
/// /// Shades eight fragments at once in shades of grey.
/// struct Greyscale;
/// impl FragmentShaderSimd<Varyings> for Greyscale {
///     type Uniforms = ();
///
///     fn shade_simd(&self, varyings: VaryingsSimd, _: &()) -> ColourSimd {
///         let brightness = varyings.brightness;
///         ColourSimd { r: brightness, g: brightness, b: brightness, a: f32x8::splat(1.0) }
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(16, 16);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = SimdPipeline::new(PassThrough, Greyscale);
///
/// let vertices = [
///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
///     Vec4::new(1.0, -1.0, 0.0, 1.0),
///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
/// ];
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
///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, ()),
///     (),
/// );
/// pass.finish();
///
/// let pixel = target.pixels()[15 * 16]; // bottom-left corner
/// assert!((pixel.r - 1.0).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
///
/// # Known issues
///
/// - The varyings can differ from those a [`Pipeline`] gives the same pixel by more than expected;
///   the cause hasn't been investigated
///   ([#21](https://github.com/costott/cpu_rasteriser/issues/21)).
pub struct SimdPipeline<VS, FS>
where
    VS: VertexShader,
    VS::Varyings: SimdInterpolate,
    FS: FragmentShaderSimd<VS::Varyings>,
{
    vertex_shader: VS,
    fragment_shader: Arc<FS>,
    state: PipelineState,
}
impl<VS, FS> SimdPipeline<VS, FS>
where
    VS: VertexShader,
    VS::Varyings: SimdInterpolate,
    FS: FragmentShaderSimd<VS::Varyings>,
{
    /// Creates a SIMD pipeline from a vertex shader and a SIMD fragment shader.
    ///
    /// The pipeline starts with [`CullingMode::BackFace`], no blending (fragment colours replace
    /// the target's), and [`DepthState::DISABLED`], as [`Pipeline::new`] does. The builder methods
    /// change these.
    ///
    /// # Examples
    ///
    /// See [`SimdPipeline`].
    pub fn new(vertex_shader: VS, fragment_shader: FS) -> Self {
        Self {
            vertex_shader,
            fragment_shader: Arc::new(fragment_shader),
            state: PipelineState {
                culling_mode: CullingMode::BackFace,
                blend_state: None,
                depth_state: DepthState::DISABLED,
            },
        }
    }

    /// Sets which triangles are culled by their winding before rasterisation.
    ///
    /// # Examples
    ///
    /// See [`CullingMode`].
    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.state.culling_mode = culling_mode;
        self
    }

    /// Sets how fragment colours are combined with the colours already in the target.
    ///
    /// # Examples
    ///
    /// See [`BlendState`].
    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.state.blend_state = Some(blend_state);
        self
    }

    /// Removes the blend state, so fragment colours replace the colours already in the target.
    ///
    /// This is the default.
    ///
    /// # Examples
    ///
    /// See [`BlendState`].
    pub fn without_blend_state(mut self) -> Self {
        self.state.blend_state = None;
        self
    }

    /// Sets whether fragments are depth tested and whether they write their depth.
    ///
    /// A pipeline that tests or writes depth can only be drawn in a pass with a depth buffer.
    ///
    /// # Examples
    ///
    /// See [`DepthState`].
    pub fn with_depth_state(mut self, depth_state: DepthState) -> Self {
        self.state.depth_state = depth_state;
        self
    }
}
impl<VS, FS> std::fmt::Debug for SimdPipeline<VS, FS>
where
    VS: VertexShader,
    VS::Varyings: SimdInterpolate,
    FS: FragmentShaderSimd<VS::Varyings>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdPipeline")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct PipelineState {
    culling_mode: CullingMode,
    /// Optional blending configuration.
    ///
    /// When `None`, the source colour replaces the destination colour.
    /// When `Some`, the configured blend factors and operation are applied.
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}

/// Which triangles a pipeline discards by their winding before rasterisation.
///
/// Which way a triangle faces follows the winding convention of the [`VertexShader`] contract:
/// counter-clockwise triangles are front-facing. For closed meshes, back-face culling skips the
/// triangles facing away from the camera, which are hidden anyway.
///
/// # Examples
///
/// A clockwise triangle is culled by [`CullingMode::BackFace`] (the default) and drawn with
/// [`CullingMode::None`]:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # struct PassThrough;
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vec4;
/// #     type Uniforms = ();
/// #     type Varyings = f32;
/// #     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
/// #         (position, 0.0)
/// #     }
/// # }
/// # struct FlatColour;
/// # impl FragmentShader<f32> for FlatColour {
/// #     type Uniforms = Colour;
/// #     fn shade(&self, _: f32, colour: &Colour) -> Colour {
/// #         *colour
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let clockwise = [
///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
///     Vec4::new(1.0, -1.0, 0.0, 1.0),
/// ];
///
/// for (culling_mode, drawn) in [(CullingMode::BackFace, false), (CullingMode::None, true)] {
///     let pipeline = Pipeline::new(PassThrough, FlatColour).with_culling_mode(culling_mode);
///     let mut target = RenderTarget::new(extent);
///
///     let mut pass = renderer.begin_render_pass(
///         &mut target,
///         RenderPassDescriptor {
///             viewport: Viewport::full(&extent),
///             colour_load_op: LoadOp::Clear(Colour::BLACK),
///             depth_load_op: None,
///         },
///     );
///     pass.draw(
///         &pipeline,
///         DrawCall::new(&clockwise, &[0, 1, 2], PrimitiveMode::TRIANGLES, Colour::RED),
///         (),
///     );
///     pass.finish();
///
///     assert_eq!(target.pixels().contains(&Colour::RED), drawn);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullingMode {
    /// Draws every triangle, whichever way it faces.
    None,
    /// Discards back-facing (clockwise) triangles, and triangles with zero area.
    BackFace,
}

/// Whether a pipeline's fragments are depth tested and whether they write their depth.
///
/// A fragment's depth is the normalised device z of the [`VertexShader`] contract, where smaller
/// is nearer. The depth test passes when a fragment's depth is less than the depth stored in the
/// target, so a pass that tests depth usually clears it to 1.0. A fragment that fails the test is
/// discarded.
///
/// A pipeline that tests or writes depth can only be drawn in a render pass with a depth buffer
/// (a [`RenderPassDescriptor::depth_load_op`] of `Some`).
///
/// # Examples
///
/// With depth testing, the nearer of two overlapping triangles is visible whichever is drawn last:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # struct PassThrough;
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vec4;
/// #     type Uniforms = ();
/// #     type Varyings = f32;
/// #     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
/// #         (position, 0.0)
/// #     }
/// # }
/// # struct FlatColour;
/// # impl FragmentShader<f32> for FlatColour {
/// #     type Uniforms = Colour;
/// #     fn shade(&self, _: f32, colour: &Colour) -> Colour {
/// #         *colour
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let triangle_at = |depth| {
///     [
///         Vec4::new(-1.0, -1.0, depth, 1.0),
///         Vec4::new(1.0, -1.0, depth, 1.0),
///         Vec4::new(-1.0, 1.0, depth, 1.0),
///     ]
/// };
/// let (near, far) = (triangle_at(0.0), triangle_at(0.5));
///
/// for (depth_state, visible) in [
///     (DepthState::DEFAULT, Colour::RED),
///     (DepthState::DISABLED, Colour::BLUE),
/// ] {
///     let pipeline = Pipeline::new(PassThrough, FlatColour).with_depth_state(depth_state);
///     let mut target = RenderTarget::new(extent).with_depth();
///
///     let mut pass = renderer.begin_render_pass(
///         &mut target,
///         RenderPassDescriptor {
///             viewport: Viewport::full(&extent),
///             colour_load_op: LoadOp::Clear(Colour::BLACK),
///             depth_load_op: Some(LoadOp::Clear(1.0)),
///         },
///     );
///     for (vertices, colour) in [(&near, Colour::RED), (&far, Colour::BLUE)] {
///         pass.draw(
///             &pipeline,
///             DrawCall::new(vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, colour),
///             (),
///         );
///     }
///     pass.finish();
///
///     assert_eq!(target.pixels()[3 * 4], visible); // bottom-left corner
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    /// Whether fragments at or behind the stored depth are discarded.
    pub test_enabled: bool,
    /// Whether fragments that are drawn write their depth to the depth buffer.
    pub write_enabled: bool,
}
impl DepthState {
    /// Tests and writes depth: the usual state for opaque geometry.
    pub const DEFAULT: DepthState = DepthState {
        test_enabled: true,
        write_enabled: true,
    };

    /// Tests depth without writing it, for example for transparent geometry drawn after opaque
    /// geometry.
    pub const READ_ONLY: DepthState = DepthState {
        test_enabled: true,
        write_enabled: false,
    };

    /// Writes depth without testing it, so every fragment is drawn and records its depth.
    pub const WRITE_ONLY: DepthState = DepthState {
        test_enabled: false,
        write_enabled: true,
    };

    /// Neither tests nor writes depth: the state of a new pipeline.
    pub const DISABLED: DepthState = DepthState {
        test_enabled: false,
        write_enabled: false,
    };
}

/// The per-channel weight a [`BlendState`] multiplies a source or destination colour by.
///
/// The source (`Src`) is the colour the fragment shader produced; the destination (`Dst`) is the
/// colour already in the target. Each factor resolves to one weight per channel.
///
/// # Examples
///
/// See [`BlendState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    /// Weights every channel by 0.
    Zero,
    /// Weights every channel by 1.
    One,
    /// Weights each channel by the source's value of that channel.
    SrcColor,
    /// Weights each channel by one minus the source's value of that channel.
    OneMinusSrcColor,
    /// Weights each channel by the destination's value of that channel.
    DstColor,
    /// Weights each channel by one minus the destination's value of that channel.
    OneMinusDstColor,
    /// Weights every channel by the source alpha.
    SrcAlpha,
    /// Weights every channel by one minus the source alpha.
    OneMinusSrcAlpha,
    /// Weights every channel by the destination alpha.
    DstAlpha,
    /// Weights every channel by one minus the destination alpha.
    OneMinusDstAlpha,
}
impl BlendFactor {
    /// Resolves this blend factor to a colour value using the source and destination colours.
    fn resolve(&self, src: Colour, dst: Colour) -> Colour {
        match self {
            BlendFactor::Zero => Colour::BLACK,
            BlendFactor::One => Colour::WHITE,
            BlendFactor::SrcColor => src,
            BlendFactor::OneMinusSrcColor => Colour::WHITE - src,
            BlendFactor::DstColor => dst,
            BlendFactor::OneMinusDstColor => Colour::WHITE - dst,
            BlendFactor::SrcAlpha => Colour::new(src.a, src.a, src.a, src.a),
            BlendFactor::OneMinusSrcAlpha => {
                let a = 1.0 - src.a;
                Colour::new(a, a, a, a)
            }
            BlendFactor::DstAlpha => Colour::new(dst.a, dst.a, dst.a, dst.a),
            BlendFactor::OneMinusDstAlpha => {
                let a = 1.0 - dst.a;
                Colour::new(a, a, a, a)
            }
        }
    }

    #[inline(always)]
    fn resolve_simd(&self, src: ColourSimd, dst: ColourSimd) -> ColourSimd {
        let one = f32x8::splat(1.0);
        let zero = f32x8::splat(0.0);

        match self {
            BlendFactor::Zero => ColourSimd {
                r: zero,
                g: zero,
                b: zero,
                a: zero,
            },

            BlendFactor::One => ColourSimd {
                r: one,
                g: one,
                b: one,
                a: one,
            },

            BlendFactor::SrcColor => src,

            BlendFactor::OneMinusSrcColor => ColourSimd {
                r: one - src.r,
                g: one - src.g,
                b: one - src.b,
                a: one - src.a,
            },

            BlendFactor::DstColor => dst,

            BlendFactor::OneMinusDstColor => ColourSimd {
                r: one - dst.r,
                g: one - dst.g,
                b: one - dst.b,
                a: one - dst.a,
            },

            BlendFactor::SrcAlpha => ColourSimd {
                r: src.a,
                g: src.a,
                b: src.a,
                a: src.a,
            },

            BlendFactor::OneMinusSrcAlpha => {
                let alpha = one - src.a;

                ColourSimd {
                    r: alpha,
                    g: alpha,
                    b: alpha,
                    a: alpha,
                }
            }

            BlendFactor::DstAlpha => ColourSimd {
                r: dst.a,
                g: dst.a,
                b: dst.a,
                a: dst.a,
            },

            BlendFactor::OneMinusDstAlpha => {
                let alpha = one - dst.a;

                ColourSimd {
                    r: alpha,
                    g: alpha,
                    b: alpha,
                    a: alpha,
                }
            }
        }
    }
}

/// The operation a [`BlendState`] combines the weighted source and destination colours with.
///
/// In the descriptions, `src` and `dst` are the source and destination colours after weighting by
/// their [`BlendFactor`]s. Each operation applies per channel.
///
/// # Examples
///
/// See [`BlendState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendOp {
    /// `src + dst`.
    Add,
    /// `src - dst`.
    Subtract,
    /// `dst - src`.
    ReverseSubtract,
    /// The smaller of `src` and `dst`.
    Min,
    /// The larger of `src` and `dst`.
    Max,
}

/// How a pipeline combines each fragment's colour with the colour already in the target.
///
/// The result is `op(src * src_factor, dst * dst_factor)`, per channel, where `src` is the
/// fragment shader's colour and `dst` the target's. It isn't clamped. A pipeline without a blend
/// state writes `src` unchanged.
///
/// # Examples
///
/// Additive blending adds each fragment's colour to the target's, for effects such as light and
/// fire:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # struct PassThrough;
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vec4;
/// #     type Uniforms = ();
/// #     type Varyings = f32;
/// #     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
/// #         (position, 0.0)
/// #     }
/// # }
/// # struct FlatColour;
/// # impl FragmentShader<f32> for FlatColour {
/// #     type Uniforms = Colour;
/// #     fn shade(&self, _: f32, colour: &Colour) -> Colour {
/// #         *colour
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = Pipeline::new(PassThrough, FlatColour).with_blend_state(BlendState::ADDITIVE);
/// let vertices = [
///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
///     Vec4::new(1.0, -1.0, 0.0, 1.0),
///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
/// ];
///
/// let mut pass = renderer.begin_render_pass(
///     &mut target,
///     RenderPassDescriptor {
///         viewport: Viewport::full(&extent),
///         colour_load_op: LoadOp::Clear(Colour::rgb(0.25, 0.0, 0.5)),
///         depth_load_op: None,
///     },
/// );
/// let orange = Colour::rgb(0.5, 0.5, 0.0);
/// pass.draw(
///     &pipeline,
///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, orange),
///     (),
/// );
/// pass.finish();
///
/// let pixel = target.pixels()[3 * 4]; // bottom-left corner
/// assert_eq!((pixel.r, pixel.g, pixel.b), (0.75, 0.5, 0.5));
/// # Ok(())
/// # }
/// ```
///
/// # Known issues
///
/// - A [`Pipeline`] doesn't apply this rule to the alpha channel, so the alpha it leaves in the
///   target differs from a [`SimdPipeline`]'s; see [`BlendState::apply`]
///   ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    /// The weight of the fragment's colour.
    pub src_factor: BlendFactor,
    /// The weight of the colour already in the target.
    pub dst_factor: BlendFactor,
    /// How the two weighted colours are combined.
    pub op: BlendOp,
}
impl BlendState {
    /// Alpha blending ("over"): `src * src.a + dst * (1 - src.a)`.
    ///
    /// Draws translucent fragments over the target. Draw translucent geometry back to front,
    /// after opaque geometry.
    ///
    /// # Known issues
    ///
    /// - The colour channels blend as stated, but the alpha channel doesn't on a [`Pipeline`],
    ///   which keeps `src.a`, so the alpha left in the target differs between [`Pipeline`] and
    ///   [`SimdPipeline`] ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
    pub const ALPHA_BLEND: BlendState = BlendState {
        // TODO: currently broken
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        op: BlendOp::Add,
    };

    /// Additive blending: `src + dst`.
    ///
    /// Brightens the target, for effects such as light, glow, and fire.
    pub const ADDITIVE: BlendState = BlendState {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        op: BlendOp::Add,
    };

    /// Blends a source colour onto a destination colour, as a [`Pipeline`] does for each
    /// fragment.
    ///
    /// The red, green, and blue channels follow the equation in [`BlendState`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let src = Colour::new(1.0, 0.0, 0.0, 0.5);
    /// let dst = Colour::new(0.0, 0.0, 1.0, 1.0);
    ///
    /// let blended = BlendState::ALPHA_BLEND.apply(src, dst);
    ///
    /// assert_eq!((blended.r, blended.g, blended.b), (0.5, 0.0, 0.5));
    /// ```
    ///
    /// # Known issues
    ///
    /// - The alpha channel isn't weighted, unlike in [`BlendState::apply_simd`]: it is `src.a` for
    ///   [`BlendOp::Add`] and [`BlendOp::Subtract`], `dst.a` for [`BlendOp::ReverseSubtract`], and
    ///   the smaller or larger of the two for [`BlendOp::Min`] and [`BlendOp::Max`]
    ///   ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
    pub fn apply(&self, src: Colour, dst: Colour) -> Colour {
        // TODO: need to decide if this should be public
        let src_term = src * self.src_factor.resolve(src, dst);
        let dst_term = dst * self.dst_factor.resolve(src, dst);

        match self.op {
            BlendOp::Add => src_term + dst_term,
            BlendOp::Subtract => src_term - dst_term,
            BlendOp::ReverseSubtract => dst_term - src_term,
            BlendOp::Min => Colour::new(
                src_term.r.min(dst_term.r),
                src_term.g.min(dst_term.g),
                src_term.b.min(dst_term.b),
                src_term.a.min(dst_term.a),
            ),
            BlendOp::Max => Colour::new(
                src_term.r.max(dst_term.r),
                src_term.g.max(dst_term.g),
                src_term.b.max(dst_term.b),
                src_term.a.max(dst_term.a),
            ),
        }
    }

    /// Blends eight source colours onto eight destination colours, as a [`SimdPipeline`] does for
    /// each group of fragments.
    ///
    /// Every channel, alpha included, follows the equation in [`BlendState`], lane by lane.
    /// [`BlendOp::Min`] and [`BlendOp::Max`] don't specify which operand they return when one is
    /// NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let src = ColourSimd::splat(Colour::new(1.0, 0.0, 0.0, 0.5));
    /// let dst = ColourSimd::splat(Colour::new(0.0, 0.0, 1.0, 1.0));
    ///
    /// let blended = BlendState::ALPHA_BLEND.apply_simd(src, dst);
    ///
    /// assert_eq!(blended.r.to_array(), [0.5; 8]);
    /// assert_eq!(blended.b.to_array(), [0.5; 8]);
    /// ```
    ///
    /// # Known issues
    ///
    /// - The alpha channel differs from [`BlendState::apply`]'s, which doesn't weight alpha, so a
    ///   [`SimdPipeline`] and a [`Pipeline`] leave different alpha in the target
    ///   ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
    #[inline(always)]
    pub fn apply_simd(&self, src: ColourSimd, dst: ColourSimd) -> ColourSimd {
        let src_factor = self.src_factor.resolve_simd(src, dst);
        let dst_factor = self.dst_factor.resolve_simd(src, dst);

        let src_term = ColourSimd {
            r: src.r * src_factor.r,
            g: src.g * src_factor.g,
            b: src.b * src_factor.b,
            a: src.a * src_factor.a,
        };

        let dst_term = ColourSimd {
            r: dst.r * dst_factor.r,
            g: dst.g * dst_factor.g,
            b: dst.b * dst_factor.b,
            a: dst.a * dst_factor.a,
        };

        match self.op {
            BlendOp::Add => ColourSimd {
                r: src_term.r + dst_term.r,
                g: src_term.g + dst_term.g,
                b: src_term.b + dst_term.b,
                a: src_term.a + dst_term.a,
            },

            BlendOp::Subtract => ColourSimd {
                r: src_term.r - dst_term.r,
                g: src_term.g - dst_term.g,
                b: src_term.b - dst_term.b,
                a: src_term.a - dst_term.a,
            },

            BlendOp::ReverseSubtract => ColourSimd {
                r: dst_term.r - src_term.r,
                g: dst_term.g - src_term.g,
                b: dst_term.b - src_term.b,
                a: dst_term.a - src_term.a,
            },

            BlendOp::Min => ColourSimd {
                r: src_term.r.fast_min(dst_term.r),
                g: src_term.g.fast_min(dst_term.g),
                b: src_term.b.fast_min(dst_term.b),
                a: src_term.a.fast_min(dst_term.a),
            },

            BlendOp::Max => ColourSimd {
                r: src_term.r.fast_max(dst_term.r),
                g: src_term.g.fast_max(dst_term.g),
                b: src_term.b.fast_max(dst_term.b),
                a: src_term.a.fast_max(dst_term.a),
            },
        }
    }
}

/// A render pass in progress: records draw calls into one [`RenderTarget`], then renders them.
///
/// Created by [`Renderer::begin_render_pass`]. [`RenderPass::draw`] and [`RenderPass::draw_simd`]
/// queue draw calls, and [`RenderPass::finish`] renders them all. [`Pipeline`] and
/// [`SimdPipeline`] draws can be mixed in one pass.
///
/// Draw calls take effect in the order they were submitted: where two draws cover the same pixel,
/// the later one is drawn over (or blended onto, or depth tested against) the earlier one.
///
/// The pass borrows the renderer and the target mutably until it is finished. `'renderer` is the
/// borrow of the renderer; `'pass` is the borrow of the target, which the pipelines and the
/// geometry of every draw call must outlive. Dropping a pass without calling
/// [`RenderPass::finish`] discards its draws and leaves the target unchanged.
///
/// # Examples
///
/// Two draws over the same pixels: the second, submitted later, is visible.
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// # struct PassThrough;
/// # impl VertexShader for PassThrough {
/// #     type Vertex = Vec4;
/// #     type Uniforms = ();
/// #     type Varyings = f32;
/// #     fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
/// #         (position, 0.0)
/// #     }
/// # }
/// # struct FlatColour;
/// # impl FragmentShader<f32> for FlatColour {
/// #     type Uniforms = Colour;
/// #     fn shade(&self, _: f32, colour: &Colour) -> Colour {
/// #         *colour
/// #     }
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut renderer = Renderer::new()?;
/// let extent = Extent::new(4, 4);
/// let mut target = RenderTarget::new(extent);
/// let pipeline = Pipeline::new(PassThrough, FlatColour);
/// let vertices = [
///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
///     Vec4::new(1.0, -1.0, 0.0, 1.0),
///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
/// ];
///
/// let mut pass = renderer.begin_render_pass(
///     &mut target,
///     RenderPassDescriptor {
///         viewport: Viewport::full(&extent),
///         colour_load_op: LoadOp::Clear(Colour::BLACK),
///         depth_load_op: None,
///     },
/// );
/// for colour in [Colour::RED, Colour::GREEN] {
///     pass.draw(
///         &pipeline,
///         DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, colour),
///         (),
///     );
/// }
/// pass.finish();
///
/// assert_eq!(target.pixels()[3 * 4], Colour::GREEN); // bottom-left corner
/// # Ok(())
/// # }
/// ```
pub struct RenderPass<'renderer, 'pass> {
    renderer: &'renderer mut Renderer,

    render_target: &'pass mut RenderTarget,
    descriptor: RenderPassDescriptor,

    queued_draws: Vec<Box<dyn RenderPassCommand + 'pass>>,

    tile_binner: TileBinner,
}
impl<'renderer, 'pass> RenderPass<'renderer, 'pass> {
    /// Queues a draw call that renders `draw_call`'s geometry with a [`Pipeline`].
    ///
    /// `vertex_uniforms` is passed to the pipeline's vertex shader for every vertex of the draw;
    /// the fragment shader's uniforms are part of the [`DrawCall`]. Nothing is rendered until
    /// [`RenderPass::finish`].
    ///
    /// # Panics
    ///
    /// Panics if the pipeline's [`DepthState`] tests or writes depth and the pass has no depth
    /// buffer (its [`RenderPassDescriptor::depth_load_op`] is `None`).
    ///
    /// # Examples
    ///
    /// The vertex uniform scales the triangle to half its size, so it leaves the top-left corner
    /// of the target uncovered:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// struct Scale;
    /// impl VertexShader for Scale {
    ///     type Vertex = Vec4;
    ///     type Uniforms = f32;
    ///     type Varyings = f32;
    ///
    ///     fn shade(&self, position: Vec4, scale: &f32) -> (Vec4, f32) {
    ///         (Vec4::new(position.x * scale, position.y * scale, position.z, position.w), 0.0)
    ///     }
    /// }
    ///
    /// struct FlatColour;
    /// impl FragmentShader<f32> for FlatColour {
    ///     type Uniforms = Colour;
    ///
    ///     fn shade(&self, _: f32, colour: &Colour) -> Colour {
    ///         *colour
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut renderer = Renderer::new()?;
    /// let extent = Extent::new(8, 8);
    /// let mut target = RenderTarget::new(extent);
    /// let pipeline = Pipeline::new(Scale, FlatColour);
    /// let vertices = [
    ///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
    ///     Vec4::new(1.0, -1.0, 0.0, 1.0),
    ///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
    /// ];
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
    ///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Colour::RED),
    ///     0.5,
    /// );
    /// pass.finish();
    ///
    /// let pixels = target.pixels();
    /// assert_eq!(pixels[5 * 8 + 2], Colour::RED);
    /// assert_eq!(pixels[0], Colour::BLACK);
    /// # Ok(())
    /// # }
    /// ```
    pub fn draw<'pipeline, VS, FS>(
        &mut self,
        pipeline: &'pipeline Pipeline<VS, FS>,
        draw_call: DrawCall<'pass, VS::Vertex, FS::Uniforms>,
        vertex_uniforms: VS::Uniforms,
    ) where
        'pipeline: 'pass,
        VS: VertexShader,
        FS: FragmentShader<VS::Varyings>,
    {
        self.assert_depth_buffer_for(pipeline.state.depth_state);
        self.queued_draws.push(Box::new(QueuedDraw {
            pipeline,
            draw_call,
            vertex_uniforms,
        }));
    }

    /// Queues a draw call that renders `draw_call`'s geometry with a [`SimdPipeline`].
    ///
    /// Fragments are shaded eight at a time with [`FragmentShaderSimd::shade_simd`]. Otherwise
    /// this behaves as [`RenderPass::draw`]: `vertex_uniforms` is passed to the vertex shader for
    /// every vertex, and nothing is rendered until [`RenderPass::finish`].
    ///
    /// # Panics
    ///
    /// Panics if the pipeline's [`DepthState`] tests or writes depth and the pass has no depth
    /// buffer (its [`RenderPassDescriptor::depth_load_op`] is `None`).
    ///
    /// # Examples
    ///
    /// The vertex uniform scales the triangle to half its size, so it leaves the top-left corner
    /// of the target uncovered:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// #[derive(Interpolate, SimdInterpolate)]
    /// struct Varyings {
    ///     unused: f32,
    /// }
    ///
    /// struct Scale;
    /// impl VertexShader for Scale {
    ///     type Vertex = Vec4;
    ///     type Uniforms = f32;
    ///     type Varyings = Varyings;
    ///
    ///     fn shade(&self, position: Vec4, scale: &f32) -> (Vec4, Varyings) {
    ///         let (x, y) = (position.x * scale, position.y * scale);
    ///         (Vec4::new(x, y, position.z, position.w), Varyings { unused: 0.0 })
    ///     }
    /// }
    ///
    /// struct FlatColour;
    /// impl FragmentShaderSimd<Varyings> for FlatColour {
    ///     type Uniforms = Colour;
    ///
    ///     fn shade_simd(&self, _: VaryingsSimd, colour: &Colour) -> ColourSimd {
    ///         ColourSimd::splat(*colour)
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut renderer = Renderer::new()?;
    /// let extent = Extent::new(8, 8);
    /// let mut target = RenderTarget::new(extent);
    /// let pipeline = SimdPipeline::new(Scale, FlatColour);
    /// let vertices = [
    ///     Vec4::new(-1.0, -1.0, 0.0, 1.0),
    ///     Vec4::new(1.0, -1.0, 0.0, 1.0),
    ///     Vec4::new(-1.0, 1.0, 0.0, 1.0),
    /// ];
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
    ///     DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Colour::RED),
    ///     0.5,
    /// );
    /// pass.finish();
    ///
    /// let pixels = target.pixels();
    /// assert_eq!(pixels[5 * 8 + 2], Colour::RED);
    /// assert_eq!(pixels[0], Colour::BLACK);
    /// # Ok(())
    /// # }
    /// ```
    pub fn draw_simd<'pipeline, VS, FS>(
        &mut self,
        pipeline: &'pipeline SimdPipeline<VS, FS>,
        draw_call: DrawCall<'pass, VS::Vertex, FS::Uniforms>,
        vertex_uniforms: VS::Uniforms,
    ) where
        'pipeline: 'pass,
        VS: VertexShader,
        VS::Varyings: SimdInterpolate + Send + Sync + 'static,
        FS: FragmentShaderSimd<VS::Varyings> + Send + Sync + 'static,
    {
        self.assert_depth_buffer_for(pipeline.state.depth_state);
        self.queued_draws.push(Box::new(QueuedDrawSimd {
            pipeline,
            draw_call,
            vertex_uniforms,
        }));
    }

    fn assert_depth_buffer_for(&self, depth_state: DepthState) {
        assert!(
            self.descriptor.depth_load_op.is_some()
                || !(depth_state.test_enabled || depth_state.write_enabled),
            "the pipeline tests or writes depth, but the render pass has no depth buffer \
             (its `depth_load_op` is `None`)"
        );
    }

    /// Renders the queued draw calls into the target and ends the pass.
    ///
    /// This blocks until the target is complete; read it afterwards with
    /// [`RenderTarget::pixels`] or [`RenderTarget::pixels_u32`]. The draws' shaders run during this
    /// call, as the [`VertexShader`] and [`FragmentShader`] contracts describe.
    ///
    /// # Panics
    ///
    /// Panics if an index that a draw call assembles into a primitive is out of range of its
    /// vertex buffer.
    ///
    /// # Examples
    ///
    /// See [`RenderPass`] and [`Renderer`].
    ///
    /// # Known issues
    ///
    /// - A panic in a fragment shader is swallowed: `finish` returns normally, and the region of
    ///   the target around the pixels it was shading keeps its previous contents, including
    ///   what other draws in the pass drew there
    ///   ([#58](https://github.com/costott/cpu_rasteriser/issues/58)).
    pub fn finish(mut self) {
        for draw in self.queued_draws {
            draw.execute(&mut self.tile_binner, &self.descriptor.viewport);
        }

        self.renderer
            .render_tiles(self.tile_binner, self.render_target);
    }
}
impl std::fmt::Debug for RenderPass<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPass")
            .field("render_target", &self.render_target)
            .field("descriptor", &self.descriptor)
            .field("queued_draws", &self.queued_draws.len())
            .finish_non_exhaustive()
    }
}

/// How a draw call's indices are assembled into primitives.
///
/// # Examples
///
/// See [`DrawCall`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveMode {
    /// Each consecutive group of three indices is one triangle; leftover indices are ignored.
    TRIANGLES,
    // TODO: support other primitive modes (LINES, POINTS, etc.)
}

/// The geometry of one draw, and the uniforms its fragment shader receives.
///
/// A draw call borrows a vertex buffer and an index buffer. `V` is the pipeline's vertex type
/// ([`VertexShader::Vertex`]) and `U` its fragment shader's uniforms. Submit it with
/// [`RenderPass::draw`] or [`RenderPass::draw_simd`].
///
/// # Examples
///
/// A quad made of two triangles that share vertices through the index buffer:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let vertices = [
///     Vec4::new(-0.5, -0.5, 0.0, 1.0),
///     Vec4::new(0.5, -0.5, 0.0, 1.0),
///     Vec4::new(0.5, 0.5, 0.0, 1.0),
///     Vec4::new(-0.5, 0.5, 0.0, 1.0),
/// ];
/// let indices = [0, 1, 2, 0, 2, 3];
///
/// let quad = DrawCall::new(&vertices, &indices, PrimitiveMode::TRIANGLES, Colour::RED);
/// # let _ = quad;
/// ```
///
/// [`Renderer`] shows a draw call rendered in a pass.
pub struct DrawCall<'a, V, U>
where
    V: Clone,
{
    primitive: Primitive<'a, V>,
    fragment_uniforms: U,
}
impl<'a, V, U> DrawCall<'a, V, U>
where
    V: Clone,
{
    /// Creates a draw call that assembles `vertices` into primitives by `indices`, according to
    /// `primitive_mode`.
    ///
    /// `fragment_uniforms` is passed to the fragment shader for every fragment of the draw. Each
    /// index selects an element of `vertices`.
    ///
    /// # Examples
    ///
    /// See [`DrawCall`].
    ///
    /// # Known issues
    ///
    /// - Indices aren't checked against `vertices`, so an out-of-range index panics later, in
    ///   [`RenderPass::finish`], instead of here
    ///   ([#59](https://github.com/costott/cpu_rasteriser/issues/59)).
    pub fn new(
        vertices: &'a [V],
        indices: &'a [u32],
        primitive_mode: PrimitiveMode,
        fragment_uniforms: U,
    ) -> DrawCall<'a, V, U> {
        DrawCall {
            primitive: Primitive::new(vertices, indices, primitive_mode),
            fragment_uniforms,
        }
    }

    fn primitive_mode(&self) -> PrimitiveMode {
        self.primitive.primitive_mode
    }
}
impl<V: Clone, U> std::fmt::Debug for DrawCall<'_, V, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawCall")
            .field("vertices", &self.primitive.vertices.len())
            .field("indices", &self.primitive.indices.len())
            .field("primitive_mode", &self.primitive.primitive_mode)
            .finish_non_exhaustive()
    }
}

/// A collection of indexed geometry.
///
/// A primitive consists of a vertex buffer, index buffer, and topology describing how vertices
/// should be assembled into renderable shapes.
///
/// Currently only triangle primitives are supported.
pub struct Primitive<'a, V>
where
    V: Clone,
{
    vertices: &'a [V],
    indices: &'a [u32],
    primitive_mode: PrimitiveMode,
}
impl<'a, V> Primitive<'a, V>
where
    V: Clone,
{
    pub fn new(
        vertices: &'a [V],
        indices: &'a [u32],
        primitive_mode: PrimitiveMode,
    ) -> Primitive<'a, V> {
        Primitive {
            vertices,
            indices,
            primitive_mode,
        }
    }

    fn triangles(&self) -> impl Iterator<Item = Triangle3D<V>> {
        self.indices.chunks_exact(3).map(|indices| Triangle3D {
            a: self.vertices[indices[0] as usize].clone(),
            b: self.vertices[indices[1] as usize].clone(),
            c: self.vertices[indices[2] as usize].clone(),
        })
    }
}

/// Internal type-erased command used to execute a scalar [`Pipeline`] draw.
///
/// `QueuedDraw` is the concrete implementation of [`RenderPassCommand`] used for
/// normal rendering operations. It stores references to a [`Pipeline`], the
/// submitted [`DrawCall`], and the vertex shader uniforms required to process
/// the geometry.
///
/// When executed, the draw call:
///
/// 1. Assembles indexed geometry into triangles.
/// 2. Runs the vertex shader and geometry processing pipeline.
/// 3. Applies fixed-function state such as back-face culling.
/// 4. Bins generated triangles into screen-space tiles for parallel rasterisation.
///
/// The generic shader parameters are erased when stored in a [`RenderPass`] through
/// the [`RenderPassCommand`] trait object.
struct QueuedDraw<'a, VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    pipeline: &'a Pipeline<VS, FS>,

    draw_call: DrawCall<'a, VS::Vertex, FS::Uniforms>,

    vertex_uniforms: VS::Uniforms,
}
impl<VS, FS> RenderPassCommand for QueuedDraw<'_, VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    fn execute(self: Box<Self>, tile_binner: &mut TileBinner, viewport: &Viewport) {
        let uniforms = Arc::new(self.draw_call.fragment_uniforms);

        for triangle in self.draw_call.primitive.triangles() {
            let triangles = GeometryProcessor::process_triangle(
                triangle,
                &self.pipeline.vertex_shader,
                &self.vertex_uniforms,
                viewport,
                self.pipeline.state.culling_mode,
            );

            for triangle in triangles {
                let command = Arc::new(TriangleRasterCommand {
                    triangle,
                    uniforms: uniforms.clone(),
                    shader: self.pipeline.fragment_shader.clone(),
                    blend_state: self.pipeline.state.blend_state,
                    depth_state: self.pipeline.state.depth_state,
                });

                tile_binner.bin_command(command);
            }
        }
    }
}

struct TriangleRasterCommand<V, FS>
where
    V: Interpolate,
    FS: FragmentShader<V>,
{
    triangle: Triangle2D<V>,
    uniforms: Arc<FS::Uniforms>,
    shader: Arc<FS>,
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}
impl<V, FS> RasterCommand for TriangleRasterCommand<V, FS>
where
    V: Interpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
{
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        mut depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        self.triangle.rasterise_segment(bounds, |mut fragment| {
            fragment.position.x -= bounds.min_x as f32;
            fragment.position.y -= bounds.min_y as f32;

            if self.depth_state.test_enabled {
                let depthbuffer = depthbuffer.as_deref_mut().expect(
                    "draw rejects depth-testing pipelines in passes without a depth buffer",
                );

                if fragment.depth >= depthbuffer.get(fragment.position) {
                    return;
                }
            }

            let src = self.shader.shade(fragment.varyings, self.uniforms.as_ref());
            let dst = framebuffer
                .get_pixel(fragment.position)
                .unwrap_or(Colour::BLACK);

            let colour = match self.blend_state {
                Some(blend_state) => blend_state.apply(src, dst),
                None => src,
            };
            framebuffer.set_pixel(fragment.position, colour);

            if self.depth_state.write_enabled {
                let depthbuffer = depthbuffer.as_deref_mut().expect(
                    "draw rejects depth-writing pipelines in passes without a depth buffer",
                );

                depthbuffer.set_depth(fragment.position, fragment.depth);
            }
        });
    }

    fn bounding_box(&self) -> (Vec2, Vec2) {
        self.triangle.bounding_box()
    }

    fn intersects(&self, rect: Rect) -> bool {
        self.triangle.intersects_rect(rect)
    }
}

/// Internal type-erased command used to execute a SIMD [`SimdPipeline`] draw.
///
/// `QueuedDrawSimd` stores the pipeline, draw call, and vertex uniforms until the
/// render pass is executed.
struct QueuedDrawSimd<'a, VS, FS>
where
    VS: VertexShader,
    VS::Varyings: SimdInterpolate,
    FS: FragmentShaderSimd<VS::Varyings>,
{
    pipeline: &'a SimdPipeline<VS, FS>,
    draw_call: DrawCall<'a, VS::Vertex, FS::Uniforms>,
    vertex_uniforms: VS::Uniforms,
}

impl<VS, FS> RenderPassCommand for QueuedDrawSimd<'_, VS, FS>
where
    VS: VertexShader,
    VS::Varyings: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShaderSimd<VS::Varyings> + Send + Sync + 'static,
{
    fn execute(self: Box<Self>, tile_binner: &mut TileBinner, viewport: &Viewport) {
        let uniforms = Arc::new(self.draw_call.fragment_uniforms);

        for triangle in self.draw_call.primitive.triangles() {
            let triangles = GeometryProcessor::process_triangle(
                triangle,
                &self.pipeline.vertex_shader,
                &self.vertex_uniforms,
                viewport,
                self.pipeline.state.culling_mode,
            );

            for triangle in triangles {
                let command = Arc::new(TriangleRasterCommandSimd {
                    triangle,
                    uniforms: uniforms.clone(),
                    shader: self.pipeline.fragment_shader.clone(),
                    blend_state: self.pipeline.state.blend_state,
                    depth_state: self.pipeline.state.depth_state,
                });

                tile_binner.bin_command(command);
            }
        }
    }
}

struct TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate,
    FS: FragmentShaderSimd<V>,
{
    triangle: Triangle2D<V>,
    uniforms: Arc<FS::Uniforms>,
    shader: Arc<FS>,
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}

impl<V, FS> TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShaderSimd<V> + Send + Sync + 'static,
{
    fn rasterise_impl<const TEST_DEPTH: bool, const WRITE_DEPTH: bool>(
        &self,
        framebuffer: &mut FrameBuffer,
        mut depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        self.triangle
            .rasterise_segment_simd(bounds, |fragment_simd| {
                let pass = if TEST_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("draw_simd rejects depth-testing pipelines in passes without a depth buffer");

                    let index = depthbuffer.index_unchecked(
                        (fragment_simd.x_start - bounds.min_x) as usize,
                        (fragment_simd.y - bounds.min_y) as usize,
                    );

                    // SAFETY: `rasterise_segment_simd` keeps `x_start` and `y` inside `bounds`, and
                    // the buffers are exactly `bounds` in size (the `RasterCommand::rasterise`
                    // contract), so `index` is an in-bounds pixel.
                    let stored = unsafe { depthbuffer.get8_unchecked(index) };

                    fragment_simd.mask & fragment_simd.depth.simd_lt(stored)
                } else {
                    fragment_simd.mask
                };

                if !pass.any() {
                    return;
                }

                let mask = pass.to_bitmask();

                let base = framebuffer.index_unchecked(
                    (fragment_simd.x_start - bounds.min_x) as usize,
                    (fragment_simd.y - bounds.min_y) as usize,
                );

                let src = self
                    .shader
                    .shade_simd(fragment_simd.varyings, self.uniforms.as_ref());

                let colour = if let Some(blend_state) = self.blend_state {
                    // SAFETY: `base` is an in-bounds pixel, as for the depth test.
                    let dst = unsafe { framebuffer.get8_unchecked(base) };

                    blend_state.apply_simd(src, dst)
                } else {
                    src
                };

                let r = colour.r.to_array();
                let g = colour.g.to_array();
                let b = colour.b.to_array();
                let a = colour.a.to_array();

                for lane in 0..8 {
                    if mask & (1 << lane) == 0 {
                        continue;
                    }

                    let frag_colour = Colour::new(r[lane], g[lane], b[lane], a[lane]);

                    // SAFETY: `pass` only includes lanes of the coverage mask, which lie within the
                    // span and so inside `bounds`. `base + lane` is therefore an in-bounds pixel,
                    // as for the depth test.
                    unsafe {
                        framebuffer.set_pixel_index_unchecked(base + lane, frag_colour);
                    }
                }

                if WRITE_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("draw_simd rejects depth-writing pipelines in passes without a depth buffer");

                    let index = depthbuffer.index_unchecked(
                        (fragment_simd.x_start - bounds.min_x) as usize,
                        (fragment_simd.y - bounds.min_y) as usize,
                    );

                    // SAFETY: `index` is an in-bounds pixel, as for the depth test.
                    unsafe {
                        depthbuffer.set8_unchecked_with_mask(index, fragment_simd.depth, pass);
                    }
                }
            });
    }
}

impl<V, FS> RasterCommand for TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShaderSimd<V> + Send + Sync + 'static,
{
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        match (
            self.depth_state.test_enabled,
            self.depth_state.write_enabled,
        ) {
            (false, false) => {
                self.rasterise_impl::<false, false>(framebuffer, depthbuffer, bounds);
            }

            (false, true) => {
                self.rasterise_impl::<false, true>(framebuffer, depthbuffer, bounds);
            }

            (true, false) => {
                self.rasterise_impl::<true, false>(framebuffer, depthbuffer, bounds);
            }

            (true, true) => {
                self.rasterise_impl::<true, true>(framebuffer, depthbuffer, bounds);
            }
        }
    }

    fn bounding_box(&self) -> (Vec2, Vec2) {
        self.triangle.bounding_box()
    }

    fn intersects(&self, rect: Rect) -> bool {
        self.triangle.intersects_rect(rect)
    }
}

/// Internal type-erased command queued during a render pass.
///
/// This allows scalar and SIMD draw commands to coexist in the same render pass
/// without exposing their concrete types to the renderer's scheduling machinery.
///
/// `RenderPassCommand` provides the interface required by [`RenderPass`] to defer rendering
/// operations until [`RenderPass::finish`] is called.
///
/// Commands are stored as trait objects because a single render pass may contain draw
/// calls using different combinations of vertex and fragment shader types. The
/// concrete shader types are hidden behind this interface until execution.
///
/// Implementors should perform any CPU-side preparation required to convert the
/// command into rasterisation work and submit it to the tile scheduler.
trait RenderPassCommand {
    fn execute(self: Box<Self>, tile_binner: &mut TileBinner, viewport: &Viewport);
}

/// A tile-based triangle scheduler.
///
/// The tile binner divides the viewport into fixed-size regions and assigns triangles to every
/// tile they overlap.
///
/// Tiles can then be rasterised independently in parallel.
struct TileBinner {
    tiles: Vec<Tile>,
    tiles_x: usize,
    tiles_y: usize,
}
impl TileBinner {
    const TILE_SIZE: i32 = 64;

    fn new(
        render_target: &RenderTarget,
        colour_load_op: &LoadOp<Colour>,
        depth_load_op: &Option<LoadOp<f32>>,
        pool: &mut TilePool,
    ) -> Self {
        let tiles_x = render_target.width().div_ceil(Self::TILE_SIZE as usize);

        let tiles_y = render_target.height().div_ceil(Self::TILE_SIZE as usize);

        let mut tiles = Vec::with_capacity(tiles_x * tiles_y);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                tiles.push(Tile::new(
                    Rect {
                        min_x: tile_x as i32 * Self::TILE_SIZE,
                        min_y: tile_y as i32 * Self::TILE_SIZE,
                        max_x: ((tile_x + 1) as i32 * Self::TILE_SIZE)
                            .min(render_target.width() as i32),
                        max_y: ((tile_y + 1) as i32 * Self::TILE_SIZE)
                            .min(render_target.height() as i32),
                    },
                    render_target,
                    colour_load_op,
                    depth_load_op,
                    pool,
                ));
            }
        }

        Self {
            tiles,
            tiles_x,
            tiles_y,
        }
    }

    fn bin_command(&mut self, command: Arc<dyn RasterCommand>) {
        let (mins, maxs) = command.bounding_box();

        let min_tile_x = (mins.x as i32 / Self::TILE_SIZE).max(0);

        let min_tile_y = (mins.y as i32 / Self::TILE_SIZE).max(0);

        let max_tile_x = (maxs.x as i32 / Self::TILE_SIZE).min(self.tiles_x as i32 - 1);

        let max_tile_y = (maxs.y as i32 / Self::TILE_SIZE).min(self.tiles_y as i32 - 1);

        for y in min_tile_y..=max_tile_y {
            for x in min_tile_x..=max_tile_x {
                let index = y as usize * self.tiles_x + x as usize;

                if command.intersects(self.tiles[index].bounds) {
                    self.tiles[index].commands.push(command.clone());
                }
            }
        }
    }
}

struct Tile {
    bounds: Rect,

    framebuffer: FrameBuffer,
    depthbuffer: Option<DepthBuffer>,

    commands: Vec<Arc<dyn RasterCommand>>,
}
impl Tile {
    fn new(
        bounds: Rect,
        render_target: &RenderTarget,
        colour_load_op: &LoadOp<Colour>,
        depth_load_op: &Option<LoadOp<f32>>,
        pool: &mut TilePool,
    ) -> Self {
        let width = (bounds.max_x - bounds.min_x) as usize;

        let height = (bounds.max_y - bounds.min_y) as usize;

        let (min_x, min_y) = (bounds.min_x as usize, bounds.min_y as usize);

        let mut framebuffer = pool.take_framebuffer(width, height);

        match colour_load_op {
            LoadOp::Load => {
                framebuffer.blit_from(
                    (0, 0),
                    &render_target.framebuffer,
                    (min_x, min_y),
                    Extent::new(width, height),
                );
            }
            LoadOp::Clear(colour) => {
                framebuffer.clear(*colour);
            }
        }

        let depthbuffer = match depth_load_op {
            Some(load_op) => {
                let src = render_target
                    .depthbuffer
                    .as_ref()
                    .expect("depth load op specified but render target has no depth buffer");

                let mut depthbuffer = pool.take_depthbuffer(width, height);

                match load_op {
                    LoadOp::Load => {
                        depthbuffer.blit_from(
                            (0, 0),
                            src,
                            (min_x, min_y),
                            Extent::new(width, height),
                        );
                    }

                    LoadOp::Clear(depth) => {
                        depthbuffer.clear(*depth);
                    }
                }

                Some(depthbuffer)
            }

            None => None,
        };

        Self {
            bounds,
            framebuffer,
            depthbuffer,
            commands: Vec::new(),
        }
    }
}

trait RasterCommand: Send + Sync {
    /// Rasterises the part of the command inside `bounds` into tile-local buffers.
    ///
    /// `framebuffer` and `depthbuffer` must be exactly `bounds` in size, with their origin at
    /// `(bounds.min_x, bounds.min_y)`. The SIMD command's unchecked buffer accesses rely on this.
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    );
    /// Returns the axis-aligned bounding box of the raster command in pixel space.
    fn bounding_box(&self) -> (Vec2, Vec2);
    /// Returns true if the raster command intersects the given rectangle in pixel space.
    fn intersects(&self, rect: Rect) -> bool;
}

/// Tile buffers reused across render passes. Buffers taken from the pool keep stale contents, so
/// callers must overwrite or clear them.
#[derive(Default)]
struct TilePool {
    framebuffers: Vec<FrameBuffer>,
    depthbuffers: Vec<DepthBuffer>,
}
impl TilePool {
    fn take_framebuffer(&mut self, width: usize, height: usize) -> FrameBuffer {
        match self.framebuffers.pop() {
            Some(mut framebuffer) => {
                framebuffer.resize(width, height);
                framebuffer
            }
            None => FrameBuffer::new(width, height),
        }
    }

    fn take_depthbuffer(&mut self, width: usize, height: usize) -> DepthBuffer {
        match self.depthbuffers.pop() {
            Some(mut depthbuffer) => {
                depthbuffer.resize(width, height);
                depthbuffer
            }
            None => DepthBuffer::new(width, height),
        }
    }

    fn recycle(&mut self, framebuffer: FrameBuffer, depthbuffer: Option<DepthBuffer>) {
        self.framebuffers.push(framebuffer);
        if let Some(depthbuffer) = depthbuffer {
            self.depthbuffers.push(depthbuffer);
        }
    }
}

/// A rectangular region of the target, in pixels: the bounds of a tile.
///
/// The minimum edges are inclusive and the maximum edges exclusive, so the region is
/// `max_x - min_x` pixels wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// The left edge (inclusive).
    pub min_x: i32,
    /// The top edge (inclusive).
    pub min_y: i32,
    /// The right edge (exclusive).
    pub max_x: i32,
    /// The bottom edge (exclusive).
    pub max_y: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PassThroughVertexShader;
    impl VertexShader for PassThroughVertexShader {
        type Vertex = Vec4;
        type Uniforms = ();
        type Varyings = f32;

        fn shade(&self, vertex: Vec4, _: &()) -> (Vec4, f32) {
            (vertex, 0.0)
        }
    }

    struct WhiteFragmentShader;
    impl FragmentShader<f32> for WhiteFragmentShader {
        type Uniforms = ();

        fn shade(&self, _: f32, _: &()) -> Colour {
            Colour::WHITE
        }
    }

    #[derive(Interpolate, SimdInterpolate)]
    struct SimdVaryings {
        value: f32,
    }

    struct SimdVaryingsVertexShader;
    impl VertexShader for SimdVaryingsVertexShader {
        type Vertex = Vec4;
        type Uniforms = ();
        type Varyings = SimdVaryings;

        fn shade(&self, vertex: Vec4, _: &()) -> (Vec4, SimdVaryings) {
            (vertex, SimdVaryings { value: 0.0 })
        }
    }

    struct WhiteFragmentShaderSimd;
    impl FragmentShaderSimd<SimdVaryings> for WhiteFragmentShaderSimd {
        type Uniforms = ();

        fn shade_simd(&self, _: <SimdVaryings as SimdInterpolate>::Simd, _: &()) -> ColourSimd {
            ColourSimd::splat(Colour::WHITE)
        }
    }

    fn no_depth_buffer_pass<'r, 't>(
        renderer: &'r mut Renderer,
        target: &'t mut RenderTarget,
    ) -> RenderPass<'r, 't> {
        let extent = target.extent();
        renderer.begin_render_pass(
            target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        )
    }

    #[test]
    #[should_panic(expected = "the render pass has no depth buffer")]
    fn draw_rejects_a_depth_testing_pipeline_in_a_pass_without_a_depth_buffer() {
        let mut renderer = Renderer::new().unwrap();
        let mut target = RenderTarget::new(Extent::new(4, 4)).with_depth();
        let pipeline = Pipeline::new(PassThroughVertexShader, WhiteFragmentShader)
            .with_depth_state(DepthState::READ_ONLY);

        let mut pass = no_depth_buffer_pass(&mut renderer, &mut target);
        pass.draw(
            &pipeline,
            DrawCall::new(&[], &[], PrimitiveMode::TRIANGLES, ()),
            (),
        );
    }

    #[test]
    #[should_panic(expected = "the render pass has no depth buffer")]
    fn draw_simd_rejects_a_depth_writing_pipeline_in_a_pass_without_a_depth_buffer() {
        let mut renderer = Renderer::new().unwrap();
        let mut target = RenderTarget::new(Extent::new(4, 4)).with_depth();
        let pipeline = SimdPipeline::new(SimdVaryingsVertexShader, WhiteFragmentShaderSimd)
            .with_depth_state(DepthState::WRITE_ONLY);

        let mut pass = no_depth_buffer_pass(&mut renderer, &mut target);
        pass.draw_simd(
            &pipeline,
            DrawCall::new(&[], &[], PrimitiveMode::TRIANGLES, ()),
            (),
        );
    }

    #[test]
    fn a_pipeline_without_depth_needs_no_depth_buffer() {
        let mut renderer = Renderer::new().unwrap();
        let mut target = RenderTarget::new(Extent::new(4, 4));
        let pipeline = Pipeline::new(PassThroughVertexShader, WhiteFragmentShader);
        let triangle = [
            Vec4::new(-1.0, -1.0, 0.0, 1.0),
            Vec4::new(1.0, -1.0, 0.0, 1.0),
            Vec4::new(-1.0, 1.0, 0.0, 1.0),
        ];

        let mut pass = no_depth_buffer_pass(&mut renderer, &mut target);
        pass.draw(
            &pipeline,
            DrawCall::new(&triangle, &[0, 1, 2], PrimitiveMode::TRIANGLES, ()),
            (),
        );
        pass.finish();

        assert!(target.pixels().contains(&Colour::WHITE));
    }

    #[test]
    fn blend_state_alpha_blend() {
        let state = BlendState::ALPHA_BLEND;

        let src = Colour::new(1.0, 0.0, 0.0, 0.5);
        let dst = Colour::new(0.0, 0.0, 1.0, 1.0);

        let result = state.apply(src, dst);

        assert!((result.r - 0.5).abs() < f32::EPSILON);
        assert!((result.g - 0.0).abs() < f32::EPSILON);
        assert!((result.b - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_state_additive() {
        let state = BlendState::ADDITIVE;

        let src = Colour::new(0.5, 0.2, 0.1, 0.5);
        let dst = Colour::new(0.1, 0.3, 0.4, 1.0);

        let result = state.apply(src, dst);

        assert!((result.r - 0.6).abs() < f32::EPSILON);
        assert!((result.g - 0.5).abs() < f32::EPSILON);
        assert!((result.b - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_state_subtract() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Subtract,
        };

        let src = Colour::new(0.5, 0.6, 0.7, 1.0);
        let dst = Colour::new(0.2, 0.4, 0.6, 1.0);

        let result = state.apply(src, dst);

        assert!((result.r - 0.3).abs() < f32::EPSILON);
        assert!((result.g - 0.2).abs() < f32::EPSILON);
        assert!((result.b - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_state_reverse_subtract() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::ReverseSubtract,
        };

        let src = Colour::new(0.2, 0.4, 0.6, 1.0);
        let dst = Colour::new(1.0, 1.5, 2.0, 2.5);

        let result = state.apply(src, dst);

        assert!((result.r - 0.8).abs() < f32::EPSILON);
        assert!((result.g - 1.1).abs() < f32::EPSILON);
        assert!((result.b - 1.4).abs() < f32::EPSILON);
        assert!((result.a - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_state_min() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Min,
        };

        let src = Colour::new(0.5, 0.2, 0.8, 0.5);
        let dst = Colour::new(0.1, 0.6, 0.6, 1.0);

        let result = state.apply(src, dst);

        assert!((result.r - 0.1).abs() < f32::EPSILON);
        assert!((result.g - 0.2).abs() < f32::EPSILON);
        assert!((result.b - 0.6).abs() < f32::EPSILON);
        assert!((result.a - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_state_max() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Max,
        };

        let src = Colour::new(0.5, 0.2, 0.8, 0.5);
        let dst = Colour::new(0.1, 0.6, 0.6, 1.0);

        let result = state.apply(src, dst);

        assert!((result.r - 0.5).abs() < f32::EPSILON);
        assert!((result.g - 0.6).abs() < f32::EPSILON);
        assert!((result.b - 0.8).abs() < f32::EPSILON);
        assert!((result.a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn blend_factor_resolve() {
        let src = Colour::new(0.5, 0.6, 0.7, 0.5);
        let dst = Colour::new(0.2, 0.4, 0.6, 0.25);

        assert_eq!(BlendFactor::Zero.resolve(src, dst), Colour::BLACK);
        assert_eq!(BlendFactor::One.resolve(src, dst), Colour::WHITE);

        assert_eq!(BlendFactor::SrcColor.resolve(src, dst), src);

        assert_eq!(
            BlendFactor::OneMinusSrcColor.resolve(src, dst),
            Colour::new(0.5, 0.4, 0.3, 0.5)
        );

        assert_eq!(BlendFactor::DstColor.resolve(src, dst), dst);

        assert_eq!(
            BlendFactor::OneMinusDstColor.resolve(src, dst),
            Colour::new(0.1, 0.4, 0.4, 0.25)
        );

        assert_eq!(
            BlendFactor::SrcAlpha.resolve(src, dst),
            Colour::new(0.5, 0.5, 0.5, 0.5)
        );

        assert_eq!(
            BlendFactor::OneMinusSrcAlpha.resolve(src, dst),
            Colour::new(0.5, 0.5, 0.5, 0.5)
        );

        assert_eq!(
            BlendFactor::DstAlpha.resolve(src, dst),
            Colour::new(0.25, 0.25, 0.25, 0.25)
        );

        assert_eq!(
            BlendFactor::OneMinusDstAlpha.resolve(src, dst),
            Colour::new(0.75, 0.75, 0.75, 0.75)
        );
    }
}
