use std::sync::Arc;
use threadpool::ThreadPool;
use wide::f32x8;

use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::triangle::Triangle2D;

/// A CPU-based renderer responsible for executing draw calls and producing a final framebuffer.
///
/// The renderer owns the framebuffer and depth buffer used during rendering, and uses a thread
/// pool to parallelise tile rasterisation. Rendering is performed in frames:
///
/// 1. Call [`Renderer::begin_render_pass`] to begin recording draw commands.
/// 2. Submit draw calls through [`RenderPass::draw`].
/// 3. Call [`RenderPass::finish`] to execute the queued commands and rasterise the frame.
///
/// # Example
///
/// ```
/// let mut renderer = Renderer::new()?;
///
/// let extent = Extent::new(WIDTH, HEIGHT);
/// let screen_target = RenderTarget::new(extent).with_depth();
///
/// let pipeline = Pipeline::new(
///     vertex_shader,
///     fragment_shader
/// ).with_culling_mode(CullingMode::BackFace)
///  .with_depth_state(DepthState::DEFAULT)
///
/// let mut render_pass = renderer.begin_render_pass(RenderPassDescriptor {
///     viewport: Viewport::full(&screen_target),
///     target: &mut screen_target,
///     colour_load_op: LoadOp::Clear(Colour::BLACK),
///     depth_load_op: LoadOp::Clear(1.0),
/// });
///
/// render_pass.draw(
///     &pipeline,
///     DrawCall::new(
///         &vertices,
///         &indices,
///         PrimitiveMode::TRIANGLES,
///         fragment_uniforms,
///     ),
///     &vertex_uniforms,
/// );
///
/// render_pass.finish();
///
/// let pixels = screen_target.pixels();
/// ```
///
/// For SIMD rendering, construct a [`SimdPipeline`] and submit it with
/// [`RenderPass::draw_simd`].
///
/// The renderer is not thread-safe and should only be accessed from the thread performing
/// rendering. Internal rasterisation work is dispatched across worker threads automatically.
pub struct Renderer {
    thread_pool: ThreadPool,
    tile_pool: TilePool,
}
impl Renderer {
    /// Creates a new renderer with one worker thread per available CPU thread.
    ///
    /// # Errors
    ///
    /// Returns the error from [`std::thread::available_parallelism`] if the number of available
    /// CPU threads can't be determined.
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
            tile_pool: TilePool::default(),
        })
    }

    /// Begins a new render pass.
    ///
    /// Only one render pass may be active at a time because the renderer is mutably
    /// borrowed while the render pass exist.
    ///
    /// # Example
    /// ```ignore
    /// let mut render_pass = renderer.begin_render_pass(RenderPassDescriptor{
    ///     viewport: Viewport::full(&render_target),
    ///     target: &mut render_target,
    ///     colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///     depth_load_op: LoadOp::Clear(1.0),
    /// });
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

pub struct RenderTarget {
    framebuffer: FrameBuffer,
    depthbuffer: Option<DepthBuffer>,
}
impl RenderTarget {
    pub fn new(extent: Extent) -> Self {
        Self {
            framebuffer: FrameBuffer::new(extent.width, extent.height),
            depthbuffer: None,
        }
    }

    pub fn with_depth(mut self) -> Self {
        self.depthbuffer = Some(DepthBuffer::new(
            self.framebuffer.width(),
            self.framebuffer.height(),
        ));
        self
    }

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

    pub fn pixels_u32(&self) -> Vec<u32> {
        self.framebuffer.pixels_u32()
    }

    /// Writes the pixels into `out` as packed `0xAARRGGBB` values, row by row, without allocating.
    ///
    /// Each channel is clamped to `[0, 1]` and NaN becomes 0, as in [`Colour::to_u32`]. Elements
    /// of `out` past `width * height` are left untouched. This is the allocation-free form of
    /// [`RenderTarget::pixels_u32`], for writing straight into a presentation buffer.
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

    pub fn width(&self) -> usize {
        self.framebuffer.width()
    }

    pub fn height(&self) -> usize {
        self.framebuffer.height()
    }

    pub fn extent(&self) -> Extent {
        self.framebuffer.extent()
    }

    /// Resizes the framebuffer and depth buffer to match the new extent.
    ///
    /// Existing framebuffer contents are discarded.
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPassDescriptor {
    pub viewport: Viewport,
    pub colour_load_op: LoadOp<Colour>,
    pub depth_load_op: Option<LoadOp<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp<T> {
    /// Keeps the existing data in the framebuffer / depth buffer.
    Load,
    /// Clears the existing data to the provided value before rendering.
    Clear(T),
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

/// A collection of shader stages and fixed-function rendering state used to process draw calls.
///
/// `Pipeline` uses the scalar fragment-shading path. Fragment shaders only need to implement
/// [`FragmentShader`].
///
/// A pipeline contains:
///
/// - A vertex shader, responsible for transforming input vertices and producing interpolated data.
/// - A fragment shader, responsible for calculating the final colour of each rasterised fragment.
/// - Rasterisation state such as back-face culling, blending, and depth testing.
///
/// Pipelines are intended to be created once and reused across multiple render passes.
///
/// For SIMD rasterisation and fragment shading, use [`SimdPipeline`] instead.
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
    /// Creates a new scalar rendering pipeline.
    ///
    /// The pipeline uses scalar fragment shading and the scalar rasterisation path.
    ///
    /// SIMD rendering is available through [`SimdPipeline`].
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

    /// Configures the back-face culling mode used by the pipeline.
    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.state.culling_mode = culling_mode;
        self
    }

    /// Configures the blending state used by the pipeline.
    ///
    /// When set, fragment colours are combined with the existing framebuffer colour
    /// according to the provided [`BlendState`].
    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.state.blend_state = Some(blend_state);
        self
    }

    /// Removes any blending state from the pipeline, causing fragment colours to replace
    /// the existing framebuffer colour.
    pub fn without_blend_state(mut self) -> Self {
        self.state.blend_state = None;
        self
    }

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

/// A rendering pipeline that uses SIMD rasterisation and fragment shading.
///
/// `SimdPipeline` requires the fragment shader to implement [`FragmentShaderSimd`].
/// Unlike [`Pipeline`], it does not require a scalar [`FragmentShader`]
/// implementation.
///
/// Rendering is performed in batches of SIMD lanes, with the rasteriser,
/// depth operations, and fragment shader operating on multiple fragments at once.
///
/// A SIMD pipeline contains the same fixed-function state as [`Pipeline`]:
///
/// - A vertex shader, responsible for transforming input vertices and producing interpolated data.
/// - A SIMD-capable fragment shader, responsible for calculating colours for multiple fragments.
/// - Rasterisation state such as back-face culling, blending, and depth testing.
///
/// Pipelines are intended to be created once and reused across multiple render passes.
///
/// # SIMD shader requirements
///
/// The fragment shader must implement both [`FragmentShader`] and [`FragmentShaderSimd`].
/// [`FragmentShader`] remains available for scalar use, while [`FragmentShaderSimd`] provides
/// the SIMD implementation used by this pipeline.
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
    /// Creates a new SIMD rendering pipeline.
    ///
    /// The fragment shader must implement [`FragmentShaderSimd`].
    /// The resulting pipeline uses SIMD rasterisation and SIMD fragment shading.
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

    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.state.culling_mode = culling_mode;
        self
    }

    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.state.blend_state = Some(blend_state);
        self
    }

    pub fn without_blend_state(mut self) -> Self {
        self.state.blend_state = None;
        self
    }

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

/// Controls whether primitives are removed based on their winding direction before rasterisation.
///
/// Back-face culling can improve rendering performance by avoiding rasterisation of triangles
/// facing away from the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullingMode {
    /// No culling is performed; all triangles are rendered.
    None,
    /// Triangles facing away from the camera are culled.
    BackFace,
}

/// Rendering depth-test and depth-write configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
}
impl DepthState {
    /// Default depth state with depth testing and depth writes enabled.
    pub const DEFAULT: DepthState = DepthState {
        test_enabled: true,
        write_enabled: true,
    };

    /// Depth testing enabled with depth writes disabled.
    pub const READ_ONLY: DepthState = DepthState {
        test_enabled: true,
        write_enabled: false,
    };

    /// Depth testing disabled with depth writes enabled.
    pub const WRITE_ONLY: DepthState = DepthState {
        test_enabled: false,
        write_enabled: true,
    };

    /// Disables both depth testing and depth writes.
    pub const DISABLED: DepthState = DepthState {
        test_enabled: false,
        write_enabled: false,
    };
}

/// Determines how a source or destination colour contributes to the blend result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
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

/// Determines the arithmetic operation used to combine the scaled source and destination colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// A fixed-function blending configuration.
///
/// A blend state determines how a fragment's source colour is combined with
/// the existing destination colour in the framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub op: BlendOp,
}
impl BlendState {
    /// Common alpha blending configuration.
    ///
    /// Uses source alpha for the source factor and one minus source alpha for
    /// the destination factor.
    pub const ALPHA_BLEND: BlendState = BlendState {
        // TODO: currently broken
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        op: BlendOp::Add,
    };

    /// Common additive blending configuration.
    ///
    /// Adds the source and destination colours together.
    pub const ADDITIVE: BlendState = BlendState {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        op: BlendOp::Add,
    };

    /// Applies this blend state to a source and destination colour.
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

/// A single render pass.
///
/// A render pass provides a temporary command recording context. Draw calls submitted through
/// [`RenderPass::draw`] or [`RenderPass::draw_simd`] are queued and converted into rendering
/// work when [`RenderPass::finish`] is called.
///
/// Both scalar [`Pipeline`] and SIMD [`SimdPipeline`] draw calls may be submitted to the same
/// render pass.
///
/// A render pass borrows the renderer mutably and must be completed before the renderer can be used
/// again.
///
/// # Example
///
/// ```ignore
/// let mut render_pass = renderer.begin_render_pass(RenderPassDescriptor {
///     target: &mut screen_target,
///     viewport: Viewport::new(0, 0, screen_target.width(), screen_target.height()),
///     colour_load_op: LoadOp::Clear(Colour::BLACK),
///     depth_load_op: LoadOp::Clear(1.0),
/// });
///
/// render_pass.draw(
///     &pipeline,
///     draw_call,
///     vertex_uniforms,
/// );
///
/// render_pass.finish();
/// ```
pub struct RenderPass<'renderer, 'pass> {
    renderer: &'renderer mut Renderer,

    render_target: &'pass mut RenderTarget,
    descriptor: RenderPassDescriptor,

    queued_draws: Vec<Box<dyn RenderPassCommand + 'pass>>,

    tile_binner: TileBinner,
}
impl<'renderer, 'pass> RenderPass<'renderer, 'pass> {
    /// Queues a draw call using a scalar [`Pipeline`].
    ///
    /// The draw call is not rendered immediately. It is stored and processed when
    /// [`RenderPass::finish`] is called.
    ///
    /// For SIMD rasterisation and fragment shading, use [`RenderPass::draw_simd`].
    ///
    /// # Panics
    ///
    /// Panics if the pipeline's [`DepthState`] tests or writes depth and the render pass has no
    /// depth buffer, because its [`RenderPassDescriptor::depth_load_op`] is `None`.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The scalar shader pipeline and rendering configuration.
    /// * `draw_call` - Geometry and fragment shader uniforms.
    /// * `vertex_uniforms` - Data passed to the vertex shader.
    ///
    /// # Example
    ///
    /// ```ignore
    /// render_pass.draw(
    ///     &pipeline,
    ///     DrawCall::new(
    ///         vertices,
    ///         indices,
    ///         PrimitiveMode::TRIANGLES,
    ///         material,
    ///     ),
    ///     transform,
    /// );
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

    /// Queues a draw call using a SIMD [`SimdPipeline`].
    ///
    /// The draw call is not rendered immediately. It is stored and processed when
    /// [`RenderPass::finish`] is called.
    ///
    /// The SIMD pipeline rasterises fragments in batches and invokes the shader's
    /// [`FragmentShaderSimd::shade_simd`] implementation for those batches.
    ///
    /// # Panics
    ///
    /// Panics if the pipeline's [`DepthState`] tests or writes depth and the render pass has no
    /// depth buffer, because its [`RenderPassDescriptor::depth_load_op`] is `None`.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The SIMD shader pipeline and rendering configuration.
    /// * `draw_call` - Geometry and fragment shader uniforms.
    /// * `vertex_uniforms` - Data passed to the vertex shader.
    ///  
    /// # Example
    ///
    /// ```ignore
    /// render_pass.draw(
    ///     &simd_pipeline,
    ///     DrawCall::new(
    ///         vertices,
    ///         indices,
    ///         PrimitiveMode::TRIANGLES,
    ///         material,
    ///     ),
    ///     transform,
    /// );
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

    /// Executes all queued draw calls and renders the completed render pass.
    ///
    /// This performs geometry processing, triangle binning, parallel tile rasterisation, and merges
    /// the resulting tiles back into the renderer framebuffer.
    ///
    /// After completion, rendered pixels can be accessed through the bound [`RenderTarget::pixels`].
    pub fn finish(mut self) {
        for draw in self.queued_draws {
            draw.execute(&mut self.tile_binner, &self.descriptor.viewport);
        }

        self.renderer
            .render_tiles(self.tile_binner, self.render_target);
    }

    fn assert_depth_buffer_for(&self, depth_state: DepthState) {
        assert!(
            self.descriptor.depth_load_op.is_some()
                || !(depth_state.test_enabled || depth_state.write_enabled),
            "the pipeline tests or writes depth, but the render pass has no depth buffer \
             (its `depth_load_op` is `None`)"
        );
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

/// The topology used when assembling indexed geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveMode {
    /// Every group of three indices defines a triangle.
    TRIANGLES,
    // TODO: support other primitive modes (LINES, POINTS, etc.)
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

/// A geometry submission describing what to render and the associated fragment shader data.
///
/// A draw call combines a primitive with the uniforms required when shading fragments.
///
/// Draw calls are submitted through [`RenderPass::draw`].
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
    /// Creates a new draw call.
    ///
    /// # Arguments
    ///
    /// * `vertices` - Vertex buffer containing geometry data.
    /// * `indices` - Index buffer describing primitive connectivity.
    /// * `primitive_mode` - Topology used to assemble primitives.
    /// * `fragment_uniforms` - Uniform data passed to the fragment shader.
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

/// A rectangular pixel-space region.
///
/// Used for tile boundaries, clipping regions, and intersection tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
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
