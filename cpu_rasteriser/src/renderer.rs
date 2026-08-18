use std::sync::Arc;
use threadpool::ThreadPool;

use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::geometry_processing::GeometryProcessor;

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
/// The renderer is not thread-safe and should only be accessed from the thread performing
/// rendering. Internal rasterisation work is dispatched across worker threads automatically.
pub struct Renderer {
    thread_pool: ThreadPool,
}
impl Renderer {
    /// Creates a new renderer.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
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
        let RenderPassDescriptor {
            viewport,
            colour_load_op,
            depth_load_op,
        } = descriptor;

        let tile_binner = TileBinner::new(target, colour_load_op, depth_load_op);

        RenderPass {
            renderer: self,
            render_target: target,
            viewport,
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
        for y in 0..result.framebuffer.height() {
            for x in 0..result.framebuffer.width() {
                let position = Vec2::new(
                    result.bounds.min_x as f32 + x as f32,
                    result.bounds.min_y as f32 + y as f32,
                );

                if let Some(colour) = result.framebuffer.get_pixel((x, y).into()) {
                    render_target.framebuffer.set_pixel(position, colour);
                }

                if let (Some(dst), Some(src)) = (
                    render_target.depthbuffer.as_mut(),
                    result.depthbuffer.as_ref(),
                ) {
                    dst.set_depth((position.x, position.y).into(), src.get((x, y).into()));
                }
            }
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

    pub fn pixels(&self) -> &[u32] {
        self.framebuffer.pixels()
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

pub struct RenderPassDescriptor {
    pub viewport: Viewport,
    pub colour_load_op: LoadOp<Colour>,
    pub depth_load_op: Option<LoadOp<f32>>,
}

pub enum LoadOp<T> {
    /// Keeps the existing data in the framebuffer / depth buffer.
    Load,
    /// Clears the existing data to the provided value before rendering.
    Clear(T),
}

/// A collection of shader stages and fixed-function rendering state used to process draw calls.
///
/// A pipeline defines how vertices are transformed and how fragments are shaded. It contains:
///
/// - A vertex shader, responsible for transforming input vertices and producing interpolated data.
/// - A fragment shader, responsible for calculating the final colour of each rasterised fragment.
/// - Rasterisation state such as back-face culling, blending, and depth testing.
///
/// Pipelines are intended to be created once and reused across multiple render passes.
pub struct Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    vertex_shader: VS,
    fragment_shader: Arc<FS>,

    culling_mode: CullingMode,
    /// Optional blending configuration.
    ///
    /// When `None`, the source colour replaces the destination colour.
    /// When `Some`, the configured blend factors and operation are applied.
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}
impl<VS, FS> Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    pub fn new(vertex_shader: VS, fragment_shader: FS) -> Self {
        Self {
            vertex_shader,
            fragment_shader: Arc::new(fragment_shader),
            culling_mode: CullingMode::BackFace,
            blend_state: None,
            depth_state: DepthState::DISABLED,
        }
    }

    /// Configures the back-face culling mode used by the pipeline.
    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.culling_mode = culling_mode;
        self
    }

    /// Configures the blending state used by the pipeline.
    ///
    /// When set, fragment colours are combined with the existing framebuffer colour
    /// according to the provided [`BlendState`].
    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.blend_state = Some(blend_state);
        self
    }

    /// Removes any blending state from the pipeline, causing fragment colours to replace
    /// the existing framebuffer colour.
    pub fn without_blend_state(mut self) -> Self {
        self.blend_state = None;
        self
    }

    pub fn with_depth_state(mut self, depth_state: DepthState) -> Self {
        self.depth_state = depth_state;
        self
    }
}

/// Controls whether primitives are removed based on their winding direction before rasterisation.
///
/// Back-face culling can improve rendering performance by avoiding rasterisation of triangles
/// facing away from the camera.
#[derive(Clone, Copy)]
pub enum CullingMode {
    /// No culling is performed; all triangles are rendered.
    None,
    /// Triangles facing away from the camera are culled.
    BackFace,
}

/// Rendering depth-test and depth-write configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
                let a = 255 - src.a;
                Colour::new(a, a, a, a)
            }
            BlendFactor::DstAlpha => Colour::new(dst.a, dst.a, dst.a, dst.a),
            BlendFactor::OneMinusDstAlpha => {
                let a = 255 - dst.a;
                Colour::new(a, a, a, a)
            }
        }
    }
}

/// Determines the arithmetic operation used to combine the scaled source and destination colours.
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
    fn apply(&self, src: Colour, dst: Colour) -> Colour {
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
}

/// A single render pass.
///
/// A render pass provides a temporary command recording context. Draw calls submitted through
/// [`RenderPass::draw`] are queued and converted into rasterisation commands when [`RenderPass::finish`]
/// is called.
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
    viewport: Viewport,

    queued_draws: Vec<Box<dyn RenderPassCommand + 'pass>>,

    tile_binner: TileBinner,
}
impl<'renderer, 'pass> RenderPass<'renderer, 'pass> {
    /// Queues a draw call for execution during this render pass.
    ///
    /// Draw calls are not rendered immediately. They are stored and processed when [`RenderPass::finish`]
    /// is called.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - Shader pipeline and rendering configuration.
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
        self.queued_draws.push(Box::new(QueuedDraw {
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
            draw.execute(&mut self.tile_binner, &self.viewport);
        }

        self.renderer
            .render_tiles(self.tile_binner, self.render_target);
    }
}

/// A type-erased rendering command queued during a render pass.
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

/// A queued draw call containing geometry, shaders, and rendering state.
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
                self.pipeline.culling_mode,
            );

            for triangle in triangles {
                let command = Arc::new(TriangleRasterCommand {
                    triangle: triangle.clone(),
                    uniforms: uniforms.clone(),
                    shader: self.pipeline.fragment_shader.clone(),
                    blend_state: self.pipeline.blend_state,
                    depth_state: self.pipeline.depth_state,
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

trait RasterCommand: Send + Sync {
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
                let depthbuffer = depthbuffer
                    .as_deref_mut()
                    .expect("depth testing enabled but render target has no depth buffer");

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
                let depthbuffer = depthbuffer
                    .as_deref_mut()
                    .expect("depth writing enabled but render target has no depth buffer");

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
        colour_load_op: LoadOp<Colour>,
        depth_load_op: Option<LoadOp<f32>>,
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
                    &colour_load_op,
                    &depth_load_op,
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
    ) -> Self {
        let width = (bounds.max_x - bounds.min_x) as usize;

        let height = (bounds.max_y - bounds.min_y) as usize;

        let mut framebuffer = FrameBuffer::new(width, height);

        match colour_load_op {
            LoadOp::Load => {
                for y in 0..height {
                    for x in 0..width {
                        let target_position = Vec2::new(
                            bounds.min_x as f32 + x as f32,
                            bounds.min_y as f32 + y as f32,
                        );

                        let tile_position = Vec2::new(x as f32, y as f32);

                        if let Some(colour) = render_target.framebuffer.get_pixel(target_position) {
                            framebuffer.set_pixel(tile_position, colour);
                        }
                    }
                }
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

                let mut depthbuffer = DepthBuffer::new(width, height);

                match load_op {
                    LoadOp::Load => {
                        for y in 0..height {
                            for x in 0..width {
                                let target_position = Vec2::new(
                                    bounds.min_x as f32 + x as f32,
                                    bounds.min_y as f32 + y as f32,
                                );

                                let tile_position = Vec2::new(x as f32, y as f32);

                                depthbuffer.set_depth(tile_position, src.get(target_position));
                            }
                        }
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

    #[test]
    fn blend_state_alpha_blend() {
        let state = BlendState::ALPHA_BLEND;

        let src = Colour::new(255, 0, 0, 128);
        let dst = Colour::new(0, 0, 255, 255);

        let result = state.apply(src, dst);

        assert_eq!(result.r, 128);
        assert_eq!(result.g, 0);
        assert_eq!(result.b, 127);
    }

    #[test]
    fn blend_state_additive() {
        let state = BlendState::ADDITIVE;

        let src = Colour::new(100, 50, 25, 128);
        let dst = Colour::new(20, 30, 40, 255);

        let result = state.apply(src, dst);

        assert_eq!(result.r, 120);
        assert_eq!(result.g, 80);
        assert_eq!(result.b, 65);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn blend_state_subtract() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Subtract,
        };

        let src = Colour::new(100, 150, 200, 255);
        let dst = Colour::new(20, 40, 60, 255);

        let result = state.apply(src, dst);

        assert_eq!(result.r, 80);
        assert_eq!(result.g, 110);
        assert_eq!(result.b, 140);
        assert_eq!(result.a, 0);
    }

    #[test]
    fn blend_state_reverse_subtract() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::ReverseSubtract,
        };

        let src = Colour::new(20, 40, 60, 255);
        let dst = Colour::new(100, 150, 200, 255);

        let result = state.apply(src, dst);

        assert_eq!(result.r, 80);
        assert_eq!(result.g, 110);
        assert_eq!(result.b, 140);
        assert_eq!(result.a, 0);
    }

    #[test]
    fn blend_state_min() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Min,
        };

        let src = Colour::new(100, 50, 200, 128);
        let dst = Colour::new(20, 100, 150, 255);

        let result = state.apply(src, dst);

        assert_eq!(result, Colour::new(20, 50, 150, 128));
    }

    #[test]
    fn blend_state_max() {
        let state = BlendState {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            op: BlendOp::Max,
        };

        let src = Colour::new(100, 50, 200, 128);
        let dst = Colour::new(20, 100, 150, 255);

        let result = state.apply(src, dst);

        assert_eq!(result, Colour::new(100, 100, 200, 255));
    }

    #[test]
    fn blend_factor_resolve() {
        let src = Colour::new(100, 150, 200, 128);
        let dst = Colour::new(20, 40, 60, 64);

        assert_eq!(BlendFactor::Zero.resolve(src, dst), Colour::BLACK);
        assert_eq!(BlendFactor::One.resolve(src, dst), Colour::WHITE);

        assert_eq!(BlendFactor::SrcColor.resolve(src, dst), src);

        assert_eq!(
            BlendFactor::OneMinusSrcColor.resolve(src, dst),
            Colour::new(155, 105, 55, 127)
        );

        assert_eq!(BlendFactor::DstColor.resolve(src, dst), dst);

        assert_eq!(
            BlendFactor::OneMinusDstColor.resolve(src, dst),
            Colour::new(235, 215, 195, 191)
        );

        assert_eq!(
            BlendFactor::SrcAlpha.resolve(src, dst),
            Colour::new(128, 128, 128, 128)
        );

        assert_eq!(
            BlendFactor::OneMinusSrcAlpha.resolve(src, dst),
            Colour::new(127, 127, 127, 127)
        );

        assert_eq!(
            BlendFactor::DstAlpha.resolve(src, dst),
            Colour::new(64, 64, 64, 64)
        );

        assert_eq!(
            BlendFactor::OneMinusDstAlpha.resolve(src, dst),
            Colour::new(191, 191, 191, 191)
        );
    }
}
