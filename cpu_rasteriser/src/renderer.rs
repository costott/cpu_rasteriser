use std::sync::Arc;
use threadpool::ThreadPool;

use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::fragment_shader::FragmentShader;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::vertex_shader::VertexShader;
use crate::viewport::Viewport;

/// A CPU-based renderer responsible for executing draw calls and producing a final framebuffer.
///
/// The renderer owns the framebuffer and depth buffer used during rendering, and uses a thread
/// pool to parallelise tile rasterisation. Rendering is performed in frames:
///
/// 1. Call [`Renderer::begin_frame`] to begin recording draw commands.
/// 2. Submit draw calls through [`Frame::draw`].
/// 3. Call [`Frame::finish`] to execute the queued commands and rasterise the frame.
///
/// # Example
///
/// ```ignore
/// let mut renderer = Renderer::new(
///     &viewport
/// )?;
///
/// let pipeline = Pipeline::new(
///     vertex_shader,
///     fragment_shader
/// ).with_culling_mode(CullingMode::BackFace);
///
/// let mut frame = renderer.begin_frame(&viewport);
///
/// frame.draw(
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
/// frame.finish();
///
/// let pixels = renderer.pixels();
/// ```
///
/// The renderer is not thread-safe and should only be accessed from the thread performing
/// rendering. Internal rasterisation work is dispatched across worker threads automatically.
pub struct Renderer {
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,

    thread_pool: ThreadPool,
}
impl Renderer {
    /// Creates a new renderer.
    ///
    /// The renderer takes ownership of the framebuffer and depth buffer used
    /// during rendering.
    pub fn new(viewport: &Viewport) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
        })
    }

    /// Returns the current framebuffer pixel data.
    ///
    /// The returned slice contains packed pixel values in framebuffer order.
    ///
    /// The returned reference remains valid until the framebuffer is modified or the renderer is
    /// resized.
    pub fn pixels(&self) -> &[u32] {
        self.framebuffer.pixels()
    }

    /// Resizes the renderer's framebuffer and depth buffer to match the viewport.
    ///
    /// Existing framebuffer contents are discarded.
    ///
    /// # Arguments
    ///
    /// * `viewport` - The new rendering dimensions.
    pub fn resize(&mut self, viewport: &Viewport) {
        self.framebuffer.resize(viewport.width, viewport.height);
        self.depthbuffer.resize(viewport.width, viewport.height);
    }

    /// Begins a new rendering frame.
    ///
    /// This clears the framebuffer and depth buffer, then returns a [`Frame`] used to submit draw
    /// commands.
    ///
    /// Only one frame may be active at a time because the renderer is mutably borrowed while the
    /// frame exists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut frame = renderer.begin_frame(&viewport);
    ///
    /// frame.draw(
    ///     &pipeline,
    ///     draw_call,
    ///     &vertex_uniforms,
    /// );
    ///
    /// frame.finish();
    /// ```
    pub fn begin_frame<'renderer, 'viewport>(
        &'renderer mut self,
        viewport: &'viewport Viewport,
    ) -> Frame<'renderer, 'viewport> {
        self.framebuffer.clear(Colour::BLACK);
        self.depthbuffer.clear();

        Frame {
            renderer: self,
            viewport,
            queued_draws: Vec::new(),
            tile_binner: TileBinner::new(viewport),
        }
    }

    fn render_tiles(&mut self, tile_binner: TileBinner) {
        let (tx, rx) = std::sync::mpsc::channel();

        for tile in tile_binner.tiles {
            let tx = tx.clone();

            self.thread_pool.execute(move || {
                tx.send(render_tile(tile)).unwrap();
            });
        }

        drop(tx);

        for result in rx {
            self.merge_tile(result);
        }

        self.thread_pool.join();
    }

    fn merge_tile(&mut self, result: TileResult) {
        for y in 0..result.framebuffer.height() {
            for x in 0..result.framebuffer.width() {
                if let Some(colour) = result.framebuffer.get_pixel((x, y).into()) {
                    self.framebuffer.set_pixel(
                        Vec2::new(
                            result.bounds.min_x as f32 + x as f32,
                            result.bounds.min_y as f32 + y as f32,
                        ),
                        colour,
                    );
                }
            }
        }
    }
}

/// A collection of shader stages and fixed-function rendering state used to process draw calls.
///
/// A pipeline defines how vertices are transformed and how fragments are shaded. It contains:
///
/// - A vertex shader, responsible for transforming input vertices and producing interpolated data.
/// - A fragment shader, responsible for calculating the final colour of each rasterised fragment.
/// - Rasterisation state such as back-face culling configuration.
///
/// Pipelines are intended to be created once and reused across multiple frames.
pub struct Pipeline<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    pub vertex_shader: VS,
    pub fragment_shader: Arc<FS>,

    pub culling_mode: CullingMode,
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
        }
    }

    pub fn with_culling_mode(mut self, culling_mode: CullingMode) -> Self {
        self.culling_mode = culling_mode;
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

/// A single rendering frame.
///
/// A frame provides a temporary command recording context. Draw calls submitted through
/// [`Frame::draw`] are queued and converted into rasterisation commands when [`Frame::finish`]
/// is called.
///
/// A frame borrows the renderer mutably and must be completed before the renderer can be used
/// again.
///
/// # Example
///
/// ```ignore
/// let mut frame = renderer.begin_frame(&viewport);
///
/// frame.draw(
///     &pipeline,
///     draw_call,
///     vertex_uniforms,
/// );
///
/// frame.finish();
/// ```
pub struct Frame<'renderer, 'viewport> {
    renderer: &'renderer mut Renderer,

    viewport: &'viewport Viewport,

    queued_draws: Vec<Box<dyn FrameCommand + 'renderer>>,

    tile_binner: TileBinner,
}
impl<'renderer, 'viewport> Frame<'renderer, 'viewport> {
    /// Queues a draw call for execution during this frame.
    ///
    /// Draw calls are not rendered immediately. They are stored and processed when [`Frame::finish`]
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
    /// frame.draw(
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
    pub fn draw<VS, FS>(
        &mut self,
        pipeline: &'renderer Pipeline<VS, FS>,
        draw_call: DrawCall<'renderer, VS::Vertex, FS::Uniforms>,
        vertex_uniforms: VS::Uniforms,
    ) where
        VS: VertexShader,
        FS: FragmentShader<VS::Varyings>,
        VS::Varyings: Interpolate + Send + Sync + 'static,
        FS::Uniforms: Send + Sync + 'static,
    {
        self.queued_draws.push(Box::new(QueuedDraw {
            pipeline,
            draw_call,
            vertex_uniforms,
        }));
    }

    /// Executes all queued draw calls and renders the completed frame.
    ///
    /// This performs geometry processing, triangle binning, parallel tile rasterisation, and merges
    /// the resulting tiles back into the renderer framebuffer.
    ///
    /// After completion, rendered pixels can be accessed through [`Renderer::pixels`].
    pub fn finish(mut self) {
        for draw in self.queued_draws {
            draw.execute(&mut self.tile_binner, self.viewport);
        }

        self.renderer.render_tiles(self.tile_binner);
    }
}

/// A type-erased rendering command queued during a frame.
///
/// `FrameCommand` provides the interface required by [`Frame`] to defer rendering
/// operations until [`Frame::finish`] is called.
///
/// Commands are stored as trait objects because a single frame may contain draw
/// calls using different combinations of vertex and fragment shader types. The
/// concrete shader types are hidden behind this interface until execution.
///
/// Implementors should perform any CPU-side preparation required to convert the
/// command into rasterisation work and submit it to the tile scheduler.
trait FrameCommand {
    fn execute(self: Box<Self>, tile_binner: &mut TileBinner, viewport: &Viewport);
}

/// A queued draw call containing geometry, shaders, and rendering state.
///
/// `QueuedDraw` is the concrete implementation of [`FrameCommand`] used for
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
/// The generic shader parameters are erased when stored in a [`Frame`] through
/// the [`FrameCommand`] trait object.
struct QueuedDraw<'a, VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    pipeline: &'a Pipeline<VS, FS>,

    draw_call: DrawCall<'a, VS::Vertex, FS::Uniforms>,

    vertex_uniforms: VS::Uniforms,
}
impl<VS, FS> FrameCommand for QueuedDraw<'_, VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
    VS::Varyings: Interpolate + Send + Sync + 'static,
    FS::Uniforms: Send + Sync + 'static,
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
                tile_binner.bin_triangle(
                    triangle.clone(),
                    Box::new(TriangleRasterCommand {
                        triangle,
                        uniforms: uniforms.clone(),
                        shader: self.pipeline.fragment_shader.clone(),
                    }),
                );
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
    pub vertices: &'a [V],
    pub indices: &'a [u32],
    pub primitive_mode: PrimitiveMode,
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
/// Draw calls are submitted through [`Frame::draw`].
pub struct DrawCall<'a, V, U>
where
    V: Clone,
{
    pub primitive: Primitive<'a, V>,
    pub fragment_uniforms: U,
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
        self: Box<Self>,
        framebuffer: &mut FrameBuffer,
        depthbuffer: &mut DepthBuffer,
        bounds: Rect,
    );
    fn clone_box(&self) -> Box<dyn RasterCommand>;
}
impl Clone for Box<dyn RasterCommand> {
    fn clone(&self) -> Self {
        self.clone_box()
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
}
impl<V, FS> RasterCommand for TriangleRasterCommand<V, FS>
where
    V: Interpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
    FS::Uniforms: Send + Sync + 'static,
{
    fn rasterise(
        self: Box<Self>,
        framebuffer: &mut FrameBuffer,
        depthbuffer: &mut DepthBuffer,
        bounds: Rect,
    ) {
        self.triangle.rasterise_segment(bounds, |mut fragment| {
            fragment.position.x -= bounds.min_x as f32;

            fragment.position.y -= bounds.min_y as f32;

            let colour = self.shader.shade(fragment.varyings, self.uniforms.as_ref());

            if fragment.depth < depthbuffer.get(fragment.position) {
                framebuffer.set_pixel(fragment.position, colour);

                depthbuffer.set_depth(fragment.position, fragment.depth);
            }
        });
    }

    fn clone_box(&self) -> Box<dyn RasterCommand> {
        Box::new(TriangleRasterCommand {
            triangle: self.triangle.clone(),
            uniforms: self.uniforms.clone(),
            shader: self.shader.clone(),
        })
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

    fn new(viewport: &Viewport) -> Self {
        let tiles_x = viewport.width.div_ceil(Self::TILE_SIZE as usize);

        let tiles_y = viewport.height.div_ceil(Self::TILE_SIZE as usize);

        let mut tiles = Vec::with_capacity(tiles_x * tiles_y);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                tiles.push(Tile {
                    bounds: Rect {
                        min_x: (tile_x * Self::TILE_SIZE as usize) as i32,

                        min_y: (tile_y * Self::TILE_SIZE as usize) as i32,

                        max_x: ((tile_x + 1) * Self::TILE_SIZE as usize) as i32,

                        max_y: ((tile_y + 1) * Self::TILE_SIZE as usize) as i32,
                    },

                    triangles: Vec::new(),
                });
            }
        }

        Self {
            tiles,
            tiles_x,
            tiles_y,
        }
    }

    fn bin_triangle(
        &mut self,
        triangle: Triangle2D<impl Interpolate>,
        command: Box<dyn RasterCommand>,
    ) {
        let (mins, maxs) = triangle.bounding_box();

        let min_tile_x = (mins.x as i32 / Self::TILE_SIZE).max(0);

        let min_tile_y = (mins.y as i32 / Self::TILE_SIZE).max(0);

        let max_tile_x = (maxs.x as i32 / Self::TILE_SIZE).min(self.tiles_x as i32 - 1);

        let max_tile_y = (maxs.y as i32 / Self::TILE_SIZE).min(self.tiles_y as i32 - 1);

        for y in min_tile_y..=max_tile_y {
            for x in min_tile_x..=max_tile_x {
                let index = y as usize * self.tiles_x + x as usize;

                if triangle.intersects_rect(self.tiles[index].bounds) {
                    self.tiles[index].triangles.push(command.clone());
                }
            }
        }
    }
}

struct Tile {
    bounds: Rect,

    triangles: Vec<Box<dyn RasterCommand>>,
}

fn render_tile(tile: Tile) -> TileResult {
    let width = (tile.bounds.max_x - tile.bounds.min_x) as usize;

    let height = (tile.bounds.max_y - tile.bounds.min_y) as usize;

    let mut framebuffer = FrameBuffer::new(width, height);

    let mut depthbuffer = DepthBuffer::new(width, height);

    for triangle in tile.triangles {
        triangle.rasterise(&mut framebuffer, &mut depthbuffer, tile.bounds);
    }

    TileResult {
        bounds: tile.bounds,
        framebuffer,
    }
}

/// A tile result is the output of rendering a single tile, used for merging into the main framebuffer.
struct TileResult {
    bounds: Rect,
    framebuffer: FrameBuffer,
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
