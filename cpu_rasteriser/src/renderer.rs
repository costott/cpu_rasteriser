use std::sync::Arc;
use std::time::Duration;
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
        let start = std::time::Instant::now();

        let (tx, rx) = std::sync::mpsc::channel();

        for tile in tile_binner.tiles {
            let tx = tx.clone();

            self.thread_pool.execute(move || {
                let result = render_tile(tile);
                tx.send(result).unwrap();
            });
        }

        drop(tx);

        println!("job submission: {:?}", start.elapsed());

        let mut aggregate = TileProfilingAggregate::default();
        let mut wall_total = std::time::Duration::ZERO;
        let mut wall_max = std::time::Duration::ZERO;

        for result in rx {
            wall_total += result.profiling.total_tile_time;
            wall_max = wall_max.max(result.profiling.total_tile_time);

            aggregate.add(&result.profiling);
            self.merge_tile(result);
        }

        println!(
            "tile wall time: total={:?}, avg={:?}, max={:?}",
            wall_total,
            average_duration(wall_total, aggregate.tile_count),
            wall_max,
        );

        println!(
            "tile profiling aggregate: tiles={}, alloc total={:?}, avg={:?}, max={:?}, triangle total={:?}, avg={:?}, max={:?}, coverage+interp total={:?}, avg={:?}, max={:?}, shader total={:?}, avg={:?}, max={:?}, depth test total={:?}, avg={:?}, max={:?}, write total={:?}, avg={:?}, max={:?}, fragments tested total={}, avg={}, max={}, fragments passed total={}, avg={}, max={}",
            aggregate.tile_count,
            aggregate.total_alloc_time,
            average_duration(aggregate.total_alloc_time, aggregate.tile_count),
            aggregate.max_alloc_time,
            aggregate.total_triangle_time,
            average_duration(aggregate.total_triangle_time, aggregate.tile_count),
            aggregate.max_triangle_time,
            aggregate.total_coverage_and_interpolation_time,
            average_duration(
                aggregate.total_coverage_and_interpolation_time,
                aggregate.tile_count
            ),
            aggregate.max_coverage_and_interpolation_time,
            aggregate.total_shader_time,
            average_duration(aggregate.total_shader_time, aggregate.tile_count),
            aggregate.max_shader_time,
            aggregate.total_depth_test_time,
            average_duration(aggregate.total_depth_test_time, aggregate.tile_count),
            aggregate.max_depth_test_time,
            aggregate.total_write_time,
            average_duration(aggregate.total_write_time, aggregate.tile_count),
            aggregate.max_write_time,
            aggregate.total_fragments_tested,
            average_usize(aggregate.total_fragments_tested, aggregate.tile_count),
            aggregate.max_fragments_tested,
            aggregate.total_fragments_passed,
            average_usize(aggregate.total_fragments_passed, aggregate.tile_count),
            aggregate.max_fragments_passed,
        );

        let start = std::time::Instant::now();

        self.thread_pool.join();

        println!("join: {:?}", start.elapsed());
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
///     &vertex_uniforms,
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
    ///     &transform,
    /// );
    /// ```
    pub fn draw<VS, FS>(
        &mut self,
        pipeline: &'renderer Pipeline<VS, FS>,
        draw_call: DrawCall<'renderer, VS::Vertex, FS::Uniforms>,
        vertex_uniforms: &'renderer VS::Uniforms,
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
        let start = std::time::Instant::now();

        for draw in self.queued_draws {
            draw.execute(&mut self.tile_binner, self.viewport);
        }

        println!("geometry + binning: {:?}", start.elapsed());

        let start = std::time::Instant::now();

        self.renderer.render_tiles(self.tile_binner);

        println!("tile rendering + merge: {:?}", start.elapsed());
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

    vertex_uniforms: &'a VS::Uniforms,
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
                self.vertex_uniforms,
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

#[derive(Debug, Clone, Copy, Default)]
struct RasterStats {
    fragments_tested: usize,
    fragments_passed: usize,
    depth_tests: usize,
    successful_writes: usize,
    coverage_and_interpolation_time: Duration,
    shader_time: Duration,
    depth_test_time: Duration,
    write_time: Duration,
}

trait RasterCommand: Send + Sync {
    fn rasterise(
        self: Box<Self>,
        framebuffer: &mut FrameBuffer,
        depthbuffer: &mut DepthBuffer,
        bounds: Rect,
    ) -> RasterStats;
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
    ) -> RasterStats {
        let total_start = std::time::Instant::now();
        let mut stats = RasterStats::default();

        self.triangle.rasterise_segment(bounds, |mut fragment| {
            fragment.position.x -= bounds.min_x as f32;
            fragment.position.y -= bounds.min_y as f32;

            stats.fragments_tested += 1;
            stats.depth_tests += 1;

            let shade_start = std::time::Instant::now();
            let colour = self.shader.shade(fragment.varyings, self.uniforms.as_ref());
            stats.shader_time += shade_start.elapsed();

            let depth_test_start = std::time::Instant::now();
            let passes_depth_test = fragment.depth < depthbuffer.get(fragment.position);
            stats.depth_test_time += depth_test_start.elapsed();

            if passes_depth_test {
                stats.fragments_passed += 1;

                let write_start = std::time::Instant::now();
                framebuffer.set_pixel(fragment.position, colour);
                depthbuffer.set_depth(fragment.position, fragment.depth);
                stats.write_time += write_start.elapsed();
                stats.successful_writes += 1;
            }
        });

        let total_time = total_start.elapsed();
        let measured_fragment_path = stats.shader_time + stats.depth_test_time + stats.write_time;
        stats.coverage_and_interpolation_time = total_time.saturating_sub(measured_fragment_path);

        stats
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

fn average_duration(total: std::time::Duration, count: usize) -> std::time::Duration {
    if count == 0 {
        std::time::Duration::ZERO
    } else {
        total / count as u32
    }
}

fn average_usize(total: usize, count: usize) -> usize {
    if count == 0 { 0 } else { total / count }
}

#[derive(Debug, Clone, Copy, Default)]
struct TileProfiling {
    total_tile_time: std::time::Duration,
    framebuffer_alloc_time: std::time::Duration,
    triangle_rasterisation_time: std::time::Duration,
    coverage_and_interpolation_time: std::time::Duration,
    shader_time: std::time::Duration,
    depth_test_time: std::time::Duration,
    write_time: std::time::Duration,
    fragments_tested: usize,
    fragments_passed: usize,
    depth_tests: usize,
    successful_writes: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct TileProfilingAggregate {
    tile_count: usize,
    total_tile_time: std::time::Duration,
    max_tile_time: std::time::Duration,
    total_alloc_time: std::time::Duration,
    max_alloc_time: std::time::Duration,
    total_triangle_time: std::time::Duration,
    max_triangle_time: std::time::Duration,
    total_coverage_and_interpolation_time: std::time::Duration,
    max_coverage_and_interpolation_time: std::time::Duration,
    total_shader_time: std::time::Duration,
    max_shader_time: std::time::Duration,
    total_depth_test_time: std::time::Duration,
    max_depth_test_time: std::time::Duration,
    total_write_time: std::time::Duration,
    max_write_time: std::time::Duration,
    total_fragments_tested: usize,
    max_fragments_tested: usize,
    total_fragments_passed: usize,
    max_fragments_passed: usize,
    total_depth_tests: usize,
    max_depth_tests: usize,
    total_successful_writes: usize,
    max_successful_writes: usize,
}

impl TileProfilingAggregate {
    fn add(&mut self, profiling: &TileProfiling) {
        self.tile_count += 1;
        self.total_tile_time += profiling.total_tile_time;
        self.max_tile_time = self.max_tile_time.max(profiling.total_tile_time);

        self.total_alloc_time += profiling.framebuffer_alloc_time;
        self.max_alloc_time = self.max_alloc_time.max(profiling.framebuffer_alloc_time);

        self.total_triangle_time += profiling.triangle_rasterisation_time;
        self.max_triangle_time = self
            .max_triangle_time
            .max(profiling.triangle_rasterisation_time);

        self.total_coverage_and_interpolation_time += profiling.coverage_and_interpolation_time;
        self.max_coverage_and_interpolation_time = self
            .max_coverage_and_interpolation_time
            .max(profiling.coverage_and_interpolation_time);

        self.total_shader_time += profiling.shader_time;
        self.max_shader_time = self.max_shader_time.max(profiling.shader_time);

        self.total_depth_test_time += profiling.depth_test_time;
        self.max_depth_test_time = self.max_depth_test_time.max(profiling.depth_test_time);

        self.total_write_time += profiling.write_time;
        self.max_write_time = self.max_write_time.max(profiling.write_time);

        self.total_fragments_tested += profiling.fragments_tested;
        self.max_fragments_tested = self.max_fragments_tested.max(profiling.fragments_tested);

        self.total_fragments_passed += profiling.fragments_passed;
        self.max_fragments_passed = self.max_fragments_passed.max(profiling.fragments_passed);

        self.total_depth_tests += profiling.depth_tests;
        self.max_depth_tests = self.max_depth_tests.max(profiling.depth_tests);

        self.total_successful_writes += profiling.successful_writes;
        self.max_successful_writes = self.max_successful_writes.max(profiling.successful_writes);
    }
}

fn render_tile(tile: Tile) -> TileResult {
    let total_start = std::time::Instant::now();

    let width = (tile.bounds.max_x - tile.bounds.min_x) as usize;
    let height = (tile.bounds.max_y - tile.bounds.min_y) as usize;

    let alloc_start = std::time::Instant::now();
    let mut framebuffer = FrameBuffer::new(width, height);
    let mut depthbuffer = DepthBuffer::new(width, height);
    let framebuffer_alloc_time = alloc_start.elapsed();

    let mut triangle_rasterisation_time = std::time::Duration::ZERO;
    let mut coverage_and_interpolation_time = std::time::Duration::ZERO;
    let mut shader_time = std::time::Duration::ZERO;
    let mut depth_test_time = std::time::Duration::ZERO;
    let mut write_time = std::time::Duration::ZERO;
    let mut fragments_tested = 0usize;
    let mut fragments_passed = 0usize;
    let mut depth_tests = 0usize;
    let mut successful_writes = 0usize;

    for triangle in tile.triangles {
        let triangle_start = std::time::Instant::now();
        let stats = triangle.rasterise(&mut framebuffer, &mut depthbuffer, tile.bounds);
        triangle_rasterisation_time += triangle_start.elapsed();

        coverage_and_interpolation_time += stats.coverage_and_interpolation_time;
        shader_time += stats.shader_time;
        depth_test_time += stats.depth_test_time;
        write_time += stats.write_time;
        fragments_tested += stats.fragments_tested;
        fragments_passed += stats.fragments_passed;
        depth_tests += stats.depth_tests;
        successful_writes += stats.successful_writes;
    }

    let total_tile_time = total_start.elapsed();

    TileResult {
        bounds: tile.bounds,
        framebuffer,
        profiling: TileProfiling {
            total_tile_time,
            framebuffer_alloc_time,
            triangle_rasterisation_time,
            coverage_and_interpolation_time,
            shader_time,
            depth_test_time,
            write_time,
            fragments_tested,
            fragments_passed,
            depth_tests,
            successful_writes,
        },
    }
}

/// A tile result is the output of rendering a single tile, used for merging into the main framebuffer.
struct TileResult {
    bounds: Rect,
    framebuffer: FrameBuffer,
    profiling: TileProfiling,
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
    fn tile_profiling_aggregate_tracks_totals() {
        let mut aggregate = TileProfilingAggregate::default();

        let sample = TileProfiling {
            total_tile_time: Duration::from_millis(4),
            framebuffer_alloc_time: Duration::from_millis(1),
            triangle_rasterisation_time: Duration::from_millis(2),
            coverage_and_interpolation_time: Duration::from_millis(1),
            shader_time: Duration::from_millis(1),
            depth_test_time: Duration::from_millis(1),
            write_time: Duration::from_millis(1),
            fragments_tested: 200,
            fragments_passed: 50,
            depth_tests: 200,
            successful_writes: 50,
        };

        aggregate.add(&sample);
        aggregate.add(&sample);

        assert_eq!(aggregate.tile_count, 2);
        assert_eq!(aggregate.total_fragments_tested, 400);
        assert_eq!(aggregate.total_fragments_passed, 100);
        assert_eq!(aggregate.max_fragments_tested, 200);
        assert_eq!(aggregate.max_fragments_passed, 50);
        assert_eq!(aggregate.total_depth_tests, 400);
        assert_eq!(aggregate.total_successful_writes, 100);
    }
}
